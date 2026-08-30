"""Paired A/B between two feature sets, on identical rows and identical folds.

NOT the referee. `backtest.run_benchmark` decides whether a STRATEGY is better
and is the only thing allowed to record an experiments row; this module decides
whether a set of COLUMNS pays, and records nothing. Its numbers must never be
quoted as a benchmark or entered into the promotion gauntlet — the arms here
can run over a restricted era, which makes them incomparable to the standard
window by construction.

Why it exists as code rather than a script: the first two attempts to answer
that question (#110's six price features, #115's four intraday ones) were both
decided by ad-hoc runs, and both first reported a wrong answer for the same
reason — an UNPAIRED ruler. A difference of arm means was compared against the
spread of 3 random seeds, which measures model-fit variance, not sampling
variance, and the range of 3 iid draws is ~1.69 sigma, so the comparison is
biased toward "no effect". Pairing is the whole point:

  * ONE feature frame, ONE fold list, ONE row set. The arms differ in the
    column list handed to the strategy and in nothing else.
  * AUC is paired by FOLD (n = folds) and precision@N by DAY (n = test days),
    because that is the unit each metric is actually computed over.
  * Seeds are averaged WITHIN a fold/day before pairing, so seed noise shrinks
    the paired quantity instead of inflating a spread it is compared against.

Everything the report needs to be read honestly rides in the result: the
multiplicity exposure, the observed coverage of the columns under test, and
whether the strategy ignored the seed.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import statistics
from collections.abc import Mapping, Sequence

import duckdb
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from twopercent import backtest, features, strategies
from twopercent.predict import LIQUIDITY_MIN_MEDIAN_VOLUME

logger = logging.getLogger(__name__)

DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_SEED_PARAM = "random_state"
# Two primary tests are reported (AUC and precision@N), so a nominal 0.05 is a
# 0.025 threshold. Stated in the result rather than left for the reader to
# remember: #115's "p = 0.039" was best-of-two with no correction, and reads as
# significant only while the second test is out of frame.
PRIMARY_TESTS = 2
NOMINAL_ALPHA = 0.05


def _paired_test(deltas: Sequence[float], alpha: float = NOMINAL_ALPHA / PRIMARY_TESTS) -> dict:
    """Paired-sample summary of one arm-vs-arm difference vector.

    Both a t-test and an exact sign test, because they fail differently: the t
    is sensitive to a few large folds, the sign test to none of the magnitudes.
    Reporting only the smaller of the two would be the same best-of-two error
    this module exists to stop, so both are always present.

    `mde_80` is the smallest true effect this many pairs at this much spread
    would have caught 80% of the time, at `alpha` two-sided (normal
    approximation). Without it "no significant difference" and "no difference"
    are indistinguishable in the output — and #115's answer was wrong precisely
    because an underpowered null read as a verdict. A measured delta far below
    mde_80 means the test was not able to settle the question, whichever way it
    came out.
    """
    values = [float(d) for d in deltas]
    n = len(values)
    positive = sum(1 for d in values if d > 0)
    negative = sum(1 for d in values if d < 0)
    zeros = n - positive - negative
    sd = float(statistics.stdev(values)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    result = {
        "n": n,
        "mean": float(sum(values) / n) if n else float("nan"),
        "sd": sd,
        "se": se,
        "alpha": alpha,
        "mde_80": (
            float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(0.80)) * se)
            if n > 1 and sd > 0
            else None
        ),
        "positive": positive,
        "negative": negative,
        "zeros": zeros,
        "t": None,
        "p_t": None,
        "p_sign": None,
    }
    if positive + negative:
        # Ties carry no directional information, so they leave the denominator
        # rather than counting as evidence against the effect.
        result["p_sign"] = float(stats.binomtest(positive, positive + negative, 0.5).pvalue)
    if n < 2 or all(math.isclose(d, values[0]) for d in values):
        # A constant difference vector has zero paired variance: the t statistic
        # is undefined (or infinite), not "extremely significant". The SIGN test
        # above is still perfectly well defined on it — dropping both would
        # throw away the one number that constant vector does support.
        return result
    # A one-sample t on the DIFFERENCES is the paired t-test; taking it here
    # keeps the two arms' vectors from ever being length-mismatched silently.
    t_stat, p_value = stats.ttest_1samp(values, 0.0)
    result["t"] = float(t_stat)
    result["p_t"] = float(p_value)
    return result


def _column_coverage(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    """Observed (non-null) fraction of each column over `frame`."""
    n = len(frame)
    return {col: (float(frame[col].notna().sum()) / n if n else 0.0) for col in columns}


def run_ab(
    con: duckdb.DuckDBPyConnection,
    arms: Mapping[str, Sequence[str]],
    strategy_name: str,
    months: int = backtest.DEFAULT_TEST_MONTHS,
    top_n: int = backtest.DEFAULT_TOP_N,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    train_start: dt.date | None = None,
    seed_param: str = DEFAULT_SEED_PARAM,
    strategy_params: Mapping[str, object] | None = None,
) -> dict:
    """Run every arm over the same folds and seeds; return paired statistics.

    `arms` maps an arm name to its feature columns (order preserved; the first
    arm is the reference every other arm is differenced against). `train_start`
    drops all rows whose target_date precedes it — the way an arm is confined to
    an era in which a column is actually observed. It moves the training window
    for EVERY arm, so the arms stay comparable to each other and stop being
    comparable to the standard-window benchmark.
    """
    if len(arms) < 2:
        raise ValueError("an A/B needs at least two arms")
    arm_names = list(arms)
    resolved = {name: list(cols) for name, cols in arms.items()}
    for name, cols in resolved.items():
        # Same whitelist the strategies enforce, applied here so a typo costs a
        # second rather than a fold's training time.
        strategies.resolve_feature_columns(f"arm {name!r}", cols)
    if len({tuple(cols) for cols in resolved.values()}) < len(resolved):
        raise ValueError("two arms have identical feature columns — the A/B would compare nothing")

    shared = set.intersection(*(set(cols) for cols in resolved.values()))
    under_test = sorted(set().union(*(set(cols) for cols in resolved.values())) - shared)
    if not under_test:
        raise ValueError("arms differ by no column")

    frame = features.feature_frame(con)
    labeled = frame[frame["did_2pct_next"].notna()].copy()
    labeled["target_date"] = pd.to_datetime(labeled["target_date"]).dt.date
    if train_start is not None:
        before = len(labeled)
        labeled = labeled[labeled["target_date"] >= train_start]
        logger.warning(
            "train_start %s drops %d of %d labeled rows — these arms are NOT comparable to "
            "a standard-window benchmark, only to each other",
            train_start,
            before - len(labeled),
            before,
        )
    if labeled.empty:
        raise RuntimeError("no labeled rows survive train_start — nothing to run")

    coverage = _column_coverage(labeled, under_test)
    blind = sorted(col for col, frac in coverage.items() if frac == 0.0)
    if blind:
        # Every strategy drops an all-NaN column (it crashes HistGBM's binner),
        # so an arm distinguished only by such columns is byte-identical to the
        # reference and the A/B would report "no difference" as a finding.
        raise RuntimeError(
            "column(s) under test have ZERO observed values over the selected rows, so the "
            f"arms would train identically: {', '.join(blind)}"
        )
    logger.info(
        "columns under test, observed fraction over %d selected rows: %s",
        len(labeled),
        ", ".join(f"{col} {frac:.3f}" for col, frac in sorted(coverage.items())),
    )

    folds = backtest.month_folds(labeled["target_date"], months)
    # fold_auc[arm][seed] -> {fold_start: auc}; day_precision[arm][seed] -> {day: hit rate}
    fold_auc: dict[str, dict[int, dict[dt.date, float]]] = {a: {s: {} for s in seeds} for a in arms}
    day_precision: dict[str, dict[int, dict[dt.date, float]]] = {
        a: {s: {} for s in seeds} for a in arms
    }
    pooled: dict[str, dict[int, list[tuple[pd.Series, pd.Series]]]] = {
        a: {s: [] for s in seeds} for a in arms
    }
    dropped: dict[str, set[str]] = {a: set() for a in arms}
    train_coverage: dict[str, float] = {col: 1.0 for col in under_test}
    folds_run = 0
    floored_row_days = 0
    unscoreable_days = 0

    for month_start, month_end in folds:
        train = labeled[labeled["target_date"] < month_start]
        test = labeled[
            (labeled["target_date"] >= month_start) & (labeled["target_date"] <= month_end)
        ]
        if len(train) < backtest.MIN_TRAIN_ROWS or test.empty:
            logger.warning(
                "fold %s skipped: %d train / %d test rows", month_start, len(train), len(test)
            )
            continue
        folds_run += 1
        for col, frac in _column_coverage(train, under_test).items():
            train_coverage[col] = min(train_coverage[col], frac)
        for arm in arm_names:
            for seed in seeds:
                params = {
                    **(strategy_params or {}),
                    seed_param: seed,
                    "feature_columns": resolved[arm],
                }
                strategy = strategies.get(strategy_name, **params)
                strategy.fit(train)
                dropped[arm].update(getattr(strategy, "dropped_columns", ()))
                probs = strategy.predict_proba(test)
                labels = test["did_2pct_next"].astype(int)
                pooled[arm][seed].append((probs, labels))
                if labels.nunique() > 1:
                    fold_auc[arm][seed][month_start] = float(roc_auc_score(labels, probs))
                for target_date, day_rows in test.assign(prob=probs).groupby("target_date"):
                    # The referee's selection rule, reused verbatim: the shipped
                    # liquidity floor applies at SELECTION only, never to
                    # training or to the AUC population.
                    eligible = day_rows[day_rows["median_vol_20"] >= LIQUIDITY_MIN_MEDIAN_VOLUME]
                    if arm == arm_names[0] and seed == seeds[0]:
                        floored_row_days += len(day_rows) - len(eligible)
                    if eligible.empty:
                        if arm == arm_names[0] and seed == seeds[0]:
                            unscoreable_days += 1
                        continue
                    top = eligible.nlargest(top_n, "prob")
                    day_precision[arm][seed][target_date] = float(top["did_2pct_next"].mean())
        logger.info(
            "fold %s..%s: %d train, %d test (%d arms x %d seeds)",
            month_start,
            month_end,
            len(train),
            len(test),
            len(arm_names),
            len(seeds),
        )

    if not folds_run:
        raise RuntimeError("no folds had enough data to run")
    for arm in arm_names:
        leaked = dropped[arm] & set(under_test)
        if leaked:
            logger.warning(
                "arm %r dropped column(s) under test in at least one fold (all-NaN in that "
                "fold's training rows): %s — that fold compared fewer features than the arm "
                "claims",
                arm,
                ", ".join(sorted(leaked)),
            )
    if floored_row_days:
        logger.warning(
            "top-N selection excluded %d row-days below the %d-share liquidity floor "
            "(%d days had no eligible names at all)",
            floored_row_days,
            LIQUIDITY_MIN_MEDIAN_VOLUME,
            unscoreable_days,
        )

    seed_blind = [
        arm
        for arm in arm_names
        if len(seeds) > 1 and len({tuple(sorted(fold_auc[arm][s].items())) for s in seeds}) == 1
    ]
    if seed_blind:
        logger.warning(
            "strategy %s ignored %r — every seed produced identical folds for arm(s) %s, so the "
            "seed averaging below removes no noise and the paired n is smaller than it looks",
            strategy_name,
            seed_param,
            ", ".join(seed_blind),
        )

    def seed_mean(per_seed: dict[int, dict[dt.date, float]]) -> dict[dt.date, float]:
        keys = set.intersection(*(set(v) for v in per_seed.values()))
        return {k: sum(per_seed[s][k] for s in per_seed) / len(per_seed) for k in sorted(keys)}

    arm_summary: dict[str, dict] = {}
    auc_by_fold: dict[str, dict[dt.date, float]] = {}
    precision_by_day: dict[str, dict[dt.date, float]] = {}
    base_rate = float(pd.concat([lab for _, lab in pooled[arm_names[0]][seeds[0]]]).mean())
    for arm in arm_names:
        auc_by_fold[arm] = seed_mean(fold_auc[arm])
        precision_by_day[arm] = seed_mean(day_precision[arm])
        pooled_auc = []
        for seed in seeds:
            probs = pd.concat([p for p, _ in pooled[arm][seed]])
            labels = pd.concat([lab for _, lab in pooled[arm][seed]])
            if labels.nunique() > 1:
                pooled_auc.append(float(roc_auc_score(labels, probs)))
        precision = (
            sum(precision_by_day[arm].values()) / len(precision_by_day[arm])
            if precision_by_day[arm]
            else float("nan")
        )
        arm_summary[arm] = {
            "auc_by_fold": {k.isoformat(): v for k, v in auc_by_fold[arm].items()},
            "precision_by_day": {k.isoformat(): v for k, v in precision_by_day[arm].items()},
            "columns": resolved[arm],
            "n_columns": len(resolved[arm]),
            "feature_set": features.feature_set_version(resolved[arm]),
            "auc": sum(pooled_auc) / len(pooled_auc) if pooled_auc else None,
            "precision_at_n": precision,
            "lift": precision / base_rate if base_rate > 0 else None,
            "dropped_columns": sorted(dropped[arm]),
        }

    reference = arm_names[0]
    comparisons: dict[str, dict] = {}
    for arm in arm_names[1:]:
        fold_keys = sorted(set(auc_by_fold[reference]) & set(auc_by_fold[arm]))
        day_keys = sorted(set(precision_by_day[reference]) & set(precision_by_day[arm]))
        auc_deltas = [auc_by_fold[arm][k] - auc_by_fold[reference][k] for k in fold_keys]
        precision_deltas = [
            precision_by_day[arm][k] - precision_by_day[reference][k] for k in day_keys
        ]
        comparisons[arm] = {
            "vs": reference,
            "auc_paired_by_fold": _paired_test(auc_deltas),
            "precision_paired_by_day": _paired_test(precision_deltas),
            # The vectors the tests above ran on. A conclusion nobody can re-test
            # without a fresh hour of fitting is a conclusion nobody re-tests.
            "auc_deltas_by_fold": {
                k.isoformat(): d for k, d in zip(fold_keys, auc_deltas, strict=True)
            },
            "precision_deltas_by_day": {
                k.isoformat(): d for k, d in zip(day_keys, precision_deltas, strict=True)
            },
            "auc_delta": arm_summary[arm]["auc"] - arm_summary[reference]["auc"],
            "precision_delta": (
                arm_summary[arm]["precision_at_n"] - arm_summary[reference]["precision_at_n"]
            ),
            "lift_delta": arm_summary[arm]["lift"] - arm_summary[reference]["lift"],
        }

    return {
        "strategy": strategy_name,
        "months": months,
        "top_n": top_n,
        "seeds": list(seeds),
        "seed_param": seed_param,
        "seed_ignored_by": seed_blind,
        "train_start": train_start.isoformat() if train_start else None,
        "folds": folds_run,
        "test_days": len(precision_by_day[reference]),
        "test_start": folds[0][0].isoformat(),
        "test_end": folds[-1][1].isoformat(),
        "labeled_rows": int(len(labeled)),
        "base_rate": base_rate,
        "columns_under_test": under_test,
        "coverage_all_rows": coverage,
        "coverage_min_fold_train": train_coverage,
        "arms": arm_summary,
        "comparisons": comparisons,
        "multiplicity": {
            "primary_tests": PRIMARY_TESTS,
            "nominal_alpha": NOMINAL_ALPHA,
            "bonferroni_alpha": NOMINAL_ALPHA / PRIMARY_TESTS,
        },
    }


def format_report(result: dict) -> str:
    """The result as a readable block — the numbers plus what qualifies them."""
    lines: list[str] = []
    era = result["train_start"] or "all history"
    lines.append(
        f"A/B {result['strategy']}: {result['folds']} folds, {result['test_days']} test days, "
        f"{result['test_start']}..{result['test_end']}, train from {era}, "
        f"seeds {result['seeds']} (averaged within fold/day before pairing)"
    )
    lines.append(f"base rate {result['base_rate']:.4f} (identical across arms — same rows)")
    lines.append("")
    lines.append("Columns under test, observed fraction:")
    for col in result["columns_under_test"]:
        lines.append(
            f"  {col:<22} {result['coverage_all_rows'][col]:.3f} over all selected rows, "
            f"{result['coverage_min_fold_train'][col]:.3f} in the thinnest fold's training rows"
        )
    lines.append("")
    width = max(len(a) for a in result["arms"])
    lines.append(f"  {'arm':<{width}}  {'cols':>4}  {'AUC':>8}  {'p@N':>8}  {'lift':>7}")
    for arm, summary in result["arms"].items():
        lines.append(
            f"  {arm:<{width}}  {summary['n_columns']:>4}  {summary['auc']:>8.5f}  "
            f"{summary['precision_at_n']:>8.5f}  {summary['lift']:>7.4f}"
        )
    for arm, comp in result["comparisons"].items():
        lines.append("")
        lines.append(f"{arm} vs {comp['vs']}:")
        for label, key in (
            ("AUC, paired by FOLD", "auc_paired_by_fold"),
            ("p@N, paired by DAY", "precision_paired_by_day"),
        ):
            test = comp[key]
            p_t = "n/a" if test["p_t"] is None else f"{test['p_t']:.4f}"
            p_sign = "n/a" if test["p_sign"] is None else f"{test['p_sign']:.4f}"
            mde = "n/a" if test["mde_80"] is None else f"{test['mde_80']:.5f}"
            lines.append(
                f"  {label:<20} n={test['n']:<4} delta {test['mean']:+.5f}  "
                f"positive {test['positive']}/{test['n']}  p(t)={p_t}  p(sign)={p_sign}  "
                f"detectable at 80% power: {mde}"
            )
        lines.append(f"  lift {comp['lift_delta']:+.4f}")
    alpha = result["multiplicity"]
    lines.append("")
    lines.append(
        f"Multiplicity: {alpha['primary_tests']} primary tests reported, so a nominal "
        f"{alpha['nominal_alpha']} is {alpha['bonferroni_alpha']} Bonferroni-corrected. "
        "These arms are not a benchmark and nothing here is recorded."
    )
    if result["seed_ignored_by"]:
        lines.append(
            f"WARNING: {result['seed_param']} had no effect on arm(s) "
            f"{', '.join(result['seed_ignored_by'])} — seed averaging removed no noise."
        )
    return "\n".join(lines)
