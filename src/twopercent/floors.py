"""The liquidity-floor study: does the edge survive a more tradeable universe?

The August paper measurement (ROADMAP "Paper trading") ended pointing here: the
edge does not clearly survive costs at the shipped selection, and "the edge
must get bigger, or the names more liquid" — a higher floor cuts per-name cost
AND cuts the universe the model picks from. This measures which effect wins.

NOT the referee, and not a strategy: the floor applies at SELECTION only (the
locked-in rule — training, labels and the AUC population never see it), so
every arm shares the SAME fitted model and probability vector per fold×seed,
and arms differ only in the eligibility predicate. That makes the sweep almost
free relative to a feature A/B, and perfectly paired: any per-day difference
between arms is the selection rule, nothing else. Records nothing; any change
to the shipped floor is a product PR with its own review — this module only
produces the evidence table (#121).

COST MODEL, stated bluntly: per-pick round-trip cost is ONE TICK over the
signal-day close (0.01/close), the ROADMAP's own convention. That is a LOWER
BOUND on real cost — the spread is at least a tick and slippage is extra — so
every net number here is OPTIMISTIC, and an arm that loses under this cost
model loses, period; an arm that wins under it has earned a harder look, not a
promotion.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence

import duckdb
import pandas as pd

from twopercent import ab, backtest, features, strategies
from twopercent.predict import LIQUIDITY_MIN_MEDIAN_VOLUME
from twopercent.scan import DEFAULT_THRESHOLD

logger = logging.getLogger(__name__)

BASELINE_ARM = f"shares>={LIQUIDITY_MIN_MEDIAN_VOLUME // 1000}k"
SHARE_FLOORS = (100_000, 250_000, 500_000, 1_000_000, 2_000_000)
DOLLAR_FLOORS = (1_000_000, 5_000_000, 20_000_000)
PRICE_FLOORS = (2.0, 5.0)
# One tick, round trip, over the signal close — see the module docstring.
TICK = 0.01


def _arms() -> dict[str, dict]:
    """Arm name -> predicate spec. Every arm is a candidate REPLACEMENT
    selection rule; the predicate is data, not code, so the report can print
    exactly what each arm filtered on."""
    arms: dict[str, dict] = {}
    for shares in SHARE_FLOORS:
        arms[f"shares>={shares // 1000}k"] = {"min_shares": shares}
    for dollars in DOLLAR_FLOORS:
        arms[f"dollars>={dollars // 1_000_000}M"] = {"min_dollars": dollars}
    for price in PRICE_FLOORS:
        # Price floors keep the shipped share floor: a price-only rule would
        # admit illiquid expensive names the current selection excludes, which
        # answers a question nobody asked.
        arms[f"price>=${price:g}+{LIQUIDITY_MIN_MEDIAN_VOLUME // 1000}k"] = {
            "min_shares": LIQUIDITY_MIN_MEDIAN_VOLUME,
            "min_price": price,
        }
    return arms


def _eligible(day_rows: pd.DataFrame, spec: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=day_rows.index)
    if "min_shares" in spec:
        mask &= day_rows["median_vol_20"] >= spec["min_shares"]
    if "min_dollars" in spec:
        mask &= day_rows["median_vol_20"] * day_rows["signal_close"] >= spec["min_dollars"]
    if "min_price" in spec:
        mask &= day_rows["signal_close"] >= spec["min_price"]
    # A row without a usable signal close cannot be cost-priced and cannot pass
    # a dollar/price arm; for pure share arms it stays eligible for PRECISION
    # (matching the shipped selection exactly) but is excluded from cost/net.
    return day_rows[mask]


def run_study(
    con: duckdb.DuckDBPyConnection,
    strategy_name: str,
    months: int = backtest.DEFAULT_TEST_MONTHS,
    top_n: int = backtest.DEFAULT_TOP_N,
    seeds: Sequence[int] = ab.DEFAULT_SEEDS,
    seed_param: str = ab.DEFAULT_SEED_PARAM,
) -> dict:
    """Sweep every arm over identical fitted models; paired stats vs baseline."""
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds: {list(seeds)}")
    arms = _arms()
    assert BASELINE_ARM in arms  # the shipped rule anchors every comparison

    frame = features.feature_frame(con)
    labeled = frame[frame["did_2pct_next"].notna()].copy()
    labeled["target_date"] = pd.to_datetime(labeled["target_date"]).dt.date
    labeled["signal_date"] = pd.to_datetime(labeled["signal_date"]).dt.date

    # Signal-day close, for tick cost and dollar volume. Guarded exactly like
    # the store's other float gates: a non-finite or non-positive close is a
    # missing price, never a divisor.
    closes = con.execute(
        "SELECT symbol, date AS signal_date, close AS signal_close FROM prices "
        "WHERE close > 0 AND isfinite(close)"
    ).df()
    closes["signal_date"] = pd.to_datetime(closes["signal_date"]).dt.date
    labeled = labeled.merge(closes, on=["symbol", "signal_date"], how="left")
    missing_close = int(labeled["signal_close"].isna().sum())
    if missing_close:
        logger.warning(
            "floors: %d of %d labeled rows have no usable signal close — they stay "
            "eligible for share arms' precision but carry NaN cost (excluded from "
            "cost/net means, counted per arm)",
            missing_close,
            len(labeled),
        )
    # Per-pick gross limit_2pct return (the paper ledger's rule) and tick cost.
    labeled["gross_limit"] = labeled["next_oc_return"].where(
        ~labeled["did_2pct_next"].astype(bool), DEFAULT_THRESHOLD
    )
    labeled["tick_cost"] = TICK / labeled["signal_close"]

    folds = backtest.month_folds(labeled["target_date"], months)
    # per arm/seed/day metric dicts
    metric_names = ("precision", "gross", "net", "cost_bps")
    per_day: dict[str, dict[int, dict[str, dict[dt.date, float]]]] = {
        a: {s: {m: {} for m in metric_names} for s in seeds} for a in arms
    }
    profile: dict[str, dict] = {
        a: {"picks": 0, "thin_days": 0, "unpriced_picks": 0, "prices": [], "dollars": []}
        for a in arms
    }
    folds_run = 0
    first_run_start: dt.date | None = None

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
        if first_run_start is None:
            first_run_start = month_start
        for seed in seeds:
            strategy = strategies.get(strategy_name, **{seed_param: seed})
            strategy.fit(train)
            probs = strategy.predict_proba(test)
            scored = test.assign(prob=probs)
            for target_date, day_rows in scored.groupby("target_date"):
                for arm, spec in arms.items():
                    eligible = _eligible(day_rows, spec)
                    if eligible.empty:
                        continue
                    top = eligible.nlargest(top_n, "prob")
                    bucket = per_day[arm][seed]
                    bucket["precision"][target_date] = float(top["did_2pct_next"].mean())
                    priced = top[top["signal_close"].notna()]
                    if seed == seeds[0]:
                        prof = profile[arm]
                        prof["picks"] += len(top)
                        prof["thin_days"] += int(len(top) < top_n)
                        prof["unpriced_picks"] += len(top) - len(priced)
                        prof["prices"].extend(priced["signal_close"].tolist())
                        prof["dollars"].extend(
                            (priced["signal_close"] * priced["median_vol_20"]).tolist()
                        )
                    if priced.empty:
                        continue
                    gross = float(priced["gross_limit"].mean())
                    cost = float(priced["tick_cost"].mean())
                    bucket["gross"][target_date] = gross
                    bucket["net"][target_date] = gross - cost
                    bucket["cost_bps"][target_date] = cost * 10_000
        logger.info("fold %s..%s: %d train, %d test", month_start, month_end, len(train), len(test))

    if not folds_run:
        raise RuntimeError("no folds had enough data to run")

    def seed_mean(arm: str, metric: str) -> dict[dt.date, float]:
        per_seed = [per_day[arm][s][metric] for s in seeds]
        keys = set.intersection(*(set(d) for d in per_seed))
        return {k: sum(d[k] for d in per_seed) / len(per_seed) for k in sorted(keys)}

    base = {m: seed_mean(BASELINE_ARM, m) for m in metric_names}
    base_rate = float(
        labeled.loc[labeled["target_date"] >= first_run_start, "did_2pct_next"].mean()
    )
    arm_rows: dict[str, dict] = {}
    for arm in arms:
        series = {m: seed_mean(arm, m) for m in metric_names}
        days = len(series["precision"])
        prof = profile[arm]
        prices = pd.Series(prof["prices"], dtype=float)
        dollars = pd.Series(prof["dollars"], dtype=float)
        row = {
            "predicate": arms[arm],
            # The seed-averaged per-day series the paired tests run on. Persisted
            # so the conclusion is re-testable — including with fold-clustered
            # errors (#119's lesson: days share a model within a month, so the
            # naive daily t overstates certainty) — without refitting for an hour.
            "by_day": {m: {k.isoformat(): v for k, v in series[m].items()} for m in metric_names},
            "days": days,
            "thin_days": prof["thin_days"],
            "unpriced_picks": prof["unpriced_picks"],
            "precision": (sum(series["precision"].values()) / days if days else None),
            "gross_daily": (
                sum(series["gross"].values()) / len(series["gross"]) if series["gross"] else None
            ),
            "net_daily": (
                sum(series["net"].values()) / len(series["net"]) if series["net"] else None
            ),
            "tick_cost_bps": (
                sum(series["cost_bps"].values()) / len(series["cost_bps"])
                if series["cost_bps"]
                else None
            ),
            "median_price": float(prices.median()) if len(prices) else None,
            "median_dollar_vol": float(dollars.median()) if len(dollars) else None,
        }
        if arm != BASELINE_ARM:
            for metric, key in (("precision", "precision_vs_baseline"), ("net", "net_vs_baseline")):
                shared = sorted(set(series[metric]) & set(base[metric]))
                row[key] = ab._paired_test([series[metric][k] - base[metric][k] for k in shared])
        arm_rows[arm] = row

    return {
        "strategy": strategy_name,
        "months": months,
        "top_n": top_n,
        "seeds": list(seeds),
        "folds": folds_run,
        "folds_requested": len(folds),
        "test_start": first_run_start.isoformat(),
        "test_end": folds[-1][1].isoformat(),
        "base_rate": base_rate,
        "baseline_arm": BASELINE_ARM,
        "cost_model": "one tick (0.01) round trip over signal close — a LOWER BOUND on cost",
        "labeled_rows": int(len(labeled)),
        "rows_without_close": missing_close,
        "arms": arm_rows,
        "multiplicity": {
            "non_baseline_arms": len(arms) - 1,
            "primary_readouts": 2,
            "note": (
                "every arm is always reported; adoption of any arm is a product PR "
                "with its own review, never an output of this study"
            ),
        },
    }


def format_report(result: dict) -> str:
    lines = [
        f"Liquidity-floor study — {result['strategy']}, top-{result['top_n']}, "
        f"{result['folds']} folds, {result['test_start']}..{result['test_end']}, "
        f"seeds {result['seeds']}",
        f"Cost model: {result['cost_model']} — net numbers are OPTIMISTIC.",
        f"All-names base rate {result['base_rate']:.4f} (floor-independent; labels never floored)",
        "",
        f"  {'arm':<18} {'days':>4} {'prec':>7} {'gross/d':>8} {'cost':>6} {'net/d':>8} "
        f"{'medPx':>7} {'med$vol':>9}  vs baseline (paired by day)",
    ]
    for arm, row in result["arms"].items():
        if not row["days"]:
            lines.append(
                f"  {arm:<18}    0 — no day had any eligible name (floor above the universe)"
            )
            continue
        cost = f"{row['tick_cost_bps']:.0f}bp" if row["tick_cost_bps"] is not None else "n/a"
        med_px = f"${row['median_price']:.2f}" if row["median_price"] is not None else "n/a"
        med_dv = (
            f"${row['median_dollar_vol'] / 1e6:.1f}M"
            if row["median_dollar_vol"] is not None
            else "n/a"
        )
        base = ""
        if "net_vs_baseline" in row:
            p = row["precision_vs_baseline"]
            nt = row["net_vs_baseline"]

            def fmt_p(test: dict) -> str:
                return "n/a" if test["p_t"] is None else f"{test['p_t']:.3f}"

            base = f"prec {p['mean']:+.4f} (p={fmt_p(p)}) net {nt['mean']:+.5f} (p={fmt_p(nt)})"
        elif arm == result["baseline_arm"]:
            base = "BASELINE (shipped)"
        gross = f"{row['gross_daily']:>8.5f}" if row["gross_daily"] is not None else "     n/a"
        net = f"{row['net_daily']:>8.5f}" if row["net_daily"] is not None else "     n/a"
        lines.append(
            f"  {arm:<18} {row['days']:>4} {row['precision']:>7.4f} "
            f"{gross} {cost:>6} {net} "
            f"{med_px:>7} {med_dv:>9}  {base}"
        )
    m = result["multiplicity"]
    lines += [
        "",
        f"Family: {m['non_baseline_arms']} non-baseline arms x {m['primary_readouts']} readouts. "
        + m["note"],
    ]
    return "\n".join(lines)
