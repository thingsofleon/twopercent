"""The referee: walk-forward benchmark harness.

Every strategy is scored on identical expanding-window monthly folds and the
same metrics, and every run is recorded in the experiments table. Strategies
must never influence this module — "better" is defined here and only here,
changed only by human-reviewed PR.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

import duckdb
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from twopercent import calibration, scan, store, strategies
from twopercent.features import feature_frame
from twopercent.predict import LIQUIDITY_MIN_MEDIAN_VOLUME

logger = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 10_000
DEFAULT_TEST_MONTHS = 12
DEFAULT_TOP_N = 20
RECORD_RANKS = 20  # per-day ranks persisted to experiment_daily for the SIM panel


def month_folds(target_dates: pd.Series, months: int) -> list[tuple[dt.date, dt.date]]:
    """The last `months` calendar months present, as (month_start, month_end)."""
    stamps = pd.to_datetime(target_dates.dropna().unique())
    periods = sorted(pd.PeriodIndex(stamps, freq="M").unique())
    return [(p.start_time.date(), p.end_time.date()) for p in periods[-months:]]


def latest_standard_experiment(
    con: duckdb.DuckDBPyConnection,
    strategy: str,
    months: int = DEFAULT_TEST_MONTHS,
    top_n: int = DEFAULT_TOP_N,
) -> tuple[int, dict, dt.date | None, dt.date | None] | None:
    """The strategy's newest recorded standard (`months`, `top_n`) DEFAULT-CONFIG
    benchmark, as (id, metrics, test_start, test_end).

    Rows with non-empty strategy_params are research variants recorded under
    the strategy's name — they must never be quoted as the strategy's own
    benchmark (pre-research rows lack the key entirely; that counts as
    default-config). Shared reader: research's champion reference and the
    signal email both quote exactly this row.

    Touch era ONLY (M1): filters event = TOUCH_EVENT, so a close-era archive row
    is never quoted, compared, or promoted against a touch challenger. Returns
    None until a champion re-benchmark under touch is recorded (required
    post-merge step) — callers degrade gracefully rather than fall back to close.
    """
    rows = con.execute(
        "SELECT id, params, metrics, test_start, test_end FROM experiments "
        "WHERE strategy = ? AND event = ? ORDER BY run_ts DESC, id DESC",
        [strategy, scan.TOUCH_EVENT],
    ).fetchall()
    for exp_id, params_json, metrics_json, test_start, test_end in rows:
        try:
            params = json.loads(params_json) if params_json else {}
            metrics = json.loads(metrics_json)
        except json.JSONDecodeError:
            logger.warning("experiments row #%s has unparseable JSON — ignored", exp_id)
            continue
        if params.get("strategy_params"):
            continue  # parameterized variant, not the strategy's own config
        if params.get("months") == months and params.get("top_n") == top_n:
            return exp_id, metrics, test_start, test_end
    return None


def run_benchmark(
    con: duckdb.DuckDBPyConnection,
    strategy_name: str,
    months: int = DEFAULT_TEST_MONTHS,
    top_n: int = DEFAULT_TOP_N,
    record: bool = True,
    strategy_params: dict | None = None,
) -> dict:
    """Walk-forward benchmark; returns metrics and records an experiments row.

    strategy_params are passed to the strategy constructor on every fold's
    fresh instance and recorded in the experiments params JSON under
    "strategy_params" — the key the research runner matches queue configs
    against, so a recorded config is never re-run.
    """
    frame = feature_frame(con)
    labeled = frame[frame["did_2pct_next"].notna()].copy()
    labeled["target_date"] = pd.to_datetime(labeled["target_date"]).dt.date

    folds = month_folds(labeled["target_date"], months)
    all_probs: list[pd.Series] = []
    all_labels: list[pd.Series] = []
    all_dates: list[pd.Series] = []
    daily_hits: list[float] = []
    fold_drops: dict[dt.date, frozenset[str]] = {}
    folds_run = 0
    floored_row_days = 0
    unscoreable_days = 0
    daily_picks: list[tuple[dt.date, int, float]] = []
    rank_rows: list[tuple[dt.date, int, float, int, float, float]] = []
    pick_probs: list[pd.Series] = []
    pick_labels: list[pd.Series] = []
    first_run_start: dt.date | None = None

    for month_start, month_end in folds:
        train = labeled[labeled["target_date"] < month_start]
        test = labeled[
            (labeled["target_date"] >= month_start) & (labeled["target_date"] <= month_end)
        ]
        if len(train) < MIN_TRAIN_ROWS or test.empty:
            logger.warning(
                "fold %s skipped: %d train / %d test rows", month_start, len(train), len(test)
            )
            continue
        folds_run += 1
        if first_run_start is None:
            # Skipped folds must not be claimed: the recorded test_start is the
            # first fold that actually RAN, not the first fold requested.
            first_run_start = month_start
        strategy = strategies.get(strategy_name, **(strategy_params or {}))
        strategy.fit(train)
        fold_drops[month_start] = frozenset(getattr(strategy, "dropped_columns", ()))
        probs = strategy.predict_proba(test)
        all_probs.append(probs)
        all_labels.append(test["did_2pct_next"])
        # Per-row target_date rides alongside the (prob, label) pair the AUC/Brier
        # already pool, so calibration keys the daily-base-rate BSS reference and
        # the regime strata to the SAME out-of-sample population — no lookahead.
        all_dates.append(test["target_date"])
        for target_date, day_rows in test.assign(prob=probs).groupby("target_date"):
            # Same liquidity floor the shipped predictions apply (predict.py):
            # only the top-N SELECTION filters — training and the AUC/brier
            # populations above stay all-names, matching the label definition.
            eligible = day_rows[day_rows["median_vol_20"] >= LIQUIDITY_MIN_MEDIAN_VOLUME]
            floored_row_days += len(day_rows) - len(eligible)
            if eligible.empty:
                unscoreable_days += 1
                continue
            top = eligible.nlargest(top_n, "prob")
            daily_hits.append(top["did_2pct_next"].mean())
            # Calibration on the population the user ACTS on: the same top-N
            # liquid picks that drive precision_at_n, out-of-sample per fold.
            # A pick called X% should reach ~X% — the all-names view answers a
            # different question (ranking discrimination) and buries this one.
            pick_probs.append(top["prob"])
            pick_labels.append(top["did_2pct_next"])
            # One frame drives both the recorded per-rank rows (experiment_daily,
            # for the dashboard trailing-window reach explorer + strategy
            # explorer) and the top-1/top-5 reach aggregates (rank 1 row, mean of
            # ranks 1-5). Per rank the target day's oh/ol/oc outcome returns are
            # persisted so any exit rule replays deterministically from the
            # stored row; no $-growth is computed HERE (the explorer's P&L view
            # is an explicit opt-in, never a benchmark headline).
            top20 = eligible.nlargest(RECORD_RANKS, "prob")
            for rank, row in enumerate(top20.itertuples(), start=1):
                rank_rows.append(
                    (
                        target_date,
                        rank,
                        float(row.next_oc_return),
                        int(row.did_2pct_next),
                        float(row.next_high_return),
                        float(row.next_low_return),
                    )
                )
            top5 = top20.iloc[:5]
            top1 = top5.iloc[0]
            daily_picks.append(
                (
                    target_date,
                    int(top1["did_2pct_next"]),
                    float(top5["did_2pct_next"].mean()),
                )
            )
        logger.info("fold %s..%s: %d train, %d test", month_start, month_end, len(train), len(test))

    if not all_probs:
        raise RuntimeError("no folds had enough data to benchmark")
    if not daily_hits:
        raise RuntimeError("every test day fell below the liquidity floor — no top-N to score")
    if floored_row_days:
        logger.warning(
            "top-N selection excluded %d row-days below the %d-share liquidity floor "
            "across %d test days (%d days had no eligible names at all; training and "
            "AUC/brier populations keep all names)",
            floored_row_days,
            LIQUIDITY_MIN_MEDIAN_VOLUME,
            len(daily_hits) + unscoreable_days,
            unscoreable_days,
        )

    dropped_columns = sorted(set().union(*fold_drops.values()))
    if len(set(fold_drops.values())) > 1:
        logger.warning(
            "benchmark mixed structurally different fits — dropped feature columns "
            "differ across folds: %s",
            "; ".join(
                f"{start}: {', '.join(sorted(cols)) or 'none'}"
                for start, cols in sorted(fold_drops.items())
            ),
        )

    probs = pd.concat(all_probs)
    labels = pd.concat(all_labels).astype(int)
    dates = pd.concat(all_dates)
    base_rate = labels.mean()
    precision_at_n = float(pd.Series(daily_hits).mean())
    picks = pd.DataFrame(daily_picks, columns=["target_date", "top1_hit", "top5_hits"])
    metrics = {
        # Reach metrics only (Stage B): this app reports how well the ranking
        # predicts a +2% intraday touch, never trading P&L. Champion selection
        # keys on lift/auc — base-rate-invariant and regime-robust.
        "precision_at_n": round(precision_at_n, 4),
        "top_n": top_n,
        "base_rate": round(float(base_rate), 4),
        "lift": round(precision_at_n / base_rate, 3) if base_rate > 0 else None,
        "auc": round(float(roc_auc_score(labels, probs)), 4) if labels.nunique() > 1 else None,
        "brier": round(float(brier_score_loss(labels, probs)), 5),
        "precision_at_1": round(float(picks["top1_hit"].mean()), 4),
        "precision_at_5": round(float(picks["top5_hits"].mean()), 4),
        "test_rows": int(len(labels)),
        "test_days": len(daily_hits),
        "folds": folds_run,
        # Walk-forward calibration. Two populations, two questions:
        #   - all-names (same rows as brier/auc): reliability + Brier SKILL score
        #     vs the daily base rate + per-regime — ranking discrimination.
        #   - picks (top-N liquid, nested under "picks"): reliability of the
        #     numbers the user acts on — a pick called X% should reach ~X%.
        #     No BSS on picks (post-selection BSS is a misleading artifact).
        # MEASURED only — Stage C does not recalibrate the model.
        "calibration": {
            **calibration.calibration_report(probs.to_numpy(), labels.to_numpy(), dates),
            "picks": calibration.pick_calibration_report(
                pd.concat(pick_probs).to_numpy(),
                pd.concat(pick_labels).to_numpy(),
                top_n=top_n,
                n_days=len(daily_hits),
            ),
        },
    }
    if record:
        params = {
            "months": months,
            "top_n": top_n,
            "selection": "liquidity_floor_100k",
            "dropped_columns": dropped_columns,
            "strategy_params": strategy_params or {},
        }
        # Which device actually trained (strategies expose resolved_device
        # when it matters, e.g. xgb's CUDA probe). A sibling of
        # strategy_params on purpose: config done-matching keys on
        # strategy_params only, so CPU-vs-GPU never changes config identity.
        device = getattr(strategy, "resolved_device", None)
        if device is not None:
            params["device"] = device
        # Both records must land atomically: an experiments row without its
        # daily rows would be counted "done" by the research queue forever
        # while the sim-panel data it promises is missing.
        con.execute("BEGIN")
        try:
            seq = store.record_experiment(
                con,
                strategy=strategy_name,
                params=params,
                train_start=labeled["target_date"].min(),
                test_start=first_run_start,
                test_end=folds[-1][1],
                metrics=metrics,
                event=scan.TOUCH_EVENT,
            )
            # Per-day per-rank pick outcomes land in experiment_daily (dashboard
            # SIM panel), never in the metrics JSON above.
            rank_frame = pd.DataFrame(
                rank_rows, columns=["target_date", "rank", "ret", "hit", "oh", "ol"]
            )
            store.record_experiment_daily(con, seq, rank_frame)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
    return metrics
