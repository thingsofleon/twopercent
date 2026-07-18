"""The referee: walk-forward benchmark harness.

Every strategy is scored on identical expanding-window monthly folds and the
same metrics, and every run is recorded in the experiments table. Strategies
must never influence this module — "better" is defined here and only here,
changed only by human-reviewed PR.
"""

from __future__ import annotations

import datetime as dt
import logging

import duckdb
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from twopercent import store, strategies, track
from twopercent.features import feature_frame
from twopercent.predict import LIQUIDITY_MIN_MEDIAN_VOLUME

logger = logging.getLogger(__name__)

MIN_TRAIN_ROWS = 10_000
DEFAULT_TEST_MONTHS = 12
DEFAULT_TOP_N = 20


def month_folds(target_dates: pd.Series, months: int) -> list[tuple[dt.date, dt.date]]:
    """The last `months` calendar months present, as (month_start, month_end)."""
    stamps = pd.to_datetime(target_dates.dropna().unique())
    periods = sorted(pd.PeriodIndex(stamps, freq="M").unique())
    return [(p.start_time.date(), p.end_time.date()) for p in periods[-months:]]


def run_benchmark(
    con: duckdb.DuckDBPyConnection,
    strategy_name: str,
    months: int = DEFAULT_TEST_MONTHS,
    top_n: int = DEFAULT_TOP_N,
    record: bool = True,
) -> dict:
    """Walk-forward benchmark; returns metrics and records an experiments row."""
    frame = feature_frame(con)
    labeled = frame[frame["did_2pct_next"].notna()].copy()
    labeled["target_date"] = pd.to_datetime(labeled["target_date"]).dt.date

    folds = month_folds(labeled["target_date"], months)
    all_probs: list[pd.Series] = []
    all_labels: list[pd.Series] = []
    daily_hits: list[float] = []
    fold_drops: dict[dt.date, frozenset[str]] = {}
    folds_run = 0
    floored_row_days = 0
    unscoreable_days = 0
    daily_picks: list[tuple[float, int, float, float]] = []

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
        strategy = strategies.get(strategy_name)
        strategy.fit(train)
        fold_drops[month_start] = frozenset(getattr(strategy, "dropped_columns", ()))
        probs = strategy.predict_proba(test)
        all_probs.append(probs)
        all_labels.append(test["did_2pct_next"])
        for _, day_rows in test.assign(prob=probs).groupby("target_date"):
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
            top5 = eligible.nlargest(5, "prob")
            top1 = top5.iloc[0]
            daily_picks.append(
                (
                    float(top1["next_oc_return"]),
                    int(top1["did_2pct_next"]),
                    float(top5["next_oc_return"].mean()),
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
    base_rate = labels.mean()
    precision_at_n = float(pd.Series(daily_hits).mean())
    picks = pd.DataFrame(daily_picks, columns=["top1_ret", "top1_hit", "top5_ret", "top5_hits"])
    sim_top1 = float((1 + picks["top1_ret"] - track.COST_ROUND_TRIP).prod())
    sim_top5 = float((1 + picks["top5_ret"] - track.COST_ROUND_TRIP).prod())
    metrics = {
        "precision_at_n": round(precision_at_n, 4),
        "top_n": top_n,
        "base_rate": round(float(base_rate), 4),
        "lift": round(precision_at_n / base_rate, 3) if base_rate > 0 else None,
        "auc": round(float(roc_auc_score(labels, probs)), 4) if labels.nunique() > 1 else None,
        "brier": round(float(brier_score_loss(labels, probs)), 5),
        "precision_at_1": round(float(picks["top1_hit"].mean()), 4),
        "precision_at_5": round(float(picks["top5_hits"].mean()), 4),
        # Growth of $1 trading the daily pick(s) open-to-close over the whole
        # test window, net of track.COST_ROUND_TRIP per day. An execution
        # upper bound (assumed costs, perfect fills at open/close) — see
        # track.py for the cost caveat.
        "sim_top1_growth": round(sim_top1, 4),
        "sim_top5_growth": round(sim_top5, 4),
        "test_rows": int(len(labels)),
        "test_days": len(daily_hits),
        "folds": folds_run,
    }
    if record:
        store.record_experiment(
            con,
            strategy=strategy_name,
            params={
                "months": months,
                "top_n": top_n,
                "selection": "liquidity_floor_100k",
                "dropped_columns": dropped_columns,
            },
            train_start=labeled["target_date"].min(),
            test_start=folds[0][0],
            test_end=folds[-1][1],
            metrics=metrics,
        )
    return metrics
