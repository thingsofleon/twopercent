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

from twopercent import store, strategies
from twopercent.features import feature_frame

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
    folds_run = 0

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
        probs = strategy.predict_proba(test)
        all_probs.append(probs)
        all_labels.append(test["did_2pct_next"])
        for _, day_rows in test.assign(prob=probs).groupby("target_date"):
            top = day_rows.nlargest(top_n, "prob")
            daily_hits.append(top["did_2pct_next"].mean())
        logger.info("fold %s..%s: %d train, %d test", month_start, month_end, len(train), len(test))

    if not all_probs:
        raise RuntimeError("no folds had enough data to benchmark")

    probs = pd.concat(all_probs)
    labels = pd.concat(all_labels).astype(int)
    base_rate = labels.mean()
    precision_at_n = float(pd.Series(daily_hits).mean())
    metrics = {
        "precision_at_n": round(precision_at_n, 4),
        "top_n": top_n,
        "base_rate": round(float(base_rate), 4),
        "lift": round(precision_at_n / base_rate, 3) if base_rate > 0 else None,
        "auc": round(float(roc_auc_score(labels, probs)), 4) if labels.nunique() > 1 else None,
        "brier": round(float(brier_score_loss(labels, probs)), 5),
        "test_rows": int(len(labels)),
        "test_days": len(daily_hits),
        "folds": folds_run,
    }
    if record:
        store.record_experiment(
            con,
            strategy=strategy_name,
            params={"months": months, "top_n": top_n},
            train_start=labeled["target_date"].min(),
            test_start=folds[0][0],
            test_end=folds[-1][1],
            metrics=metrics,
        )
    return metrics
