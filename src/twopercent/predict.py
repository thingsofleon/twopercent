"""Shared prediction logic: train a strategy, score a signal date, log results."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import duckdb
import pandas as pd

from twopercent import store, strategies
from twopercent.features import feature_frame

logger = logging.getLogger(__name__)


@dataclass
class PredictResult:
    strategy: str
    signal_date: dt.date
    scored: pd.DataFrame  # feature columns + prob + rank, sorted by prob desc
    trained_rows: int


def predict_for(
    con: duckdb.DuckDBPyConnection,
    strategy_name: str,
    signal_date: dt.date | None = None,
    save: bool = True,
) -> PredictResult:
    """Score every symbol for the trading day after `signal_date`.

    Defaults to the latest date in the store. For a PAST signal date
    (track-record backfill), training uses only outcomes with
    target_date <= signal_date — what was knowable at that day's close —
    so backfilled predictions stay walk-forward honest.
    """
    frame = feature_frame(con)
    if frame.empty:
        raise ValueError("no feature rows — is the store ingested?")
    frame = frame.assign(
        signal_date=pd.to_datetime(frame["signal_date"]).dt.date,
        target_date=pd.to_datetime(frame["target_date"]).dt.date,
    )
    signal_date = signal_date or frame["signal_date"].max()

    rows = frame[frame["signal_date"] == signal_date]
    if rows.empty:
        raise ValueError(f"no feature rows for signal date {signal_date}")
    train = frame[frame["did_2pct_next"].notna() & (frame["target_date"] <= signal_date)]
    if train.empty:
        raise ValueError("no labeled history to train on — ingest more data")

    strategy = strategies.get(strategy_name)
    strategy.fit(train)
    scored = (
        rows.assign(prob=strategy.predict_proba(rows))
        .sort_values("prob", ascending=False)
        .reset_index(drop=True)
    )
    scored["rank"] = range(1, len(scored) + 1)
    if save:
        store.save_predictions(con, strategy_name, signal_date, scored)
    logger.info(
        "predicted %s for day after %s: %d symbols, trained on %d rows",
        strategy_name,
        signal_date,
        len(scored),
        len(train),
    )
    return PredictResult(strategy_name, signal_date, scored, len(train))
