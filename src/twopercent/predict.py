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

# Liquidity floor, applied at selection time only (issue #25): symbols whose
# trailing median volume (the median_vol_20 metadata column from features.py,
# 20 bars ending at signal_date) is below LIQUIDITY_MIN_MEDIAN_VOLUME shares
# are too thin to trade and are excluded from the ranking and the saved rows.
# The benchmark applies the same floor to its top-N selection (backtest.py)
# so reported precision matches the shipped product. Labels and training keep
# illiquid names — see predict_for's docstring for the asymmetry.
LIQUIDITY_WINDOW_BARS = 20
LIQUIDITY_MIN_MEDIAN_VOLUME = 100_000


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

    Liquidity floor (selection time ONLY): after scoring, symbols whose
    median_vol_20 (trailing median volume over the LIQUIDITY_WINDOW_BARS
    bars ending at signal_date, computed in features.py) is below
    LIQUIDITY_MIN_MEDIAN_VOLUME shares are excluded from the ranking AND
    the saved rows — too thin to trade at any size. This is deliberately
    asymmetric: labels and training keep illiquid names (their moves are
    real observations that would otherwise bias the base rate), so trained
    probabilities include names the ranking will never surface. The window
    is strictly trailing — no lookahead — and the benchmark's top-N
    selection applies the same floor (backtest.py), so reported precision
    describes the list this function actually ships.
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
    scored = rows.assign(prob=strategy.predict_proba(rows))

    liquid = scored["median_vol_20"] >= LIQUIDITY_MIN_MEDIAN_VOLUME  # NaN median -> excluded
    if (~liquid).any():
        logger.warning(
            "%d of %d symbols excluded from ranking and saved predictions: median "
            "volume over last %d bars < %d shares (labels/training keep them): %s",
            int((~liquid).sum()),
            len(scored),
            LIQUIDITY_WINDOW_BARS,
            LIQUIDITY_MIN_MEDIAN_VOLUME,
            ", ".join(sorted(scored.loc[~liquid, "symbol"])[:20]),
        )
    scored = (
        scored[liquid]
        .sort_values("prob", ascending=False, kind="mergesort")  # stable: ties match the referee
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
