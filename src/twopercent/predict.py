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

# Liquidity floor, applied at PREDICTION time only (issue #25): symbols whose
# median volume over the last LIQUIDITY_WINDOW_BARS bars ending at signal_date
# is below LIQUIDITY_MIN_MEDIAN_VOLUME shares are too thin to trade and are
# excluded from the ranking and the saved rows. Labels and training keep
# illiquid names — see predict_for's docstring for the asymmetry.
LIQUIDITY_WINDOW_BARS = 20
LIQUIDITY_MIN_MEDIAN_VOLUME = 100_000

_MEDIAN_VOLUME_SQL = """
WITH recent AS (
    SELECT symbol, volume,
           row_number() OVER (PARTITION BY symbol ORDER BY date DESC) AS rn
    FROM prices
    WHERE date <= ?
)
SELECT symbol, median(volume) AS median_volume
FROM recent
WHERE rn <= ?
GROUP BY symbol
"""


def _median_volumes(con: duckdb.DuckDBPyConnection, signal_date: dt.date) -> dict[str, float]:
    """Median volume per symbol over the last LIQUIDITY_WINDOW_BARS bars with
    date <= signal_date. Trailing bars only — never reads past the signal date,
    so backfilled predictions stay walk-forward honest."""
    rows = con.execute(_MEDIAN_VOLUME_SQL, [signal_date, LIQUIDITY_WINDOW_BARS]).fetchall()
    return dict(rows)


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

    Liquidity floor (prediction time ONLY): after scoring, symbols whose
    median volume over the last LIQUIDITY_WINDOW_BARS bars ending at
    signal_date is below LIQUIDITY_MIN_MEDIAN_VOLUME shares are excluded
    from the ranking AND the saved rows — too thin to trade at any size.
    This is deliberately asymmetric: labels and training keep illiquid
    names (their moves are real observations that would otherwise bias the
    base rate), so trained probabilities include names the ranking will
    never surface. The window is strictly trailing (date <= signal_date) —
    no lookahead.
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

    median_volume = scored["symbol"].map(_median_volumes(con, signal_date))
    liquid = median_volume >= LIQUIDITY_MIN_MEDIAN_VOLUME  # NaN median -> excluded
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
    scored = scored[liquid].sort_values("prob", ascending=False).reset_index(drop=True)
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
