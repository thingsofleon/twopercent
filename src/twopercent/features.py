"""Canonical feature frame for prediction strategies.

Timing model: a row is keyed by (symbol, signal_date). All features are
computed from data through the END of signal_date S — they are known after S's
close and used to predict the NEXT trading day. The label `did_2pct_next` is
the next trading day's +2% outcome (explicitly a LEAD; everything else must
never look forward). Predictions for "tomorrow" are the rows at the latest
signal_date, which have no label yet.

Caveat to the claim above: price-derived features honor it strictly, but
sector labels, sector-aggregate membership, and log_mcap come from the LATEST
universe snapshot applied to all of history. That is survivorship in feature
values (today's sector/cap assignments were not knowable on past signal
dates), and historical values of those features change when the universe
refreshes — reproduce a logged experiment only against the same snapshot.
"""

from __future__ import annotations

import datetime as dt
import logging

import duckdb
import pandas as pd

from twopercent.scan import _THRESHOLD_EPSILON, DEFAULT_THRESHOLD

logger = logging.getLogger(__name__)

MIN_HISTORY_DAYS = 20

FEATURE_COLUMNS = [
    "oc_return_today",
    "ret_5d",
    "vol_20d",
    "volume_ratio",
    "close_pos",
    "cnt_2pct_20d",
    "breadth",
    "market_heat",
    "log_mcap",
    "sector_breadth",
    "sector_excess",
]

# Metadata, NOT a feature: trailing median volume over the last 20 bars
# through signal_date. Prediction and benchmark top-N selection apply the
# liquidity floor to it (see predict.py); models must never train on it, so
# it stays out of FEATURE_COLUMNS.
METADATA_COLUMNS = ["median_vol_20"]

_SQL = """
WITH per_symbol AS (
    SELECT
        symbol, date, oc_return, volume,
        row_number() OVER w AS history_days,
        close / nullif(LAG(close, 5) OVER w, 0) - 1 AS ret_5d,
        stddev_samp(oc_return) OVER w20 AS vol_20d,
        volume / nullif(avg(volume) OVER w20, 0) AS volume_ratio,
        median(volume) OVER w20 AS median_vol_20,
        CASE WHEN high > low THEN (close - low) / (high - low) END AS close_pos,
        sum(CASE WHEN oc_return >= ? THEN 1 ELSE 0 END) OVER w20 AS cnt_2pct_20d,
        LEAD(date) OVER w AS target_date,
        LEAD(oc_return) OVER w AS next_oc_return
    FROM daily_returns
    WINDOW
        w AS (PARTITION BY symbol ORDER BY date),
        w20 AS (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
),
market AS (
    SELECT
        date,
        avg(CASE WHEN oc_return > 0 THEN 1.0 ELSE 0.0 END) AS breadth,
        avg(CASE WHEN oc_return >= ? THEN 1.0 ELSE 0.0 END) AS market_heat
    FROM daily_returns
    GROUP BY date
),
sector_day AS (
    SELECT
        d.date,
        u.sector,
        avg(CASE WHEN d.oc_return > 0 THEN 1.0 ELSE 0.0 END) AS sector_breadth,
        avg(d.oc_return) AS sector_mean_oc
    FROM daily_returns d
    JOIN latest_universe u USING (symbol)
    WHERE u.sector IS NOT NULL AND u.sector <> ''
    GROUP BY d.date, u.sector
)
SELECT
    s.symbol,
    s.date AS signal_date,
    s.target_date,
    CASE
        WHEN s.next_oc_return IS NULL THEN NULL
        WHEN s.next_oc_return >= ? THEN 1
        ELSE 0
    END AS did_2pct_next,
    s.oc_return AS oc_return_today,
    s.ret_5d,
    s.vol_20d,
    s.volume_ratio,
    s.close_pos,
    s.cnt_2pct_20d,
    m.breadth,
    m.market_heat,
    ln(u.market_cap) AS log_mcap,
    sd.sector_breadth,
    s.oc_return - sd.sector_mean_oc AS sector_excess,
    s.median_vol_20,
    s.history_days
FROM per_symbol s
JOIN market m ON s.date = m.date
LEFT JOIN latest_universe u USING (symbol)
LEFT JOIN sector_day sd ON sd.date = s.date AND sd.sector = u.sector
WHERE s.date >= ? AND s.date <= ?
ORDER BY s.date, s.symbol
"""


def feature_frame(
    con: duckdb.DuckDBPyConnection,
    start: dt.date = dt.date.min,
    end: dt.date = dt.date.max,
) -> pd.DataFrame:
    """Feature rows for all symbols with signal_date in [start, end].

    Rows with under MIN_HISTORY_DAYS of history are dropped (loudly): their
    rolling features are unstable and would teach the model IPO artifacts.
    """
    threshold = DEFAULT_THRESHOLD - _THRESHOLD_EPSILON
    df = con.execute(_SQL, [threshold, threshold, threshold, start, end]).df()
    thin = df["history_days"] < MIN_HISTORY_DAYS
    if thin.any():
        logger.warning(
            "%d feature rows dropped: under %d days of history",
            int(thin.sum()),
            MIN_HISTORY_DAYS,
        )
    out = df[~thin].drop(columns="history_days").reset_index(drop=True)
    nan_sector = out["sector_breadth"].isna()
    if not out.empty and nan_sector.all():
        logger.warning(
            "no sector data in the latest universe snapshot: sector_breadth/sector_excess "
            "are all NaN (refresh the universe to populate sectors; all-NaN columns crash "
            "sklearn's HistGradientBoosting binner)"
        )
    elif nan_sector.any():
        logger.warning(
            "%d feature rows across %d symbols have NaN sector features "
            "(blank sector or symbol missing from the latest universe snapshot)",
            int(nan_sector.sum()),
            out.loc[nan_sector, "symbol"].nunique(),
        )
    return out
