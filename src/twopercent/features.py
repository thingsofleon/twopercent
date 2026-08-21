"""Canonical feature frame for prediction strategies.

Timing model: a row is keyed by (symbol, signal_date). All features are
computed from data through the END of signal_date S — they are known after S's
close and used to predict the NEXT trading day. The label `did_2pct_next` is
the next trading day's TOUCH outcome — reached +2% intraday (open-to-high) and
not a high-spike glitch (scan.touch_event_predicate), explicitly a LEAD;
everything else must never look forward. oc_return / cnt_2pct_20d remain as
FEATURES (momentum), not the event. Predictions for "tomorrow" are the rows at
the latest signal_date, which have no label yet.

Caveat to the claim above: price-derived features honor it strictly, but
sector labels, sector-aggregate membership, and log_mcap come from the LATEST
universe snapshot applied to all of history. That is survivorship in feature
values (today's sector/cap assignments were not knowable on past signal
dates), and historical values of those features change when the universe
refreshes — reproduce a logged experiment only against the same snapshot.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging

import duckdb
import pandas as pd

from twopercent.scan import _THRESHOLD_EPSILON, DEFAULT_THRESHOLD, touch_event_predicate

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
    # #110: six price-derived additions. The first two close the range gap --
    # the label is a range event and nothing measured range.
    "range_20d",
    "high_return_mean_20d",
    "gap_prior",
    "days_since_2pct",
    "volume_accel",
    "dist_52w_high",
]

# Metadata, NOT a feature: trailing median volume over the last 20 bars
# through signal_date. Prediction and benchmark top-N selection apply the
# liquidity floor to it (see predict.py); models must never train on it, so
# it stays out of FEATURE_COLUMNS.
METADATA_COLUMNS = ["median_vol_20"]
# Label-side columns (like did_2pct_next and target_date): FUTURE information.
# next_oc_return is the label's magnitude; next_high_return / next_low_return
# are the target day's open-to-high / open-to-low moves (the strategy
# explorer's exit-rule inputs, recorded per pick by the benchmark). They exist
# for scoring/simulation and must never appear in FEATURE_COLUMNS or
# METADATA_COLUMNS (the lookahead canary deliberately excludes label columns,
# which legitimately change when the future changes) — an explicit test in
# test_features.py pins their absence.

_SQL = f"""
WITH marked AS (
    -- Bar index and touch-flagged index, computed BEFORE per_symbol so
    -- days_since_2pct can take max() over them: DuckDB forbids nesting a
    -- window call inside another (row_number inside max would be nested).
    SELECT *,
           row_number() OVER (PARTITION BY symbol ORDER BY date) AS rn
    FROM daily_returns
),
touched AS (
    SELECT *,
           CASE WHEN {touch_event_predicate()} THEN rn END AS touch_rn
    FROM marked
),
per_symbol AS (
    SELECT
        symbol, date, oc_return, volume,
        rn AS history_days,
        close / nullif(LAG(close, 5) OVER w, 0) - 1 AS ret_5d,
        stddev_samp(oc_return) OVER w20 AS vol_20d,
        volume / nullif(avg(volume) OVER w20, 0) AS volume_ratio,
        median(volume) OVER w20 AS median_vol_20,
        CASE WHEN high > low THEN (close - low) / (high - low) END AS close_pos,
        -- cnt_2pct_20d is now a count of TOUCH days (open-to-high reached +2%
        -- and not a high-spike glitch), the same event as the label.
        sum(CASE WHEN {touch_event_predicate()} THEN 1 ELSE 0 END) OVER w20 AS cnt_2pct_20d,
        LEAD(date) OVER w AS target_date,
        LEAD(oc_return) OVER w AS next_oc_return,
        -- LEAD of the touch event's inputs → the NEXT day's reached-2% label.
        LEAD(high_return) OVER w AS next_high_return,
        LEAD(high_glitch_suspect) OVER w AS next_high_glitch_suspect,
        -- LEAD of the open-to-low move: outcome-side stop-rule input, label-only.
        LEAD(low_return) OVER w AS next_low_return,
        -- #110. The label is an intraday RANGE event -- did the high reach +2%
        -- above the open -- and nothing measured range. vol_20d is close-to-close
        -- dispersion, a different quantity: a name can have quiet closes and wide
        -- days. Every window below ends at the CURRENT bar inclusive, so all of
        -- these are known at signal_date's close; none reads the target day.
        avg((high - low) / nullif(open, 0)) OVER w20 AS range_20d,
        -- The label's own quantity, lagged. cnt_2pct_20d counts crossings and
        -- discards magnitude, so a name averaging 1.9% open-to-high and one
        -- averaging 0.3% look identical when neither crossed.
        avg(high_return) OVER w20 AS high_return_mean_20d,
        -- The SIGNAL day's gap. Deliberately open-vs-PRIOR-close: the target
        -- day's open is unknown at the pre-open prediction moment and is the
        -- classic lookahead trap here.
        (open - LAG(close) OVER w) / nullif(LAG(close) OVER w, 0) AS gap_prior,
        -- Recency, which a 20-day count cannot express: five touches last week
        -- and five touches three weeks ago score the same today.
        rn - max(touch_rn) OVER w_all AS days_since_2pct,
        -- Building interest vs a one-day spike, which volume_ratio conflates.
        avg(volume) OVER w5 / nullif(avg(volume) OVER w20, 0) AS volume_accel,
        -- Position in the annual range: the standard breakout/momentum axis,
        -- entirely absent until now.
        close / nullif(max(close) OVER w252, 0) AS dist_52w_high
    FROM touched
    WINDOW
        w AS (PARTITION BY symbol ORDER BY date),
        w20 AS (PARTITION BY symbol ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
        w5 AS (PARTITION BY symbol ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
        w252 AS (PARTITION BY symbol ORDER BY date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW),
        w_all AS (PARTITION BY symbol ORDER BY date
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
),
market AS (
    SELECT
        date,
        avg(CASE WHEN oc_return > 0 THEN 1.0 ELSE 0.0 END) AS breadth,
        -- market_heat stays an open-to-CLOSE breadth FEATURE (a market-state
        -- predictor, deliberately NOT an event site — it never defines a label).
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
    -- The label is the NEXT day's TOUCH event (reached +2% intraday, not a
    -- glitch). next_high_return IS NULL means there is no next bar yet (newest
    -- signal row) → NULL label, exactly as the close-era label keyed on the next
    -- bar's absence. next_oc_return is kept only as the scoring/sim magnitude.
    CASE
        WHEN s.next_high_return IS NULL THEN NULL
        WHEN {touch_event_predicate("s.next_high_return", "s.next_high_glitch_suspect")} THEN 1
        ELSE 0
    END AS did_2pct_next,
    s.next_oc_return,
    s.next_high_return,
    s.next_low_return,
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
    s.range_20d,
    s.high_return_mean_20d,
    s.gap_prior,
    s.days_since_2pct,
    s.volume_accel,
    s.dist_52w_high,
    s.median_vol_20,
    s.history_days
FROM per_symbol s
JOIN market m ON s.date = m.date
LEFT JOIN latest_universe u USING (symbol)
LEFT JOIN sector_day sd ON sd.date = s.date AND sd.sector = u.sector
WHERE s.date >= ? AND s.date <= ?
ORDER BY s.date, s.symbol
"""


def feature_set_version() -> str:
    """Short fingerprint of the active feature set, for experiment identity.

    A recorded benchmark is only comparable to another run over the SAME
    features. The research loop's done-ledger keyed on (strategy, params) alone,
    so adding features left every past config still counted "done" and the
    overnight loop kept no-op'ing on results that no longer described the model
    (#78/#110). The event filter in research.recorded_configs already does this
    for a LABEL change; this is the same idea for a FEATURE change.

    LIMIT, stated because it is not obvious: this hashes NAMES, not semantics.
    Redefining what an existing column MEANS without renaming it will not
    invalidate the ledger. Rename the column, or purge the affected rows by hand.
    Sorted, so reordering the list alone does not trigger a pointless re-sweep.
    """
    joined = ",".join(sorted(FEATURE_COLUMNS))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


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
    # Four epsilon-guarded thresholds, in SQL order: cnt_2pct_20d,
    # days_since_2pct (#110), market_heat, and the label predicate.
    df = con.execute(_SQL, [threshold, threshold, threshold, threshold, start, end]).df()
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
