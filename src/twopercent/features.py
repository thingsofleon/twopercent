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
from collections.abc import Sequence

import duckdb
import pandas as pd

from twopercent.scan import _THRESHOLD_EPSILON, DEFAULT_THRESHOLD, touch_event_predicate

logger = logging.getLogger(__name__)

# A regular 1h session is exactly 7 bars (09:30..15:30). Require ALL of them.
#
# This started as "require most of them" with a >=5 floor, and every weakening
# of that idea turned out to admit a fabricated session shape:
#   * a bar COUNT says nothing about WHICH bars. 1,948 real sessions clear a
#     >=5 count with no 15:30 bar, so last_hour_drift described the 13:30 hour
#     and close_volume_share divided by the wrong numerator.
#   * adding "and the closing bar is present" still admits INTERIOR holes:
#     14,181 real sessions have both ends and only 5-6 bars, so the VWAP and the
#     session volume are computed over a day with an hour missing from the
#     middle. Measured, they are a different population -- reach 29.9% vs 33.0%
#     -- and the distortion has no known sign: close_volume_share is LOWER on
#     them (0.200 vs 0.243), not inflated as a shrunken denominator would
#     suggest, because the holed sessions are disproportionately illiquid names.
#
# Requiring the complete session costs 1.2% of otherwise-usable rows (97.7% ->
# 96.5% coverage) and removes the whole class. A half day (4 bars) is therefore
# NULL, which is correct: it is a different session, not a damaged one.
INTRADAY_SESSION_BARS = 7
INTRADAY_CLOSE_HOUR = 15
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

# #79 phase 2: the SIGNAL day's intraday shape (1h bars). COMPUTED, CANARY-
# WATCHED, and deliberately NOT model inputs.
#
# They are safe -- the timing survived four adversarial passes -- but they do
# not pay for themselves yet. Paired walk-forward: AUC +0.0022 (p=0.039, best of
# two metrics on 12 folds, no multiplicity correction) and NOTHING on what the
# product ships (lift 2.0997 -> 2.1108, precision@20 p=0.76). Listing them here
# would change feature_set_version(), which invalidates all 56 recorded research
# configs (#110) and forces a champion re-benchmark -- a permanent cost for a
# maybe.
#
# The measurement is also underpowered BY CONSTRUCTION and cannot yet settle it:
# Yahoo serves ~730 days of 1h against a daily history reaching 2021, so these
# are observable in only 21-36% of training rows in every fold. The decisive
# test is both arms restricted to the intraday era (train from 2024-10-01),
# where coverage is ~95%. Promoting them is one line plus a re-benchmark;
# un-spending the research ledger is not, and that asymmetry is the whole
# argument for waiting.
#
# They stay in the frame and in the canary's watch list so they cannot rot
# silently while they wait.
INTRADAY_FEATURE_COLUMNS = [
    "close_vwap_gap",
    "last_hour_drift",
    "intraday_vol",
    "close_volume_share",
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
intraday_day AS (
    -- #79 phase 2. The SIGNAL day's own intraday shape, from 1h bars. Every
    -- column below is an aggregate over bars of signal_date S ONLY, so all are
    -- known at S's close -- the same standing as oc_return_today. The TARGET
    -- day's intraday bars are never read; that would be the most direct
    -- lookahead available in this codebase, which is why the lookahead canary
    -- now mutates intraday_prices (it previously mutated only `prices`, so a
    -- leak from here would have passed it VACUOUSLY).
    --
    -- 1h is the only interval deep enough to train on: the referee's standard
    -- window is 12 months and 5m/1m reach ~53 and ~34 days.
    SELECT
        symbol,
        date,
        sum(close * volume) / nullif(sum(volume), 0) AS vwap,
        count(*) AS bars,
        stddev_samp((close - open) / nullif(open, 0)) AS intraday_vol,
        last(close ORDER BY ts) AS last_close,
        last((close - open) / nullif(open, 0) ORDER BY ts) AS last_bar_ret,
        last(volume ORDER BY ts) AS last_bar_vol,
        sum(volume) AS session_vol,
        bool_or(EXTRACT(hour FROM ts) = {INTRADAY_CLOSE_HOUR}) AS has_close_bar
    FROM intraday_prices
    WHERE interval = '1h'
    GROUP BY symbol, date
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
    -- Phase-2 intraday features. NULL wherever the signal day has no 1h
    -- record, which is most of history until the backfill completes -- the
    -- existing dropped-column semantics already handle a column the model
    -- cannot use, loudly.
    CASE WHEN i.bars = {INTRADAY_SESSION_BARS} AND i.has_close_bar
         THEN (i.last_close - i.vwap) / nullif(i.vwap, 0) END AS close_vwap_gap,
    CASE WHEN i.bars = {INTRADAY_SESSION_BARS} AND i.has_close_bar
         THEN i.last_bar_ret END AS last_hour_drift,
    CASE WHEN i.bars = {INTRADAY_SESSION_BARS} AND i.has_close_bar
         THEN i.intraday_vol END AS intraday_vol,
    CASE WHEN i.bars = {INTRADAY_SESSION_BARS} AND i.has_close_bar
         THEN i.last_bar_vol / nullif(i.session_vol, 0) END AS close_volume_share,
    s.median_vol_20,
    s.history_days
FROM per_symbol s
JOIN market m ON s.date = m.date
LEFT JOIN latest_universe u USING (symbol)
LEFT JOIN sector_day sd ON sd.date = s.date AND sd.sector = u.sector
LEFT JOIN intraday_day i ON i.symbol = s.symbol AND i.date = s.date
WHERE s.date >= ? AND s.date <= ?
ORDER BY s.date, s.symbol
"""


def _warn_intraday_coverage(out: pd.DataFrame) -> None:
    """Say out loud how much of the frame has NO intraday features, and why.

    CLAUDE.md: anything that skips or filters must warn loudly about what it
    excluded. But loudly is not the same as constantly, and this warning has to
    survive the alarm-fatigue test the doctor's checks just failed twice.

    So it separates two populations that mean opposite things:

      * STRUCTURAL -- signal dates before the provider's 1h horizon. Daily
        history reaches 2021-07 and Yahoo serves ~730 days of 1h, so roughly
        63% of the frame can NEVER have these features. Nothing is wrong and
        nothing can be done; a standing 63% WARNING on every single call would
        train the operator to ignore the line that also carries the real news.
        Reported once, at INFO, as a fact about the data.

      * RECENT -- dates the 1h capture was supposed to cover and did not. THIS
        is the one that decays forward: it is what a stopped capture, a
        picks-only regression, or a provider outage looks like, and it is worth
        a WARNING every time it appears.
    """
    if out.empty:
        return
    missing = out["close_vwap_gap"].isna()
    if not missing.any():
        return
    signal = pd.to_datetime(out["signal_date"])
    covered = signal[~missing]
    if covered.empty:
        logger.warning(
            "intraday features are unavailable on ALL %d rows — the 1h capture has "
            "never run, or its record does not overlap this frame (build it with "
            "`twopercent intraday --interval 1h --days 700`)",
            len(out),
        )
        return

    horizon = covered.min()
    structural = missing & (signal < horizon)
    recent = missing & (signal >= horizon)
    if structural.any():
        logger.info(
            "intraday features are structurally absent before %s on %d of %d rows "
            "(%.0f%%): the provider serves ~730 days of 1h and daily history is "
            "longer — expected, not a fault",
            horizon.date(),
            int(structural.sum()),
            len(out),
            100.0 * structural.mean(),
        )
    if not recent.any():
        return

    in_era = signal >= horizon
    by_day = recent.groupby(signal.dt.date).sum()
    day_totals = in_era.groupby(signal.dt.date).sum()
    blank = by_day[(day_totals > 0) & (by_day == day_totals)]
    detail = ""
    if len(blank):
        shown = ", ".join(str(d) for d in list(blank.index)[:5])
        detail = (
            f"; {len(blank)} day(s) since {horizon.date()} have NO usable 1h session "
            f"for ANY symbol ({shown}{', ...' if len(blank) > 5 else ''})"
        )
    logger.warning(
        "intraday features missing on %d of %d rows (%.1f%%) INSIDE the covered era "
        "(since %s): no 1h record, or a session that is not a complete %d-bar "
        "regular session%s",
        int(recent.sum()),
        int(in_era.sum()),
        100.0 * recent.sum() / max(int(in_era.sum()), 1),
        horizon.date(),
        INTRADAY_SESSION_BARS,
        detail,
    )


def feature_set_version(columns: Sequence[str] | None = None) -> str:
    """Short fingerprint of a feature set, for experiment identity.

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

    `columns` defaults to FEATURE_COLUMNS — the shipped set, and the only value
    that has ever been hashed, so the fingerprint of a normal run is unchanged.
    It is passed explicitly by the referee when a strategy was pointed at a
    different set (`configured_columns`): a recorded row whose feature_set named
    the shipped list while the model trained on another one would be a ledger
    that lies, which is precisely the failure this fingerprint exists to prevent.
    """
    joined = ",".join(sorted(FEATURE_COLUMNS if columns is None else columns))
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
    # Four epsilon-guarded thresholds. SQL order is days_since_2pct's predicate
    # FIRST (it lives in the `touched` CTE, above per_symbol), then
    # cnt_2pct_20d, then market_heat, then the label. All four bind the same
    # value today, which is exactly why a wrong order here would go unnoticed
    # until one of them needs a different threshold.
    df = con.execute(_SQL, [threshold, threshold, threshold, threshold, start, end]).df()
    thin = df["history_days"] < MIN_HISTORY_DAYS
    if thin.any():
        logger.warning(
            "%d feature rows dropped: under %d days of history",
            int(thin.sum()),
            MIN_HISTORY_DAYS,
        )
    out = df[~thin].drop(columns="history_days").reset_index(drop=True)
    _warn_intraday_coverage(out)
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
