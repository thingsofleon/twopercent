"""The 2% scanner: which tickers REACHED +N% intraday (open-to-high) on a day."""

from __future__ import annotations

import datetime as dt

import duckdb
import pandas as pd

DEFAULT_THRESHOLD = 0.02
# Absolute tolerance on the threshold comparison: (high - open) / open for a
# move of exactly 2% can land a few ULPs below 0.02 in double arithmetic
# (e.g. open 5.00 → 0.019999999999999928), which would silently drop
# exactly-at-threshold reachers.
_THRESHOLD_EPSILON = 1e-9

# Metric-definition tags (M1). The touch era (open-to-high) is stamped on every
# experiments/predictions/shadow row from Stage A on; pre-pivot rows carry NULL
# (= the open-to-close era) and are walled off from touch comparisons. Reads
# that quote a "champion benchmark" or "live track record" filter to TOUCH_EVENT
# so a close-era row is never scored, compared, or promoted against a touch row.
TOUCH_EVENT = "open_to_high"
CLOSE_EVENT = "open_to_close"


def touch_event_predicate(
    high_return: str = "high_return", glitch: str = "high_glitch_suspect"
) -> str:
    """The ONE SQL predicate for the touch EVENT ("reached +2% intraday").

    A bar is a touch event iff its high reached the threshold AND it is not a
    high-spike glitch (store.high_glitch_suspect, the M2 guard). Every consumer
    — the training label and cnt_2pct_20d feature (features.py), the scanner
    (scan.daily_movers), the base rate / precision / lift (track.py, backtest.py)
    — embeds THIS predicate so they can never disagree (quant-skeptic N3). The
    caller binds the epsilon-guarded threshold (DEFAULT_THRESHOLD - _THRESHOLD_EPSILON)
    to the single `?`; column names default to store.daily_returns but are
    overridable for LEAD/aliased references (e.g. next_high_return, dr.high_return).
    Parenthesized so it drops into a WHERE/CASE/AND context unchanged.
    """
    return f"({high_return} >= ? AND NOT {glitch})"


def latest_price_date(con: duckdb.DuckDBPyConnection) -> dt.date | None:
    return con.execute("SELECT max(date) FROM prices").fetchone()[0]


def price_count_on(con: duckdb.DuckDBPyConnection, date: dt.date) -> int:
    """Raw price rows stored for a date (including rows daily_returns excludes)."""
    return con.execute("SELECT count(*) FROM prices WHERE date = ?", [date]).fetchone()[0]


def returns_count_on(con: duckdb.DuckDBPyConnection, date: dt.date) -> int:
    """Scannable rows for a date (what daily_returns actually covers)."""
    return con.execute("SELECT count(*) FROM daily_returns WHERE date = ?", [date]).fetchone()[0]


def daily_movers(
    con: duckdb.DuckDBPyConnection,
    date: dt.date | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Tickers that REACHED `threshold` intraday (open-to-high) on `date`.

    The touch EVENT (touch_event_predicate): high >= open × (1 + threshold) and
    not a high-spike glitch — a pre-placed +2% limit would have filled on the
    day's high. oc_return is still reported (momentum context) but no longer
    defines the event. Defaults to the latest date in the store. Names come from
    the latest universe snapshot (null for symbols no longer in it). Ordered by
    high_return descending.
    """
    date = date or latest_price_date(con)
    columns = [
        "symbol",
        "name",
        "date",
        "open",
        "high",
        "close",
        "oc_return",
        "high_return",
        "volume",
    ]
    if date is None:
        return pd.DataFrame(columns=columns)
    return con.execute(
        f"""
        SELECT r.symbol, u.name, r.date, r.open, r.high, r.close,
               r.oc_return, r.high_return, r.volume
        FROM daily_returns r
        LEFT JOIN latest_universe u USING (symbol)
        WHERE r.date = ? AND {touch_event_predicate("r.high_return", "r.high_glitch_suspect")}
        ORDER BY r.high_return DESC
        """,
        [date, threshold - _THRESHOLD_EPSILON],
    ).df()
