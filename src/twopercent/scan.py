"""The 2% scanner: which tickers moved +N% open-to-close on a given day."""

from __future__ import annotations

import datetime as dt

import duckdb
import pandas as pd

DEFAULT_THRESHOLD = 0.02
# Absolute tolerance on the threshold comparison: (close - open) / open for a
# move of exactly 2% can land a few ULPs below 0.02 in double arithmetic
# (e.g. open 5.00 → 0.019999999999999928), which would silently drop
# exactly-at-threshold movers.
_THRESHOLD_EPSILON = 1e-9


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
    """Tickers whose open-to-close return reached `threshold` on `date`.

    Defaults to the latest date in the store. Names come from the latest
    universe snapshot (null for symbols no longer in it). Ordered by return
    descending.
    """
    date = date or latest_price_date(con)
    columns = ["symbol", "name", "date", "open", "close", "oc_return", "volume"]
    if date is None:
        return pd.DataFrame(columns=columns)
    return con.execute(
        """
        SELECT r.symbol, u.name, r.date, r.open, r.close, r.oc_return, r.volume
        FROM daily_returns r
        LEFT JOIN latest_universe u USING (symbol)
        WHERE r.date = ? AND r.oc_return >= ?
        ORDER BY r.oc_return DESC
        """,
        [date, threshold - _THRESHOLD_EPSILON],
    ).df()
