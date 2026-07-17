"""The 2% scanner: which tickers moved +N% open-to-close on a given day."""

from __future__ import annotations

import datetime as dt

import duckdb
import pandas as pd

DEFAULT_THRESHOLD = 0.02


def latest_price_date(con: duckdb.DuckDBPyConnection) -> dt.date | None:
    return con.execute("SELECT max(date) FROM prices").fetchone()[0]


def price_count_on(con: duckdb.DuckDBPyConnection, date: dt.date) -> int:
    return con.execute("SELECT count(*) FROM prices WHERE date = ?", [date]).fetchone()[0]


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
        LEFT JOIN (
            SELECT symbol, name FROM universe
            WHERE as_of = (SELECT max(as_of) FROM universe)
        ) u USING (symbol)
        WHERE r.date = ? AND r.oc_return >= ?
        ORDER BY r.oc_return DESC
        """,
        [date, threshold],
    ).df()
