"""DuckDB storage for the ticker universe and daily prices."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_DB_PATH = Path("data/twopercent.duckdb")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS universe (
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    market_cap DOUBLE NOT NULL,
    as_of DATE NOT NULL,
    PRIMARY KEY (symbol, as_of)
);
CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    date DATE NOT NULL,
    open DOUBLE,
    high DOUBLE,
    low DOUBLE,
    close DOUBLE,
    adj_close DOUBLE,
    volume BIGINT,
    PRIMARY KEY (symbol, date)
);
CREATE TABLE IF NOT EXISTS ingest_meta (
    symbol TEXT NOT NULL PRIMARY KEY,
    from_date DATE NOT NULL
);
CREATE OR REPLACE VIEW daily_returns AS
    SELECT symbol, date, open, high, low, close, volume,
           (close - open) / open AS oc_return
    FROM prices
    WHERE open > 0 AND isfinite(open) AND isfinite(close);
CREATE SEQUENCE IF NOT EXISTS experiment_id_seq;
CREATE TABLE IF NOT EXISTS experiments (
    id BIGINT PRIMARY KEY DEFAULT nextval('experiment_id_seq'),
    run_ts TIMESTAMP NOT NULL,
    strategy TEXT NOT NULL,
    params TEXT,
    train_start DATE,
    test_start DATE,
    test_end DATE,
    metrics TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    strategy TEXT NOT NULL,
    signal_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    prob DOUBLE NOT NULL,
    rank INTEGER NOT NULL,
    created_ts TIMESTAMP NOT NULL,
    PRIMARY KEY (strategy, signal_date, symbol)
);
CREATE OR REPLACE VIEW latest_universe AS
    SELECT symbol, name, market_cap, as_of
    FROM universe
    WHERE as_of = (SELECT max(as_of) FROM universe);
"""


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(_SCHEMA)
    return con


def upsert_universe(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, as_of: dt.date) -> int:
    """Store a universe snapshot (columns: symbol, name, market_cap) for a date."""
    snapshot = df[["symbol", "name", "market_cap"]].copy()
    snapshot["as_of"] = as_of
    con.register("universe_in", snapshot)
    con.execute("INSERT OR REPLACE INTO universe SELECT * FROM universe_in")
    con.unregister("universe_in")
    return len(snapshot)


def latest_universe(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The most recent universe snapshot, ranked by market cap descending."""
    return con.execute("SELECT * FROM latest_universe ORDER BY market_cap DESC").df()


def all_universe_symbols(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Union of symbols across ALL universe snapshots, largest cap first.

    Ingest keys off this rather than the latest snapshot so a symbol that
    churns out around the rank-3000 boundary keeps its price history current.
    """
    rows = con.execute(
        """
        SELECT symbol FROM universe
        GROUP BY symbol
        ORDER BY max(market_cap) DESC
        """
    ).fetchall()
    return [r[0] for r in rows]


def upsert_prices(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Idempotently upsert price rows.

    Expects columns: symbol, date, open, high, low, close, adj_close, volume.
    """
    if df.empty:
        return 0
    con.register("prices_in", df)
    con.execute(
        """
        INSERT OR REPLACE INTO prices
        SELECT symbol, date, open, high, low, close, adj_close, volume FROM prices_in
        """
    )
    con.unregister("prices_in")
    return len(df)


def last_price_dates(con: duckdb.DuckDBPyConnection) -> dict[str, dt.date]:
    """Map each stored symbol to its most recent price date (for resume logic)."""
    rows = con.execute("SELECT symbol, max(date) FROM prices GROUP BY symbol").fetchall()
    return dict(rows)


def ingest_from_dates(con: duckdb.DuckDBPyConnection) -> dict[str, dt.date]:
    """Map each symbol to the earliest window start it was ever ingested from."""
    rows = con.execute("SELECT symbol, from_date FROM ingest_meta").fetchall()
    return dict(rows)


def record_ingest_from(
    con: duckdb.DuckDBPyConnection, symbols: list[str], from_date: dt.date
) -> None:
    """Record that `symbols` now have coverage from `from_date` (keeps the earliest)."""
    if not symbols:
        return
    df = pd.DataFrame({"symbol": symbols, "from_date": from_date})
    con.register("meta_in", df)
    con.execute(
        """
        INSERT INTO ingest_meta SELECT symbol, from_date FROM meta_in
        ON CONFLICT (symbol)
        DO UPDATE SET from_date = least(ingest_meta.from_date, excluded.from_date)
        """
    )
    con.unregister("meta_in")


def price_row_count(con: duckdb.DuckDBPyConnection) -> int:
    return con.execute("SELECT count(*) FROM prices").fetchone()[0]


def save_predictions(
    con: duckdb.DuckDBPyConnection, strategy: str, signal_date: dt.date, df: pd.DataFrame
) -> int:
    """Idempotently store scored rows (columns: symbol, prob, rank) for a signal date."""
    if df.empty:
        return 0
    rows = df[["symbol", "prob", "rank"]].copy()
    rows.insert(0, "strategy", strategy)
    rows.insert(1, "signal_date", signal_date)
    con.register("predictions_in", rows)
    con.execute(
        """
        INSERT OR REPLACE INTO predictions
        SELECT strategy, signal_date, symbol, prob, rank, now() FROM predictions_in
        """
    )
    con.unregister("predictions_in")
    return len(rows)


def predicted_signal_dates(con: duckdb.DuckDBPyConnection, strategy: str) -> list[dt.date]:
    rows = con.execute(
        "SELECT DISTINCT signal_date FROM predictions WHERE strategy = ? ORDER BY signal_date",
        [strategy],
    ).fetchall()
    return [r[0] for r in rows]


def record_experiment(
    con: duckdb.DuckDBPyConnection,
    strategy: str,
    params: dict,
    train_start: dt.date,
    test_start: dt.date,
    test_end: dt.date,
    metrics: dict,
) -> None:
    con.execute(
        """
        INSERT INTO experiments (run_ts, strategy, params, train_start, test_start,
                                 test_end, metrics)
        VALUES (now(), ?, ?, ?, ?, ?, ?)
        """,
        [strategy, json.dumps(params), train_start, test_start, test_end, json.dumps(metrics)],
    )


def list_experiments(con: duckdb.DuckDBPyConnection, limit: int = 20) -> pd.DataFrame:
    return con.execute("SELECT * FROM experiments ORDER BY run_ts DESC LIMIT ?", [limit]).df()
