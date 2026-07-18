"""DuckDB storage for the ticker universe and daily prices."""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/twopercent.duckdb")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS universe (
    symbol TEXT NOT NULL,
    name TEXT NOT NULL,
    market_cap DOUBLE NOT NULL,
    as_of DATE NOT NULL,
    sector TEXT,
    PRIMARY KEY (symbol, as_of)
);
ALTER TABLE universe ADD COLUMN IF NOT EXISTS sector TEXT;
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
CREATE TABLE IF NOT EXISTS experiment_daily (
    seq BIGINT NOT NULL,
    target_date DATE NOT NULL,
    rank INTEGER NOT NULL,
    ret DOUBLE NOT NULL,
    hit INTEGER NOT NULL,
    PRIMARY KEY (seq, target_date, rank)
);
CREATE TABLE IF NOT EXISTS predictions (
    strategy TEXT NOT NULL,
    signal_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    prob DOUBLE NOT NULL,
    rank INTEGER NOT NULL,
    created_ts TIMESTAMP NOT NULL,
    universe_as_of DATE,
    PRIMARY KEY (strategy, signal_date, symbol)
);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS universe_as_of DATE;
CREATE OR REPLACE VIEW latest_universe AS
    SELECT symbol, name, market_cap, as_of, sector
    FROM universe
    WHERE as_of = (SELECT max(as_of) FROM universe);
"""


def _drop_pre_release_experiment_daily(con: duckdb.DuckDBPyConnection) -> None:
    """Drop the short-lived per-aggregate experiment_daily shape (never released).

    The table changed to per-rank rows while its introducing PR was still open;
    a dev store that connected in that window has the old columns, which
    CREATE TABLE IF NOT EXISTS would silently keep. Rows are regenerable by
    rerunning `twopercent benchmark`. No other table is touched.
    """
    cols = {
        r[0]
        for r in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'experiment_daily'"
        ).fetchall()
    }
    if cols and "rank" not in cols:
        n_rows = con.execute("SELECT count(*) FROM experiment_daily").fetchone()[0]
        logger.warning(
            "experiment_daily has the pre-release aggregate shape — dropping it and "
            "discarding %d sim row(s); rerun `twopercent benchmark` to regenerate them",
            n_rows,
        )
        con.execute("DROP TABLE experiment_daily")


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> duckdb.DuckDBPyConnection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    _drop_pre_release_experiment_daily(con)
    con.execute(_SCHEMA)
    return con


def upsert_universe(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, as_of: dt.date) -> int:
    """Store a universe snapshot (columns: symbol, name, market_cap[, sector]) for a date.

    Frames without a sector column (pre-sector callers) store an empty string.
    """
    snapshot = df[["symbol", "name", "market_cap"]].copy()
    snapshot["sector"] = df["sector"].fillna("") if "sector" in df.columns else ""
    snapshot["as_of"] = as_of
    con.register("universe_in", snapshot)
    con.execute(
        """
        INSERT OR REPLACE INTO universe (symbol, name, market_cap, sector, as_of)
        SELECT symbol, name, market_cap, sector, as_of FROM universe_in
        """
    )
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


def last_price_bars(con: duckdb.DuckDBPyConnection) -> dict[str, tuple[dt.date, float | None]]:
    """Map each stored symbol to (last price date, close on that date).

    One query serving both the ingest resume logic and the split-artifact
    prev_close seed: a tail fetch's first bar has no in-frame prior bar, so
    without the stored close the artifact rule is blind on exactly the daily
    updates that will ever see a new artifact.
    """
    rows = con.execute(
        "SELECT symbol, max(date), arg_max(close, date) FROM prices GROUP BY symbol"
    ).fetchall()
    return {sym: (last, close) for sym, last, close in rows}


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
    """Replace the (strategy, signal_date) slice with `df` (columns: symbol, prob, rank).

    Delete-then-insert, not upsert: a re-run that scores FEWER symbols (e.g.
    the liquidity floor now excludes one) must not leave the missing symbols
    behind as phantom ranks from an earlier save.
    """
    con.execute(
        "DELETE FROM predictions WHERE strategy = ? AND signal_date = ?",
        [strategy, signal_date],
    )
    if df.empty:
        return 0
    rows = df[["symbol", "prob", "rank"]].copy()
    rows.insert(0, "strategy", strategy)
    rows.insert(1, "signal_date", signal_date)
    # Which universe snapshot the features were built against: without this,
    # a logged prediction can't be reproduced after the next refresh (feature
    # values are snapshot-dependent — see features.py docstring).
    as_of = con.execute("SELECT max(as_of) FROM universe").fetchone()[0]
    con.register("predictions_in", rows)
    con.execute(
        """
        INSERT INTO predictions
        SELECT strategy, signal_date, symbol, prob, rank, now(), ? FROM predictions_in
        """,
        [as_of],
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
) -> int:
    """Insert an experiments row and return its id (the seq daily rows key on)."""
    return con.execute(
        """
        INSERT INTO experiments (run_ts, strategy, params, train_start, test_start,
                                 test_end, metrics)
        VALUES (now(), ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        [strategy, json.dumps(params), train_start, test_start, test_end, json.dumps(metrics)],
    ).fetchone()[0]


def record_experiment_daily(con: duckdb.DuckDBPyConnection, seq: int, rows: pd.DataFrame) -> int:
    """Store a benchmark's per-day per-rank pick outcomes keyed to its experiments row.

    Expects columns: target_date, rank, ret, hit. Rows with a non-finite ret
    or a null hit are REJECTED with ValueError — a benchmark producing corrupt
    sim rows must die loudly, never persist quietly (a NaN would later vanish
    into skipna aggregations looking like a clean shorter window).
    """
    if rows.empty:
        return 0
    daily = rows[["target_date", "rank", "ret", "hit"]].copy()
    bad = int((~np.isfinite(daily["ret"].astype(float))).sum() + daily["hit"].isna().sum())
    if bad:
        raise ValueError(
            f"refusing to record experiment_daily for seq {seq}: {bad} row(s) with "
            "non-finite ret or null hit — corrupt sim rows must not be persisted"
        )
    daily["target_date"] = pd.to_datetime(daily["target_date"])
    daily.insert(0, "seq", seq)
    con.register("experiment_daily_in", daily)
    con.execute(
        """
        INSERT OR REPLACE INTO experiment_daily
        SELECT seq, CAST(target_date AS DATE), rank, ret, hit
        FROM experiment_daily_in
        """
    )
    con.unregister("experiment_daily_in")
    return len(daily)


def latest_experiment_daily(
    con: duckdb.DuckDBPyConnection, strategy: str
) -> tuple[dict, pd.DataFrame] | None:
    """The best-recorded experiment for `strategy` that HAS daily rows, plus those rows.

    "Best" = most daily rows first, then newest run_ts: a later short run
    (`benchmark --months 2`, a compare) must not silently displace the
    12-month record the dashboard windows need. Returns (experiment metadata
    dict, per-rank daily frame ordered by target_date, rank), or None when no
    experiment of this strategy recorded daily rows — experiments predating
    the experiment_daily table have aggregates only.
    """
    row = con.execute(
        """
        SELECT e.id, e.run_ts, e.params, e.test_start, e.test_end
        FROM experiments e
        JOIN (
            SELECT seq, count(DISTINCT target_date) AS n_days
            FROM experiment_daily GROUP BY seq
        ) d ON d.seq = e.id
        WHERE e.strategy = ?
        ORDER BY d.n_days DESC, e.run_ts DESC, e.id DESC
        LIMIT 1
        """,
        [strategy],
    ).fetchone()
    if row is None:
        return None
    seq, run_ts, params, test_start, test_end = row
    daily = con.execute(
        """
        SELECT target_date, rank, ret, hit
        FROM experiment_daily WHERE seq = ? ORDER BY target_date, rank
        """,
        [seq],
    ).df()
    meta = {
        "seq": seq,
        "run_ts": run_ts,
        "params": json.loads(params) if params else {},
        "test_start": test_start,
        "test_end": test_end,
    }
    return meta, daily


def list_experiments(con: duckdb.DuckDBPyConnection, limit: int = 20) -> pd.DataFrame:
    return con.execute("SELECT * FROM experiments ORDER BY run_ts DESC LIMIT ?", [limit]).df()
