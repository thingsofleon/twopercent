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
    WHERE open > 0 AND isfinite(open) AND isfinite(close)
      -- isfinite(high)/isfinite(low) MUST precede the >=/<= comparisons: in
      -- DuckDB total ordering NaN >= x is TRUE, so a NaN high would otherwise
      -- pass high >= open and leak an OHLC-impossible bar into oc_return.
      AND isfinite(high) AND isfinite(low)
      AND high >= open AND high >= close AND low <= open AND low <= close;
-- Which trading days are COMPLETE (trustworthy to score/predict from) vs
-- PROVISIONAL (an in-progress day whose pre-market/partial bar covers only a
-- fraction of the universe). The ONE definition, reused by track.py scoring,
-- base rates, and predict's default signal day. A date D is complete iff its
-- valid-bar coverage is not materially short of the recent norm:
--   count(valid bars on D) >= 0.9 * median(count over the 20 dates before D).
-- The 0.9 / 20 literals below are the SQL copy of track.COMPLETENESS_MIN_FRACTION
-- and track.COMPLETENESS_MEDIAN_WINDOW (that module is the source of truth; keep
-- them in sync). Counts come from daily_returns (post-OHLC-gate valid bars — the
-- rows that would actually score), never raw prices. The window is TRAILING ONLY
-- (ROWS ... 20 PRECEDING AND 1 PRECEDING — strictly before D), so a date's
-- verdict uses no future data and never changes when later dates arrive. A date
-- is JUDGED as soon as it has >= 5 prior trading dates (the median is taken over
-- whatever exists, up to 20 trailing dates); a date with FEWER than 5 priors (a
-- from-scratch backfill / brand-new store) can't be judged against a baseline and
-- is treated complete — but that regime is made LOUD at the use sites (predict +
-- routine WARN when the latest day is used yet unjudgeable), never silent. The
-- 0.9 / 20 / 5 literals are the SQL copy of track.COMPLETENESS_MIN_FRACTION,
-- track.COMPLETENESS_MEDIAN_WINDOW, and track.COMPLETENESS_MIN_PRIOR_DATES (that
-- module is the source of truth). bar_count/trailing_median are counts of
-- non-null rows, always finite, so no isfinite() guard is needed; the 1e-9 slack
-- keeps an exactly-0.9 boundary day complete despite FP (0.9 is not representable).
--
-- LOAD-BEARING ASSUMPTION: this is a COVERAGE proxy for FINALITY — a symbol
-- present in daily_returns for date D is treated as posting its FINAL bar for D.
-- One known path violates it: ingest.classify_missing RETAINS a morning partial
-- bar when a later refetch returns empty (a provider rate-limit), so a stale
-- mid-session bar can count toward coverage and let a day pass as complete. That
-- is a pre-existing limit of the coverage approach (not introduced by #65); the
-- real fix is refetch/finality tracking at ingest — see #34 / #31 — and is out of
-- scope here. SCOPE: this view gates track.py scoring, daily_base_rates, and
-- predict's default signal day. It deliberately does NOT gate features.py
-- (training labels, built from raw daily_returns via per-symbol LEAD) or
-- backtest.py (consumes did_2pct_next/target_date directly) — the benchmark and
-- training are NOT completeness-gated on the live edge day. That is harmless for
-- the default predict path (predict uses the latest COMPLETE day and training is
-- filtered target_date <= signal_date, so the incomplete edge is excluded), but
-- an EXPLICIT signal_date on the provisional day, or a benchmark run whose window
-- reaches it, would train/score labels off the partial bar.
CREATE OR REPLACE VIEW complete_trading_days AS
    WITH daily_counts AS (
        SELECT date, count(*) AS bar_count FROM daily_returns GROUP BY date
    ),
    windowed AS (
        SELECT date, bar_count,
               count(*) OVER w AS prior_days,
               median(bar_count) OVER w AS trailing_median
        FROM daily_counts
        WINDOW w AS (ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
    )
    SELECT date, bar_count, prior_days, trailing_median
    FROM windowed
    WHERE prior_days < 5  -- < 5 priors: too little history to judge, treat complete
       OR bar_count >= 0.9 * trailing_median - 1e-9;
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
CREATE TABLE IF NOT EXISTS shadow_predictions (
    challenger TEXT NOT NULL,
    strategy TEXT NOT NULL,
    params TEXT NOT NULL,
    signal_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    prob DOUBLE NOT NULL,
    rank INTEGER NOT NULL,
    created_ts TIMESTAMP NOT NULL,
    universe_as_of DATE,
    PRIMARY KEY (challenger, signal_date, symbol)
);
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


def trading_day_coverage(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-date valid-bar coverage, the trailing-median norm, and the complete
    verdict — the diagnostic frame behind complete_trading_days, with the
    INCOMPLETE dates retained and a `complete` flag, for callers that must WARN
    about a held-back day (predict, routine).

    Columns: date, bar_count (valid rows in daily_returns), prior_days,
    trailing_median, complete. bar_count/trailing_median are recomputed here for
    display only; the `complete` flag is derived by joining the view, so the
    0.9/20 THRESHOLD lives in exactly one place (the view, #65) and this helper
    can never disagree with the gate the scoring queries use.
    """
    return con.execute(
        """
        WITH daily_counts AS (
            SELECT date, count(*) AS bar_count FROM daily_returns GROUP BY date
        ),
        windowed AS (
            SELECT date, bar_count,
                   count(*) OVER w AS prior_days,
                   median(bar_count) OVER w AS trailing_median
            FROM daily_counts
            WINDOW w AS (ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
        )
        SELECT w.date, w.bar_count, w.prior_days, w.trailing_median,
               (c.date IS NOT NULL) AS complete
        FROM windowed w
        LEFT JOIN complete_trading_days c ON c.date = w.date
        ORDER BY w.date
        """
    ).df()


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


def save_shadow_predictions(
    con: duckdb.DuckDBPyConnection,
    challenger: str,
    strategy: str,
    params_json: str,
    signal_date: dt.date,
    df: pd.DataFrame,
) -> int:
    """Replace the (challenger, signal_date) slice of shadow_predictions with `df`.

    The shadow twin of save_predictions: same delete-then-insert (a re-run
    scoring fewer symbols must not leave phantom ranks behind), same
    created_ts=now() and universe_as_of handling. Keyed by CHALLENGER identity,
    not strategy — a challenger can share a strategy name with the champion or
    another challenger and differ only by params. Writes ONLY shadow_predictions,
    never the champion's predictions table.
    """
    con.execute(
        "DELETE FROM shadow_predictions WHERE challenger = ? AND signal_date = ?",
        [challenger, signal_date],
    )
    if df.empty:
        return 0
    rows = df[["symbol", "prob", "rank"]].copy()
    rows.insert(0, "challenger", challenger)
    rows.insert(1, "strategy", strategy)
    rows.insert(2, "params", params_json)
    rows.insert(3, "signal_date", signal_date)
    as_of = con.execute("SELECT max(as_of) FROM universe").fetchone()[0]
    con.register("shadow_predictions_in", rows)
    con.execute(
        """
        INSERT INTO shadow_predictions
        SELECT challenger, strategy, params, signal_date, symbol, prob, rank, now(), ?
        FROM shadow_predictions_in
        """,
        [as_of],
    )
    con.unregister("shadow_predictions_in")
    return len(rows)


def shadow_signal_dates(con: duckdb.DuckDBPyConnection, challenger: str) -> list[dt.date]:
    rows = con.execute(
        "SELECT DISTINCT signal_date FROM shadow_predictions WHERE challenger = ? "
        "ORDER BY signal_date",
        [challenger],
    ).fetchall()
    return [r[0] for r in rows]


def shadow_challengers(con: duckdb.DuckDBPyConnection) -> list[tuple[str, str, str]]:
    """(challenger, strategy, params_json) for every challenger with shadow picks."""
    rows = con.execute(
        "SELECT DISTINCT challenger, strategy, params FROM shadow_predictions ORDER BY challenger"
    ).fetchall()
    return [(c, s, p) for c, s, p in rows]


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
    """The best-recorded DEFAULT-CONFIG experiment for `strategy` that HAS
    daily rows, plus those rows.

    "Best" = most daily rows first, then newest run_ts: a later short run
    (`benchmark --months 2`, a compare) must not silently displace the
    12-month record the dashboard windows need. Experiments with non-empty
    `strategy_params` are EXCLUDED: a parameterized research variant recorded
    under the same strategy name (the nightly sweep does this) must never
    masquerade as the strategy's own reference run — it can tie on day count
    and win on recency. Returns (experiment metadata dict, per-rank daily
    frame ordered by target_date, rank), or None when no qualifying experiment
    recorded daily rows — experiments predating the experiment_daily table
    have aggregates only.
    """
    rows = con.execute(
        """
        SELECT e.id, e.run_ts, e.params, e.test_start, e.test_end
        FROM experiments e
        JOIN (
            SELECT seq, count(DISTINCT target_date) AS n_days
            FROM experiment_daily GROUP BY seq
        ) d ON d.seq = e.id
        WHERE e.strategy = ?
        ORDER BY d.n_days DESC, e.run_ts DESC, e.id DESC
        """,
        [strategy],
    ).fetchall()
    row = None
    for candidate in rows:
        try:
            parsed = json.loads(candidate[2]) if candidate[2] else {}
        except json.JSONDecodeError:
            logger.warning(
                "experiments row #%s has unparseable params — skipped as reference run",
                candidate[0],
            )
            continue
        if parsed.get("strategy_params"):
            continue  # parameterized variant, not the strategy's default config
        row = candidate
        break
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
