"""DuckDB storage for the ticker universe and daily prices."""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from twopercent import scan

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/twopercent.duckdb")

# High-spike glitch guard (M2), used inside the daily_returns view. A touch
# decided by the day's HIGH alone is a GLITCH-SUSPECT (excluded from the touch
# event) ONLY at the narrow intersection quant-skeptic sized at ~150 bars / 5yr
# (~0.014% of touch bars): an isolated high AND a close that did not confirm the
# move AND volume that did not corroborate. Real high-volume squeezes are KEPT.
_HIGH_GLITCH_MULTIPLE = 1.15  # high >= 1.15 × max(open, close, prev_close)
_HIGH_GLITCH_VOL_WINDOW = 20  # trailing bars (STRICTLY prior) for the volume norm

_SCHEMA = f"""
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
-- daily_returns carries BOTH return definitions plus the touch-event guard:
--   oc_return   = (close - open) / open  — the open-to-close move, now a FEATURE
--                 (momentum) and the guard's close-confirmation input, no longer
--                 the event.
--   high_return = (high  - open) / open  — the open-to-high (intraday reach) move.
--                 The touch EVENT (scan.touch_event_predicate) is
--                 high_return >= threshold-eps AND NOT high_glitch_suspect.
--   low_return  = (low   - open) / open  — the open-to-low move (always <= 0 given
--                 the OHLC gate). OUTCOME-side input to the strategy explorer's
--                 stop rules; like high_return it must never become a feature.
--   high_glitch_suspect (M2) = TRUE only at the isolated-high / close-unconfirmed
--                 / volume-uncorroborated intersection (see below). ONE definition
--                 here so every consumer agrees.
CREATE OR REPLACE VIEW daily_returns AS
    SELECT symbol, date, open, high, low, close, volume, oc_return, high_return, low_return,
           -- high_glitch_suspect: the touch was decided by the HIGH alone (close
           -- did NOT confirm: oc_return < threshold), the high is an isolated
           -- outlier vs the same bar + PRIOR close only, AND volume did not
           -- corroborate (below its trailing-{_HIGH_GLITCH_VOL_WINDOW} PRIOR
           -- average). prev_close is LAG(close) over VALID bars — same-bar +
           -- prior data ONLY; NEVER next_open or any future bar, because this
           -- column feeds the training LABEL and future data would be lookahead
           -- (quant-skeptic flagged next_open as diagnostic-only). coalesce(...,
           -- FALSE) keeps the flag a definite boolean: a NULL prev_close (first
           -- valid bar) or NULL/absent volume norm means "no prior reference to
           -- call it a glitch" → NOT suspect. The {scan.DEFAULT_THRESHOLD} /
           -- {_HIGH_GLITCH_MULTIPLE} literals are the SQL copy of
           -- scan.DEFAULT_THRESHOLD and store._HIGH_GLITCH_MULTIPLE (raw
           -- threshold, no epsilon — this is a heuristic guard, not the event).
           coalesce(
               high_return >= {scan.DEFAULT_THRESHOLD}
               AND oc_return < {scan.DEFAULT_THRESHOLD}
               AND high >= {_HIGH_GLITCH_MULTIPLE} * greatest(open, close, coalesce(prev_close, 0))
               AND trailing_avg_vol IS NOT NULL AND isfinite(trailing_avg_vol)
               AND volume < trailing_avg_vol,
               FALSE
           ) AS high_glitch_suspect
    FROM (
        SELECT symbol, date, open, high, low, close, volume,
               (close - open) / open AS oc_return,
               (high - open) / open AS high_return,
               (low - open) / open AS low_return,
               LAG(close) OVER w AS prev_close,
               avg(volume) OVER wv AS trailing_avg_vol
        FROM prices
        WHERE open > 0 AND isfinite(open) AND isfinite(close)
          -- isfinite(high)/isfinite(low) MUST precede the >=/<= comparisons: in
          -- DuckDB total ordering NaN >= x is TRUE, so a NaN high would otherwise
          -- pass high >= open and leak an OHLC-impossible bar into the returns.
          AND isfinite(high) AND isfinite(low)
          AND high >= open AND high >= close AND low <= open AND low <= close
        WINDOW
            w AS (PARTITION BY symbol ORDER BY date),
            wv AS (PARTITION BY symbol ORDER BY date
                   ROWS BETWEEN {_HIGH_GLITCH_VOL_WINDOW} PRECEDING AND 1 PRECEDING)
    );
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
    metrics TEXT NOT NULL,
    -- Metric-definition tag (M1): 'open_to_high' for touch-era rows; NULL for
    -- pre-pivot (close-era) rows. Champion-benchmark reads filter to the touch
    -- era so a close-based row is never quoted or compared against a touch row.
    event TEXT
);
ALTER TABLE experiments ADD COLUMN IF NOT EXISTS event TEXT;
CREATE TABLE IF NOT EXISTS experiment_daily (
    seq BIGINT NOT NULL,
    target_date DATE NOT NULL,
    rank INTEGER NOT NULL,
    ret DOUBLE NOT NULL,
    hit INTEGER NOT NULL,
    PRIMARY KEY (seq, target_date, rank)
);
-- Strategy-explorer OUTCOME columns (per pick, per test day): oh = the target
-- day's open-to-high return, ol = open-to-low. With ret (the open-to-close
-- return) and hit (the guarded touch event = a +2% limit fill), any daily exit
-- rule is a deterministic function of the stored row — no re-join. These are
-- LABEL-SIDE quantities: they must never appear in features.FEATURE_COLUMNS or
-- METADATA_COLUMNS. Rows recorded before this upgrade keep NULL and the
-- dashboard's strategy views degrade loudly until a re-benchmark.
ALTER TABLE experiment_daily ADD COLUMN IF NOT EXISTS oh DOUBLE;
ALTER TABLE experiment_daily ADD COLUMN IF NOT EXISTS ol DOUBLE;
CREATE TABLE IF NOT EXISTS predictions (
    strategy TEXT NOT NULL,
    signal_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    prob DOUBLE NOT NULL,
    rank INTEGER NOT NULL,
    created_ts TIMESTAMP NOT NULL,
    universe_as_of DATE,
    -- Metric-definition tag (M1): 'open_to_high' for touch-era predictions; NULL
    -- for pre-cutover predictions. The live touch record counts ONLY touch-era
    -- rows (clean reset); NULL-event days are archived, never re-scored as touch.
    event TEXT,
    PRIMARY KEY (strategy, signal_date, symbol)
);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS universe_as_of DATE;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS event TEXT;
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
    event TEXT,  -- metric-definition tag (M1); see predictions.event
    PRIMARY KEY (challenger, signal_date, symbol)
);
ALTER TABLE shadow_predictions ADD COLUMN IF NOT EXISTS event TEXT;
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
    con: duckdb.DuckDBPyConnection,
    strategy: str,
    signal_date: dt.date,
    df: pd.DataFrame,
    event: str = scan.TOUCH_EVENT,
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
        SELECT strategy, signal_date, symbol, prob, rank, now(), ?, ? FROM predictions_in
        """,
        [as_of, event],
    )
    con.unregister("predictions_in")
    return len(rows)


def touch_record_bounds(
    con: duckdb.DuckDBPyConnection, strategy: str
) -> tuple[dt.date | None, int]:
    """(first touch-era signal_date, count of pre-cutover NULL-event signal_dates)
    for a strategy's predictions — the clean-reset note inputs (M1).

    The live touch record begins at the first `event = 'open_to_high'` prediction;
    earlier days targeted open-to-close and are archived (excluded from the touch
    tiles), never silently re-scored. The dashboard discloses both so the reset is
    visible. Both are None/0 on a store with no cutover (fresh touch-only store)."""
    first_touch = con.execute(
        "SELECT min(signal_date) FROM predictions WHERE strategy = ? AND event = ?",
        [strategy, scan.TOUCH_EVENT],
    ).fetchone()[0]
    archived = con.execute(
        "SELECT count(DISTINCT signal_date) FROM predictions "
        "WHERE strategy = ? AND event IS DISTINCT FROM ?",
        [strategy, scan.TOUCH_EVENT],
    ).fetchone()[0]
    return first_touch, int(archived)


def predicted_signal_dates(con: duckdb.DuckDBPyConnection, strategy: str) -> list[dt.date]:
    """Touch-era signal dates only (M1): the pending list is derived from this, so
    a pre-cutover (NULL-event) close-era day must NOT appear — it can never resolve
    to a touch score and would otherwise render "Awaiting outcomes" forever."""
    rows = con.execute(
        "SELECT DISTINCT signal_date FROM predictions WHERE strategy = ? AND event = ? "
        "ORDER BY signal_date",
        [strategy, scan.TOUCH_EVENT],
    ).fetchall()
    return [r[0] for r in rows]


def save_shadow_predictions(
    con: duckdb.DuckDBPyConnection,
    challenger: str,
    strategy: str,
    params_json: str,
    signal_date: dt.date,
    df: pd.DataFrame,
    event: str = scan.TOUCH_EVENT,
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
        SELECT challenger, strategy, params, signal_date, symbol, prob, rank, now(), ?, ?
        FROM shadow_predictions_in
        """,
        [as_of, event],
    )
    con.unregister("shadow_predictions_in")
    return len(rows)


def shadow_signal_dates(con: duckdb.DuckDBPyConnection, challenger: str) -> list[dt.date]:
    """Touch-era signal dates only (M1) — same clean-reset reason as
    predicted_signal_dates: a pre-cutover close-era shadow day can never resolve
    to a touch score and must not linger in the challenger's pending list."""
    rows = con.execute(
        "SELECT DISTINCT signal_date FROM shadow_predictions WHERE challenger = ? AND event = ? "
        "ORDER BY signal_date",
        [challenger, scan.TOUCH_EVENT],
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
    event: str = scan.TOUCH_EVENT,
) -> int:
    """Insert an experiments row and return its id (the seq daily rows key on).

    `event` stamps the metric definition (M1): touch-era rows carry
    scan.TOUCH_EVENT so champion-benchmark reads never compare a close-era row
    to a touch row. Pre-pivot rows in the store keep NULL.
    """
    return con.execute(
        """
        INSERT INTO experiments (run_ts, strategy, params, train_start, test_start,
                                 test_end, metrics, event)
        VALUES (now(), ?, ?, ?, ?, ?, ?, ?)
        RETURNING id
        """,
        [
            strategy,
            json.dumps(params),
            train_start,
            test_start,
            test_end,
            json.dumps(metrics),
            event,
        ],
    ).fetchone()[0]


def record_experiment_daily(con: duckdb.DuckDBPyConnection, seq: int, rows: pd.DataFrame) -> int:
    """Store a benchmark's per-day per-rank pick outcomes keyed to its experiments row.

    Expects columns: target_date, rank, ret, hit, and (from the strategy-explorer
    upgrade on) oh, ol. Rows with a non-finite ret/oh/ol or a null hit are
    REJECTED with ValueError — a benchmark producing corrupt sim rows must die
    loudly, never persist quietly (a NaN would later vanish into skipna
    aggregations looking like a clean shorter window). A frame WITHOUT the oh/ol
    pair (a legacy caller) stores NULL and WARNS: those rows render the
    dashboard's strategy views as "no strategy data recorded" until a
    re-benchmark. A frame with only one of the pair is corrupt shape → rejected.
    """
    if rows.empty:
        return 0
    has_strategy_cols = {"oh", "ol"} <= set(rows.columns)
    if not has_strategy_cols and ({"oh", "ol"} & set(rows.columns)):
        raise ValueError(
            f"refusing to record experiment_daily for seq {seq}: frame carries only "
            "one of the oh/ol outcome columns — corrupt shape, record both or neither"
        )
    cols = ["target_date", "rank", "ret", "hit"] + (["oh", "ol"] if has_strategy_cols else [])
    daily = rows[cols].copy()
    checked = ["ret", "oh", "ol"] if has_strategy_cols else ["ret"]
    bad = int(
        (~np.isfinite(daily[checked].astype(float))).any(axis=1).sum() + daily["hit"].isna().sum()
    )
    if bad:
        raise ValueError(
            f"refusing to record experiment_daily for seq {seq}: {bad} row(s) with "
            f"non-finite {'/'.join(checked)} or null hit — corrupt sim rows must "
            "not be persisted"
        )
    if not has_strategy_cols:
        logger.warning(
            "experiment_daily rows for seq %s carry no oh/ol outcome columns — "
            "stored as NULL; the dashboard strategy explorer will say 'no strategy "
            "data recorded' for this experiment until a re-benchmark",
            seq,
        )
    daily["target_date"] = pd.to_datetime(daily["target_date"])
    daily.insert(0, "seq", seq)
    oh_ol = "oh, ol" if has_strategy_cols else "NULL AS oh, NULL AS ol"
    con.register("experiment_daily_in", daily)
    con.execute(
        f"""
        INSERT OR REPLACE INTO experiment_daily (seq, target_date, rank, ret, hit, oh, ol)
        SELECT seq, CAST(target_date AS DATE), rank, ret, hit, {oh_ol}
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

    Touch era ONLY (M1): rows with event <> scan.TOUCH_EVENT (a close-era
    NULL-event archive row) are excluded, so the dashboard SIM panel reads a
    touch benchmark or none — it never re-scores a close-era sim under the touch
    definition. Returns None until a touch benchmark is recorded.
    """
    rows = con.execute(
        """
        SELECT e.id, e.run_ts, e.params, e.test_start, e.test_end
        FROM experiments e
        JOIN (
            SELECT seq, count(DISTINCT target_date) AS n_days
            FROM experiment_daily GROUP BY seq
        ) d ON d.seq = e.id
        WHERE e.strategy = ? AND e.event = ?
        ORDER BY d.n_days DESC, e.run_ts DESC, e.id DESC
        """,
        [strategy, scan.TOUCH_EVENT],
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
        SELECT target_date, rank, ret, hit, oh, ol
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
