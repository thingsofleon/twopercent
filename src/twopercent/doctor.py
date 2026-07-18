"""Data-quality doctor: gaps, staleness, suspicious bars, and coverage checks."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import duckdb
import pandas as pd

from twopercent import scan
from twopercent.ingest import _SPLIT_EPSILON, SPLIT_ARTIFACT_OC, SPLIT_ARTIFACT_SCALE

logger = logging.getLogger(__name__)

DEFAULT_STALE_DAYS = 2
EXTREME_RETURN_THRESHOLD = 0.5
ZERO_VOLUME_MIN_RUN = 3


def gap_counts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-symbol count of missing bars, measured against the store's own calendar.

    The calendar is the set of dates present in daily_returns across all
    symbols. A symbol has a gap if a calendar date between its own first and
    last bar has no bar for it. Ordered worst-first.
    """
    return con.execute(
        """
        WITH calendar AS (SELECT DISTINCT date FROM daily_returns),
             spans AS (
                 SELECT symbol, min(date) AS first_date, max(date) AS last_date
                 FROM daily_returns GROUP BY symbol
             )
        SELECT s.symbol,
               count(*) AS missing,
               min(c.date) AS first_missing,
               max(c.date) AS last_missing
        FROM spans s
        JOIN calendar c ON c.date > s.first_date AND c.date < s.last_date
        LEFT JOIN daily_returns r ON r.symbol = s.symbol AND r.date = c.date
        WHERE r.symbol IS NULL
        GROUP BY s.symbol
        ORDER BY missing DESC, s.symbol
        """
    ).df()


def stale_symbols(
    con: duckdb.DuckDBPyConnection, stale_days: int = DEFAULT_STALE_DAYS
) -> pd.DataFrame:
    """Symbols whose last bar is more than `stale_days` TRADING days behind the store max.

    Measured against the store's own calendar (dates present in daily_returns),
    not calendar days — a symbol missing only its most recent bars would
    otherwise pass silently for up to a week (gaps are interior-only).
    """
    return con.execute(
        """
        WITH calendar AS (SELECT DISTINCT date FROM daily_returns),
             last_bars AS (SELECT symbol, max(date) AS last_date FROM prices GROUP BY symbol)
        SELECT l.symbol, l.last_date, count(c.date) AS trading_days_behind
        FROM last_bars l
        LEFT JOIN calendar c ON c.date > l.last_date
        GROUP BY l.symbol, l.last_date
        HAVING count(c.date) > ?
        ORDER BY trading_days_behind DESC, symbol
        """,
        [stale_days],
    ).df()


def invalid_bars(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-symbol count of price rows the daily_returns view silently excludes.

    These rows (open <= 0/NULL or non-finite open/close) are invisible to the
    gap and extreme checks yet make a symbol look fresh to the stale check —
    corrupt recent bars would otherwise report a healthy store.
    """
    return con.execute(
        """
        SELECT symbol, count(*) AS invalid,
               min(date) AS first_invalid, max(date) AS last_invalid
        FROM prices
        WHERE open IS NULL OR open <= 0 OR NOT isfinite(open)
           OR close IS NULL OR NOT isfinite(close)
        GROUP BY symbol
        ORDER BY invalid DESC, symbol
        """
    ).df()


def extreme_bars(
    con: duckdb.DuckDBPyConnection, threshold: float = EXTREME_RETURN_THRESHOLD
) -> pd.DataFrame:
    """Bars whose open-to-close move exceeds `threshold` in either direction."""
    return con.execute(
        """
        SELECT symbol, date, oc_return
        FROM daily_returns
        WHERE isfinite(oc_return) AND abs(oc_return) > ?
        ORDER BY abs(oc_return) DESC, symbol, date
        """,
        [threshold],
    ).df()


def split_artifacts(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Bars matching the ingest split-artifact rule already sitting in `prices`.

    Same rule as ingest.frames_to_rows: extreme open-to-close move AND an open
    on a different price scale than the PRIOR bar's close (prior-bar only — no
    lookahead). Reads the raw prices table, not the daily_returns view, and
    guards every float comparison with isfinite() (DuckDB total ordering:
    NaN > x is TRUE). Thresholds carry the same FP epsilon as ingest so both
    implementations agree at exact boundaries.
    """
    return con.execute(
        """
        WITH seq AS (
            SELECT symbol, date, open, close,
                   LAG(close) OVER (PARTITION BY symbol ORDER BY date) AS prev_close
            FROM prices
        )
        SELECT symbol, date, (close - open) / open AS oc_return
        FROM seq
        WHERE open > 0 AND isfinite(open) AND isfinite(close)
          AND abs((close - open) / open) > ?
          AND prev_close > 0 AND isfinite(prev_close)
          AND (open / prev_close > ? OR open / prev_close < ?)
        ORDER BY symbol, date
        """,
        [
            SPLIT_ARTIFACT_OC + _SPLIT_EPSILON,
            SPLIT_ARTIFACT_SCALE + _SPLIT_EPSILON,
            1 / SPLIT_ARTIFACT_SCALE - _SPLIT_EPSILON,
        ],
    ).df()


def _delete_bars(con: duckdb.DuckDBPyConnection, flagged: pd.DataFrame) -> None:
    con.register("split_artifacts_in", flagged[["symbol", "date"]])
    con.execute(
        """
        DELETE FROM prices
        WHERE (symbol, date) IN (SELECT symbol, date::DATE FROM split_artifacts_in)
        """
    )
    con.unregister("split_artifacts_in")


def repair_splits(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Delete split-artifact bars from `prices` to a FIXPOINT; returns all deletions.

    Deleting a bar can newly expose its neighbor (the neighbor's prev close
    becomes an earlier bar on the original price scale), so detection and
    deletion loop until a pass finds nothing. The fixpoint makes the repair
    idempotent — a subsequent call finds and deletes nothing — and it stays
    recoverable: a re-ingest of the affected symbols restores the bars if any
    deletion was ever wrong. The only mutating doctor operation; everything
    else is read-only.
    """
    flagged = split_artifacts(con)
    passes: list[pd.DataFrame] = []
    while not flagged.empty:
        _delete_bars(con, flagged)
        passes.append(flagged)
        flagged = split_artifacts(con)
    if not passes:
        return flagged  # empty, correctly-shaped
    removed = (
        pd.concat(passes, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    )
    logger.warning(
        "%d split-artifact bars deleted from prices across %d symbols in %d passes "
        "(re-ingest restores)",
        len(removed),
        removed["symbol"].nunique(),
        len(passes),
    )
    return removed


def zero_volume_runs(
    con: duckdb.DuckDBPyConnection, min_run: int = ZERO_VOLUME_MIN_RUN
) -> pd.DataFrame:
    """Runs of >= `min_run` consecutive zero/NULL-volume bars (consecutive in
    the symbol's own bar sequence, not the calendar)."""
    return con.execute(
        """
        WITH seq AS (
            SELECT symbol, date, volume,
                   row_number() OVER (PARTITION BY symbol ORDER BY date) AS rn
            FROM prices
        ),
        zero AS (
            SELECT symbol, date,
                   rn - row_number() OVER (PARTITION BY symbol ORDER BY date) AS grp
            FROM seq WHERE volume = 0 OR volume IS NULL
        )
        SELECT symbol, min(date) AS run_start, max(date) AS run_end,
               count(*) AS run_length
        FROM zero
        GROUP BY symbol, grp
        HAVING count(*) >= ?
        ORDER BY run_length DESC, symbol, run_start
        """,
        [min_run],
    ).df()


def universe_symbols_without_prices(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Symbols in the latest universe snapshot with no price rows at all."""
    rows = con.execute(
        """
        SELECT symbol FROM latest_universe
        EXCEPT SELECT DISTINCT symbol FROM prices
        ORDER BY symbol
        """
    ).fetchall()
    return [r[0] for r in rows]


def price_symbols_without_meta(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Symbols with price rows but no ingest_meta row (unknown coverage window)."""
    rows = con.execute(
        """
        SELECT DISTINCT symbol FROM prices
        EXCEPT SELECT symbol FROM ingest_meta
        ORDER BY symbol
        """
    ).fetchall()
    return [r[0] for r in rows]


@dataclass(frozen=True)
class DoctorReport:
    max_date: dt.date | None
    stale_days: int
    has_universe: bool
    gaps: pd.DataFrame
    stale: pd.DataFrame
    extreme: pd.DataFrame
    zero_runs: pd.DataFrame
    invalid: pd.DataFrame
    universe_missing_prices: list[str]
    prices_missing_meta: list[str]

    @property
    def problem_count(self) -> int:
        return (
            len(self.gaps)
            + len(self.stale)
            + len(self.extreme)
            + len(self.zero_runs)
            + len(self.invalid)
            + len(self.universe_missing_prices)
            + len(self.prices_missing_meta)
        )

    @property
    def ok(self) -> bool:
        return self.problem_count == 0


def run(con: duckdb.DuckDBPyConnection, stale_days: int = DEFAULT_STALE_DAYS) -> DoctorReport:
    """Run every check and collect the findings."""
    has_universe = con.execute("SELECT count(*) FROM latest_universe").fetchone()[0] > 0
    return DoctorReport(
        max_date=scan.latest_price_date(con),
        stale_days=stale_days,
        has_universe=has_universe,
        gaps=gap_counts(con),
        stale=stale_symbols(con, stale_days=stale_days),
        extreme=extreme_bars(con),
        zero_runs=zero_volume_runs(con),
        invalid=invalid_bars(con),
        universe_missing_prices=universe_symbols_without_prices(con),
        prices_missing_meta=price_symbols_without_meta(con),
    )


def _mark(problems: int) -> str:
    return "[ OK ]" if problems == 0 else "[FAIL]"


def _overflow(lines: list[str], total: int, examples: int) -> None:
    if total > examples:
        lines.append(f"    ... and {total - examples} more")


def format_report(report: DoctorReport, examples: int = 10) -> list[str]:
    """Human-readable summary: one section per check, worst examples first."""
    lines: list[str] = [f"store max date: {report.max_date}"]

    missing_total = 0 if report.gaps.empty else int(report.gaps["missing"].sum())
    lines.append(
        f"{_mark(len(report.gaps))} gaps: {len(report.gaps)} symbols missing "
        f"{missing_total} bars present in the store calendar"
    )
    for row in report.gaps.head(examples).itertuples():
        lines.append(
            f"    {row.symbol:<8} {row.missing} missing between "
            f"{row.first_missing:%Y-%m-%d} and {row.last_missing:%Y-%m-%d}"
        )
    _overflow(lines, len(report.gaps), examples)

    lines.append(
        f"{_mark(len(report.stale))} stale: {len(report.stale)} symbols with last bar "
        f"> {report.stale_days} trading days behind store max"
    )
    for row in report.stale.head(examples).itertuples():
        lines.append(
            f"    {row.symbol:<8} last bar {row.last_date:%Y-%m-%d} "
            f"({row.trading_days_behind} trading days behind)"
        )
    _overflow(lines, len(report.stale), examples)

    suspicious = len(report.extreme) + len(report.zero_runs)
    lines.append(
        f"{_mark(suspicious)} suspicious: {len(report.extreme)} bars with "
        f"|oc_return| > {EXTREME_RETURN_THRESHOLD:.0%}, {len(report.zero_runs)} "
        f"zero-volume runs of >= {ZERO_VOLUME_MIN_RUN} bars"
    )
    for row in report.extreme.head(examples).itertuples():
        lines.append(f"    {row.symbol:<8} {row.date:%Y-%m-%d} oc_return {row.oc_return:+.1%}")
    _overflow(lines, len(report.extreme), examples)
    for row in report.zero_runs.head(examples).itertuples():
        lines.append(
            f"    {row.symbol:<8} zero volume x{row.run_length} "
            f"{row.run_start:%Y-%m-%d}..{row.run_end:%Y-%m-%d}"
        )
    _overflow(lines, len(report.zero_runs), examples)

    invalid_total = 0 if report.invalid.empty else int(report.invalid["invalid"].sum())
    lines.append(
        f"{_mark(len(report.invalid))} invalid: {len(report.invalid)} symbols with "
        f"{invalid_total} bars excluded from scans (open <= 0/NULL or non-finite open/close)"
    )
    for row in report.invalid.head(examples).itertuples():
        lines.append(
            f"    {row.symbol:<8} {row.invalid} invalid bars between "
            f"{row.first_invalid:%Y-%m-%d} and {row.last_invalid:%Y-%m-%d}"
        )
    _overflow(lines, len(report.invalid), examples)

    coverage = len(report.universe_missing_prices) + len(report.prices_missing_meta)
    lines.append(
        f"{_mark(coverage)} coverage: {len(report.universe_missing_prices)} universe "
        f"symbols with no prices, {len(report.prices_missing_meta)} price symbols "
        f"with no ingest_meta row"
    )
    if not report.has_universe:
        lines.append(
            "    warning: no universe stored — universe coverage could not be checked "
            "(run `twopercent universe --refresh`)"
        )
    for symbol in report.universe_missing_prices[:examples]:
        lines.append(f"    {symbol:<8} in latest universe but has no price rows")
    _overflow(lines, len(report.universe_missing_prices), examples)
    for symbol in report.prices_missing_meta[:examples]:
        lines.append(f"    {symbol:<8} has price rows but no ingest_meta row")
    _overflow(lines, len(report.prices_missing_meta), examples)

    return lines
