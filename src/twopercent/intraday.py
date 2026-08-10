"""Intraday bars for exit-path resolution (#79 phase 1).

Daily bars say WHETHER the +2% limit and the -1% stop were both touched; they
carry no clock, so they cannot say WHICH CAME FIRST. On 61% of the sim picks
both were touched, and the two orderings are entirely different trades (out at
+2% and never seeing the dip, versus out at -1% and never seeing the rally), so
strategy.pick_return_band has to return a (worst, best) band. This module fills
that gap by replaying 5-minute bars.

OUTCOME SIDE ONLY. These bars describe the TARGET day — the day being predicted
— so any feature computed from them would be lookahead of the most direct kind.
Nothing here may be imported by features.py, and `intraday_prices` must never
appear in FEATURE_COLUMNS. The one legitimate consumer is the dashboard's exit
-rule explorer, which is a display-only what-if and never a decision number
(see validate-new-strategy §4). Prior-day intraday FEATURES are a separate,
unbuilt phase 2 that additionally requires extending the lookahead canary to
mutate this table — the canary today mutates only `prices`, so a leak here
would pass it vacuously.

Three integrity gates, all non-destructive and all LOUD, because a silently
incomplete intraday record does not just add noise — it moves the answer:

  * Validity (_flatten). The same law the daily path enforces: open > 0 and
    OHLC ordering. Impossible bars are dropped, counted and warned.
  * Completeness (session_agreement_sql). Yahoo omits no-trade intervals rather
    than emitting zero-volume bars, and this model picks illiquid small caps
    (measured: 21-27% of AAPL's 1m bar count on the thinnest names). Bar
    counting is the wrong test — a thin name legitimately has fewer bars — so
    the record must REPRODUCE the daily bar's HIGH and LOW (not its open; see
    session_agreement_sql for why). This also catches unadjusted splits, which
    scale every intraday price.
  * Contiguity (resolve). Reproducing the extremes proves the record SAW both
    triggers; it does not prove it saw them FIRST. A no-trade gap before the
    earlier trigger biases that timestamp late and can INVERT the verdict, so
    the record must additionally be unbroken from the 09:30 open through the
    earlier trigger. Only then is the ordering proven rather than inferred.
    Without this gate the failure is silent and unsigned — it flatters or
    penalises depending on which gap is longer.

An unresolvable day is never dropped and never assumed — it keeps the daily
band it already had, and every reason is counted separately so "we resolved 3
of 400" can never read as "resolved".

WHAT A COLLAPSED BAND MEANS. Before path resolution the band was a PROOF from
daily bars: the truth was inside it by construction. A collapsed endpoint is a
narrower claim — it is correct given the replay, and the gates above exist to
make that conditional as tight as possible. Resolution is also NOT random: it
depends on record quality, which tracks liquidity, which tracks outcomes. So
the amount of narrowing describes the resolvable subset, not the exit rule.
The dashboard must say so wherever it shows a collapsed band.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field

import duckdb
import pandas as pd
import yfinance as yf

from twopercent import scan, strategy

logger = logging.getLogger(__name__)

INTERVAL = "5m"
# Yahoo serves 5m bars for roughly the last 60 TRADING days (~88 calendar),
# measured 2026-08-10. Beyond that the API returns an empty frame, not an error.
MAX_LOOKBACK_DAYS = 88
# Yahoo caps a single 5m request at 60 days; stay under it.
REQUEST_SPAN_DAYS = 55
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0
BATCH_SIZE = 40

# Relative tolerance when checking the intraday record against its daily bar.
# Sized against the DECISION, not against splits: the whole ordering question
# lives in the 3-percentage-point corridor between the -1% stop and the +2%
# limit, so a tolerance of 0.5% was 1/6th of the corridor — a uniform 0.4%
# scale error passed the gate on both extremes and manufactured an early limit
# fill (quant-skeptic finding 3, reproduced). 1e-3 still catches every real
# split by a factor of hundreds.
AGREEMENT_TOLERANCE = 0.001

# Regular-session open, minutes past midnight exchange-local, and the bar width.
# The contiguity gate counts expected bars from the OPEN rather than from the
# first bar present, so an un-traded opening window cannot hide a trigger.
SESSION_OPEN_MINUTES = 9 * 60 + 30
BAR_MINUTES = 5

# Path verdicts. None/UNRESOLVED keeps the daily (worst, best) band.
LIMIT_FIRST = strategy.SEQ_LIMIT_FIRST
STOP_FIRST = strategy.SEQ_STOP_FIRST
UNRESOLVED = None

_LIMIT = scan.DEFAULT_THRESHOLD - 1e-9
_STOP = strategy.STOP_LEVEL + strategy._STOP_EPSILON


@dataclass
class IntradayResult:
    """What an ingest run actually stored, and everything it left out."""

    rows: int = 0
    symbols_requested: int = 0
    symbols_returned: int = 0
    symbols_empty: list[str] = field(default_factory=list)
    batches_failed: int = 0

    @property
    def ok(self) -> bool:
        """No batch failed. Empty symbols are REPORTED, not failures.

        Delisted or halted names in an old pick set return nothing every single
        run, so treating any empty symbol as failure would pin the exit code at
        1 forever and train the operator to ignore it. The count still rides in
        summary() and the warning still fires.
        """
        return self.batches_failed == 0

    def summary(self) -> str:
        parts = [
            f"{self.rows} bars",
            f"{self.symbols_returned}/{self.symbols_requested} symbols",
        ]
        if self.symbols_empty:
            shown = ", ".join(sorted(self.symbols_empty)[:10])
            parts.append(f"{len(self.symbols_empty)} EMPTY ({shown})")
        if self.batches_failed:
            parts.append(f"{self.batches_failed} batch(es) FAILED")
        return "; ".join(parts)


@dataclass
class ResolutionResult:
    """Per-(symbol, session) verdicts plus why the rest could not be resolved."""

    frame: pd.DataFrame
    both_touched: int = 0
    resolved: int = 0
    same_bar: int = 0
    no_intraday: int = 0
    disagreed: int = 0
    gappy: int = 0

    def summary(self) -> str:
        pct = 100 * self.resolved / self.both_touched if self.both_touched else 0.0
        return (
            f"{self.resolved}/{self.both_touched} ambiguous pick-days resolved "
            f"({pct:.0f}%); {self.reasons()}"
        )

    def reasons(self) -> str:
        """Only WHY days went unresolved — no coverage fraction.

        This line renders beside the per-selection coverage note, and that note
        counts a different population (the chosen basket and window). Two
        differing fractions inches apart read as a contradiction, so the whole-
        frame fraction stays in summary() for the logs.
        """
        return (
            f"unresolved: {self.no_intraday} no intraday, {self.same_bar} same 5m "
            f"bar, {self.gappy} gap before the first trigger, {self.disagreed} "
            "failed the daily-bar check"
        )


def _download(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """One yfinance 5m request, with the empty-response trap made loud.

    An over-long intraday range does NOT raise and does NOT truncate: yfinance
    returns an EMPTY frame and buries Yahoo's reason ("Only 8 days worth of 1m
    granularity data are allowed to be fetched per request") in a log line. A
    caller that treated empty as "nothing traded" would ingest nothing and
    report success — the exact silent-success shape this project keeps paying
    for. Empty is an error here; the caller decides whether it is fatal.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            data = yf.download(
                tickers=symbols,
                start=start.isoformat(),
                end=end.isoformat(),
                interval=INTERVAL,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
                prepost=False,
            )
            if data is not None and not data.empty:
                return data
            last_error = ValueError(
                f"empty {INTERVAL} response for {len(symbols)} symbol(s) "
                f"{start}..{end} — Yahoo serves only ~{MAX_LOOKBACK_DAYS} calendar "
                "days at this interval and returns EMPTY (not an error) beyond it"
            )
        except Exception as exc:  # yfinance raises a grab-bag of exception types
            last_error = exc
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(f"{INTERVAL} download failed after {MAX_RETRIES} attempts: {last_error}")


def _flatten(data: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Flatten a group_by='ticker' intraday frame into long bars.

    The tz-aware index is converted to exchange-local naive time and the session
    date is derived from THAT, never from a UTC timestamp — a 09:30 ET bar is
    13:30 UTC and would file under the right day by luck and the wrong one after
    any DST or venue change.
    """
    rows: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            sub = data[sym] if isinstance(data.columns, pd.MultiIndex) else data
        except KeyError:
            continue
        before = len(sub)
        sub = sub.dropna(subset=["Open", "High", "Low", "Close"])
        # The SAME validity law the daily path enforces (CLAUDE.md): a bar is
        # usable only if open > 0 and the OHLC ordering holds. An impossible bar
        # (ENHA 2026-07-24 shape: high below open) would otherwise feed the
        # trigger comparisons directly. Dropped rows are COUNTED and warned —
        # silent row loss on the path to a reported number is the failure mode
        # this project keeps paying for.
        sub = sub[
            (sub["Open"] > 0)
            & (sub["High"] >= sub["Open"])
            & (sub["High"] >= sub["Close"])
            & (sub["Low"] <= sub["Open"])
            & (sub["Low"] <= sub["Close"])
        ]
        dropped = before - len(sub)
        if dropped:
            logger.warning(
                "intraday %s: dropped %d of %d bar(s) as null or OHLC-impossible",
                sym,
                dropped,
                before,
            )
        if sub.empty:
            continue
        idx = pd.DatetimeIndex(sub.index)
        if idx.tz is not None:
            idx = idx.tz_convert("America/New_York").tz_localize(None)
        frame = pd.DataFrame(
            {
                "symbol": sym,
                "ts": idx,
                "date": idx.date,
                "interval": INTERVAL,
                "open": sub["Open"].to_numpy(dtype=float),
                "high": sub["High"].to_numpy(dtype=float),
                "low": sub["Low"].to_numpy(dtype=float),
                "close": sub["Close"].to_numpy(dtype=float),
                "volume": sub["Volume"].fillna(0).to_numpy(dtype="int64"),
            }
        )
        rows.append(frame)
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "ts", "date", "interval", "open", "high", "low", "close", "volume"]
        )
    return pd.concat(rows, ignore_index=True)


def ingest(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str],
    start: dt.date,
    end: dt.date,
    downloader=None,
) -> IntradayResult:
    """Fetch and upsert 5m bars for `symbols` over [start, end).

    Symbols that come back empty are REPORTED, never silently skipped: at this
    interval an empty response means either "beyond Yahoo's window" or "this
    name did not trade", and the caller must be able to tell the difference
    from the counts rather than from a green exit code.
    """
    symbols = sorted(set(symbols))
    result = IntradayResult(symbols_requested=len(symbols))
    if not symbols:
        return result
    # Yahoo caps a single 5m request near 60 days, so walk the range in windows
    # rather than rejecting it. Refusing was why `--days 88` — the value the CLI
    # help advertised — raised, and why the design doc's "60 trading days" was
    # unreachable (55 calendar is only ~38 trading days).
    windows: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + dt.timedelta(days=REQUEST_SPAN_DAYS), end)
        windows.append((cursor, stop))
        cursor = stop
    fetch = downloader or _download
    returned: set[str] = set()
    for win_start, win_end in windows:
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i : i + BATCH_SIZE]
            try:
                raw = fetch(batch, win_start, win_end)
            except Exception as exc:
                result.batches_failed += 1
                logger.error(
                    "intraday batch %d-%d %s..%s failed: %s",
                    i,
                    i + len(batch),
                    win_start,
                    win_end,
                    exc,
                )
                continue
            frame = _flatten(raw, batch)
            if frame.empty:
                continue
            returned.update(frame["symbol"].unique().tolist())
            con.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _intraday_stage AS "
                "SELECT * FROM intraday_prices LIMIT 0"
            )
            con.execute("DELETE FROM _intraday_stage")
            con.register("_intraday_frame", frame)
            con.execute("INSERT INTO _intraday_stage SELECT * FROM _intraday_frame")
            con.unregister("_intraday_frame")
            con.execute(
                "DELETE FROM intraday_prices WHERE (symbol, ts, interval) IN "
                "(SELECT symbol, ts, interval FROM _intraday_stage)"
            )
            con.execute("INSERT INTO intraday_prices SELECT * FROM _intraday_stage")
            result.rows += len(frame)
    result.symbols_returned = len(returned)
    result.symbols_empty = [s for s in symbols if s not in returned]
    if result.symbols_empty:
        logger.warning(
            "intraday: %d of %d symbol(s) returned NO %s bars for %s..%s: %s",
            len(result.symbols_empty),
            len(symbols),
            INTERVAL,
            start,
            end,
            ", ".join(result.symbols_empty[:20]),
        )
    return result


def session_agreement_sql() -> str:
    """Per-(symbol, session) intraday aggregate joined to its daily bar.

    `agrees` is the completeness gate: the intraday record must reproduce the
    daily bar's HIGH and LOW within AGREEMENT_TOLERANCE. A sparse record that
    never saw the day's low fails it (so a hidden stop cannot be scored as
    un-triggered), and an unadjusted split fails it too — a split scales every
    intraday price, so both extremes miss by the split factor.

    The OPEN is deliberately NOT part of the test, though it is selected for
    diagnostics. Yahoo's daily open is the official opening auction print while
    the first 5m bar's open is that bar's first trade, and on thin names the two
    legitimately differ by percent — measured on the real store, requiring them
    to agree rejected 726 of 4,504 sessions whose high AND low matched the daily
    bar EXACTLY. Nothing downstream reads the intraday open either: resolve()
    measures both triggers against the DAILY open, because that is the entry
    price the exit rules assume. Testing it would only manufacture false
    negatives, which here means silently declining to fix bands that are fixable.

    isfinite() guards precede every comparison — DuckDB uses total ordering,
    where NaN > x is TRUE, so an unguarded NaN would sail through as agreement
    (CLAUDE.md).
    """
    return f"""
        SELECT i.symbol, i.date,
               d.open AS daily_open,
               first(i.open ORDER BY i.ts) AS intra_open,
               max(i.high) AS intra_high,
               min(i.low) AS intra_low,
               count(*) AS bars,
               (isfinite(d.open) AND isfinite(d.high) AND isfinite(d.low)
                AND d.open > 0
                AND isfinite(max(i.high)) AND isfinite(min(i.low))
                AND abs(max(i.high) - d.high) <= {AGREEMENT_TOLERANCE} * d.open
                AND abs(min(i.low) - d.low) <= {AGREEMENT_TOLERANCE} * d.open
               ) AS agrees
        FROM intraday_prices i
        JOIN prices d ON d.symbol = i.symbol AND d.date = i.date
        WHERE i.interval = '{INTERVAL}'
        GROUP BY i.symbol, i.date, d.open, d.high, d.low
    """


def resolve(con: duckdb.DuckDBPyConnection, pairs: pd.DataFrame) -> ResolutionResult:
    """Verdict per (symbol, target_date) for days that touched BOTH triggers.

    `pairs` needs columns symbol, date. Only rows whose DAILY bar touched both
    the limit and the stop are ambiguous; everything else is already exact from
    daily data and simply never appears here, which leaves
    strategy.pick_return_band's existing behaviour untouched.

    Ordering compares the FIRST bar whose high reached the limit against the
    FIRST bar whose low reached the stop, both measured against the DAILY open
    (the entry price the exit rules assume). Equal timestamps mean both happened
    inside one 5-minute bar: genuinely unresolvable at this interval, counted as
    `same_bar`, never broken by a coin flip.

    Every unresolved reason is counted separately so a caller can tell "no data
    yet" from "data that failed its own integrity check" — they need different
    fixes, and a single "unresolved" total would hide the second one.
    """
    empty = pd.DataFrame(columns=["symbol", "date", "seq"])
    if pairs.empty:
        return ResolutionResult(frame=empty)

    pairs = pairs[["symbol", "date"]].drop_duplicates().copy()
    pairs["date"] = pd.to_datetime(pairs["date"]).dt.date
    con.register("_resolve_pairs", pairs)
    try:
        frame = con.execute(
            f"""
            WITH want AS (
                SELECT DISTINCT symbol, CAST(date AS DATE) AS date FROM _resolve_pairs
            ),
            ambiguous AS (
                SELECT w.symbol, w.date, d.open
                FROM want w
                JOIN daily_returns d ON d.symbol = w.symbol AND d.date = w.date
                WHERE {scan.touch_event_predicate("d.high_return", "d.high_glitch_suspect")}
                  AND isfinite(d.low_return) AND d.low_return <= {_STOP}
            ),
            agree AS ({session_agreement_sql()}),
            hits AS (
                SELECT a.symbol, a.date, ag.agrees,
                       min(CASE WHEN isfinite(i.high)
                                 AND (i.high - a.open) / a.open >= {_LIMIT}
                                THEN i.ts END) AS limit_ts,
                       min(CASE WHEN isfinite(i.low)
                                 AND (i.low - a.open) / a.open <= {_STOP}
                                THEN i.ts END) AS stop_ts
                FROM ambiguous a
                LEFT JOIN agree ag ON ag.symbol = a.symbol AND ag.date = a.date
                LEFT JOIN intraday_prices i
                       ON i.symbol = a.symbol AND i.date = a.date
                      AND i.interval = '{INTERVAL}'
                GROUP BY a.symbol, a.date, ag.agrees
            ),
            -- CONTIGUITY. Reproducing the day's extremes proves the record saw
            -- both triggers; it does NOT prove it saw them FIRST. A no-trade gap
            -- before the earlier trigger biases that timestamp LATE by an
            -- unbounded amount, and the verdict flips whenever the two lateness
            -- amounts differ — quant-skeptic inverted a verdict this way while
            -- the agreement gate reported success. Only bars up to the EARLIER
            -- trigger can change the answer, so require the record to be
            -- unbroken from the session open through that bar: then no missing
            -- bar could have carried the other trigger earlier, and the
            -- ordering is proven rather than inferred. Anchored at 09:30, not
            -- at the first recorded bar — a name that did not trade at the open
            -- has an unobserved window that could contain either trigger.
            span AS (
                SELECT h.symbol, h.date, h.agrees, h.limit_ts, h.stop_ts,
                       least(h.limit_ts, h.stop_ts) AS first_ts,
                       -- DISTINCT grid slots, floored at the open: a raw row
                       -- count let one off-grid or pre-open print mask one
                       -- missing 5m slot and pass the gate (quant-skeptic B).
                       -- Counting slots makes duplicates and off-grid prints
                       -- collapse onto the slot they belong to.
                       (SELECT count(DISTINCT
                            date_diff('minute',
                                      h.date + INTERVAL '{SESSION_OPEN_MINUTES}' MINUTE,
                                      b.ts) / {BAR_MINUTES})
                          FROM intraday_prices b
                         WHERE b.symbol = h.symbol AND b.date = h.date
                           AND b.interval = '{INTERVAL}'
                           AND b.ts >= h.date + INTERVAL '{SESSION_OPEN_MINUTES}' MINUTE
                           AND b.ts <= least(h.limit_ts, h.stop_ts)) AS bars_before
                FROM hits h
            )
            SELECT symbol, date, agrees, limit_ts, stop_ts,
                   bars_before,
                   CASE WHEN first_ts IS NULL THEN NULL ELSE
                        date_diff('minute', date + INTERVAL '{SESSION_OPEN_MINUTES}' MINUTE,
                                  first_ts) / {BAR_MINUTES} + 1 END AS bars_expected
            FROM span
        """,
            [_LIMIT],
        ).fetchdf()
    finally:
        con.unregister("_resolve_pairs")

    out = ResolutionResult(frame=empty, both_touched=len(frame))
    verdicts: list[dict] = []
    for row in frame.itertuples(index=False):
        if row.agrees is None or pd.isna(row.agrees):
            out.no_intraday += 1  # no bars stored for this session at all
            continue
        if not bool(row.agrees):
            out.disagreed += 1  # bars exist but do not reproduce the daily bar
            continue
        if pd.isna(row.limit_ts) or pd.isna(row.stop_ts):
            # The daily bar says both triggers were touched and the intraday
            # record agrees on open/high/low, yet one trigger has no bar. Not a
            # coin flip to break: trust neither ordering.
            out.disagreed += 1
            continue
        if row.limit_ts == row.stop_ts:
            out.same_bar += 1
            continue
        # Contiguity from the open through the EARLIER trigger. Without it the
        # record can reproduce both extremes yet still be missing the bar that
        # actually carried the first trigger, which silently INVERTS the verdict
        # (quant-skeptic finding 2). Only bars up to the earlier trigger matter:
        # if none are missing there, no unseen bar could have fired the other
        # trigger sooner.
        if (
            pd.isna(row.bars_expected)
            or row.bars_before < row.bars_expected
            or row.bars_expected <= 0
        ):
            out.gappy += 1
            continue
        verdicts.append(
            {
                "symbol": row.symbol,
                "date": row.date,
                "seq": LIMIT_FIRST if row.limit_ts < row.stop_ts else STOP_FIRST,
            }
        )
    out.resolved = len(verdicts)
    if verdicts:
        out.frame = pd.DataFrame(verdicts, columns=["symbol", "date", "seq"])
    logger.info("intraday resolution: %s", out.summary())
    return out
