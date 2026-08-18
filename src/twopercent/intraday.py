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

INTERVAL = "5m"  # the working resolution; 1m exists for validation and ground truth

# Per-interval provider limits, each MEASURED against the live API rather than
# taken from documentation (2026-08-10). The three numbers differ per interval
# and every one of them is a trap if guessed:
#   lookback   how far back start/end requests are served, in CALENDAR days
#   span       the most one request will return
#   minutes    the bar width, which the contiguity gates count slots in
_SPECS = {
    "5m": {"lookback": 60, "span": 55, "minutes": 5},
    # 1m is the finest Yahoo serves and the fastest to expire. Its documented
    # limit is "within the last 30 days" and that bound is EXCLUSIVE and
    # measured at request time: probing 2026-08-18, start=-30d returns EMPTY
    # with that exact error while -25d returns bars. A lookback of 30 therefore
    # aimed the first chunk AT the boundary, which failed on every single run --
    # 7 failed batches a night, 3 retries with backoff each, reported as ERROR
    # on a completely healthy system. That is the alarm-fatigue trap #87 fixed
    # for 5m and this reintroduced for 1m.
    #
    # 25 keeps a 5-day margin so the oldest chunk stays comfortably inside the
    # window even as it slides during a run. The cost is nil: the daily capture
    # means nothing older than a few days is ever missing anyway.
    "1m": {"lookback": 25, "span": 7, "minutes": 1},
}
INTERVALS = tuple(_SPECS)


def spec(interval: str) -> dict:
    """Provider limits for an interval, or a loud failure for an unknown one."""
    try:
        return _SPECS[interval]
    except KeyError:
        raise ValueError(
            f"unsupported interval {interval!r}; measured limits exist only for "
            f"{', '.join(INTERVALS)} — add a _SPECS entry from a real measurement, "
            "never a guess"
        ) from None


# How far back Yahoo actually serves 5m bars for an explicit start/end request:
# ~60 CALENDAR days. The earlier value of 88 came from measuring `period="60d"`,
# where Yahoo counts 60 TRADING days (~88 calendar) — a different API path with a
# different limit. Sending start/end beyond this returns an EMPTY frame, so a
# request for 88 days produced a whole window of "failed" batches on a healthy
# system and only ~36 trading days of usable data (#87).
MAX_LOOKBACK_DAYS = _SPECS[INTERVAL]["lookback"]  # back-compat alias for 5m
# Yahoo caps a single 5m request at 60 days; stay under it.
REQUEST_SPAN_DAYS = _SPECS[INTERVAL]["span"]  # back-compat alias for 5m
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
BAR_MINUTES = _SPECS[INTERVAL]["minutes"]  # back-compat alias for 5m

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
    # The interval these counts were computed at. Hardcoding "5m" in the text
    # mislabelled a 1m same-bar count as a 5m one once resolution went layered.
    interval: str = INTERVAL

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
            f"unresolved: {self.no_intraday} no intraday, {self.same_bar} same "
            f"{self.interval} bar, {self.gappy} gap before the first trigger, "
            f"{self.disagreed} failed the daily-bar check"
        )


def _download(
    symbols: list[str], start: dt.date, end: dt.date, interval: str = INTERVAL
) -> pd.DataFrame:
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
                interval=interval,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
                prepost=False,
            )
            if data is not None and not data.empty:
                return data
            last_error = ValueError(
                f"empty {interval} response for {len(symbols)} symbol(s) "
                f"{start}..{end} — Yahoo serves only ~{spec(interval)['lookback']} calendar "
                "days at this interval and returns EMPTY (not an error) beyond it"
            )
        except Exception as exc:  # yfinance raises a grab-bag of exception types
            last_error = exc
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(f"{interval} download failed after {MAX_RETRIES} attempts: {last_error}")


def _flatten(data: pd.DataFrame, symbols: list[str], interval: str = INTERVAL) -> pd.DataFrame:
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
                "interval": interval,
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
    today: dt.date | None = None,
    interval: str = INTERVAL,
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
    # Clamp to what Yahoo will actually serve. Asking for more does not fail
    # loudly at the API — it returns empty frames that look exactly like "this
    # symbol did not trade", so an unclamped request buries a real signal under
    # boundary noise. Clamping is announced; silently shortening would be the
    # very failure this project keeps paying for.
    # `today` is injectable so tests can exercise real historical sessions: the
    # clamp is a property of the PROVIDER's rolling window, not of the code.
    limits = spec(interval)
    earliest = (today or dt.date.today()) - dt.timedelta(days=limits["lookback"])
    if start < earliest:
        logger.warning(
            "intraday: requested from %s but Yahoo serves only ~%d calendar days "
            "of %s bars — clamping to %s (%d day(s) dropped)",
            start,
            limits["lookback"],
            interval,
            earliest,
            (earliest - start).days,
        )
        start = earliest
    if start >= end:
        logger.warning("intraday: nothing to fetch — %s is not before %s", start, end)
        return result
    # Yahoo caps a single 5m request near 60 days, so walk the range in windows
    # rather than rejecting it.
    windows: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor < end:
        stop = min(cursor + dt.timedelta(days=limits["span"]), end)
        windows.append((cursor, stop))
        cursor = stop
    fetch = downloader or _download
    returned: set[str] = set()
    for win_start, win_end in windows:
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i : i + BATCH_SIZE]
            try:
                raw = fetch(batch, win_start, win_end, interval)
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
            frame = _flatten(raw, batch, interval)
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
            interval,
            start,
            end,
            ", ".join(result.symbols_empty[:20]),
        )
    return result


def session_agreement_sql(interval: str = INTERVAL) -> str:
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
        WHERE i.interval = '{interval}'
        GROUP BY i.symbol, i.date, d.open, d.high, d.low
    """


def resolve(
    con: duckdb.DuckDBPyConnection, pairs: pd.DataFrame, interval: str = INTERVAL
) -> ResolutionResult:
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

    bar_minutes = spec(interval)["minutes"]
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
            agree AS ({session_agreement_sql(interval)}),
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
                      AND i.interval = '{interval}'
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
                                      b.ts) / {bar_minutes})
                          FROM intraday_prices b
                         WHERE b.symbol = h.symbol AND b.date = h.date
                           AND b.interval = '{interval}'
                           AND b.ts >= h.date + INTERVAL '{SESSION_OPEN_MINUTES}' MINUTE
                           AND b.ts <= least(h.limit_ts, h.stop_ts)) AS bars_before
                FROM hits h
            )
            SELECT symbol, date, agrees, limit_ts, stop_ts,
                   bars_before,
                   CASE WHEN first_ts IS NULL THEN NULL ELSE
                        date_diff('minute', date + INTERVAL '{SESSION_OPEN_MINUTES}' MINUTE,
                                  first_ts) / {bar_minutes} + 1 END AS bars_expected
            FROM span
        """,
            [_LIMIT],
        ).fetchdf()
    finally:
        con.unregister("_resolve_pairs")

    out = ResolutionResult(frame=empty, both_touched=len(frame), interval=interval)
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


def stop_fills(
    con: duckdb.DuckDBPyConnection, pairs: pd.DataFrame, interval: str = INTERVAL
) -> pd.DataFrame:
    """Measured stop EXIT return per (symbol, date), or absent when unmeasurable.

    strategy.pick_return_band books a stopped pick at exactly -1%, which is the
    trigger, not a fill: a stop becomes a market order and executes at the next
    available price (#86). That assumption is optimistic by construction, and
    the band's lower edge was therefore not a worst case.

    The proxy is the OPEN OF THE NEXT 5m BAR after the trigger. Measured by
    running THIS function, on data/twopercent.duckdb as of 2026-08-10, over the
    pairs from:

        SELECT DISTINCT symbol, date FROM daily_returns
        WHERE isfinite(low_return) AND low_return <= -0.01 + 1e-9

    -> 6,968 sessions survive all three gates:

        mean -1.01%   median -1.02%   p10 -2.49%   max +16.78%
        49.2% land BETTER than the trigger, 17.0% are outright gains
        after the clamp in strategy.pick_return_band: mean -1.49%

    Quote those numbers only with that query and that store. Two earlier
    revisions of this docstring got the provenance wrong in two different ways:
    the first cited an UNGATED population (8,589 sessions), the second was
    measured against a PARTIAL copy of the store holding 126 symbols instead of
    504 (2,712 sessions). Both were plausible and both were wrong, which is why
    the query and the file now sit next to the figures.

    The adjacency gate costs 2.0% of coverage (7,112 -> 6,968), not the large
    share an earlier note guessed at.

    The next bar's open is the first price at which a market order sent at the
    trigger could plausibly print, so it is preferred over the trigger bar's
    close (which is up to 5 minutes of further drift) and over the trigger bar's
    low (the worst tick of the bar, which no order is guaranteed). But it is up
    to five minutes AFTER the breach, so it prices in any bounce: ungated, half
    of these come out BETTER than the trigger and 17% are gains. A stop-market
    order cannot fill above its trigger, so strategy.pick_return_band CLAMPS the
    measured exit at STOP_LEVEL. Read these numbers as the distribution of a
    proxy, not of realised fills.

    Only sessions that pass the SAME gates as ordering are used: the record must
    reproduce the daily bar's extremes, and it must be unbroken from the open
    through the stop bar (a gap before it means the real trigger may have been
    earlier, at a price we never saw). The next bar must also be the adjacent
    SLOT. A trigger in the final bar is left unmeasured. Everything unmeasured
    keeps the -1% assumption, labelled — never silently mixed in as observed.
    """
    empty = pd.DataFrame(columns=["symbol", "date", "fill"])
    if pairs.empty:
        return empty
    bar_minutes = spec(interval)["minutes"]
    pairs = pairs[["symbol", "date"]].drop_duplicates().copy()
    pairs["date"] = pd.to_datetime(pairs["date"]).dt.date
    con.register("_fill_pairs", pairs)
    try:
        return con.execute(f"""
            WITH want AS (
                SELECT DISTINCT symbol, CAST(date AS DATE) AS date FROM _fill_pairs
            ),
            stopped AS (
                SELECT w.symbol, w.date, d.open
                FROM want w
                JOIN daily_returns d ON d.symbol = w.symbol AND d.date = w.date
                WHERE isfinite(d.low_return) AND d.low_return <= {_STOP}
            ),
            agree AS ({session_agreement_sql(interval)}),
            trig AS (
                SELECT s.symbol, s.date, s.open,
                       min(CASE WHEN isfinite(i.low)
                                 AND (i.low - s.open) / s.open <= {_STOP}
                                THEN i.ts END) AS stop_ts
                FROM stopped s
                JOIN agree ag ON ag.symbol = s.symbol AND ag.date = s.date AND ag.agrees
                JOIN intraday_prices i
                  ON i.symbol = s.symbol AND i.date = s.date AND i.interval = '{interval}'
                GROUP BY s.symbol, s.date, s.open
            ),
            gated AS (
                SELECT t.*,
                       (SELECT count(DISTINCT
                            date_diff('minute',
                                      t.date + INTERVAL '{SESSION_OPEN_MINUTES}' MINUTE,
                                      b.ts) / {bar_minutes})
                          FROM intraday_prices b
                         WHERE b.symbol = t.symbol AND b.date = t.date
                           AND b.interval = '{interval}'
                           AND b.ts >= t.date + INTERVAL '{SESSION_OPEN_MINUTES}' MINUTE
                           AND b.ts <= t.stop_ts) AS bars_before,
                       date_diff('minute',
                                 t.date + INTERVAL '{SESSION_OPEN_MINUTES}' MINUTE,
                                 t.stop_ts) / {bar_minutes} + 1 AS bars_expected
                FROM trig t
                WHERE t.stop_ts IS NOT NULL
            )
            SELECT g.symbol, g.date, (nxt.open - g.open) / g.open AS fill
            FROM gated g
            JOIN intraday_prices nxt
              ON nxt.symbol = g.symbol AND nxt.date = g.date AND nxt.interval = '{interval}'
             AND nxt.ts = (SELECT min(z.ts) FROM intraday_prices z
                            WHERE z.symbol = g.symbol AND z.date = g.date
                              AND z.interval = '{interval}' AND z.ts > g.stop_ts)
            WHERE g.bars_before >= g.bars_expected AND g.bars_expected > 0
              -- ADJACENCY. "Next bar" must be the NEXT SLOT, not merely the next
              -- print. 2% of sessions have a hole after the trigger (max 140
              -- minutes), and on a name that stops trading after the breach the
              -- following print is the least fill-like price available -- on
              -- exactly the illiquid names #86 was about. The contiguity gate
              -- covers only what happens BEFORE the trigger.
              AND date_diff('minute', g.stop_ts, nxt.ts) = {bar_minutes}
              AND isfinite(nxt.open) AND nxt.open > 0 AND g.open > 0
        """).fetchdf()
    finally:
        con.unregister("_fill_pairs")


def picked_symbols(
    con: duckdb.DuckDBPyConnection, top_n: int = 20, since: dt.date | None = None
) -> list[str]:
    """Symbols the model actually picked — logged predictions and sim picks.

    Picks-only by design: the explorer needs the ordering of the +2% limit and
    the -1% stop on days the model traded, not the whole universe.
    """
    # Bounded to the RETENTION WINDOW. The pick set grows ~20 names a day
    # forever, and every symbol costs requests on a provider this project
    # already has rate-limit issues with (#31/#34). Symbols last picked outside
    # the window have no fetchable bars anyway, so including them buys nothing.
    horizon = since or dt.date.today() - dt.timedelta(
        days=max(s["lookback"] for s in _SPECS.values())
    )
    return [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT symbol FROM predictions WHERE rank <= ? AND signal_date >= ? "
            "UNION SELECT DISTINCT symbol FROM experiment_daily "
            "WHERE symbol IS NOT NULL AND rank <= ? AND target_date >= ?",
            [top_n, horizon, top_n, horizon],
        ).fetchall()
    ]


@dataclass
class ValidationResult:
    """5m verdicts cross-checked against the 1m record on sessions holding both.

    NOT an accuracy check. See validate_against_1m for why agreement here is
    largely forced, and for the 21% of shipped verdicts it cannot reach.
    """

    compared: int = 0
    agreed: int = 0
    disagreed: int = 0
    # Recovery, DECOMPOSED. The headline "unresolvable at 5m, resolvable at 1m"
    # conflated three different things and was rendered against a denominator it
    # was not a subset of (442 reported "of 462 same-5m-bar" when only 350 were
    # same-bar). Finer resolution genuinely winning is a different claim from
    # routing around a defective 5m capture, and only the first is about 1m.
    recovered_same_bar: int = 0
    recovered_no_5m_record: int = 0
    recovered_5m_failed_gate: int = 0
    same_bar_at_5m: int = 0
    sessions_with_both: int = 0
    unchecked_5m_verdicts: int = 0

    @property
    def resolved_only_at_1m(self) -> int:
        return self.recovered_same_bar + self.recovered_no_5m_record + self.recovered_5m_failed_gate

    @property
    def agreement(self) -> float:
        return self.agreed / self.compared if self.compared else float("nan")

    def summary(self) -> str:
        pct = (
            f" ({100 * self.recovered_same_bar / self.same_bar_at_5m:.0f}%)"
            if self.same_bar_at_5m
            else ""
        )
        recovery = (
            f"1m recovers {self.resolved_only_at_1m} day(s) 5m could not order: "
            f"{self.recovered_same_bar} of {self.same_bar_at_5m} same-5m-bar{pct}, "
            f"{self.recovered_no_5m_record} with no 5m record, "
            f"{self.recovered_5m_failed_gate} whose 5m record failed its gate; "
            f"{self.unchecked_5m_verdicts} shipped 5m verdict(s) UNCHECKED "
            f"(1m too gappy); {self.sessions_with_both} session(s) hold both"
        )
        if not self.compared:
            return f"no 5m verdict has 1m cover yet — the 5m orderings are UNCHECKED; {recovery}"
        return (
            f"{self.agreed}/{self.compared} 5m verdicts confirmed by 1m "
            f"({100 * self.agreement:.1f}%); {self.disagreed} INVERTED; {recovery}"
        )


def validate_against_1m(con: duckdb.DuckDBPyConnection, pairs: pd.DataFrame) -> ValidationResult:
    """Cross-check 5m verdicts against the 1m record. A COMPLETENESS check.

    READ THE LIMITS BEFORE QUOTING THE NUMBER. This is not independent ground
    truth and it is not an accuracy measurement, for two reasons that between
    them account for almost all of the agreement it reports.

    1. 5m is a server-side RESAMPLE of 1m, not a separate observation. Measured
       on the real store, a 5m bar's high equals the max of its five constituent
       1m bars in 99.82% of slots (low: 99.83%). Under that identity the first
       5m bar to reach a trigger is just the 5m bucket containing the first 1m
       bar to reach it — and resolve() only emits a verdict when the two
       triggers land in DIFFERENT buckets, in which case the 1m ordering must
       agree. Zero inversions is very largely forced by arithmetic, not
       evidence that the verdicts are right.

    2. The comparison excludes exactly the risky cases. A 5m verdict is only
       checked where the 1m record is ALSO gapless to the first trigger, and on
       the real store 649 of 3,014 shipped verdicts (21.5%) fail that — 647 of
       them because the 1m record is gappy. Those are the illiquid sessions
       (median $12.6M dollar volume vs $38.1M for the checked cohort), which is
       precisely where a gap before the trigger is plausible. Agreement is a
       result about the easy cohort.

    What this DID reveal, and it is not reassuring: in the unchecked cohort a
    typical session has 77 of 78 5m slots present but only 264 of 390 minutes
    traded. On an illiquid name the 5m contiguity gate passes trivially, because
    a name trading a third of the day's minutes still prints in every 5m slot.
    The 1m data is the first evidence of how WEAK that gate is at 5m — read this
    function as measuring the gate's coverage, never as confirming its verdicts.

    A disagreement is not automatically a 5m failure — 1m is sparser, so its
    record can be the defective one. Disagreements are counted, never used to
    silently overwrite anything.
    """
    both = con.execute(
        """
        SELECT count(DISTINCT (symbol, date)) FROM intraday_prices a
        WHERE a.interval = '1m'
          AND EXISTS (SELECT 1 FROM intraday_prices b
                      WHERE b.symbol = a.symbol AND b.date = a.date AND b.interval = '5m')
        """
    ).fetchone()[0]
    out = ValidationResult(sessions_with_both=int(both))
    if pairs.empty:
        return out

    five = resolve(con, pairs, interval="5m")
    one = resolve(con, pairs, interval="1m")
    out.same_bar_at_5m = five.same_bar

    key = ["symbol", "date"]

    def _norm(frame: pd.DataFrame, seq_name: str) -> pd.DataFrame:
        # DuckDB returns DATE as datetime64 while callers pass datetime.date, so
        # an un-normalised set comparison or join silently matches NOTHING —
        # which reads as "1m recovered no days" rather than as a bug.
        if frame.empty:
            return frame
        out_ = frame.rename(columns={"seq": seq_name}).copy()
        out_["date"] = pd.to_datetime(out_["date"]).dt.date
        return out_

    f = _norm(five.frame, "seq_5m")
    o = _norm(one.frame, "seq_1m")

    def _decompose(recovered: pd.DataFrame) -> None:
        """Why could 5m not order these? Ask 5m directly, rather than assuming.

        Reporting one lumped count against a same-bar denominator overstated the
        same-bar recovery as 442 of 462 (96%) when the truth was 350 (76%); the
        other 92 were sessions whose 5m CAPTURE was missing or failed its gate,
        which is a statement about the data, not about resolution.
        """
        if recovered.empty:
            return
        back = resolve(con, recovered[key], interval="5m")
        out.recovered_same_bar = back.same_bar
        out.recovered_no_5m_record = back.no_intraday
        out.recovered_5m_failed_gate = back.disagreed + back.gappy

    if f.empty or o.empty:
        if not o.empty:
            resolved_5m = set(map(tuple, f[key].values)) if not f.empty else set()
            _decompose(o[[tuple(r) not in resolved_5m for r in o[key].values]])
        return out

    merged = f.merge(o, on=key, how="outer")
    both_known = merged.dropna(subset=["seq_5m", "seq_1m"])
    out.compared = len(both_known)
    out.agreed = int((both_known["seq_5m"] == both_known["seq_1m"]).sum())
    out.disagreed = out.compared - out.agreed
    # Shipped 5m verdicts the 1m record could NOT check — the cohort that
    # carries the risk, and the one agreement says nothing about.
    out.unchecked_5m_verdicts = int(merged["seq_1m"].isna().sum())
    _decompose(merged[merged["seq_5m"].isna()])
    logger.info("intraday 1m validation: %s", out.summary())
    return out


# Finest first, for ORDERING only. A 1m record separates triggers that share a
# 5m bar: 350 of 462 same-5m-bar sessions (76%) on the real store. Each interval
# is gated independently, so a finer verdict is not a weaker one.
#
# Deliberately NOT used for stop FILLS. Switching those to 1m moves a displayed
# money number in the flattering direction (+0.145pp per stopped pick on the
# live record) while adding almost no coverage (117 -> 117 there), because the
# 1m proxy has a thinner left tail and pick_return_band's clamp truncates only
# the right one. It is ~97% a re-pricing and ~3% a coverage gain, and nothing
# measures whether the 60-second proxy is more accurate than the 5-minute one --
# only that it is smaller. This project's standard is measured, not argued (#95).
RESOLUTION_ORDER = ("1m", "5m")


def resolve_best(
    con: duckdb.DuckDBPyConnection,
    pairs: pd.DataFrame,
    intervals: tuple[str, ...] = RESOLUTION_ORDER,
) -> ResolutionResult:
    """Resolve at the FINEST interval that can, falling back to coarser ones.

    Each interval is gated independently — a session only contributes a verdict
    if its own record reproduces the daily extremes and is contiguous to the
    first trigger at THAT resolution. So a fallback is never a downgrade in
    rigour, only in precision.

    Counts are reported against the FINAL state: a session that 1m could not
    order but 5m could is resolved, not same-bar. The unresolved reasons come
    from the coarsest attempt, which is the one that had the most chances.
    """
    # Collect only NON-EMPTY frames and concat once: concatenating onto an empty
    # template makes `date` object-dtyped, which then fails to merge against the
    # caller's datetime64 column.
    parts: list[pd.DataFrame] = []
    remaining = pairs[["symbol", "date"]].drop_duplicates().copy()
    remaining["date"] = pd.to_datetime(remaining["date"]).dt.date
    last: ResolutionResult | None = None
    total_touched = 0
    for interval in intervals:
        if remaining.empty:
            break
        res = resolve(con, remaining, interval=interval)
        total_touched = max(total_touched, res.both_touched)
        if not res.frame.empty:
            got = res.frame.copy()
            # NORMALISE before comparing: DuckDB hands back datetime64 while
            # `remaining` holds datetime.date, so an un-normalised set match
            # silently removes NOTHING — the coarser pass then re-resolves days
            # the finer one already did and the verdicts double-count (observed:
            # "132/120 resolved (110%)").
            got["date"] = pd.to_datetime(got["date"]).dt.date
            parts.append(got)
            done = set(map(tuple, got[["symbol", "date"]].values))
            remaining = remaining[
                [tuple(r) not in done for r in remaining[["symbol", "date"]].values]
            ]
        last = res
    verdicts = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["symbol", "date", "seq"])
    )
    out = ResolutionResult(
        frame=verdicts,
        both_touched=total_touched,
        resolved=len(verdicts),
        interval=last.interval if last else intervals[-1],
        same_bar=last.same_bar if last else 0,
        no_intraday=last.no_intraday if last else 0,
        disagreed=last.disagreed if last else 0,
        gappy=last.gappy if last else 0,
    )
    logger.info("intraday resolution (%s): %s", "->".join(intervals), out.summary())
    return out


def stop_fills_best(
    con: duckdb.DuckDBPyConnection,
    pairs: pd.DataFrame,
    intervals: tuple[str, ...] = RESOLUTION_ORDER,
) -> pd.DataFrame:
    """Price the stop exit at the finest interval available.

    Finer is strictly better here: the proxy is the next bar's open, so a 1m bar
    is 60 seconds after the breach rather than up to 5 minutes, leaving far less
    room for a bounce to be priced in as though it were a fill.
    """
    parts: list[pd.DataFrame] = []
    remaining = pairs[["symbol", "date"]].drop_duplicates().copy()
    remaining["date"] = pd.to_datetime(remaining["date"]).dt.date
    for interval in intervals:
        if remaining.empty:
            break
        got = stop_fills(con, remaining, interval=interval)
        if got.empty:
            continue
        got = got.copy()
        got["date"] = pd.to_datetime(got["date"]).dt.date  # see resolve_best
        parts.append(got)
        done = set(map(tuple, got[["symbol", "date"]].values))
        remaining = remaining[[tuple(r) not in done for r in remaining[["symbol", "date"]].values]]
    # See resolve_best: never concat onto an empty template (dtype trap).
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["symbol", "date", "fill"])
    )


# A trailing stop exits when price falls this far below its running high-water
# mark, measured from the entry (the day's open). Matches the fixed stop's 1%
# so the two rules are comparable on the explorer.
TRAILING_DROP = 0.01


def trailing_exits(
    con: duckdb.DuckDBPyConnection,
    pairs: pd.DataFrame,
    drop: float = TRAILING_DROP,
    intervals: tuple[str, ...] = RESOLUTION_ORDER,
) -> pd.DataFrame:
    """Replay a trailing stop bar by bar. Returns per-session exit returns.

    This is the rule the dropdown has been advertising as impossible: daily bars
    genuinely cannot replay it, because a trailing stop depends on the ORDER of
    every move, not just the extremes.

    A STRICTER gate than ordering. resolve() needs the record intact only up to
    the FIRST trigger — nothing after it can change which came first. A trailing
    stop is path-dependent for the whole session: a missing bar can hide the
    high that would have raised the stop, or the dip that would have hit it. So
    this demands contiguity from the open through the CLOSE, and sessions that
    fall short are simply absent (the caller shows no number rather than a
    number built on gaps).

    INTRA-BAR ASSUMPTION — and the claim that it is conservative was WRONG.
    Within one bar we do not know whether the high or the low came first, and
    this assumes the HIGH did. The original reasoning was that ratcheting the
    high-water mark first makes the stop fire sooner, so a long exits earlier
    and captures less. Measured on 448 real sessions, the opposite holds: the
    high-first ordering returns +0.62pp PER SESSION more than low-first (better
    on 195, worse on 62), because firing sooner at a HIGHER trailing level
    frequently beats riding the same bar down. Cross-checked the other way, a 5m
    replay averages +0.66pp above the same sessions replayed at 1m.

    So the assumption FLATTERS. It is not a conservatism; it is an unpriced
    choice of the favourable branch, which is why the dropdown option is
    disabled until the result can be shown as a band (low-first to high-first)
    the way limit_stop bands an unresolvable ordering.

    Fills follow the same discipline as the fixed stop (#86): the exit is the
    next bar's open, capped at the trigger price, because a stop-market order
    cannot execute above its trigger. A trigger in the final bar has no next
    bar and uses that bar's close, capped the same way.
    """
    empty = pd.DataFrame(columns=["symbol", "date", "trail"])
    if pairs.empty:
        return empty
    want = pairs[["symbol", "date"]].drop_duplicates().copy()
    want["date"] = pd.to_datetime(want["date"]).dt.date

    out_rows: list[dict] = []
    done: set[tuple] = set()
    for interval in intervals:
        remaining = want[[tuple(r) not in done for r in want[["symbol", "date"]].values]]
        if remaining.empty:
            break
        bar_minutes = spec(interval)["minutes"]
        con.register("_trail_pairs", remaining)
        try:
            bars = con.execute(f"""
                WITH want AS (
                    SELECT DISTINCT symbol, CAST(date AS DATE) AS date FROM _trail_pairs
                ),
                agree AS ({session_agreement_sql(interval)}),
                gated AS (
                    SELECT w.symbol, w.date, d.open
                    FROM want w
                    JOIN prices d ON d.symbol = w.symbol AND d.date = w.date
                    JOIN agree ag ON ag.symbol = w.symbol AND ag.date = w.date AND ag.agrees
                    WHERE d.open > 0 AND isfinite(d.open)
                )
                SELECT g.symbol, g.date, g.open, i.ts, i.open AS bar_open,
                       i.high, i.low, i.close,
                       date_diff('minute',
                                 g.date + INTERVAL '{SESSION_OPEN_MINUTES}' MINUTE,
                                 i.ts) / {bar_minutes} AS slot
                FROM gated g
                JOIN intraday_prices i
                  ON i.symbol = g.symbol AND i.date = g.date AND i.interval = '{interval}'
                 AND i.ts >= g.date + INTERVAL '{SESSION_OPEN_MINUTES}' MINUTE
                ORDER BY g.symbol, g.date, i.ts
            """).fetchdf()
        finally:
            con.unregister("_trail_pairs")
        if bars.empty:
            continue
        for (sym, day), grp in bars.groupby(["symbol", "date"], sort=False):
            slots = grp["slot"].to_numpy()
            # FULL-session contiguity: every slot from the open present, no holes.
            if slots[0] != 0 or len(set(slots)) != int(slots[-1]) + 1:
                continue
            open_px = float(grp["open"].iloc[0])
            highs = grp["high"].to_numpy(dtype=float)
            lows = grp["low"].to_numpy(dtype=float)
            closes = grp["close"].to_numpy(dtype=float)
            bar_opens = grp["bar_open"].to_numpy(dtype=float)
            hwm = open_px
            exit_px = None
            for k in range(len(highs)):
                hwm = max(hwm, highs[k])  # conservative: high before low
                trigger = hwm * (1.0 - drop)
                if lows[k] <= trigger:
                    nxt = bar_opens[k + 1] if k + 1 < len(bar_opens) else closes[k]
                    exit_px = min(float(nxt), trigger)  # cannot fill above the trigger
                    break
            if exit_px is None:
                exit_px = float(closes[-1])  # never stopped out: exit at the close
            # NORMALISE before recording: `day` arrives from a DuckDB DATE as a
            # pandas Timestamp while `want` holds datetime.date, so an
            # un-normalised set match removes NOTHING — the coarse pass then
            # re-replays every session the fine pass already did and BOTH rows
            # are returned. The identical trap is documented 160 lines above and
            # handled in resolve_best and stop_fills_best; this function shipped
            # without it, and the duplicates corrupted every payload consumer.
            day = pd.Timestamp(day).date()
            out_rows.append({"symbol": sym, "date": day, "trail": (exit_px - open_px) / open_px})
            done.add((sym, day))
    return pd.DataFrame(out_rows, columns=["symbol", "date", "trail"]) if out_rows else empty


# Regular session close, minutes past midnight exchange-local. The last bar of a
# complete session STARTS one bar-width before it (15:55 at 5m, 15:59 at 1m).
SESSION_CLOSE_MINUTES = 16 * 60
# A session is short if its last bar starts more than this many minutes before
# the expected final slot. One bar of slack absorbs a venue's early close on a
# half day without hiding a real truncation.
TRUNCATION_SLACK_MINUTES = 5
# Coverage floor against the trailing median, mirroring the daily path's
# completeness gate (store.complete_trading_days uses 0.9 over 20 dates).
COVERAGE_MIN_FRACTION = 0.9
# SHORT window, deliberately. The check exists to catch a session that silently
# lost symbols — a failed batch, a rate limit. It is not meant to flag a
# sustained, intentional change in how many symbols we capture.
#
# It did exactly that: #94 bounded picked_symbols to the retention window, the
# nightly count dropped ~522 -> ~215 on 2026-08-11, and against a 20-session
# median every subsequent session read THIN for eleven sessions running. A check
# that cries wolf for a fortnight teaches the operator to ignore it, which is
# the same alarm-fatigue trap as the 1m boundary above.
#
# Five sessions absorbs a level shift within a week while leaving a one-day
# dropout just as loud: a session at 25% of its neighbours still fires
# immediately, because its neighbours are the comparison.
COVERAGE_MEDIAN_WINDOW = 5


def session_coverage(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per (interval, date) capture health, read from intraday_prices RAW.

    Diagnostics must read raw tables, never filtered views (CLAUDE.md): the
    agreement gate hides exactly the defective sessions a coverage check exists
    to find, and makes a truncated capture look like a merely un-resolvable one.

    Two independent failures, because they look identical downstream and need
    different fixes:

      * TRUNCATED — the session's last bar starts well before the close. A
        provider cutting the feed at 15:10 leaves a session that passes every
        per-bar check and is simply missing its final hour.
      * THIN — symbol count materially below the trailing median. A batch that
        failed silently, or a rate limit, removes names rather than minutes.

    Neither is visible in row counts alone, which is why the ingest step's
    "N bars; M/M symbols" could report success on both.
    """
    rows = []
    for interval in INTERVALS:
        minutes = spec(interval)["minutes"]
        expected_last = SESSION_CLOSE_MINUTES - minutes
        frame = con.execute(
            f"""
            WITH per_day AS (
                SELECT date,
                       count(*) AS bars,
                       count(DISTINCT symbol) AS symbols,
                       max(date_diff('minute', date, ts)) AS last_minute
                FROM intraday_prices
                WHERE interval = '{interval}'
                GROUP BY date
            )
            SELECT date, bars, symbols, last_minute,
                   median(symbols) OVER (
                       ORDER BY date ROWS BETWEEN {COVERAGE_MEDIAN_WINDOW} PRECEDING
                                              AND 1 PRECEDING
                   ) AS trailing_median_symbols
            FROM per_day ORDER BY date
            """
        ).fetchdf()
        if frame.empty:
            continue
        frame.insert(0, "interval", interval)
        frame["expected_last_minute"] = expected_last
        frame["truncated"] = frame["last_minute"] < expected_last - TRUNCATION_SLACK_MINUTES
        # A NULL trailing median (the first sessions ever captured) cannot be
        # judged against a baseline, so those are NOT called thin — same rule the
        # daily completeness gate uses, and made explicit rather than implied.
        frame["thin"] = frame["trailing_median_symbols"].notna() & (
            frame["symbols"] < COVERAGE_MIN_FRACTION * frame["trailing_median_symbols"]
        )
        rows.append(frame)
    if not rows:
        return pd.DataFrame(
            columns=[
                "interval",
                "date",
                "bars",
                "symbols",
                "last_minute",
                "trailing_median_symbols",
                "expected_last_minute",
                "truncated",
                "thin",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def coverage_problems(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Only the sessions that failed a coverage check — newest first."""
    frame = session_coverage(con)
    if frame.empty:
        return frame
    bad = frame[frame["truncated"] | frame["thin"]].copy()
    return bad.sort_values(["date", "interval"], ascending=[False, True])
