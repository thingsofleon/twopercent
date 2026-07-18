"""Batched download of daily OHLCV from yfinance into the store."""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd
import yfinance as yf

from twopercent import store

logger = logging.getLogger(__name__)

BATCH_SIZE = 150
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0
# Split-artifact rejection (issue #25): yfinance sometimes serves a bar whose
# open is on the pre-split price scale while the close is post-split (e.g.
# DRUG 2024-10-15, +1369% "intraday"). A bar is rejected when BOTH its
# open-to-close move is extreme AND its open sits on a different price scale
# than the PRIOR bar's close. Deliberately conservative: a genuine extreme
# move with a continuous open is never touched; only same-bar and prior-bar
# data are consulted (no lookahead).
SPLIT_ARTIFACT_OC = 0.5
SPLIT_ARTIFACT_SCALE = 2.0
# FP guard on the strict > / < threshold comparisons: an exactly-at-boundary
# bar (e.g. a true 50% move at open 5.70) computes a few ULPs either side of
# the threshold in double arithmetic. Flagging is destructive, so the epsilon
# widens the KEEP region — boundary bars are never flagged, and the pandas
# (ingest) and SQL (doctor) implementations agree.
_SPLIT_EPSILON = 1e-9


DORMANT_AFTER_DAYS = 30


@dataclass
class IngestResult:
    rows_written: int = 0
    symbols_ok: list[str] = field(default_factory=list)
    symbols_skipped: list[str] = field(default_factory=list)
    symbols_failed: list[str] = field(default_factory=list)
    symbols_dormant: list[str] = field(default_factory=list)


def to_yf_symbol(symbol: str) -> str:
    """NASDAQ-style class shares (BRK.B, BF/B) use dashes on Yahoo."""
    return symbol.strip().replace("/", "-").replace(".", "-")


def frames_to_rows(
    data: pd.DataFrame,
    yf_to_symbol: dict[str, str],
    last_bars: dict[str, tuple[dt.date, float | None]] | None = None,
) -> pd.DataFrame:
    """Flatten a yf.download frame (group_by='ticker') into long price rows.

    `last_bars` maps symbol -> (last stored date, close) and seeds the
    split-artifact prev_close for the FIRST in-frame bar — without it the
    artifact rule is blind on tail fetches (a single-bar daily update has no
    in-frame prior bar). The seed applies only when the first bar is strictly
    after the stored date, so it stays correct whether tail fetches start the
    day after the last stored bar or at the last stored bar itself.
    """
    if not isinstance(data.columns, pd.MultiIndex):
        if len(yf_to_symbol) == 1:
            data = pd.concat({next(iter(yf_to_symbol)): data}, axis=1)
        else:
            raise ValueError("expected group_by='ticker' MultiIndex columns")
    out: list[pd.DataFrame] = []
    dropped_invalid = 0
    dropped_split = 0
    split_symbols: list[str] = []
    for yf_sym in data.columns.get_level_values(0).unique():
        sub = data[yf_sym].dropna(subset=["Open", "Close"], how="any")
        valid = (sub["Open"] > 0) & np.isfinite(sub["Open"]) & np.isfinite(sub["Close"])
        dropped_invalid += int((~valid).sum())
        sub = sub[valid]
        if sub.empty:
            continue
        oc_return = (sub["Close"] - sub["Open"]) / sub["Open"]
        prev_close = sub["Close"].shift(1)
        seed = (last_bars or {}).get(yf_to_symbol[yf_sym])
        if seed is not None:
            seed_date, seed_close = seed
            if (
                seed_close is not None
                and np.isfinite(seed_close)
                and seed_close > 0
                and sub.index[0].date() > seed_date
            ):
                prev_close.iloc[0] = seed_close
        scale = sub["Open"] / prev_close
        artifact = (
            (oc_return.abs() > SPLIT_ARTIFACT_OC + _SPLIT_EPSILON)
            & (prev_close > 0)
            & (
                (scale > SPLIT_ARTIFACT_SCALE + _SPLIT_EPSILON)
                | (scale < 1 / SPLIT_ARTIFACT_SCALE - _SPLIT_EPSILON)
            )
        )
        if artifact.any():
            dropped_split += int(artifact.sum())
            split_symbols.append(yf_to_symbol[yf_sym])
            sub = sub[~artifact]
        if sub.empty:
            continue
        frame = pd.DataFrame(
            {
                "symbol": yf_to_symbol[yf_sym],
                "date": pd.to_datetime(sub.index).date,
                "open": sub["Open"].to_numpy(),
                "high": sub["High"].to_numpy(),
                "low": sub["Low"].to_numpy(),
                "close": sub["Close"].to_numpy(),
                "adj_close": sub["Adj Close"].to_numpy(),
                "volume": sub["Volume"].fillna(0).astype("int64").to_numpy(),
            }
        )
        out.append(frame)
    if dropped_invalid:
        logger.warning(
            "%d rows dropped for invalid open/close (<=0 or non-finite)", dropped_invalid
        )
    if dropped_split:
        logger.warning(
            "%d bars dropped as split artifacts (|oc_return| > %.0f%% with open "
            "off-scale vs prior close): %s",
            dropped_split,
            SPLIT_ARTIFACT_OC * 100,
            ", ".join(sorted(split_symbols)),
        )
    if not out:
        return pd.DataFrame(
            columns=["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
        )
    return pd.concat(out, ignore_index=True)


def _download_batch(
    yf_symbols: list[str], start: dt.date, end: dt.date, retries: int = MAX_RETRIES
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            data = yf.download(
                tickers=yf_symbols,
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if data is not None and not data.empty:
                return data
            last_error = ValueError("empty response")
        except Exception as exc:  # yfinance raises a grab-bag of exception types
            last_error = exc
        if attempt < retries - 1:
            time.sleep(RETRY_BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError(f"batch download failed after {retries} attempts: {last_error}")


def ingest(
    con: duckdb.DuckDBPyConnection,
    symbols: list[str],
    years: float = 5,
    end: dt.date | None = None,
    batch_size: int = BATCH_SIZE,
) -> IngestResult:
    """Download daily bars for `symbols` and upsert into the store.

    Symbols covered from `start` (tracked in ingest_meta) tail-fetch from
    their LAST stored bar inclusive — never skipped, so a partial bar from a
    mid-session run is healed by the next run's overwrite. Everything else
    fetches the full window; interrupted or shorter prior runs never block a
    backfill. The last stored close per symbol seeds split-artifact detection
    for the first bar of each tail fetch.
    """
    end = end or dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=round(years * 365.25))
    result = IngestResult()

    last_bars = store.last_price_bars(con)
    last_dates = {sym: bar[0] for sym, bar in last_bars.items()}
    covered_from = store.ingest_from_dates(con)

    dormant_cutoff = end - dt.timedelta(days=DORMANT_AFTER_DAYS)
    plan: list[tuple[str, dt.date]] = []  # (symbol, fetch start)
    for sym in symbols:
        last = last_dates.get(sym)
        covers_start = covered_from.get(sym) is not None and covered_from[sym] <= start
        if covers_start and last is not None and last < dormant_cutoff:
            # Delisted/halted names kept for history: requesting their empty
            # tail every day would count as a "failure" forever and train
            # operators to ignore the failure-rate gate. Excluded loudly; a
            # full backfill (fresh ingest_meta) still refetches them.
            result.symbols_dormant.append(sym)
        elif covers_start and last is not None:
            # Always refetch from the LAST stored bar inclusive — never skip.
            # The one bar a same-day skip would preserve is exactly the one
            # that can be a partial (mid-session) bar; refetching heals it.
            plan.append((sym, last))
        else:
            plan.append((sym, start))
    if result.symbols_dormant:
        logger.warning(
            "%d symbols dormant (no bar in %d days) — not fetched: %s%s",
            len(result.symbols_dormant),
            DORMANT_AFTER_DAYS,
            ", ".join(result.symbols_dormant[:10]),
            " ..." if len(result.symbols_dormant) > 10 else "",
        )
    # Sort by fetch start so tail-fetches batch together instead of dragging a
    # full-history window for the whole batch.
    plan.sort(key=lambda item: item[1])

    planned_start = dict(plan)

    def classify_missing(syms: list[str]) -> None:
        # A symbol that returned nothing but whose stored last bar already
        # covers its requested start is current-with-retained-bar, not failed:
        # refetch-empties happen in bursts (provider rate limiting reads as
        # "possibly delisted") and would otherwise trip the routine's failure
        # gate on healthy data. We keep the stored bar and count it loudly.
        for s in syms:
            last = last_dates.get(s)
            if last is not None and last >= planned_start[s]:
                result.symbols_skipped.append(s)
            else:
                result.symbols_failed.append(s)

    n_batches = -(-len(plan) // batch_size)
    for i in range(0, len(plan), batch_size):
        batch = plan[i : i + batch_size]
        batch_syms = [sym for sym, _ in batch]
        batch_start = batch[0][1]
        yf_map = {to_yf_symbol(s): s for s in batch_syms}
        try:
            data = _download_batch(list(yf_map), batch_start, end)
            rows = frames_to_rows(data, yf_map, last_bars)
            result.rows_written += store.upsert_prices(con, rows)
            got = set(rows["symbol"].unique())
        except Exception:
            logger.exception("batch %s..%s failed", batch_syms[0], batch_syms[-1])
            classify_missing(batch_syms)
            continue
        ok = [s for s in batch_syms if s in got]
        result.symbols_ok.extend(ok)
        classify_missing([s for s in batch_syms if s not in got])
        store.record_ingest_from(con, ok, start)
        logger.info(
            "batch %d/%d: %d rows, %d/%d symbols",
            i // batch_size + 1,
            n_batches,
            len(rows),
            len(ok),
            len(batch_syms),
        )
    return result
