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


@dataclass
class IngestResult:
    rows_written: int = 0
    symbols_ok: list[str] = field(default_factory=list)
    symbols_skipped: list[str] = field(default_factory=list)
    symbols_failed: list[str] = field(default_factory=list)


def to_yf_symbol(symbol: str) -> str:
    """NASDAQ-style class shares (BRK.B, BF/B) use dashes on Yahoo."""
    return symbol.strip().replace("/", "-").replace(".", "-")


def frames_to_rows(data: pd.DataFrame, yf_to_symbol: dict[str, str]) -> pd.DataFrame:
    """Flatten a yf.download frame (group_by='ticker') into long price rows."""
    if not isinstance(data.columns, pd.MultiIndex):
        if len(yf_to_symbol) == 1:
            data = pd.concat({next(iter(yf_to_symbol)): data}, axis=1)
        else:
            raise ValueError("expected group_by='ticker' MultiIndex columns")
    out: list[pd.DataFrame] = []
    dropped_invalid = 0
    for yf_sym in data.columns.get_level_values(0).unique():
        sub = data[yf_sym].dropna(subset=["Open", "Close"], how="any")
        valid = (sub["Open"] > 0) & np.isfinite(sub["Open"]) & np.isfinite(sub["Close"])
        dropped_invalid += int((~valid).sum())
        sub = sub[valid]
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

    A symbol is skipped only when its stored coverage spans the requested
    window: ingested from at or before `start` (tracked in ingest_meta) AND
    with a last bar close to `end`. Symbols covered from `start` but stale at
    the end fetch only the missing tail; everything else fetches the full
    window. Interrupted or shorter prior runs therefore never block a backfill.
    """
    end = end or dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=round(years * 365.25))
    result = IngestResult()

    last_dates = store.last_price_dates(con)
    covered_from = store.ingest_from_dates(con)

    plan: list[tuple[str, dt.date]] = []  # (symbol, fetch start)
    for sym in symbols:
        last = last_dates.get(sym)
        covers_start = covered_from.get(sym) is not None and covered_from[sym] <= start
        if covers_start and last is not None:
            # Always refetch from the LAST stored bar inclusive — never skip.
            # The one bar a same-day skip would preserve is exactly the one
            # that can be a partial (mid-session) bar; refetching heals it.
            plan.append((sym, last))
        else:
            plan.append((sym, start))
    # Sort by fetch start so tail-fetches batch together instead of dragging a
    # full-history window for the whole batch.
    plan.sort(key=lambda item: item[1])

    n_batches = -(-len(plan) // batch_size)
    for i in range(0, len(plan), batch_size):
        batch = plan[i : i + batch_size]
        batch_syms = [sym for sym, _ in batch]
        batch_start = batch[0][1]
        yf_map = {to_yf_symbol(s): s for s in batch_syms}
        try:
            data = _download_batch(list(yf_map), batch_start, end)
            rows = frames_to_rows(data, yf_map)
            result.rows_written += store.upsert_prices(con, rows)
            got = set(rows["symbol"].unique())
        except Exception:
            logger.exception("batch %s..%s failed", batch_syms[0], batch_syms[-1])
            result.symbols_failed.extend(batch_syms)
            continue
        ok = [s for s in batch_syms if s in got]
        result.symbols_ok.extend(ok)
        result.symbols_failed.extend(s for s in batch_syms if s not in got)
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
