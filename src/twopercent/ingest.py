"""Batched download of daily OHLCV from yfinance into the store."""

from __future__ import annotations

import datetime as dt
import logging
import time
from dataclasses import dataclass, field

import duckdb
import pandas as pd
import yfinance as yf

from twopercent import store

logger = logging.getLogger(__name__)

BATCH_SIZE = 150
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 5.0
# A symbol whose last stored bar is at least this recent is considered current
# and skipped, which makes interrupted runs resumable.
CURRENT_WITHIN_DAYS = 4


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
    out: list[pd.DataFrame] = []
    if not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("expected group_by='ticker' MultiIndex columns")
    for yf_sym in data.columns.get_level_values(0).unique():
        sub = data[yf_sym].dropna(subset=["Open", "Close"], how="any")
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

    Symbols whose stored data is already current are skipped, so a crashed or
    interrupted run picks up where it left off.
    """
    end = end or dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=round(years * 365.25))
    result = IngestResult()

    last_dates = store.last_price_dates(con)
    current_cutoff = dt.date.today() - dt.timedelta(days=CURRENT_WITHIN_DAYS)
    pending: list[str] = []
    for sym in symbols:
        if last_dates.get(sym) and last_dates[sym] >= current_cutoff:
            result.symbols_skipped.append(sym)
        else:
            pending.append(sym)

    for i in range(0, len(pending), batch_size):
        batch = pending[i : i + batch_size]
        yf_map = {to_yf_symbol(s): s for s in batch}
        try:
            data = _download_batch(list(yf_map), start, end)
        except RuntimeError:
            logger.exception("batch %s..%s failed", batch[0], batch[-1])
            result.symbols_failed.extend(batch)
            continue
        rows = frames_to_rows(data, yf_map)
        result.rows_written += store.upsert_prices(con, rows)
        got = set(rows["symbol"].unique())
        result.symbols_ok.extend(s for s in batch if s in got)
        result.symbols_failed.extend(s for s in batch if s not in got)
        logger.info(
            "batch %d/%d: %d rows, %d/%d symbols",
            i // batch_size + 1,
            -(-len(pending) // batch_size),
            len(rows),
            len(got),
            len(batch),
        )
    return result
