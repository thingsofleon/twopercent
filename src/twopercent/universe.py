"""Ticker universe: top N US-listed common stocks by market cap (Russell 3000 proxy).

Source is the NASDAQ stock screener API, which lists all US-listed stocks with
market caps and needs no API key. Known limitations (see ROADMAP.md):
survivorship bias (today's constituents applied to history), and unlike the
real Russell 3000, foreign-domiciled US-listed ordinaries are included — a
deliberate widening, since they trade here and can do 2% days.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
EXCHANGES = ("NASDAQ", "NYSE", "AMEX")
TOP_N = 3000
_RETRY_SLEEP_SECONDS = 2.0

_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

# Listing types that aren't common stock. Word-bounded so company names like
# "Preferred Bank", "Wright Medical", or "MidCap Funding" survive.
_EXCLUDE_NAME_PATTERNS = (
    r"\betfs?\b",
    r"\bfunds?\b",
    r"\bwarrants?\b",
    r"\brights?\b",
    r"\bunits?(?:,| each\b| consisting\b|$)",
    r"\bpreferred (?:stock|shares?)\b",
    r"\bdepositary\b",
    r"\bnotes? due\b",
    r"%",
)
_EXCLUDE_PATTERN = "|".join(_EXCLUDE_NAME_PATTERNS)


def fetch_screener_rows(
    exchange: str,
    session: requests.Session | None = None,
    timeout: int = 30,
    retries: int = 3,
) -> list[dict]:
    """All screener rows for one exchange.

    The API sometimes answers HTTP 200 with `{"data": null}` when throttling,
    so an empty payload is retried and then raised as a clear error.
    """
    ses = session or requests.Session()
    last_status = None
    for attempt in range(retries):
        resp = ses.get(
            SCREENER_URL,
            params={"exchange": exchange, "limit": 25000, "download": "true"},
            headers=_HEADERS,
            timeout=timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        rows = (payload.get("data") or {}).get("rows")
        if rows:
            return rows
        last_status = payload.get("status")
        if attempt < retries - 1:
            time.sleep(_RETRY_SLEEP_SECONDS * (attempt + 1))
    raise RuntimeError(
        f"NASDAQ screener returned no rows for {exchange} (throttled?); status: {last_status}"
    )


def _parse_market_cap(raw: str | float | None) -> float:
    if raw is None or raw == "":
        return 0.0
    if isinstance(raw, int | float):
        return float(raw)
    try:
        return float(str(raw).replace(",", ""))
    except ValueError:
        return 0.0


def build_universe(rows: list[dict], top_n: int = TOP_N) -> pd.DataFrame:
    """Rank screener rows by market cap and keep the top N common stocks.

    Returns columns: symbol, name, market_cap.
    """
    df = pd.DataFrame(rows)
    df["symbol"] = df["symbol"].astype(str).str.strip()
    df["name"] = df["name"].astype(str).str.strip()
    df["market_cap"] = df["marketCap"].map(_parse_market_cap)

    bad_cap = df["market_cap"] <= 0
    if bad_cap.any():
        logger.warning(
            "%d screener rows dropped for missing/invalid market cap", int(bad_cap.sum())
        )
    df = df[~bad_cap]
    df = df[~df["symbol"].str.contains(r"[\^~]", regex=True)]
    df = df[~df["name"].str.lower().str.contains(_EXCLUDE_PATTERN, regex=True)]

    df = df.sort_values("market_cap", ascending=False)
    df = df.drop_duplicates(subset="symbol", keep="first")
    out = df[["symbol", "name", "market_cap"]].head(top_n).reset_index(drop=True)
    if len(out) < top_n:
        logger.warning("universe smaller than requested: %d < %d symbols", len(out), top_n)
    return out


def refresh_universe(top_n: int = TOP_N, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch all exchanges and build the current top-N universe."""
    ses = session or requests.Session()
    rows: list[dict] = []
    for exchange in EXCHANGES:
        rows.extend(fetch_screener_rows(exchange, session=ses))
    return build_universe(rows, top_n=top_n)
