"""Ticker universe: top N US common stocks by market cap (Russell 3000 proxy).

Source is the NASDAQ stock screener API, which lists all US-listed stocks with
market caps and needs no API key. Known limitation (see ROADMAP.md): applying
today's constituents to historical data introduces survivorship bias.
"""

from __future__ import annotations

import pandas as pd
import requests

SCREENER_URL = "https://api.nasdaq.com/api/screener/stocks"
EXCHANGES = ("NASDAQ", "NYSE", "AMEX")
TOP_N = 3000

_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
}

# Name fragments that indicate non-common-stock listings to exclude.
_EXCLUDE_NAME_FRAGMENTS = (
    " etf",
    " fund",
    " trust units",
    "warrant",
    " right",
    " unit ",
    "preferred",
    "depositary",
    "notes due",
)


def fetch_screener_rows(
    exchange: str, session: requests.Session | None = None, timeout: int = 30
) -> list[dict]:
    """All screener rows for one exchange."""
    ses = session or requests.Session()
    resp = ses.get(
        SCREENER_URL,
        params={"exchange": exchange, "limit": 25000, "download": "true"},
        headers=_HEADERS,
        timeout=timeout,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload["data"]["rows"]


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

    df = df[df["market_cap"] > 0]
    df = df[~df["symbol"].str.contains(r"[\^~]", regex=True)]
    lowered = df["name"].str.lower()
    for fragment in _EXCLUDE_NAME_FRAGMENTS:
        df = df[~lowered.str.contains(fragment, regex=False)]
        lowered = df["name"].str.lower()

    df = df.sort_values("market_cap", ascending=False)
    df = df.drop_duplicates(subset="symbol", keep="first")
    return df[["symbol", "name", "market_cap"]].head(top_n).reset_index(drop=True)


def refresh_universe(top_n: int = TOP_N, session: requests.Session | None = None) -> pd.DataFrame:
    """Fetch all exchanges and build the current top-N universe."""
    ses = session or requests.Session()
    rows: list[dict] = []
    for exchange in EXCHANGES:
        rows.extend(fetch_screener_rows(exchange, session=ses))
    return build_universe(rows, top_n=top_n)
