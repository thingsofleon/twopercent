from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twopercent import store


@pytest.fixture
def con(tmp_path):
    return store.connect(tmp_path / "test.duckdb")


@pytest.fixture
def screener_rows():
    """Canned NASDAQ screener payload rows, deliberately messy."""
    return [
        {
            "symbol": "NVDA",
            "name": "NVIDIA Corporation Common Stock",
            "marketCap": "4,974,496,340,000",
        },
        {"symbol": "AAPL", "name": "Apple Inc. Common Stock", "marketCap": "4,853,994,909,728"},
        {"symbol": "SMALL", "name": "Small Co Common Stock", "marketCap": "1,000,000"},
        {"symbol": "TINY", "name": "Tiny Co Common Stock", "marketCap": "500,000"},
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "marketCap": "600,000,000,000"},
        {"symbol": "FOO.W", "name": "Foo Inc Warrant", "marketCap": "10,000,000"},
        {"symbol": "NOCAP", "name": "No Cap Inc Common Stock", "marketCap": ""},
        {"symbol": "AAPL", "name": "Apple Inc. Common Stock", "marketCap": "4,853,994,909,728"},
        {
            "symbol": "BRK/B",
            "name": "Berkshire Hathaway Class B Common Stock",
            "marketCap": "1,100,000,000,000",
        },
    ]


def seed_history(
    con, oc_returns: dict[str, list[float]], start="2026-01-05", vary_volume: bool = False
) -> pd.DataFrame:
    """Seed prices for symbols with exact open-to-close returns per business day.

    open is always 100.0, so close = 100 * (1 + oc). vary_volume avoids
    constant feature columns (sklearn's binner rejects single-valued columns).
    """
    from twopercent import store

    frames = []
    for symbol, ocs in oc_returns.items():
        n = len(ocs)
        dates = pd.bdate_range(start, periods=n)
        opens = np.full(n, 100.0)
        closes = opens * (1 + np.asarray(ocs))
        volume = 1_000_000 + (np.arange(n) % 17) * 1_000 if vary_volume else 1_000_000
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates.date,
                    "open": opens,
                    "high": np.maximum(opens, closes) * 1.001,
                    "low": np.minimum(opens, closes) * 0.999,
                    "close": closes,
                    "adj_close": closes,
                    "volume": volume,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    store.upsert_prices(con, df)
    return df


def make_yf_frame(symbols: list[str], days: int = 5, start_price: float = 100.0) -> pd.DataFrame:
    """Synthetic yf.download output: MultiIndex (ticker, field) columns."""
    dates = pd.bdate_range("2026-01-05", periods=days)
    frames = {}
    for i, sym in enumerate(symbols):
        base = start_price * (1 + i)
        opens = np.linspace(base, base * 1.05, days)
        closes = opens * 1.01
        frames[sym] = pd.DataFrame(
            {
                "Open": opens,
                "High": closes * 1.01,
                "Low": opens * 0.99,
                "Close": closes,
                "Adj Close": closes * 0.98,
                "Volume": np.full(days, 1_000_000.0),
            },
            index=dates,
        )
    return pd.concat(frames, axis=1)
