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
