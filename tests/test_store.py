import datetime as dt

import pandas as pd

from twopercent import store


def _price_rows():
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "AAPL", "NVDA"],
            "date": [dt.date(2026, 1, 5), dt.date(2026, 1, 6), dt.date(2026, 1, 6)],
            "open": [100.0, 101.0, 500.0],
            "high": [102.0, 103.0, 510.0],
            "low": [99.0, 100.0, 495.0],
            "close": [101.5, 102.5, 505.0],
            "adj_close": [101.0, 102.0, 504.0],
            "volume": [1_000_000, 1_100_000, 2_000_000],
        }
    )


def test_upsert_prices_is_idempotent(con):
    assert store.upsert_prices(con, _price_rows()) == 3
    store.upsert_prices(con, _price_rows())
    assert store.price_row_count(con) == 3


def test_last_price_dates(con):
    store.upsert_prices(con, _price_rows())
    dates = store.last_price_dates(con)
    assert dates["AAPL"] == dt.date(2026, 1, 6)
    assert dates["NVDA"] == dt.date(2026, 1, 6)


def test_universe_snapshot_roundtrip(con):
    df = pd.DataFrame(
        {"symbol": ["NVDA", "AAPL"], "name": ["NVIDIA", "Apple"], "market_cap": [5e12, 4e12]}
    )
    store.upsert_universe(con, df, as_of=dt.date(2026, 7, 17))
    latest = store.latest_universe(con)
    assert latest["symbol"].tolist() == ["NVDA", "AAPL"]  # market-cap order

    # A newer snapshot supersedes the old one.
    df2 = df.assign(market_cap=[6e12, 5e12])
    store.upsert_universe(con, df2, as_of=dt.date(2026, 7, 18))
    latest = store.latest_universe(con)
    assert latest["as_of"].iloc[0].date() == dt.date(2026, 7, 18)
    assert latest["market_cap"].iloc[0] == 6e12
