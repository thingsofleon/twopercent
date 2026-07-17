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


def test_universe_sector_roundtrip(con):
    df = pd.DataFrame(
        {
            "symbol": ["NVDA", "XOM", "MYST"],
            "name": ["NVIDIA", "Exxon", "Mystery Co"],
            "market_cap": [5e12, 5e11, 1e9],
            "sector": ["Technology", "Energy", ""],
        }
    )
    store.upsert_universe(con, df, as_of=dt.date(2026, 7, 17))
    latest = store.latest_universe(con).set_index("symbol")
    assert latest.loc["NVDA", "sector"] == "Technology"
    assert latest.loc["XOM", "sector"] == "Energy"
    assert latest.loc["MYST", "sector"] == ""


def test_upsert_universe_without_sector_column(con):
    # Pre-sector callers pass frames without the column; sector defaults to "".
    df = pd.DataFrame({"symbol": ["AAPL"], "name": ["Apple"], "market_cap": [4e12]})
    assert store.upsert_universe(con, df, as_of=dt.date(2026, 7, 17)) == 1
    latest = store.latest_universe(con)
    assert latest["sector"].tolist() == [""]


def test_connect_migrates_pre_sector_database(tmp_path):
    # Simulate a database created before the sector column existed.
    import duckdb

    path = tmp_path / "old.duckdb"
    old = duckdb.connect(str(path))
    old.execute(
        """
        CREATE TABLE universe (
            symbol TEXT NOT NULL,
            name TEXT NOT NULL,
            market_cap DOUBLE NOT NULL,
            as_of DATE NOT NULL,
            PRIMARY KEY (symbol, as_of)
        );
        INSERT INTO universe VALUES ('AAPL', 'Apple', 4e12, DATE '2026-07-16');
        """
    )
    old.close()

    con = store.connect(path)  # must add the sector column on connect
    latest = store.latest_universe(con)
    assert latest["symbol"].tolist() == ["AAPL"]
    assert latest["sector"].isna().all()  # migrated rows have NULL sector

    df = pd.DataFrame(
        {"symbol": ["NVDA"], "name": ["NVIDIA"], "market_cap": [5e12], "sector": ["Technology"]}
    )
    store.upsert_universe(con, df, as_of=dt.date(2026, 7, 17))
    assert store.latest_universe(con)["sector"].tolist() == ["Technology"]


def test_all_universe_symbols_unions_snapshots(con):
    df1 = pd.DataFrame(
        {"symbol": ["AAPL", "EDGE"], "name": ["Apple", "Edge Co"], "market_cap": [4e12, 1e9]}
    )
    store.upsert_universe(con, df1, as_of=dt.date(2026, 7, 16))
    # EDGE churns out of the next snapshot but must remain ingestable.
    df2 = pd.DataFrame(
        {"symbol": ["AAPL", "NVDA"], "name": ["Apple", "NVIDIA"], "market_cap": [4e12, 5e12]}
    )
    store.upsert_universe(con, df2, as_of=dt.date(2026, 7, 17))

    symbols = store.all_universe_symbols(con)
    assert set(symbols) == {"AAPL", "NVDA", "EDGE"}
    assert symbols[0] == "NVDA"  # largest cap first


def test_record_ingest_from_keeps_earliest(con):
    store.record_ingest_from(con, ["AAPL"], dt.date(2024, 1, 1))
    store.record_ingest_from(con, ["AAPL"], dt.date(2025, 1, 1))  # later; must not regress
    assert store.ingest_from_dates(con)["AAPL"] == dt.date(2024, 1, 1)

    store.record_ingest_from(con, ["AAPL"], dt.date(2020, 1, 1))  # earlier; must win
    assert store.ingest_from_dates(con)["AAPL"] == dt.date(2020, 1, 1)
