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


def _experiment(con, strategy="s", test_start=dt.date(2026, 1, 5), test_end=dt.date(2026, 2, 27)):
    return store.record_experiment(
        con,
        strategy=strategy,
        params={"months": 2},
        train_start=dt.date(2025, 1, 2),
        test_start=test_start,
        test_end=test_end,
        metrics={"lift": 1.5},
    )


def _daily_frame(dates, rets_by_rank: dict[int, list[float]]):
    """Rank-level daily rows: {rank: [ret per date]} → target_date, rank, ret, hit."""
    rows = [
        {"target_date": d, "rank": rank, "ret": rets[i], "hit": 1 if rets[i] >= 0.021 else 0}
        for i, d in enumerate(dates)
        for rank, rets in sorted(rets_by_rank.items())
    ]
    return pd.DataFrame(rows)


def test_record_experiment_returns_seq(con):
    seq1 = _experiment(con)
    seq2 = _experiment(con)
    assert isinstance(seq1, int)
    assert seq2 > seq1


def test_latest_experiment_daily_none_when_no_daily_rows(con):
    _experiment(con)  # aggregates only — predates the experiment_daily table
    assert store.latest_experiment_daily(con, "s") is None


def test_experiment_daily_roundtrip_most_days_wins(con):
    long_dates = sorted(pd.bdate_range("2026-01-05", periods=3).date)
    short_dates = long_dates[:2]
    long_seq = _experiment(con)
    store.record_experiment_daily(
        con, long_seq, _daily_frame(long_dates, {1: [0.031, -0.013, 0.022], 2: [0.011] * 3})
    )
    short_seq = _experiment(con)  # NEWER but shorter — must not displace the long record
    store.record_experiment_daily(con, short_seq, _daily_frame(short_dates, {1: [0.09, 0.09]}))
    _experiment(con)  # newest of all, but no daily rows — must be skipped
    other_seq = _experiment(con, strategy="other")
    store.record_experiment_daily(con, other_seq, _daily_frame(long_dates, {1: [0.05] * 3}))

    result = store.latest_experiment_daily(con, "s")
    assert result is not None
    meta, daily = result
    assert meta["seq"] == long_seq
    assert meta["test_start"] == dt.date(2026, 1, 5)
    assert meta["test_end"] == dt.date(2026, 2, 27)
    assert meta["params"]["months"] == 2
    assert "run_ts" in meta
    # Ordered by (target_date, rank); values roundtrip exactly.
    assert [pd.Timestamp(d).date() for d in daily["target_date"]] == [
        d for d in long_dates for _ in (1, 2)
    ]
    assert daily["rank"].tolist() == [1, 2, 1, 2, 1, 2]
    assert daily[daily["rank"] == 1]["ret"].tolist() == [0.031, -0.013, 0.022]
    assert daily[daily["rank"] == 1]["hit"].tolist() == [1, 0, 1]


def test_latest_experiment_daily_tiebreak_is_newest(con):
    dates = sorted(pd.bdate_range("2026-01-05", periods=2).date)
    old_seq = _experiment(con)
    store.record_experiment_daily(con, old_seq, _daily_frame(dates, {1: [0.01, 0.01]}))
    new_seq = _experiment(con)
    store.record_experiment_daily(con, new_seq, _daily_frame(dates, {1: [0.02, 0.02]}))
    meta, _ = store.latest_experiment_daily(con, "s")
    assert meta["seq"] == new_seq  # same day count → newest run wins


def test_record_experiment_daily_empty_frame_is_noop(con):
    seq = _experiment(con)
    assert store.record_experiment_daily(con, seq, pd.DataFrame()) == 0
    assert store.latest_experiment_daily(con, "s") is None


def test_record_experiment_daily_rejects_corrupt_rows(con):
    import numpy as np
    import pytest

    dates = sorted(pd.bdate_range("2026-01-05", periods=2).date)
    seq = _experiment(con)
    nan_ret = _daily_frame(dates, {1: [0.01, 0.02]})
    nan_ret.loc[0, "ret"] = np.nan
    with pytest.raises(ValueError, match="1 row"):
        store.record_experiment_daily(con, seq, nan_ret)

    null_hit = _daily_frame(dates, {1: [0.01, 0.02]})
    null_hit["hit"] = null_hit["hit"].astype("float64")
    null_hit.loc[1, "hit"] = np.nan
    with pytest.raises(ValueError, match="non-finite ret or null hit"):
        store.record_experiment_daily(con, seq, null_hit)

    # Nothing was persisted by the rejected writes.
    assert con.execute("SELECT count(*) FROM experiment_daily").fetchone()[0] == 0


def test_connect_drops_pre_release_experiment_daily(tmp_path, caplog):
    import duckdb

    path = tmp_path / "old.duckdb"
    old = duckdb.connect(str(path))
    old.execute(
        """
        CREATE TABLE experiment_daily (
            seq BIGINT NOT NULL, target_date DATE NOT NULL,
            top1_ret DOUBLE, top1_hit INTEGER, top5_ret DOUBLE, top5_hits DOUBLE,
            PRIMARY KEY (seq, target_date)
        );
        INSERT INTO experiment_daily VALUES (1, DATE '2026-01-05', 0.01, 0, 0.01, 0.2);
        """
    )
    old.close()

    with caplog.at_level("WARNING", logger="twopercent.store"):
        con = store.connect(path)
    assert "pre-release" in caplog.text and "1 sim row" in caplog.text
    # The new shape is usable immediately.
    seq = _experiment(con)
    dates = sorted(pd.bdate_range("2026-01-05", periods=2).date)
    assert store.record_experiment_daily(con, seq, _daily_frame(dates, {1: [0.01, 0.03]})) == 2
    meta, daily = store.latest_experiment_daily(con, "s")
    assert meta["seq"] == seq and daily["rank"].tolist() == [1, 1]


def test_record_ingest_from_keeps_earliest(con):
    store.record_ingest_from(con, ["AAPL"], dt.date(2024, 1, 1))
    store.record_ingest_from(con, ["AAPL"], dt.date(2025, 1, 1))  # later; must not regress
    assert store.ingest_from_dates(con)["AAPL"] == dt.date(2024, 1, 1)

    store.record_ingest_from(con, ["AAPL"], dt.date(2020, 1, 1))  # earlier; must win
    assert store.ingest_from_dates(con)["AAPL"] == dt.date(2020, 1, 1)
