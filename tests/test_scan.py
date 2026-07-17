import datetime as dt

import pandas as pd

from twopercent import scan, store

DAY1 = dt.date(2026, 7, 14)
DAY2 = dt.date(2026, 7, 15)


def _seed(con):
    rows = [
        # symbol, date, open, close  → oc_return
        ("BIG", DAY1, 100.0, 104.0),  # +4.0%
        ("EDGE", DAY1, 100.0, 102.0),  # exactly +2.0% — boundary, included
        ("MEH", DAY1, 100.0, 101.0),  # +1.0% — below threshold
        ("DOWN", DAY1, 100.0, 95.0),  # −5.0% — negative, excluded
        ("ZERO", DAY1, 0.0, 5.0),  # open=0 — excluded by the view guard
        ("BIG", DAY2, 104.0, 110.0),  # +5.77% on a later day
    ]
    store.upsert_prices(
        con,
        pd.DataFrame(
            {
                "symbol": [r[0] for r in rows],
                "date": [r[1] for r in rows],
                "open": [r[2] for r in rows],
                "high": [max(r[2], r[3]) for r in rows],
                "low": [min(r[2], r[3]) for r in rows],
                "close": [r[3] for r in rows],
                "adj_close": [r[3] for r in rows],
                "volume": [1_000_000] * len(rows),
            }
        ),
    )
    store.upsert_universe(
        con,
        pd.DataFrame(
            {
                "symbol": ["BIG", "EDGE"],
                "name": ["Big Corp", "Edge Inc"],
                "market_cap": [2e9, 1e9],
            }
        ),
        as_of=DAY2,
    )


def test_daily_movers_math_threshold_and_order(con):
    _seed(con)
    movers = scan.daily_movers(con, date=DAY1)

    assert movers["symbol"].tolist() == ["BIG", "EDGE"]  # ordered by return desc
    assert movers["oc_return"].tolist() == [0.04, 0.02]  # exact per definition
    assert "MEH" not in set(movers["symbol"])  # below threshold
    assert "DOWN" not in set(movers["symbol"])  # negative excluded
    assert "ZERO" not in set(movers["symbol"])  # open=0 guarded


def test_daily_movers_defaults_to_latest_date(con):
    _seed(con)
    movers = scan.daily_movers(con)
    assert movers["date"].iloc[0].date() == DAY2
    assert movers["symbol"].tolist() == ["BIG"]


def test_daily_movers_custom_threshold(con):
    _seed(con)
    movers = scan.daily_movers(con, date=DAY1, threshold=0.03)
    assert movers["symbol"].tolist() == ["BIG"]


def test_daily_movers_names_null_safe(con):
    _seed(con)
    con.execute("DELETE FROM universe")
    movers = scan.daily_movers(con, date=DAY1)
    assert movers["symbol"].tolist() == ["BIG", "EDGE"]  # still listed
    assert movers["name"].isna().all()  # names simply absent


def test_daily_movers_empty_store(con):
    movers = scan.daily_movers(con)
    assert movers.empty


def test_price_count_and_latest_date(con):
    assert scan.latest_price_date(con) is None
    _seed(con)
    assert scan.latest_price_date(con) == DAY2
    assert scan.price_count_on(con, DAY1) == 5
    assert scan.price_count_on(con, dt.date(2026, 7, 13)) == 0
