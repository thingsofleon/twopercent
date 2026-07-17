import datetime as dt
import math

import pandas as pd
import pytest

from twopercent import scan, store

DAY1 = dt.date(2026, 7, 14)
DAY2 = dt.date(2026, 7, 15)


@pytest.fixture
def seeded(con):
    rows = [
        # symbol, date, open, close  → oc_return
        ("BIG", DAY1, 100.0, 104.0),  # +4.0%
        ("EDGE", DAY1, 100.0, 102.0),  # exactly +2.0% — boundary, included
        ("ULP", DAY1, 5.0, 5.1),  # exactly +2.0% at an open whose FP result lands below 0.02
        ("MEH", DAY1, 100.0, 101.0),  # +1.0% — below threshold
        ("DOWN", DAY1, 100.0, 95.0),  # −5.0% — negative, excluded
        ("ZERO", DAY1, 0.0, 5.0),  # open=0 — excluded by the view guard
        ("NANO", DAY1, math.nan, 9.0),  # NaN open — excluded (DuckDB NaN > 0 is true!)
        ("BIG", DAY2, 104.0, 110.0),  # +5.77% on a later day
    ]
    store.upsert_prices(
        con,
        pd.DataFrame(
            {
                "symbol": [r[0] for r in rows],
                "date": [r[1] for r in rows],
                "open": [r[2] for r in rows],
                "high": [r[3] for r in rows],
                "low": [r[2] for r in rows],
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
    return con


def test_daily_movers_math_threshold_and_order(seeded):
    movers = scan.daily_movers(seeded, date=DAY1)

    assert movers["symbol"].tolist() == ["BIG", "EDGE", "ULP"]  # ordered by return desc
    assert movers["oc_return"].tolist()[:2] == [0.04, 0.02]
    excluded = {"MEH", "DOWN", "ZERO", "NANO"}
    assert not excluded & set(movers["symbol"])


def test_exact_threshold_included_regardless_of_open_price(seeded):
    # (5.1 - 5.0) / 5.0 is a hair below 0.02 in doubles; the epsilon keeps it in.
    movers = scan.daily_movers(seeded, date=DAY1)
    assert "ULP" in set(movers["symbol"])


def test_nan_open_rows_never_surface(seeded):
    # DuckDB's total ordering makes NaN > 0 true and NaN sort first — the view
    # must exclude such rows or they'd top every scan.
    movers = scan.daily_movers(seeded, date=DAY1)
    assert "NANO" not in set(movers["symbol"])
    assert not movers["oc_return"].isna().any()


def test_daily_movers_defaults_to_latest_date(seeded):
    movers = scan.daily_movers(seeded)
    assert movers["date"].iloc[0].date() == DAY2
    assert movers["symbol"].tolist() == ["BIG"]


def test_daily_movers_custom_threshold(seeded):
    movers = scan.daily_movers(seeded, date=DAY1, threshold=0.03)
    assert movers["symbol"].tolist() == ["BIG"]


def test_daily_movers_names_null_safe(seeded):
    seeded.execute("DELETE FROM universe")
    movers = scan.daily_movers(seeded, date=DAY1)
    assert movers["symbol"].tolist() == ["BIG", "EDGE", "ULP"]  # still listed
    assert movers["name"].isna().all()  # names simply absent


def test_daily_movers_empty_store(con):
    movers = scan.daily_movers(con)
    assert movers.empty


def test_counts_and_latest_date(seeded):
    assert scan.latest_price_date(seeded) == DAY2
    assert scan.price_count_on(seeded, DAY1) == 7
    assert scan.returns_count_on(seeded, DAY1) == 5  # ZERO and NANO excluded
    assert scan.price_count_on(seeded, dt.date(2026, 7, 13)) == 0


def test_latest_date_empty_store(con):
    assert scan.latest_price_date(con) is None
