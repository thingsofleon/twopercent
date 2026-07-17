import pytest

from twopercent.universe import _parse_market_cap, build_universe, fetch_screener_rows


def test_parse_market_cap():
    assert _parse_market_cap("4,974,496,340,000") == 4_974_496_340_000.0
    assert _parse_market_cap("") == 0.0
    assert _parse_market_cap(None) == 0.0
    assert _parse_market_cap("n/a") == 0.0
    assert _parse_market_cap(123.0) == 123.0


def test_build_universe_ranks_and_filters(screener_rows):
    df = build_universe(screener_rows, top_n=10)
    symbols = df["symbol"].tolist()

    assert symbols[0] == "NVDA"  # ranked by market cap
    assert "SPY" not in symbols  # ETF excluded
    assert "FOO.W" not in symbols  # warrant excluded
    assert "NOCAP" not in symbols  # missing market cap excluded
    assert symbols.count("AAPL") == 1  # deduplicated
    assert "BRK/B" in symbols  # class shares kept


def test_build_universe_respects_top_n(screener_rows):
    df = build_universe(screener_rows, top_n=2)
    assert df["symbol"].tolist() == ["NVDA", "AAPL"]


def test_build_universe_keeps_sector_and_tolerates_missing():
    rows = [
        {
            "symbol": "NVDA",
            "name": "NVIDIA Corporation Common Stock",
            "marketCap": "4,974,496,340,000",
            "sector": "Technology",
        },
        {
            "symbol": "XOM",
            "name": "Exxon Mobil Corporation Common Stock",
            "marketCap": "500,000,000,000",
            "sector": " Energy ",  # screener whitespace stripped
        },
        {
            "symbol": "NOSEC",
            "name": "No Sector Inc Common Stock",
            "marketCap": "400,000,000,000",
            # no sector key at all — must NOT drop the row
        },
        {
            "symbol": "NULLSEC",
            "name": "Null Sector Inc Common Stock",
            "marketCap": "300,000,000,000",
            "sector": None,
        },
    ]
    df = build_universe(rows, top_n=10).set_index("symbol")
    assert df.loc["NVDA", "sector"] == "Technology"
    assert df.loc["XOM", "sector"] == "Energy"
    assert df.loc["NOSEC", "sector"] == ""  # kept, empty sector
    assert df.loc["NULLSEC", "sector"] == ""


def test_build_universe_without_any_sector_field(screener_rows):
    # Pre-sector payloads (no sector key anywhere) still build, with empty sectors.
    df = build_universe(screener_rows, top_n=10)
    assert "sector" in df.columns
    assert (df["sector"] == "").all()


def test_build_universe_warns_on_missing_sectors(caplog):
    rows = [
        {
            "symbol": "NVDA",
            "name": "NVIDIA Corporation Common Stock",
            "marketCap": "4,974,496,340,000",
            "sector": "Technology",
        },
        {
            "symbol": "NOSEC",
            "name": "No Sector Inc Common Stock",
            "marketCap": "400,000,000,000",
        },
        {
            "symbol": "NULLSEC",
            "name": "Null Sector Inc Common Stock",
            "marketCap": "300,000,000,000",
            "sector": None,
        },
    ]
    build_universe(rows, top_n=10)
    assert "2 of 3 universe rows have no sector" in caplog.text


def test_build_universe_full_sector_coverage_stays_quiet(caplog):
    rows = [
        {
            "symbol": "NVDA",
            "name": "NVIDIA Corporation Common Stock",
            "marketCap": "4,974,496,340,000",
            "sector": "Technology",
        },
    ]
    build_universe(rows, top_n=1)
    assert "have no sector" not in caplog.text


def test_exclude_patterns_are_word_bounded():
    rows = [
        {"symbol": "PFBC", "name": "Preferred Bank Common Stock", "marketCap": "2,000,000,000"},
        {"symbol": "WMGI", "name": "Wright Medical Group N.V.", "marketCap": "3,000,000,000"},
        {"symbol": "BSIG", "name": "BrightSphere Investment Group", "marketCap": "1,500,000,000"},
        {"symbol": "MCF", "name": "MidCap Funding Corp", "marketCap": "1,200,000,000"},
        {"symbol": "BAD1", "name": "Foo Inc 5.25% Preferred Stock", "marketCap": "9,000,000,000"},
        {
            "symbol": "BAD2",
            "name": "Acme Acquisition Corp Units, each consisting of one share",
            "marketCap": "8,000,000,000",
        },
        {"symbol": "BAD3", "name": "Bar Capital Rights", "marketCap": "7,000,000,000"},
        {"symbol": "BAD4", "name": "Baz Global Fund", "marketCap": "6,000,000,000"},
    ]
    symbols = set(build_universe(rows, top_n=10)["symbol"])
    assert {"PFBC", "WMGI", "BSIG", "MCF"} <= symbols  # legitimate companies kept
    assert not {"BAD1", "BAD2", "BAD3", "BAD4"} & symbols  # listing types excluded


def test_fetch_screener_rows_raises_clearly_on_null_data():
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": None, "status": {"rCode": 403}}

    class FakeSession:
        def get(self, *args, **kwargs):
            return FakeResponse()

    with pytest.raises(RuntimeError, match="no rows for NASDAQ"):
        fetch_screener_rows("NASDAQ", session=FakeSession(), retries=1)
