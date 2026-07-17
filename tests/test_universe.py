from twopercent.universe import _parse_market_cap, build_universe


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
