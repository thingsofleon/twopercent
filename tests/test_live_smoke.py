"""Live network smoke tests — run explicitly with `pytest -m live`."""

import pytest

from twopercent import ingest, store, universe


@pytest.mark.live
def test_screener_returns_broad_universe():
    df = universe.refresh_universe(top_n=3000)
    assert len(df) == 3000
    assert {"AAPL", "NVDA"} <= set(df["symbol"])


@pytest.mark.live
def test_ingest_three_real_tickers(tmp_path):
    con = store.connect(tmp_path / "live.duckdb")
    result = ingest.ingest(con, ["AAPL", "MSFT", "BRK/B"], years=0.1)
    assert result.rows_written > 0
    assert not result.symbols_failed
