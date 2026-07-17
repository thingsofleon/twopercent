import datetime as dt

import pandas as pd
import pytest
from tests.conftest import make_yf_frame

from twopercent import ingest, store


def test_to_yf_symbol():
    assert ingest.to_yf_symbol("BRK/B") == "BRK-B"
    assert ingest.to_yf_symbol("BF.B") == "BF-B"
    assert ingest.to_yf_symbol(" AAPL ") == "AAPL"


def test_frames_to_rows_flattens_multiindex():
    frame = make_yf_frame(["AAPL", "BRK-B"], days=4)
    rows = ingest.frames_to_rows(frame, {"AAPL": "AAPL", "BRK-B": "BRK/B"})
    assert len(rows) == 8
    assert set(rows["symbol"]) == {"AAPL", "BRK/B"}
    assert rows["volume"].dtype == "int64"


def test_frames_to_rows_drops_all_nan_symbols():
    frame = make_yf_frame(["AAPL", "DEAD"], days=4)
    frame["DEAD"] = float("nan")
    rows = ingest.frames_to_rows(frame, {"AAPL": "AAPL", "DEAD": "DEAD"})
    assert set(rows["symbol"]) == {"AAPL"}


def test_frames_to_rows_rejects_flat_columns():
    flat = pd.DataFrame({"Open": [1.0], "Close": [1.0]})
    with pytest.raises(ValueError, match="MultiIndex"):
        ingest.frames_to_rows(flat, {})


def test_ingest_writes_and_reports(con, monkeypatch):
    def fake_download(tickers, **kwargs):
        return make_yf_frame(sorted(tickers), days=5)

    monkeypatch.setattr(ingest.yf, "download", fake_download)
    result = ingest.ingest(con, ["AAPL", "NVDA"], years=1)

    assert result.rows_written == 10
    assert sorted(result.symbols_ok) == ["AAPL", "NVDA"]
    assert result.symbols_failed == []
    assert store.price_row_count(con) == 10


def test_ingest_resume_skips_current_symbols(con, monkeypatch):
    today = dt.date.today()
    store.upsert_prices(
        con,
        pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "date": [today],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "adj_close": [100.0],
                "volume": [1],
            }
        ),
    )

    calls = []

    def fake_download(tickers, **kwargs):
        calls.append(sorted(tickers))
        return make_yf_frame(sorted(tickers), days=5)

    monkeypatch.setattr(ingest.yf, "download", fake_download)
    result = ingest.ingest(con, ["AAPL", "NVDA"], years=1)

    assert result.symbols_skipped == ["AAPL"]
    assert calls == [["NVDA"]]  # AAPL never re-downloaded


def test_ingest_records_failed_batch(con, monkeypatch):
    def failing_download(tickers, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr(ingest.yf, "download", failing_download)
    monkeypatch.setattr(ingest, "RETRY_BACKOFF_SECONDS", 0.0)
    result = ingest.ingest(con, ["AAPL", "NVDA"], years=1)

    assert sorted(result.symbols_failed) == ["AAPL", "NVDA"]
    assert result.rows_written == 0
