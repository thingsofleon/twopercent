import datetime as dt

import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_yf_frame
from twopercent import ingest, store


def _yf_frame_from_bars(symbol: str, bars: list[tuple[float, float]]) -> pd.DataFrame:
    """Single-symbol yf.download-shaped frame from explicit (open, close) bars."""
    opens = np.array([b[0] for b in bars])
    closes = np.array([b[1] for b in bars])
    frame = pd.DataFrame(
        {
            "Open": opens,
            "High": np.maximum(opens, closes),
            "Low": np.minimum(opens, closes),
            "Close": closes,
            "Adj Close": closes,
            "Volume": np.full(len(bars), 1_000_000.0),
        },
        index=pd.bdate_range("2026-01-05", periods=len(bars)),
    )
    return pd.concat({symbol: frame}, axis=1)


def _seed_price(con, symbol: str, date: dt.date) -> None:
    store.upsert_prices(
        con,
        pd.DataFrame(
            {
                "symbol": [symbol],
                "date": [date],
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.5],
                "adj_close": [100.0],
                "volume": [1],
            }
        ),
    )


@pytest.fixture
def download_calls(monkeypatch):
    """Monkeypatch yf.download; records (sorted tickers, start) per call."""
    calls: list[tuple[list[str], str]] = []

    def fake_download(tickers, start=None, **kwargs):
        calls.append((sorted(tickers), start))
        return make_yf_frame(sorted(tickers), days=5)

    monkeypatch.setattr(ingest.yf, "download", fake_download)
    return calls


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


def test_frames_to_rows_drops_invalid_opens_loudly(caplog):
    frame = make_yf_frame(["AAPL"], days=4)
    frame.loc[frame.index[0], ("AAPL", "Open")] = 0.0
    rows = ingest.frames_to_rows(frame, {"AAPL": "AAPL"})
    assert len(rows) == 3  # zero-open row rejected at ingest, not just hidden by the view
    assert "dropped for invalid open/close" in caplog.text


def test_frames_to_rows_drops_split_artifacts_loudly_both_directions(caplog):
    # Bar 2: open 10x the prior close (pre-split scale), -90% "intraday".
    # Bar 4: open at a tenth of the prior close, +940% "intraday".
    bars = [
        (10.0, 10.1),
        (102.0, 10.2),
        (10.2, 10.3),
        (1.0, 10.4),
    ]
    rows = ingest.frames_to_rows(_yf_frame_from_bars("SPLITX", bars), {"SPLITX": "SPLITX"})
    assert len(rows) == 2
    assert rows["open"].tolist() == [10.0, 10.2]
    assert "2 bars dropped as split artifacts" in caplog.text
    assert "SPLITX" in caplog.text


def test_frames_to_rows_keeps_genuine_extreme_move_with_continuous_open(caplog):
    # +60% open-to-close, but the open agrees with the prior close: real move.
    bars = [(10.0, 10.1), (10.1, 16.16)]
    rows = ingest.frames_to_rows(_yf_frame_from_bars("MOON", bars), {"MOON": "MOON"})
    assert len(rows) == 2
    assert rows["close"].tolist() == [10.1, 16.16]
    assert "split artifact" not in caplog.text


def test_frames_to_rows_never_flags_first_bar_of_symbol(caplog):
    # First bar has no prior close to disagree with — even a +300% bar stays.
    bars = [(10.0, 40.0), (40.0, 40.4)]
    rows = ingest.frames_to_rows(_yf_frame_from_bars("IPO", bars), {"IPO": "IPO"})
    assert len(rows) == 2
    assert "split artifact" not in caplog.text


def test_frames_to_rows_keeps_scale_gap_without_extreme_oc_move(caplog):
    # An open off-scale vs prior close but a calm intraday bar (a real split
    # with clean OHLC, or a huge overnight gap) must NOT be dropped.
    bars = [(100.0, 101.0), (10.1, 10.2)]
    rows = ingest.frames_to_rows(_yf_frame_from_bars("REV", bars), {"REV": "REV"})
    assert len(rows) == 2
    assert "split artifact" not in caplog.text


def test_frames_to_rows_accepts_flat_columns_for_single_symbol():
    flat = make_yf_frame(["AAPL"], days=4)["AAPL"]
    rows = ingest.frames_to_rows(flat, {"AAPL": "AAPL"})
    assert len(rows) == 4

    with pytest.raises(ValueError, match="MultiIndex"):
        ingest.frames_to_rows(flat, {"AAPL": "AAPL", "NVDA": "NVDA"})


def test_ingest_writes_and_reports(con, download_calls):
    result = ingest.ingest(con, ["AAPL", "NVDA"], years=1)

    assert result.rows_written == 10
    assert sorted(result.symbols_ok) == ["AAPL", "NVDA"]
    assert result.symbols_failed == []
    assert store.price_row_count(con) == 10


def test_ingest_skips_only_fully_covered_symbols(con, download_calls):
    today = dt.date.today()
    _seed_price(con, "AAPL", today)
    store.record_ingest_from(con, ["AAPL"], dt.date(2000, 1, 1))

    result = ingest.ingest(con, ["AAPL", "NVDA"], years=1)

    assert result.symbols_skipped == ["AAPL"]
    assert [c[0] for c in download_calls] == [["NVDA"]]  # AAPL never re-downloaded


def test_ingest_backfills_when_prior_run_was_shorter(con, download_calls):
    # A recent bar alone (no coverage back to the requested start) must NOT
    # cause a skip — this was the bug that let a 1-month run block a 5-year one.
    today = dt.date.today()
    _seed_price(con, "AAPL", today)
    store.record_ingest_from(con, ["AAPL"], today - dt.timedelta(days=30))

    result = ingest.ingest(con, ["AAPL"], years=5)

    assert result.symbols_skipped == []
    assert [c[0] for c in download_calls] == [["AAPL"]]
    expected_start = (dt.date.today() + dt.timedelta(days=1)) - dt.timedelta(days=round(5 * 365.25))
    assert download_calls[0][1] == expected_start.isoformat()


def test_ingest_historical_window_ignores_todays_freshness(con, download_calls):
    # Explicit historical end: freshness must be judged against that end, not
    # today — current data with late coverage must still fetch the old window.
    _seed_price(con, "AAPL", dt.date.today())
    store.record_ingest_from(con, ["AAPL"], dt.date(2025, 1, 1))

    result = ingest.ingest(con, ["AAPL"], years=1, end=dt.date(2024, 1, 1))

    assert result.symbols_skipped == []
    assert (
        download_calls[0][1] == (dt.date(2024, 1, 1) - dt.timedelta(days=round(365.25))).isoformat()
    )


def test_ingest_fetches_only_missing_tail(con, download_calls):
    # Covered-from-start but stale symbols fetch from their last bar forward,
    # not the whole window again.
    last = dt.date.today() - dt.timedelta(days=30)
    _seed_price(con, "AAPL", last)
    store.record_ingest_from(con, ["AAPL"], dt.date(2000, 1, 1))

    ingest.ingest(con, ["AAPL"], years=1)

    assert download_calls[0][1] == (last + dt.timedelta(days=1)).isoformat()


def test_ingest_continues_after_postprocessing_error(con, monkeypatch, download_calls):
    original = ingest.frames_to_rows
    failed_once = []

    def flaky(data, yf_map):
        if not failed_once:
            failed_once.append(True)
            raise KeyError("malformed batch")
        return original(data, yf_map)

    monkeypatch.setattr(ingest, "frames_to_rows", flaky)
    result = ingest.ingest(con, ["AAPL", "NVDA"], years=1, batch_size=1)

    assert result.symbols_failed == ["AAPL"]
    assert result.symbols_ok == ["NVDA"]  # run continued past the bad batch


def test_ingest_records_failed_batch(con, monkeypatch):
    def failing_download(tickers, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr(ingest.yf, "download", failing_download)
    monkeypatch.setattr(ingest, "RETRY_BACKOFF_SECONDS", 0.0)
    result = ingest.ingest(con, ["AAPL", "NVDA"], years=1)

    assert sorted(result.symbols_failed) == ["AAPL", "NVDA"]
    assert result.rows_written == 0
