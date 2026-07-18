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


def test_single_bar_tail_fetch_drops_artifact_via_store_seeded_prev_close(caplog):
    # Daily tail fetches are single-bar frames with no in-frame prior bar —
    # without the store-seeded prev_close the artifact rule is blind on exactly
    # the path that will ever see a NEW artifact (reviewer reproduction).
    frame = _yf_frame_from_bars("DRUG", [(102.0, 10.2)])
    rows = ingest.frames_to_rows(frame, {"DRUG": "DRUG"}, {"DRUG": (dt.date(2026, 1, 2), 10.15)})
    assert rows.empty
    assert "1 bars dropped as split artifacts" in caplog.text
    assert "DRUG" in caplog.text


def test_single_bar_tail_fetch_keeps_continuous_bar(caplog):
    frame = _yf_frame_from_bars("OKAY", [(10.2, 10.3)])
    rows = ingest.frames_to_rows(frame, {"OKAY": "OKAY"}, {"OKAY": (dt.date(2026, 1, 2), 10.15)})
    assert len(rows) == 1
    assert "split artifact" not in caplog.text


def test_seed_ignored_when_frame_starts_at_last_stored_bar(caplog):
    # In-flight change makes tail fetches start AT the last stored bar: the
    # re-fetched first bar must not use its own stored close as prev_close —
    # the seed applies only strictly after the stored date.
    frame = _yf_frame_from_bars("OVLP", [(102.0, 10.2), (10.2, 10.3)])
    last_bars = {"OVLP": (dt.date(2026, 1, 5), 10.2)}  # first frame bar IS the stored bar
    rows = ingest.frames_to_rows(frame, {"OVLP": "OVLP"}, last_bars)
    assert len(rows) == 2  # no usable prev for the overlap bar; second bar continuous
    assert "split artifact" not in caplog.text


def test_seed_with_invalid_stored_close_is_ignored(caplog):
    # A NaN stored close cannot establish a scale — conservative keep, no crash.
    frame = _yf_frame_from_bars("BADC", [(102.0, 10.2)])
    rows = ingest.frames_to_rows(
        frame, {"BADC": "BADC"}, {"BADC": (dt.date(2026, 1, 2), float("nan"))}
    )
    assert len(rows) == 1
    assert "split artifact" not in caplog.text


def test_split_thresholds_not_flipped_by_fp_at_non_round_prices(caplog):
    # CLAUDE.md FP-boundary rule: exactly-at-threshold bars must never be
    # flagged, whichever side double arithmetic lands them on.
    exact_oc = _yf_frame_from_bars("EDGEOC", [(0.57, 0.57), (5.70, 8.55)])  # +50% at 10x scale
    assert len(ingest.frames_to_rows(exact_oc, {"EDGEOC": "EDGEOC"})) == 2
    exact_scale = _yf_frame_from_bars("EDGESC", [(5.70, 5.70), (11.40, 2.85)])  # 2.0x, -75%
    assert len(ingest.frames_to_rows(exact_scale, {"EDGESC": "EDGESC"})) == 2
    assert "split artifact" not in caplog.text
    # just past BOTH thresholds still flags
    over = _yf_frame_from_bars("OVER", [(0.57, 0.57), (5.70, 8.56)])  # +50.2% at 10x scale
    assert len(ingest.frames_to_rows(over, {"OVER": "OVER"})) == 1
    assert "1 bars dropped as split artifacts" in caplog.text


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


def test_ingest_always_refetches_last_bar_even_when_current(con, download_calls):
    # No skip path: a same-day bar could be a partial from a mid-session run,
    # so even a fully-covered current symbol refetches from its last bar.
    today = dt.date.today()
    _seed_price(con, "AAPL", today)
    store.record_ingest_from(con, ["AAPL"], dt.date(2000, 1, 1))

    result = ingest.ingest(con, ["AAPL", "NVDA"], years=1)

    assert result.symbols_skipped == []
    assert download_calls[0][0] == ["AAPL", "NVDA"]  # AAPL re-downloaded, not skipped


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


def test_ingest_fetches_tail_including_last_bar(con, download_calls):
    # Covered-from-start but stale symbols fetch from their LAST stored bar
    # (inclusive) — refetching it heals partial bars from mid-session runs.
    # (10 days: stale enough to tail-fetch, inside the 30-day dormancy window.)
    last = dt.date.today() - dt.timedelta(days=10)
    _seed_price(con, "AAPL", last)
    store.record_ingest_from(con, ["AAPL"], dt.date(2000, 1, 1))

    ingest.ingest(con, ["AAPL"], years=1)

    assert download_calls[0][1] == last.isoformat()


def test_ingest_continues_after_postprocessing_error(con, monkeypatch, download_calls):
    original = ingest.frames_to_rows
    failed_once = []

    def flaky(data, yf_map, last_bars=None):
        if not failed_once:
            failed_once.append(True)
            raise KeyError("malformed batch")
        return original(data, yf_map, last_bars)

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


def test_ingest_dormant_symbols_excluded_loudly(con, download_calls, caplog):
    # A delisted name (last bar months ago, fully covered) must not be fetched
    # or counted as failed — it would trip the routine's failure gate forever.
    old = dt.date.today() - dt.timedelta(days=90)
    _seed_price(con, "DEAD", old)
    store.record_ingest_from(con, ["DEAD"], dt.date(2000, 1, 1))

    result = ingest.ingest(con, ["DEAD", "NVDA"], years=1)

    assert result.symbols_dormant == ["DEAD"]
    assert result.symbols_failed == []
    assert [c[0] for c in download_calls] == [["NVDA"]]  # DEAD never requested
    assert "dormant" in caplog.text


def test_ingest_dormant_still_backfills_without_coverage(con, download_calls):
    # Dormancy only applies to COVERED symbols: a fresh backfill (no
    # ingest_meta) must still fetch the full window even for an old last bar.
    old = dt.date.today() - dt.timedelta(days=90)
    _seed_price(con, "DEAD", old)

    result = ingest.ingest(con, ["DEAD"], years=1)

    assert result.symbols_dormant == []
    assert [c[0] for c in download_calls] == [["DEAD"]]
