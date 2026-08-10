"""Intraday path resolution (#79 phase 1).

Offline throughout: `_download` is replaced by a canned-payload builder, per the
project's network-code pattern (live smoke tests live in test_live_smoke.py).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from tests.conftest import seed_history
from twopercent import intraday, strategy

# Well inside seed_history's default 40 business days from 2026-01-05, so the
# session has the prior bars daily_returns' glitch guard needs.
SESSION = dt.date(2026, 2, 16)


def _bars(rows: list[tuple[str, float, float, float, float]], symbol: str = "AAA") -> pd.DataFrame:
    """A yfinance-shaped 5m frame: tz-aware index, single-symbol columns."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp(f"{SESSION} {t}", tz="America/New_York") for t, *_ in rows]
    )
    frame = pd.DataFrame(
        {
            "Open": [r[1] for r in rows],
            "High": [r[2] for r in rows],
            "Low": [r[3] for r in rows],
            "Close": [r[4] for r in rows],
            "Volume": [1000] * len(rows),
        },
        index=idx,
    )
    return pd.concat({symbol: frame}, axis=1)


def _seed_daily(con, open_=100.0, high=103.0, low=98.0, close=101.0, symbol="AAA"):
    """A daily bar that touches BOTH triggers: +3% high, -2% low off a 100 open."""
    seed_history(con, {symbol: [0.001] * 40}, vary_volume=True)
    con.execute(
        "UPDATE prices SET open = ?, high = ?, low = ?, close = ? WHERE symbol = ? AND date = ?",
        [open_, high, low, close, symbol, SESSION],
    )


def _ingest(con, frame, symbol="AAA"):
    return intraday.ingest(
        con,
        [symbol],
        SESSION,
        SESSION + dt.timedelta(days=1),
        downloader=lambda syms, s, e: frame,
    )


def _resolve_one(con, symbol="AAA"):
    return intraday.resolve(con, pd.DataFrame({"symbol": [symbol], "date": [SESSION]}))


# --- ingest -------------------------------------------------------------------


def test_ingest_stores_bars_with_session_dates_from_exchange_time(con):
    _seed_daily(con)
    result = _ingest(con, _bars([("09:30", 100.0, 101.0, 99.5, 100.5)]))

    assert result.rows == 1 and result.ok
    row = con.execute("SELECT symbol, ts, date, interval FROM intraday_prices").fetchone()
    # 09:30 ET must file under SESSION, not the UTC date it converts to.
    assert row[0] == "AAA" and row[2] == SESSION and row[3] == "5m"
    assert row[1].hour == 9 and row[1].minute == 30


def test_ingest_reports_symbols_that_returned_nothing(con, caplog):
    _seed_daily(con)
    result = intraday.ingest(
        con,
        ["AAA", "GHOST"],
        SESSION,
        SESSION + dt.timedelta(days=1),
        downloader=lambda syms, s, e: _bars([("09:30", 100.0, 101.0, 99.5, 100.5)]),
    )

    assert result.symbols_empty == ["GHOST"]
    # ok stays True: delisted picks return empty on EVERY run, so failing the
    # exit code on that would train the operator to ignore it. The count and the
    # warning are the signal.
    assert result.ok
    assert "GHOST" in result.summary()
    assert "GHOST" in caplog.text


def test_ingest_chunks_a_span_longer_than_one_request(con):
    """Yahoo caps a 5m request near 60 days; the range is walked, not refused.

    Refusing was why `--days 88` (the CLI's own advertised value) raised, and
    why the design doc's 60-trading-day backfill was unreachable.
    """
    _seed_daily(con)
    seen: list[tuple] = []

    def spy(syms, s, e):
        seen.append((s, e))
        return _bars([("09:30", 100.0, 101.0, 99.5, 100.5)])

    intraday.ingest(con, ["AAA"], SESSION, SESSION + dt.timedelta(days=90), downloader=spy)

    assert len(seen) == 2  # 90 days -> 55 + 35
    assert seen[0][0] == SESSION
    assert seen[-1][1] == SESSION + dt.timedelta(days=90)
    assert all((e - s).days <= intraday.REQUEST_SPAN_DAYS for s, e in seen)


def test_empty_response_is_an_error_not_an_empty_success(con, monkeypatch):
    """yfinance returns an EMPTY frame past its window instead of raising."""
    monkeypatch.setattr(intraday, "MAX_RETRIES", 1)
    monkeypatch.setattr(intraday.yf, "download", lambda **kw: pd.DataFrame())

    with pytest.raises(RuntimeError, match="empty"):
        intraday._download(["AAA"], SESSION, SESSION + dt.timedelta(days=1))


def test_ingest_is_idempotent(con):
    _seed_daily(con)
    frame = _bars([("09:30", 100.0, 101.0, 99.5, 100.5)])
    _ingest(con, frame)
    _ingest(con, frame)

    assert con.execute("SELECT count(*) FROM intraday_prices").fetchone()[0] == 1


# --- resolution ---------------------------------------------------------------


def test_limit_before_stop_collapses_the_band_to_the_limit(con):
    _seed_daily(con)
    _ingest(
        con,
        _bars(
            [
                ("09:30", 100.0, 102.5, 99.9, 102.0),  # +2.5% high — limit first
                ("09:35", 102.0, 103.0, 98.0, 98.5),  # -2% low later
            ]
        ),
    )
    res = _resolve_one(con)

    assert res.resolved == 1 and res.both_touched == 1
    assert res.frame.iloc[0]["seq"] == intraday.LIMIT_FIRST
    band = strategy.pick_return_band("limit_stop", -0.02, 0.01, True, intraday.LIMIT_FIRST)
    assert band == (strategy.LIMIT_PROFIT, strategy.LIMIT_PROFIT)


def test_stop_before_limit_collapses_the_band_to_the_stop(con):
    _seed_daily(con)
    _ingest(
        con,
        _bars(
            [
                ("09:30", 100.0, 100.5, 98.0, 98.2),  # -2% low first
                ("09:35", 98.2, 103.0, 98.0, 102.0),  # +3% high later
            ]
        ),
    )
    res = _resolve_one(con)

    assert res.frame.iloc[0]["seq"] == intraday.STOP_FIRST
    band = strategy.pick_return_band("limit_stop", -0.02, 0.01, True, intraday.STOP_FIRST)
    assert band == (strategy.STOP_LEVEL, strategy.STOP_LEVEL)


def test_both_triggers_inside_one_bar_stays_unresolved(con):
    """5m is not infinitely fine — a bar spanning both triggers cannot be ordered."""
    _seed_daily(con)
    _ingest(con, _bars([("09:30", 100.0, 103.0, 98.0, 101.0)]))
    res = _resolve_one(con)

    assert res.same_bar == 1 and res.resolved == 0
    assert res.frame.empty
    assert strategy.pick_return_band("limit_stop", -0.02, 0.01, True, None) == (
        strategy.STOP_LEVEL,
        strategy.LIMIT_PROFIT,
    )


def test_no_intraday_bars_leaves_the_band_open(con):
    _seed_daily(con)
    res = _resolve_one(con)

    assert res.both_touched == 1 and res.no_intraday == 1 and res.resolved == 0
    assert "no intraday" in res.summary()


def test_sparse_record_that_misses_the_days_low_refuses_to_resolve(con):
    """The bias this gate exists for.

    Yahoo omits no-trade intervals, so a thin name's record can miss the very
    dip that triggered the stop — which would look like "the stop never fired"
    and resolve LIMIT_FIRST. The day's own low (-2%) is not reproduced by these
    bars (lowest is -0.1%), so the session must not be trusted to order anything.
    """
    _seed_daily(con)  # daily low = 98.0 (-2%)
    _ingest(con, _bars([("09:30", 100.0, 103.0, 99.9, 102.0)]))  # never sees 98.0
    res = _resolve_one(con)

    assert res.resolved == 0 and res.disagreed == 1
    assert "failed the daily-bar check" in res.summary()


def test_unadjusted_split_scale_mismatch_refuses_to_resolve(con):
    """Intraday bars are unadjusted; a later split shifts the whole scale."""
    _seed_daily(con)
    _ingest(
        con,
        _bars(
            [
                ("09:30", 50.0, 51.25, 49.0, 51.0),  # every price halved by a 2:1
                ("09:35", 51.0, 51.5, 49.0, 50.5),
            ]
        ),
    )
    res = _resolve_one(con)

    assert res.resolved == 0 and res.disagreed == 1


def test_only_ambiguous_days_are_considered(con):
    """A day that touched the limit but never the stop needs no resolution."""
    _seed_daily(con, high=103.0, low=99.8)  # low only -0.2%: stop never triggered
    _ingest(con, _bars([("09:30", 100.0, 103.0, 99.8, 102.0)]))
    res = _resolve_one(con)

    assert res.both_touched == 0 and res.resolved == 0


def test_stop_boundary_at_an_adversarial_open(con):
    """Exactly -1% at open=5.00, where FP rounding bites (CLAUDE.md).

    (4.95 - 5.00) / 5.00 computes to -0.009999999999999964, ABOVE -0.01. The
    epsilon shared with strategy.stop_triggered is what makes this a trigger;
    without it the bar silently reads as un-stopped.
    """
    _seed_daily(con, open_=5.00, high=5.20, low=4.95, close=5.10)
    _ingest(
        con,
        _bars(
            [
                ("09:30", 5.00, 5.02, 4.95, 4.96),  # exactly -1% low first
                ("09:35", 4.96, 5.20, 4.95, 5.10),  # +4% high later
            ]
        ),
    )
    res = _resolve_one(con)

    assert res.resolved == 1
    assert res.frame.iloc[0]["seq"] == intraday.STOP_FIRST


def test_resolve_with_no_pairs_is_a_noop(con):
    res = intraday.resolve(con, pd.DataFrame(columns=["symbol", "date"]))
    assert res.both_touched == 0 and res.frame.empty


def test_verdict_constants_match_the_strategy_module(con):
    """The JS mirror injects strategy's constants; intraday must not drift."""
    assert intraday.LIMIT_FIRST == strategy.SEQ_LIMIT_FIRST
    assert intraday.STOP_FIRST == strategy.SEQ_STOP_FIRST
    assert intraday.LIMIT_FIRST != intraday.STOP_FIRST


def test_glitch_suspect_high_is_not_an_ambiguous_day(con):
    """A fake print cannot fill an order, so it cannot create an ordering question."""
    _seed_daily(con, open_=100.0, high=130.0, low=98.0, close=99.0)
    con.execute("UPDATE prices SET volume = 1 WHERE symbol = 'AAA' AND date = ?", [SESSION])
    res = _resolve_one(con)

    assert res.both_touched == 0


# --- the gates that bound the ERROR, not just the coverage --------------------


def test_gap_before_the_first_trigger_refuses_to_resolve(con):
    """The inversion quant-skeptic reproduced (finding 2).

    Truth: +2% at 09:35, -1% at 10:00 -> LIMIT_FIRST. Yahoo omits the 09:35
    no-trade interval, so the earliest RECORDED limit bar is 15:00 and the
    record still reproduces the day's high and low exactly. Ordering on what is
    present returns STOP_FIRST — confidently wrong, with the completeness gate
    reporting success. Contiguity from 09:30 through the earlier trigger is what
    catches it.
    """
    _seed_daily(con, open_=100.0, high=104.0, low=98.0, close=101.0)
    _ingest(
        con,
        _bars(
            [
                ("09:30", 100.0, 100.5, 99.9, 100.2),
                ("10:00", 100.2, 100.6, 98.0, 98.4),  # daily LOW reproduced
                ("15:00", 98.4, 104.0, 98.1, 101.0),  # daily HIGH reproduced
            ]
        ),
    )
    res = _resolve_one(con)

    assert res.resolved == 0, "a gap before the first trigger must not be ordered"
    assert res.gappy == 1
    assert "gap before the first trigger" in res.summary()


def test_contiguous_record_still_resolves(con):
    """The gate must not reject everything — an unbroken record still resolves."""
    _seed_daily(con, open_=100.0, high=104.0, low=98.0, close=101.0)
    _ingest(
        con,
        _bars(
            [
                ("09:30", 100.0, 100.5, 99.9, 100.2),
                ("09:35", 100.2, 104.0, 100.0, 103.0),  # limit here, no gap before it
                ("09:40", 103.0, 103.5, 98.0, 98.4),  # stop later
            ]
        ),
    )
    res = _resolve_one(con)

    assert res.resolved == 1 and res.gappy == 0
    assert res.frame.iloc[0]["seq"] == intraday.LIMIT_FIRST


def test_untraded_opening_window_is_a_gap(con):
    """Anchored at 09:30, not at the first bar present.

    If the name did not trade until 09:45, the unobserved 09:30-09:45 window
    could contain either trigger, so the ordering is not proven.
    """
    _seed_daily(con, open_=100.0, high=104.0, low=98.0, close=101.0)
    _ingest(
        con,
        _bars(
            [
                ("09:45", 100.0, 104.0, 100.0, 103.0),
                ("09:50", 103.0, 103.5, 98.0, 98.4),
            ]
        ),
    )
    res = _resolve_one(con)

    assert res.resolved == 0 and res.gappy == 1


def test_uniform_scale_error_cannot_manufacture_a_fill(con):
    """Finding 3: a 0.4% scale error used to pass a 0.5% tolerance.

    Every intraday price is inflated 0.4%. The 09:35 bar's true high is +1.7%
    (no fill) but reads as +2.08% (fill), which would invert a genuine
    STOP_FIRST day. The tolerance must be small relative to the 3-point corridor
    between the stop and the limit, not merely small relative to a split.
    """
    scale = 1.004
    _seed_daily(con, open_=100.0, high=103.0, low=98.0, close=101.0)
    _ingest(
        con,
        _bars(
            [
                ("09:30", 100.0 * scale, 100.4 * scale, 99.9 * scale, 100.2 * scale),
                ("09:35", 100.2 * scale, 101.7 * scale, 100.0 * scale, 101.0 * scale),
                ("09:40", 101.0 * scale, 101.2 * scale, 98.0 * scale, 98.4 * scale),
                ("09:45", 98.4 * scale, 103.0 * scale, 98.2 * scale, 101.0 * scale),
            ]
        ),
    )
    res = _resolve_one(con)

    assert res.resolved == 0 and res.disagreed == 1


def test_ohlc_impossible_intraday_bar_is_dropped_and_warned(con, caplog):
    """The daily path's validity law applies here too (CLAUDE.md)."""
    _seed_daily(con)
    result = _ingest(
        con,
        _bars(
            [
                ("09:30", 100.0, 100.5, 99.5, 100.2),
                ("09:35", 3.51, 3.40, 3.30, 3.35),  # ENHA shape: high below open
            ]
        ),
    )

    assert result.rows == 1
    assert "OHLC-impossible" in caplog.text
