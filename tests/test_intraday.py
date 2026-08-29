"""Intraday path resolution (#79 phase 1).

Offline throughout: `_download` is replaced by a canned-payload builder, per the
project's network-code pattern (live smoke tests live in test_live_smoke.py).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from tests.conftest import seed_history
from twopercent import intraday, scan, store, strategy

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
        downloader=lambda syms, s, e, iv: frame,
        today=SESSION,  # these sessions are historical; the clamp is provider-relative
    )


def _pairs_one(symbol="AAA"):
    return pd.DataFrame({"symbol": [symbol], "date": [SESSION]})


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
        downloader=lambda syms, s, e, iv: _bars([("09:30", 100.0, 101.0, 99.5, 100.5)]),
        today=SESSION,
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

    def spy(syms, s, e, iv):
        seen.append((s, e))
        return _bars([("09:30", 100.0, 101.0, 99.5, 100.5)])

    intraday.ingest(
        con, ["AAA"], SESSION, SESSION + dt.timedelta(days=90), downloader=spy, today=SESSION
    )

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


def test_off_grid_bar_cannot_mask_a_missing_slot(con):
    """Contiguity counts DISTINCT grid slots, not rows (quant-skeptic B).

    An extra off-grid 09:32 print alongside a MISSING 09:35 slot used to make a
    raw row count reach its expected total, passing the gate and restoring the
    inverted verdict. Slot counting collapses the stray print onto 09:30.
    """
    _seed_daily(con, open_=100.0, high=104.0, low=98.0, close=101.0)
    _ingest(
        con,
        _bars(
            [
                ("09:30", 100.0, 100.5, 99.9, 100.2),
                ("09:32", 100.2, 100.4, 100.0, 100.3),  # off-grid filler
                # 09:35 slot MISSING — the limit actually fired here
                ("10:00", 100.3, 100.6, 98.0, 98.4),
                ("15:00", 98.4, 104.0, 98.1, 101.0),
            ]
        ),
    )
    res = _resolve_one(con)

    assert res.resolved == 0 and res.gappy == 1


def test_ingest_clamps_a_request_beyond_yahoos_window_and_says_so(con, caplog):
    """#87: asking beyond the served window returns EMPTY, not an error.

    An unclamped request therefore produced a whole window of "failed" batches
    on a healthy system, burying the signal that a batch failure is supposed to
    carry. Clamping is announced — silently shortening the range would be the
    silent-success shape itself.
    """
    _seed_daily(con)
    seen: list[tuple] = []

    def spy(syms, s, e, iv):
        seen.append((s, e))
        return _bars([("09:30", 100.0, 101.0, 99.5, 100.5)])

    today = dt.date.today()
    intraday.ingest(con, ["AAA"], today - dt.timedelta(days=400), today, downloader=spy)

    assert seen, "the clamped range must still be fetched"
    assert seen[0][0] == today - dt.timedelta(days=intraday.MAX_LOOKBACK_DAYS)
    assert "clamping" in caplog.text
    assert "340 day(s) dropped" in caplog.text


def test_ingest_with_nothing_left_after_clamping_is_a_noop(con, caplog):
    _seed_daily(con)
    old = dt.date.today() - dt.timedelta(days=400)
    result = intraday.ingest(
        con,
        ["AAA"],
        old,
        old + dt.timedelta(days=5),
        downloader=lambda *a, **k: pytest.fail("fetched"),
    )

    assert result.rows == 0
    assert "nothing to fetch" in caplog.text


# --- #86: the stop books a TRIGGER, not a fill --------------------------------


def _stop_fill(con, symbol="AAA"):
    return intraday.stop_fills(con, pd.DataFrame({"symbol": [symbol], "date": [SESSION]}))


def test_stop_fill_is_measured_from_the_next_bar_not_the_trigger(con):
    """A stop becomes a market order; it executes at the next available price.

    Booking it at exactly -1% made the band's lower edge the BEST outcome
    available to a stopped pick, presented as the worst.
    """
    _seed_daily(con, open_=100.0, high=101.0, low=97.0, close=98.0)
    _ingest(
        con,
        _bars(
            [
                # Reproduces the daily high (101.0) and low (97.0), so the
                # session passes the completeness gate.
                ("09:30", 100.0, 101.0, 98.9, 99.0),  # -1.1% low: stop triggers here
                ("09:35", 97.5, 98.0, 97.0, 97.4),  # fill at this bar's OPEN: -2.5%
            ]
        ),
    )
    fills = _stop_fill(con)

    assert len(fills) == 1
    assert abs(float(fills.iloc[0]["fill"]) - (-0.025)) < 1e-9
    # And it flows through the band as the exit, replacing the -1% assumption.
    band = strategy.pick_return_band("limit_stop", -0.03, -0.02, False, None, -0.025)
    assert band == (-0.025, -0.025)


def test_unmeasurable_stop_keeps_the_labelled_assumption(con):
    """A trigger in the FINAL bar has no next bar — no fill may be invented."""
    _seed_daily(con, open_=100.0, high=101.0, low=97.0, close=98.0)
    _ingest(con, _bars([("09:30", 100.0, 101.0, 97.0, 98.0)]))

    assert _stop_fill(con).empty
    assert strategy.pick_return_band("limit_stop", -0.03, -0.02, False, None, None) == (
        strategy.STOP_LEVEL,
        strategy.STOP_LEVEL,
    )


def test_stop_fill_requires_the_same_gates_as_ordering(con):
    """A VALID record that disagrees with its daily bar cannot price the exit.

    The first version of this test seeded low=98.5 above high=98.0 — an
    OHLC-impossible bar that _flatten drops at ingest, so it passed through the
    validity gate and never exercised the agreement gate it claimed to test.
    """
    _seed_daily(con, open_=100.0, high=101.0, low=97.0, close=98.0)
    _ingest(
        con,
        _bars(
            [
                ("09:30", 100.0, 101.0, 98.9, 99.0),
                # Valid bars, but the record never reaches the daily low of 97.0.
                ("09:35", 98.5, 98.6, 98.4, 98.5),
            ]
        ),
    )

    assert _stop_fill(con).empty


def test_a_bad_fill_cannot_invert_the_band(con):
    """worst <= best must hold even when the measured exit is below the trigger."""
    worst, best = strategy.pick_return_band("limit_stop", -0.05, -0.04, True, None, -0.045)
    assert worst <= best
    assert worst == -0.045 and best == strategy.LIMIT_PROFIT


def test_a_fill_better_than_the_trigger_is_clamped(con):
    """A stop-market order cannot execute above its trigger.

    The proxy is the next bar's OPEN, up to five minutes after the breach, so it
    prices in whatever bounce followed. Ungated, half the measured population
    came out better than −1% and 17% were gains — a stopped pick booking a
    profit, and the "worst-case" win rate rising because of a change sold as a
    conservatism fix. The clamp is what keeps it monotone.
    """
    worst, best = strategy.pick_return_band("limit_stop", -0.04, -0.03, False, None, +0.031)
    assert worst == best == strategy.STOP_LEVEL

    # And a stopped pick can never be counted as a win.
    days = [{"d": "a", "base": 0.3, "picks": [[1, 0.03, -0.04, -0.03, 0, None, 0.031]]}]
    s = strategy.summarize_strategy_days(days, 1, "limit_stop")
    assert s["ww"] == 0.0 and s["wb"] == 0.0


def test_next_bar_must_be_the_adjacent_slot(con):
    """A print two hours after the breach is not a fill.

    2% of real sessions have a hole after the trigger (max 140 minutes), and on
    a name that stops trading after the breach the following print is the least
    fill-like price available — on exactly the illiquid names #86 was about.
    """
    _seed_daily(con, open_=100.0, high=101.0, low=97.0, close=98.0)
    _ingest(
        con,
        _bars(
            [
                ("09:30", 100.0, 101.0, 98.9, 99.0),  # stop triggers here
                ("11:30", 97.5, 98.0, 97.0, 97.4),  # next PRINT, not next SLOT
            ]
        ),
    )

    assert _stop_fill(con).empty


# --- 1m ground truth and layered resolution -----------------------------------


def _ingest_iv(con, frame, interval, symbol="AAA"):
    return intraday.ingest(
        con,
        [symbol],
        SESSION,
        SESSION + dt.timedelta(days=1),
        downloader=lambda syms, s, e, iv: frame,
        today=SESSION,
        interval=interval,
    )


def test_finer_interval_resolves_what_a_coarser_one_cannot(con):
    """The 29% same-5m-bar residual is mostly recoverable at 1m.

    Measured on the real store: 442 of 462 same-5m-bar sessions (96%) resolve at
    1m, and 1m confirmed 2,365 of 2,365 5m verdicts with zero inversions.
    """
    _seed_daily(con, open_=100.0, high=103.0, low=98.0, close=101.0)
    # One 5m bar spans BOTH triggers — unresolvable at 5m.
    _ingest_iv(con, _bars([("09:30", 100.0, 103.0, 98.0, 101.0)]), "5m")
    assert intraday.resolve(con, _pairs_one(), interval="5m").same_bar == 1

    # The same session at 1m separates them: stop first, then the limit.
    _ingest_iv(
        con,
        _bars(
            [
                ("09:30", 100.0, 100.2, 98.0, 98.4),  # stop
                ("09:31", 98.4, 103.0, 98.2, 101.0),  # limit
            ]
        ),
        "1m",
    )
    best = intraday.resolve_best(con, _pairs_one())

    assert best.resolved == 1
    assert best.frame.iloc[0]["seq"] == intraday.STOP_FIRST


def test_layered_resolution_falls_back_to_5m(con):
    """No 1m record must not lose a verdict 5m could produce."""
    _seed_daily(con, open_=100.0, high=104.0, low=98.0, close=101.0)
    _ingest_iv(
        con,
        _bars(
            [
                ("09:30", 100.0, 100.5, 99.9, 100.2),
                ("09:35", 100.2, 104.0, 100.0, 103.0),
                ("09:40", 103.0, 103.5, 98.0, 98.4),
            ]
        ),
        "5m",
    )
    best = intraday.resolve_best(con, _pairs_one())

    assert best.resolved == 1
    assert best.frame.iloc[0]["seq"] == intraday.LIMIT_FIRST


def test_validation_reports_agreement_and_recoverable_days(con):
    _seed_daily(con, open_=100.0, high=103.0, low=98.0, close=101.0)
    _ingest_iv(con, _bars([("09:30", 100.0, 103.0, 98.0, 101.0)]), "5m")  # same bar
    _ingest_iv(
        con,
        _bars([("09:30", 100.0, 100.2, 98.0, 98.4), ("09:31", 98.4, 103.0, 98.2, 101.0)]),
        "1m",
    )
    v = intraday.validate_against_1m(con, _pairs_one())

    # DECOMPOSED: this one is a genuine same-bar recovery, not a 5m capture
    # defect. Lumping the two overstated same-bar recovery as 442/462 (96%)
    # when the truth was 350 (76%) — and rendered a numerator against a
    # denominator it was not a subset of.
    assert v.recovered_same_bar == 1
    assert v.recovered_no_5m_record == 0 and v.recovered_5m_failed_gate == 0
    assert v.disagreed == 0
    assert "1 of 1 same-5m-bar" in v.summary()


def test_validation_says_so_when_there_is_no_1m_cover(con):
    """An unvalidated verdict must never read as a validated one."""
    _seed_daily(con)
    v = intraday.validate_against_1m(con, _pairs_one())
    assert v.compared == 0
    assert "UNCHECKED" in v.summary()


def test_unknown_interval_fails_loudly(con):
    with pytest.raises(ValueError, match="never a guess"):
        intraday.spec("3m")


def test_layered_resolution_never_double_counts(con):
    """A coarser pass must not re-resolve what a finer one already did.

    The "already resolved" filter compared DuckDB's datetime64 dates against
    datetime.date keys, matched nothing, and let 5m re-resolve every day 1m had
    handled — the real store reported "132/120 ambiguous pick-days resolved
    (110%)". A resolution count above the ambiguous count is arithmetically
    impossible and must fail loudly.
    """
    _seed_daily(con, open_=100.0, high=103.0, low=98.0, close=101.0)
    both = _bars([("09:30", 100.0, 100.2, 98.0, 98.4), ("09:35", 98.4, 103.0, 98.2, 101.0)])
    _ingest_iv(con, both, "5m")
    _ingest_iv(con, both, "1m")

    best = intraday.resolve_best(con, _pairs_one())

    assert best.resolved <= best.both_touched
    assert len(best.frame) == len(best.frame.drop_duplicates(subset=["symbol", "date"]))
    assert best.resolved == 1


def test_recovery_separates_finer_resolution_from_a_defective_5m_capture(con):
    """A 1m win over a BROKEN 5m record is not a win for finer resolution.

    Reported as one lumped count, 442 sessions were presented as "of 462
    same-5m-bar" when only 350 were: 45 had no 5m record at all and 47 had a 5m
    record that failed its own agreement gate. Routing around bad data is a
    different claim from resolving what 5m genuinely could not order, and only
    the second says anything about interval choice.
    """
    _seed_daily(con, open_=100.0, high=103.0, low=98.0, close=101.0)
    # 5m record that FAILS the agreement gate (never reaches the daily low).
    _ingest_iv(con, _bars([("09:30", 100.0, 103.0, 99.9, 101.0)]), "5m")
    # 1m record that is complete and orders the triggers.
    _ingest_iv(
        con,
        _bars([("09:30", 100.0, 100.2, 98.0, 98.4), ("09:31", 98.4, 103.0, 98.2, 101.0)]),
        "1m",
    )

    v = intraday.validate_against_1m(con, _pairs_one())

    assert v.recovered_5m_failed_gate == 1
    assert v.recovered_same_bar == 0, "a broken 5m capture must not count as same-bar recovery"
    assert v.same_bar_at_5m == 0
    # And the summary must never print a numerator against a zero denominator.
    assert "0 of 0 same-5m-bar" in v.summary()


# --- #97 post-merge: the duplicate that corrupted every view ------------------


def test_trailing_exits_never_returns_duplicate_sessions(con):
    """A session replayable at BOTH intervals must yield exactly one row.

    `day` arrives from a DuckDB DATE as a pandas Timestamp while the pending set
    holds datetime.date, so the un-normalised membership test removed nothing
    and the 5m pass re-replayed every session 1m had already done. The identical
    trap is documented in resolve_best and stop_fills_best; this function
    shipped without it.

    The blast radius was not the trailing view: `_attach_trailing` left-merges
    into the SHARED payload, so a duplicate becomes a duplicate PICK for every
    consumer — the default reach card and the three other exit rules all moved,
    in the flattering direction, on numbers that were correct before.
    """
    _seed_daily(con, open_=100.0, high=103.0, low=98.0, close=101.0)
    both = _bars([("09:30", 100.0, 100.2, 99.6, 100.1), ("09:35", 100.1, 103.0, 98.0, 101.0)])
    _ingest_iv(con, both, "5m")
    _ingest_iv(con, both, "1m")

    out = intraday.trailing_exits(con, _pairs_one())

    assert len(out) == len(out.drop_duplicates(subset=["symbol", "date"]))
    assert len(out) == 1
    # And the key is a plain date, so downstream merges cannot silently miss.
    assert isinstance(out.iloc[0]["date"], dt.date)


def test_trailing_exits_requires_a_gapless_session(con):
    """Path-dependent rules cannot be evaluated on a path with holes."""
    _seed_daily(con, open_=100.0, high=103.0, low=98.0, close=101.0)
    _ingest_iv(
        con,
        # 09:35 slot missing entirely.
        _bars([("09:30", 100.0, 100.2, 99.6, 100.1), ("09:40", 100.1, 103.0, 98.0, 101.0)]),
        "5m",
    )

    assert intraday.trailing_exits(con, _pairs_one()).empty


def test_every_lookback_keeps_a_margin_inside_the_provider_limit():
    """A lookback aimed AT the provider's boundary fails on every single run.

    Yahoo's 1m bound is "within the last 30 days", EXCLUSIVE and measured at
    request time: probing 2026-08-18, start=-30d returned EMPTY with that exact
    error while -25d returned bars. A lookback of 30 therefore made the oldest
    chunk fail nightly — 7 failed batches, 3 retries with backoff each, reported
    as ERROR on a healthy system. That is the alarm-fatigue trap #87 fixed for
    5m and which was reintroduced for 1m.

    Documented provider limits, so the margin is asserted rather than trusted.
    """
    documented = {"5m": 60, "1m": 30}
    for interval, limit in documented.items():
        lookback = intraday.spec(interval)["lookback"]
        assert lookback <= limit, f"{interval} lookback {lookback} exceeds the {limit}d limit"
        if interval == "1m":
            # The 1m window slides fastest and is the one that bit; require real
            # daylight, not merely "not over".
            assert lookback <= limit - 5, (
                f"1m lookback {lookback} sits within 5 days of the {limit}d boundary"
            )


def test_a_span_never_exceeds_its_interval_request_cap():
    """Chunking must respect the per-interval request cap, not the 5m alias."""
    for interval in intraday.INTERVALS:
        limits = intraday.spec(interval)
        assert limits["span"] <= limits["lookback"]
        assert limits["span"] > 0 and limits["minutes"] > 0


def test_picked_symbols_bounds_each_interval_by_its_own_retention_window(con):
    """A deep-lookback interval must not inflate the shallow ones' symbol list.

    #94 bounded picked_symbols to the retention window because every extra
    symbol costs requests on a rate-limited provider. That bound was
    `max(lookback)` across ALL specs, so adding the 1h spec (700 days) silently
    became the horizon for 5m (60) and 1m (25) too -- doubling the symbol count
    on the two captures that CANNOT be rebuilt after their window closes, for
    names that have no fetchable bars at those intervals anyway.

    There was no test on picked_symbols at all, which is exactly why that was
    invisible. This is it.
    """
    today = dt.date.today()
    rows = pd.DataFrame(
        {
            "strategy": ["baseline_gbm_v1"] * 2,
            # One pick inside every window, one only 1h could ever want.
            "signal_date": [today - dt.timedelta(days=3), today - dt.timedelta(days=400)],
            "symbol": ["FRESH", "OLD1H"],
            "prob": [0.9, 0.9],
            "rank": [1, 1],
            "created_ts": [pd.Timestamp("2026-08-28")] * 2,
            "universe_as_of": [today] * 2,
            "event": [scan.TOUCH_EVENT] * 2,
        }
    )
    con.register("_picks", rows)
    con.execute("INSERT INTO predictions BY NAME SELECT * FROM _picks")
    con.unregister("_picks")

    assert intraday.picked_symbols(con, interval="1m") == ["FRESH"]
    assert intraday.picked_symbols(con, interval="5m") == ["FRESH"]
    # Only the deep interval reaches back far enough to want the old pick.
    assert sorted(intraday.picked_symbols(con, interval="1h")) == ["FRESH", "OLD1H"]


def test_1h_captures_the_universe_while_5m_and_1m_stay_picks_only(con):
    """1h feeds FEATURES, so it must cover every symbol the model SCORES.

    Left picks-only, the phase-2 columns would go NaN for ~70% of the universe
    at predict time while the model trained on rows where they were populated,
    and "has 1h data" would decay into "was picked by this model before" — a
    cohort reaching +2% at roughly twice the base rate, i.e. a feedback loop the
    model can learn. 5m/1m stay picks-only: they serve the explorer, and every
    extra symbol is a request on a rate-limited provider.
    """
    today = dt.date.today()
    universe = pd.DataFrame(
        {
            "symbol": ["PICKED", "NEVERPICKED"],
            "name": ["a", "b"],
            "market_cap": [2e9, 1e9],
            "sector": ["Tech", "Tech"],
        }
    )
    store.upsert_universe(con, universe, as_of=today)
    picks = pd.DataFrame(
        {
            "strategy": ["baseline_gbm_v1"],
            "signal_date": [today - dt.timedelta(days=2)],
            "symbol": ["PICKED"],
            "prob": [0.9],
            "rank": [1],
            "created_ts": [pd.Timestamp("2026-08-28")],
            "universe_as_of": [today],
            "event": [scan.TOUCH_EVENT],
        }
    )
    con.register("_p", picks)
    con.execute("INSERT INTO predictions BY NAME SELECT * FROM _p")
    con.unregister("_p")

    for interval in ("5m", "1m"):
        symbols, scope = intraday.capture_symbols(con, interval)
        assert symbols == ["PICKED"], f"{interval} must stay picks-only"
        assert scope == "picked"

    symbols, scope = intraday.capture_symbols(con, "1h")
    assert sorted(symbols) == ["NEVERPICKED", "PICKED"]
    assert scope == "universe"


def _seed_session(con, symbols, day, hours, interval="1h"):
    rows = pd.DataFrame(
        [
            {
                "symbol": sym,
                "ts": dt.datetime.combine(day, dt.time(h, 30)),
                "date": day,
                "interval": interval,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000,
            }
            for sym in symbols
            for h in hours
        ]
    )
    con.register("_s", rows)
    con.execute("INSERT INTO intraday_prices SELECT * FROM _s")
    con.unregister("_s")


def test_a_calendared_half_day_is_not_a_coverage_fault(con):
    """A 13:00 half day ends three hours early BY DESIGN.

    Treated as a truncation it produced six permanent, unclearable doctor
    problems — a check that cries wolf forever teaches the operator to ignore
    it, the same alarm-fatigue trap #94 and the 1m boundary already cost us.
    """
    syms = [f"S{i:02d}" for i in range(10)]
    full, half = dt.date(2026, 3, 2), dt.date(2026, 3, 3)
    _seed_session(con, syms, full, [9, 10, 11, 12, 13, 14, 15])
    _seed_session(con, syms, half, [9, 10, 11, 12])  # last bar 12:30, a 13:00 close

    cov = intraday.session_coverage(con)
    # session_coverage returns DATE as datetime64; comparing it to a
    # datetime.date matches nothing SILENTLY (the trap this project keeps
    # re-hitting). Normalise both sides.
    row = cov[pd.to_datetime(cov["date"]).dt.date == half].iloc[0]
    assert row["early_close"] and not row["truncated"]
    bad = intraday.coverage_problems(con)
    assert half not in set(pd.to_datetime(bad["date"]).dt.date)


def test_a_feed_that_dies_mid_session_is_still_truncated(con):
    """Accepting the half-day end must not accept every short session."""
    syms = [f"S{i:02d}" for i in range(10)]
    good, dead = dt.date(2026, 3, 2), dt.date(2026, 3, 3)
    _seed_session(con, syms, good, [9, 10, 11, 12, 13, 14, 15])
    _seed_session(con, syms, dead, [9, 10])  # feed died at 10:30 — neither end

    cov = intraday.session_coverage(con)
    row = cov[pd.to_datetime(cov["date"]).dt.date == dead].iloc[0]
    assert row["truncated"] and not row["early_close"]
    assert intraday.coverage_verdict(row) == "TRUNCATED"


def test_a_session_missing_its_morning_is_caught_despite_stragglers(con):
    """2026-02-02 lost 09:30–12:30 market-wide yet closed at 15:30.

    Every existing check read it clean: the last bar was on time, the symbol
    count was normal, and a handful of names DID print at 09:30 — so a min()
    over first bars saw 09:30 and shrugged. The statistic has to be robust.
    """
    syms = [f"S{i:02d}" for i in range(20)]
    good, gap = dt.date(2026, 3, 2), dt.date(2026, 3, 3)
    _seed_session(con, syms, good, [9, 10, 11, 12, 13, 14, 15])
    _seed_session(con, syms[:2], gap, [9, 10, 11, 12, 13, 14, 15])  # the stragglers
    _seed_session(con, syms[2:], gap, [13, 14, 15])  # everyone else: afternoon only

    cov = intraday.session_coverage(con)
    row = cov[pd.to_datetime(cov["date"]).dt.date == gap].iloc[0]
    assert row["late_open"], "a market-wide missing morning must be a fault"
    assert not row["truncated"], "it closed on time — that is not a truncation"
    assert intraday.coverage_verdict(row) == "LATE_OPEN"
    bad = intraday.coverage_problems(con)
    assert gap in set(pd.to_datetime(bad["date"]).dt.date)
