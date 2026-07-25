"""Stage A: the touch (open-to-high) event, its high-spike guard, and that
every consumer reads the SAME event. This is the project's core definition, so
it is tested at adversarial values, on the unhappy paths, and for lookahead."""

from __future__ import annotations

import datetime as dt

import pandas as pd

from tests.conftest import seed_history
from twopercent import backtest, doctor, scan, store, track
from twopercent.features import feature_frame

_THR = scan.DEFAULT_THRESHOLD - scan._THRESHOLD_EPSILON
_EVENT_SQL = scan.touch_event_predicate("high_return", "high_glitch_suspect")


def _touch_symbols(con, date: dt.date) -> set[str]:
    """Symbols whose bar on `date` is a touch event, straight from the shared
    predicate against the daily_returns view."""
    rows = con.execute(
        f"SELECT symbol FROM daily_returns WHERE date = ? AND {_EVENT_SQL}",
        [date, _THR],
    ).df()
    return set(rows["symbol"])


# --- the touch label separates from the old close label -----------------------


def test_touch_label_is_intraday_reach_not_close(con):
    # TCH reaches +2.5% intraday but closes only +0.5% — NOT a close-2% day, but
    # IS a touch. CLS closes +3% (its high >= close >= +2%, still a touch). FLT
    # does neither. Under the old close label TCH would be a miss every day.
    seed_history(
        con,
        {"TCH": [0.005] * 26, "CLS": [0.03] * 26, "FLT": [0.005] * 26},
        vary_volume=True,
        high_returns={"TCH": [0.025] * 26},
    )
    frame = feature_frame(con).set_index(["symbol", "signal_date"])
    tch = frame.xs("TCH")["did_2pct_next"].dropna()
    cls = frame.xs("CLS")["did_2pct_next"].dropna()
    flt = frame.xs("FLT")["did_2pct_next"].dropna()
    assert (tch == 1).all()  # touched intraday every day, though it closed < 2%
    assert (cls == 1).all()  # closing +3% implies a +2% touch
    assert (flt == 0).all()


def test_touch_boundary_epsilon_at_adversarial_open(con):
    # open=5.00 (NOT a round 100): (high-open)/open for a +2% high lands a few
    # ULPs below 0.02 in double math (CLAUDE.md). The epsilon guard must still
    # count it as a touch. close +0.2% keeps it a touch-only (not close) event.
    store.upsert_prices(
        con,
        pd.DataFrame(
            {
                "symbol": ["BND"],
                "date": [dt.date(2026, 1, 5)],
                "open": [5.00],
                "high": [5.10],  # exactly +2.0% off a 5.00 open
                "low": [4.99],
                "close": [5.01],  # +0.2%: close does NOT confirm
                "adj_close": [5.01],
                "volume": [1_000_000],
            }
        ),
    )
    raw = (5.10 - 5.00) / 5.00  # documents the sub-0.02 FP landing
    assert raw < scan.DEFAULT_THRESHOLD  # would be dropped without the epsilon
    assert _touch_symbols(con, dt.date(2026, 1, 5)) == {"BND"}


# --- the high-spike guard is an INTERSECTION (M2) -----------------------------


def _seed_guard_symbol(con, symbol: str, open_, high, low, close, volume) -> dt.date:
    """21 flat baseline bars (close 100, volume 1e6) then ONE special bar. The
    baseline gives prev_close=100 and a trailing-20 average volume of 1e6, so the
    special bar's isolation / volume clauses have a reference. Returns the
    special bar's date."""
    dates = pd.bdate_range("2026-01-05", periods=22)
    base = 21
    store.upsert_prices(
        con,
        pd.DataFrame(
            {
                "symbol": symbol,
                "date": dates.date,
                "open": [100.0] * base + [open_],
                "high": [100.1] * base + [high],
                "low": [99.9] * base + [low],
                "close": [100.0] * base + [close],
                "adj_close": [100.0] * base + [close],
                "volume": [1_000_000] * base + [volume],
            }
        ),
    )
    return dates[-1].date()


def test_high_spike_guard_intersection(con):
    # GLTCH: isolated high (+20%), close unconfirmed (+0.5%), LOW volume → glitch.
    # SQUEZ: the SAME isolated high but HIGH volume → a real squeeze a +2% limit
    #        WOULD have filled — must NOT be flagged (the key M2 case).
    # CONFIRM: isolated high but the CLOSE confirmed the move (+25%) → not a glitch.
    d = _seed_guard_symbol(con, "GLTCH", 100, 120, 99, 100.5, 100_000)
    _seed_guard_symbol(con, "SQUEZ", 100, 120, 99, 100.5, 3_000_000)
    _seed_guard_symbol(con, "CONFIRM", 100, 130, 99, 125, 100_000)

    flags = (
        con.execute(
            "SELECT symbol, high_glitch_suspect FROM daily_returns WHERE date = ? ORDER BY symbol",
            [d],
        )
        .df()
        .set_index("symbol")["high_glitch_suspect"]
    )
    assert bool(flags["GLTCH"]) is True
    assert bool(flags["SQUEZ"]) is False  # real high-volume squeeze kept
    assert bool(flags["CONFIRM"]) is False  # close-confirmed, not a glitch

    # The event follows the guard: only the glitch is NOT a touch event.
    touch = _touch_symbols(con, d)
    assert "GLTCH" not in touch
    assert {"SQUEZ", "CONFIRM"} <= touch


def test_doctor_glitch_check_reads_raw_and_flags_only_the_glitch(con):
    d = _seed_guard_symbol(con, "GLTCH", 100, 120, 99, 100.5, 100_000)
    _seed_guard_symbol(con, "SQUEZ", 100, 120, 99, 100.5, 3_000_000)
    _seed_guard_symbol(con, "CONFIRM", 100, 130, 99, 125, 100_000)
    flagged = doctor.glitch_bars(con)
    assert set(flagged["symbol"]) == {"GLTCH"}
    row = flagged.iloc[0]
    assert pd.Timestamp(row["date"]).date() == d
    assert row["volume"] < row["trailing_avg_vol"]  # below-average volume clause
    # It rides in the DoctorReport problem_count, like the OHLC-ordering gate.
    report = doctor.run(con)
    assert len(report.glitch) == 1
    assert not report.ok


# --- all event sites agree on the same bar ------------------------------------


def test_all_event_sites_agree_on_the_same_bar(con):
    seed_history(
        con,
        {"TCH": [0.005] * 26, "CLS": [0.03] * 26, "FLT": [0.005] * 26},
        vary_volume=True,
        high_returns={"TCH": [0.025] * 26},
    )
    dates = sorted(
        pd.to_datetime(con.execute("SELECT DISTINCT date FROM daily_returns").df()["date"]).dt.date
    )
    target, signal = dates[-1], dates[-2]
    expected = {"TCH", "CLS"}

    # (1) the shared predicate on the view, and (2) the scanner, must agree.
    assert _touch_symbols(con, target) == expected
    assert set(scan.daily_movers(con, target)["symbol"]) == expected

    # (3) the training label at the prior signal day matches membership.
    frame = feature_frame(con)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.date
    frame = frame.set_index(["signal_date", "symbol"])
    for sym in ("TCH", "CLS", "FLT"):
        assert frame.loc[(signal, sym), "did_2pct_next"] == (1 if sym in expected else 0)

    # (4) the base rate is the touch share of ALL names.
    assert abs(track.daily_base_rates(con, [target])[target] - 2 / 3) < 1e-9

    # (5) the rolling cnt_2pct_20d feature counts touch days.
    assert frame.loc[(signal, "TCH"), "cnt_2pct_20d"] == 20
    assert frame.loc[(signal, "FLT"), "cnt_2pct_20d"] == 0


def test_touch_base_rate_exceeds_close_base_rate(con):
    # Sanity: touching +2% is far more common than closing +2%. TCH touches every
    # day but closes +0.5%; FLT does neither. Close base rate is 0, touch is ~0.5.
    seed_history(
        con,
        {"TCH": [0.005] * 30, "FLT": [0.005] * 30},
        vary_volume=True,
        high_returns={"TCH": [0.025] * 30},
    )
    close_br = con.execute(
        "SELECT avg(CASE WHEN oc_return >= ? THEN 1.0 ELSE 0.0 END) FROM daily_returns", [_THR]
    ).fetchone()[0]
    touch_br = con.execute(
        f"SELECT avg(CASE WHEN {_EVENT_SQL} THEN 1.0 ELSE 0.0 END) FROM daily_returns", [_THR]
    ).fetchone()[0]
    assert close_br == 0.0
    assert touch_br > close_br
    assert abs(touch_br - 0.5) < 1e-9


# --- no lookahead: the label uses only the target day + prior -----------------


def test_touch_label_has_no_lookahead_beyond_target_day(con):
    # Mutating any bar AFTER the target day must not change a signal row's label
    # (the label is the target day's OWN touch; the guard uses same-bar + prior
    # only, never next_open). Distinct from the features canary in test_features.
    seed_history(con, {"AAA": [0.001] * 30}, high_returns={"AAA": [0.025] * 30})
    before = feature_frame(con).set_index("signal_date")["did_2pct_next"]
    dates = sorted(before.index)
    cutoff = dates[len(dates) // 2]
    con.execute("UPDATE prices SET high = high * 5, close = close * 5 WHERE date > ?", [cutoff])
    after = feature_frame(con).set_index("signal_date")["did_2pct_next"]
    # Labels for signal rows strictly before the cutoff are unchanged.
    early = [d for d in dates if d < cutoff]
    assert before.loc[early].equals(after.loc[early])


# --- metric-definition versioning (M1) ----------------------------------------


def _std_experiment(con, strategy: str, lift: float, event: str | None) -> int:
    return store.record_experiment(
        con,
        strategy,
        {"months": 12, "top_n": 20},
        dt.date(2026, 1, 1),
        dt.date(2026, 2, 1),
        dt.date(2026, 3, 1),
        {"lift": lift, "top_n": 20},
        event=event,
    )


def test_champion_reader_is_touch_only_and_degrades_gracefully(con):
    # A close-era row (event NULL) must NEVER be quoted as the champion benchmark.
    _std_experiment(con, "gbm", 2.0, event=None)
    assert backtest.latest_standard_experiment(con, "gbm") is None  # graceful: no touch row yet

    touch_id = _std_experiment(con, "gbm", 2.2, event=scan.TOUCH_EVENT)
    got = backtest.latest_standard_experiment(con, "gbm")
    assert got is not None and got[0] == touch_id and got[1]["lift"] == 2.2

    # The research done-ledger also ignores the close-era row, so the config is
    # re-benchmarked under touch (the forced re-benchmark after the cutover).
    from twopercent import research

    with_close_only = store.connect(":memory:")
    _std_experiment(with_close_only, "gbm", 2.0, event=None)
    assert research.recorded_configs(with_close_only) == set()


def test_live_record_clean_reset_excludes_pre_cutover(con):
    seed_history(con, {"HIT1": [0.001] * 23 + [0.03, 0.05], "MISS": [0.001] * 25})
    dates = sorted(pd.bdate_range("2026-01-05", periods=25).date)
    picks = pd.DataFrame({"symbol": ["HIT1", "MISS"], "prob": [0.9, 0.1], "rank": [1, 2]})
    store.save_predictions(con, "s", dates[-3], picks)  # touch era (target dates[-2])
    store.save_predictions(con, "s", dates[-2], picks, event=None)  # archived close era

    record = track.score_predictions(con, "s", top_n=2)
    scored_targets = set(pd.to_datetime(record.scored["target_date"]).dt.date)
    assert dates[-2] in scored_targets  # touch-era day scored
    assert dates[-1] not in scored_targets  # pre-cutover day excluded from the touch record

    first_touch, archived = store.touch_record_bounds(con, "s")
    assert first_touch == dates[-3]
    assert archived == 1  # the NULL-event signal day is counted as archived


def test_pending_excludes_close_era_signal_dates(con):
    # A store with ONLY close-era (NULL-event) predictions must yield empty
    # predicted_signal_dates and empty pending — never perpetual "Awaiting
    # outcomes" for days that can never resolve to a touch score (F1).
    seed_history(con, {"HIT1": [0.001] * 23 + [0.03, 0.05], "MISS": [0.001] * 25})
    dates = sorted(pd.bdate_range("2026-01-05", periods=25).date)
    picks = pd.DataFrame({"symbol": ["HIT1", "MISS"], "prob": [0.9, 0.1], "rank": [1, 2]})
    store.save_predictions(con, "s", dates[-3], picks, event=None)  # close era only

    assert store.predicted_signal_dates(con, "s") == []
    assert track.score_predictions(con, "s", top_n=2).pending == []

    # A touch-era prediction with no next trading day yet pends correctly.
    store.save_predictions(con, "s", dates[-1], picks)  # touch era
    assert store.predicted_signal_dates(con, "s") == [dates[-1]]
    assert dates[-1] in track.score_predictions(con, "s", top_n=2).pending


def test_shadow_pending_excludes_close_era_signal_dates(con):
    seed_history(con, {"AAA": [0.001] * 25})
    dates = sorted(pd.bdate_range("2026-01-05", periods=25).date)
    picks = pd.DataFrame({"symbol": ["AAA"], "prob": [0.9], "rank": [1]})
    store.save_shadow_predictions(con, "ch", "s", "{}", dates[-3], picks, event=None)
    assert store.shadow_signal_dates(con, "ch") == []
    assert track.score_shadow_predictions(con, "ch", top_n=1).pending == []


def test_benchmark_stamps_touch_event(con, monkeypatch):
    from tests.conftest import seed_planted

    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5)
    events = con.execute("SELECT DISTINCT event FROM experiments").fetchall()
    assert events == [(scan.TOUCH_EVENT,)]
