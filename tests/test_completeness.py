"""The completeness/finality gate (#65): never score or predict from an
incomplete (provisional, in-progress) trading day.

A date is COMPLETE iff its valid-bar coverage is >= 0.9 * the median coverage
over the 20 dates STRICTLY BEFORE it (trailing-only, no lookahead). The gate is
ONE definition (store.complete_trading_days) reused by scoring, base rates, and
predict's default signal day, and it must be a NO-OP on uniformly-covered data.
"""

import datetime as dt
import logging

import pandas as pd
import pytest

from tests.conftest import seed_history, seed_planted
from twopercent import doctor, predict, store, track


def _seed_uniform(con, n_symbols: int = 30, n_days: int = 25, start: str = "2026-01-05"):
    """n_symbols each with n_days of flat, fully-covered history."""
    oc = {f"S{i:02d}": [0.001] * n_days for i in range(n_symbols)}
    seed_history(con, oc, start=start, vary_volume=True)
    return oc, sorted(pd.bdate_range(start, periods=n_days).date)


def _drop_last_day_bars(con, symbols: list[str], day: dt.date) -> None:
    con.register("drop_in", pd.DataFrame({"symbol": symbols}))
    con.execute(
        "DELETE FROM prices WHERE date = ? AND symbol IN (SELECT symbol FROM drop_in)", [day]
    )
    con.unregister("drop_in")


def _complete_dates(con) -> set[dt.date]:
    rows = con.execute("SELECT date FROM complete_trading_days").fetchall()
    return {r[0] for r in rows}


# --- the view: definition, boundary, no-op ------------------------------------


def test_uniform_coverage_marks_every_judged_day_complete(con):
    _, dates = _seed_uniform(con)
    complete = _complete_dates(con)
    assert complete == set(dates)  # no-op on uniform data — nothing held


def test_short_latest_day_is_held_back(con):
    oc, dates = _seed_uniform(con, n_symbols=30, n_days=25)
    # ~60% coverage on the last day: 18 of 30 symbols (a provisional day).
    _drop_last_day_bars(con, [f"S{i:02d}" for i in range(12)], dates[-1])
    complete = _complete_dates(con)
    assert dates[-1] not in complete
    assert dates[-2] in complete  # the prior full day is untouched


def test_boundary_at_exactly_nine_tenths_median(con):
    # Adversarial exact boundary: median 30 → 0.9*30 = 27. Exactly 27 present is
    # COMPLETE (>=, epsilon-guarded); 26 is INCOMPLETE.
    _, dates = _seed_uniform(con, n_symbols=30, n_days=25)
    last = dates[-1]

    _drop_last_day_bars(con, [f"S{i:02d}" for i in range(3)], last)  # 27 remain
    assert last in _complete_dates(con), "exactly 0.9*median must stay complete"

    _drop_last_day_bars(con, ["S03"], last)  # now 26
    assert last not in _complete_dates(con), "just below 0.9*median must be held"


def test_earliest_days_without_min_window_are_complete(con):
    # Fewer than COMPLETENESS_MIN_PRIOR_DATES (5) prior dates → can't judge against
    # a baseline → treated complete, even though these early days are wildly uneven.
    seed_history(
        con,
        {"A": [0.001] * 5, "B": [0.001] * 3, "C": [0.001] * 1},
        vary_volume=True,
    )
    dates = sorted(pd.bdate_range("2026-01-05", periods=5).date)
    # every seeded date has < 5 prior days, so all are treated complete
    assert _complete_dates(con) == set(dates)


def test_short_history_store_holds_provisional_latest_day(con):
    # The reviewer's silent blind spot: a store YOUNGER than a full 20-day window
    # but with >= 5 prior dates must still judge — and HOLD — a provisional latest
    # day. Before FIX 1 (prior_days < 20 → complete) this 8-day store silently
    # scored the 17%-coverage edge day; now it is held.
    oc, dates = _seed_uniform(con, n_symbols=12, n_days=8)
    _save(con, "strat", dates[-2], ["S00", "S01", "S02"])
    # ~17% coverage on the last day: 2 of 12 symbols (a provisional pre-market bar).
    _drop_last_day_bars(con, [f"S{i:02d}" for i in range(2, 12)], dates[-1])

    assert dates[-1] not in _complete_dates(con)  # judged on the 7-date partial window
    record = track.score_predictions(con, "strat", top_n=3)
    assert record.scored.empty  # not scored against the partial bar...
    assert dates[-2] in record.pending  # ...stays pending instead


def test_very_young_store_latest_treated_complete_but_warns(con, caplog):
    from twopercent import routine

    # A brand-new store (< 5 trading days, a from-scratch backfill): the latest day
    # CANNOT be judged, so it is treated complete and USED — but that must be LOUD,
    # never silent, in both the routine step and predict's default signal day.
    _, dates = _seed_uniform(con, n_symbols=10, n_days=4)
    assert _complete_dates(con) == set(dates)  # all treated complete (too young)

    report = routine.RoutineReport()
    routine._completeness_step(report, con)
    step = report.steps[-1]
    assert step.status == routine.WARN
    assert "UNJUDGEABLE" in step.detail and str(dates[-1]) in step.detail

    # predict._default_signal_date fires the same loud warning (called directly:
    # feature_frame needs >= 20 days of history, so a 4-day store never reaches
    # predict_for — the warning is the defensive disclosure for that regime).
    frame = pd.DataFrame({"signal_date": dates})
    with caplog.at_level(logging.WARNING, logger="twopercent.predict"):
        chosen = predict._default_signal_date(con, frame)
    assert chosen == dates[-1]  # used, since there is nothing better
    assert any("UNJUDGEABLE" in r.message and str(dates[-1]) in r.message for r in caplog.records)


# --- no lookahead: the key methodological test --------------------------------


def test_verdict_uses_no_future_data(con):
    # A trailing-only window means a past date's verdict can NEVER change when
    # later dates arrive. Craft an incomplete day, snapshot every verdict up to
    # it, append many full days, and prove the snapshot is byte-for-byte the same.
    _, dates = _seed_uniform(con, n_symbols=30, n_days=22)
    judged = dates[-1]  # has a full 20-date prior window
    _drop_last_day_bars(con, [f"S{i:02d}" for i in range(20)], judged)  # 10 of 30 → incomplete

    def snapshot():
        return con.execute(
            "SELECT date, bar_count, prior_days, trailing_median "
            "FROM complete_trading_days WHERE date <= ? ORDER BY date",
            [judged],
        ).df()

    before = snapshot()
    assert judged not in set(before["date"])  # incomplete before any future data

    # Append eight fully-covered later days — a forward/centered window would
    # now re-rank the judged day; the trailing-only window must not.
    later = sorted(pd.bdate_range("2026-01-05", periods=30).date)[22:30]
    for day in later:
        store.upsert_prices(
            con,
            pd.DataFrame(
                {
                    "symbol": [f"S{i:02d}" for i in range(30)],
                    "date": day,
                    "open": 100.0,
                    "high": 100.2,
                    "low": 99.8,
                    "close": 100.1,
                    "adj_close": 100.1,
                    "volume": 1_000_000,
                }
            ),
        )

    after = snapshot()
    assert before.equals(after), "appending later dates changed an earlier verdict — lookahead!"
    assert judged not in _complete_dates(con)  # still held after the future arrived


# --- scoring gate -------------------------------------------------------------


def _save(con, strategy, signal_date, symbols):
    store.save_predictions(
        con,
        strategy,
        signal_date,
        pd.DataFrame({"symbol": symbols, "prob": [0.9, 0.5, 0.1], "rank": [1, 2, 3]}),
    )


def test_prediction_targeting_incomplete_day_stays_pending(con):
    oc, dates = _seed_uniform(con, n_symbols=30, n_days=25)
    # Make the last day do 2% for the picks so a completed day scores a real hit.
    for s in ("S00", "S01"):
        oc[s] = [0.001] * 24 + [0.03]
    seed_history(con, {"S00": oc["S00"], "S01": oc["S01"]}, vary_volume=True)  # overwrite

    _save(con, "strat", dates[-2], ["S00", "S01", "S02"])

    # Last day provisional (18 of 30): the only candidate target is incomplete.
    _drop_last_day_bars(con, [f"S{i:02d}" for i in range(12, 24)], dates[-1])

    record = track.score_predictions(con, "strat", top_n=3)
    assert record.scored.empty
    assert dates[-2] in record.pending  # awaiting outcomes, not scored on a partial bar

    perf = track.daily_pick_performance(con, "strat", top_n=3)
    assert perf.daily.empty and perf.live.empty
    assert perf.precision_at_1() is None

    ranks = track.daily_rank_outcomes(con, "strat", top_n=3)
    assert ranks.empty

    # the degradation detector sees no live day from the held day
    verdict = track.degradation_verdict(record.scored)
    assert not verdict.armed and verdict.live_days == 0

    # ...now COMPLETE the day (restore full coverage) → it resolves and scores.
    seed_history(con, oc, vary_volume=True)
    record2 = track.score_predictions(con, "strat", top_n=3)
    assert not record2.scored.empty
    row = record2.scored.iloc[0]
    assert pd.Timestamp(row["target_date"]).date() == dates[-1]
    assert row["hits"] == 2  # S00, S01 did 2%


def test_scoring_is_noop_when_all_days_complete(con):
    # A prediction on the second-to-last day of a fully-covered store scores
    # exactly as before the gate existed (target = last day).
    oc, dates = _seed_uniform(con, n_symbols=30, n_days=25)
    _save(con, "strat", dates[-2], ["S00", "S01", "S02"])
    record = track.score_predictions(con, "strat", top_n=3)
    assert len(record.scored) == 1
    assert pd.Timestamp(record.scored.iloc[0]["target_date"]).date() == dates[-1]
    assert record.pending == []


# --- base rates ---------------------------------------------------------------


def test_base_rate_absent_for_incomplete_date(con):
    _, dates = _seed_uniform(con, n_symbols=30, n_days=25)
    _drop_last_day_bars(con, [f"S{i:02d}" for i in range(15)], dates[-1])  # 50%

    rates = track.daily_base_rates(con, [dates[-2], dates[-1]])
    assert dates[-2] in rates  # complete day has a base rate
    assert dates[-1] not in rates  # incomplete day is absent (unknown), never a number


# --- predict default signal day -----------------------------------------------


def test_predict_default_uses_latest_complete_day_and_warns(con, caplog):
    seed_planted(con, n_each=10)  # 20 names, 100 days, sectors populated
    dates = sorted(pd.bdate_range("2026-01-05", periods=100).date)
    # Provisional latest day: only 5 of 20 names have posted.
    _drop_last_day_bars(
        con, [f"RUN{i:02d}" for i in range(10)] + [f"FLT{i:02d}" for i in range(5)], dates[-1]
    )

    with caplog.at_level(logging.WARNING, logger="twopercent.predict"):
        result = predict.predict_for(con, "logreg_v1", save=False)

    assert result.signal_date == dates[-2], "must forecast from the latest COMPLETE day"
    msgs = [r.message for r in caplog.records if "held back as INCOMPLETE" in r.message]
    assert msgs and str(dates[-1]) in msgs[0] and str(dates[-2]) in msgs[0]


def test_explicit_signal_date_honored_even_when_provisional(con):
    # Backfill/testing must keep working: an EXPLICIT signal_date bypasses the
    # gate entirely, including the provisional day itself.
    seed_planted(con, n_each=10)
    dates = sorted(pd.bdate_range("2026-01-05", periods=100).date)
    _drop_last_day_bars(
        con, [f"RUN{i:02d}" for i in range(10)] + [f"FLT{i:02d}" for i in range(5)], dates[-1]
    )

    result = predict.predict_for(con, "logreg_v1", signal_date=dates[-1], save=False)
    assert result.signal_date == dates[-1]  # honored verbatim, no fallback


# --- doctor -------------------------------------------------------------------


def test_doctor_reports_incomplete_day_from_raw_prices(con):
    _, dates = _seed_uniform(con, n_symbols=30, n_days=25)
    last = dates[-1]
    # 22 raw bars remain (8 deleted); of those, 4 are OHLC-corrupt so daily_returns
    # counts only 18. bar_count must be the RAW 22 (proves it reads prices, not the
    # view); valid_count the view's 18 (shown for context).
    _drop_last_day_bars(con, [f"S{i:02d}" for i in range(8)], last)
    con.execute(
        "UPDATE prices SET open = 3.51, high = 3.40, low = 2.90, close = 2.92 "
        "WHERE date = ? AND symbol IN ('S08','S09','S10','S11')",
        [last],
    )

    inc = doctor.incomplete_days(con)
    assert pd.Timestamp(inc.iloc[0]["date"]).date() == last
    assert int(inc.iloc[0]["bar_count"]) == 22  # RAW prices count
    assert int(inc.iloc[0]["valid_count"]) == 18  # daily_returns, for context
    assert float(inc.iloc[0]["trailing_median"]) == 30.0
    assert float(inc.iloc[0]["ratio"]) == pytest.approx(22 / 30)

    report = doctor.run(con)
    assert not report.ok
    text = "\n".join(doctor.format_report(report))
    assert "incomplete:" in text and str(last) in text


def test_doctor_clean_on_uniform_coverage(con):
    _seed_uniform(con, n_symbols=30, n_days=25)
    assert doctor.incomplete_days(con).empty  # no-op on complete data


# --- routine step -------------------------------------------------------------


def test_routine_completeness_step_warns_not_fails_when_held(con):
    from twopercent import routine

    _, dates = _seed_uniform(con, n_symbols=30, n_days=25)
    _drop_last_day_bars(con, [f"S{i:02d}" for i in range(15)], dates[-1])

    report = routine.RoutineReport()
    routine._completeness_step(report, con)
    step = report.steps[-1]
    assert step.name == "completeness"
    assert step.status == routine.WARN  # loud, but never FAIL — a held edge day is expected
    assert str(dates[-1]) in step.detail
    assert report.status != routine.FAIL  # a held day never forces a routine failure


def test_routine_completeness_step_ok_on_complete_store(con):
    from twopercent import routine

    _, dates = _seed_uniform(con, n_symbols=30, n_days=25)
    report = routine.RoutineReport()
    routine._completeness_step(report, con)
    step = report.steps[-1]
    assert step.status == routine.OK and str(dates[-1]) in step.detail
