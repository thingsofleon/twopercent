"""Shadow-trading engine: isolation from the champion (the crown-jewel
invariant), roster hygiene (cap/malformed/missing all warn loudly), run_shadow
crash isolation + write-nothing-to-champion guarantee, the forward live-only
score record, predict_for's strategy_params plumbing, the non-gating routine
steps, and the canonical.py extraction (no import cycle, re-export intact).

Offline: seeded fixtures only (conftest seed_history/seed_planted); no network.
"""

import datetime as dt
import json
import logging
from zoneinfo import ZoneInfo

import pandas as pd

from tests.conftest import seed_history, seed_planted
from twopercent import champion, predict, shadow, store, track

_EASTERN = ZoneInfo("America/New_York")

CHALLENGER = "logreg_v1 {}"  # identity of logreg_v1 with default params


def _roster(tmp_path, configs) -> str:
    path = tmp_path / "shadow.json"
    path.write_text(json.dumps(configs))
    return str(path)


# --- canonical.py extraction: no import cycle, re-export intact ---------------


def test_canonical_extraction_no_cycle_and_reexport():
    # Importing all three together must not deadlock on a cycle...
    import twopercent.canonical as canonical
    import twopercent.research as research
    import twopercent.routine as routine
    import twopercent.shadow as shadow_mod

    assert routine and shadow_mod
    # ...and research.canonical_params must still work as a re-export so the
    # existing research/generate tests keep importing it from research.
    assert research.canonical_params is canonical.canonical_params
    assert research.canonical_params({"b": 2, "a": 200.0}) == canonical.canonical_params(
        {"a": 200, "b": 2}
    )


# --- predict_for strategy_params plumbing -------------------------------------


def test_predict_for_forwards_strategy_params(con, monkeypatch):
    seed_planted(con, n_each=8)
    captured = {}
    real_get = predict.strategies.get

    def spy(name, **params):
        captured[name] = params
        return real_get(name, **params)

    monkeypatch.setattr(predict.strategies, "get", spy)
    predict.predict_for(
        con, "baseline_gbm_v1", save=False, strategy_params={"max_iter": 30, "max_depth": 2}
    )
    assert captured["baseline_gbm_v1"] == {"max_iter": 30, "max_depth": 2}


def test_predict_for_default_params_unchanged(con, monkeypatch):
    # The champion caller passes nothing → strategy constructed with no kwargs.
    seed_planted(con, n_each=8)
    captured = {}
    real_get = predict.strategies.get

    def spy(name, **params):
        captured[name] = params
        return real_get(name, **params)

    monkeypatch.setattr(predict.strategies, "get", spy)
    predict.predict_for(con, "baseline_gbm_v1", save=False)
    assert captured["baseline_gbm_v1"] == {}


# --- roster hygiene -----------------------------------------------------------


def test_roster_over_cap_drops_loudly(tmp_path, caplog):
    configs = [{"strategy": "logreg_v1", "params": {"x": i}} for i in range(6)]
    path = _roster(tmp_path, configs)
    with caplog.at_level(logging.WARNING):
        entries, malformed, dropped = shadow.load_roster(path)
    assert len(entries) == shadow.MAX_SHADOW == 4
    assert dropped == 2
    assert malformed == 0
    # Loud, and names the drop count — never a silent truncation.
    assert "over the MAX_SHADOW cap" in caplog.text
    assert "DROPPING 2" in caplog.text


def test_roster_malformed_entries_skipped_loudly(tmp_path, caplog):
    configs = [
        {"strategy": "logreg_v1", "params": {}},  # valid
        {"note": "no strategy key"},  # malformed
        "just a string",  # malformed
        {"strategy": "x", "params": "not a dict"},  # malformed
        {"strategy": "", "params": {}},  # malformed (empty strategy)
    ]
    path = _roster(tmp_path, configs)
    with caplog.at_level(logging.WARNING):
        entries, malformed, dropped = shadow.load_roster(path)
    assert [e.strategy for e in entries] == ["logreg_v1"]
    assert malformed == 4
    assert dropped == 0
    assert caplog.text.count("is malformed — SKIPPED") == 4


def test_roster_missing_file_is_empty_not_fatal(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        entries, malformed, dropped = shadow.load_roster(tmp_path / "absent.json")
    assert entries == [] and malformed == 0 and dropped == 0
    assert "does not exist" in caplog.text


def test_roster_empty_list_is_empty_no_warn(tmp_path):
    path = _roster(tmp_path, [])
    entries, malformed, dropped = shadow.load_roster(path)
    assert entries == [] and malformed == 0 and dropped == 0


def test_roster_not_a_list_is_empty_with_warn(tmp_path, caplog):
    path = tmp_path / "shadow.json"
    path.write_text('{"strategy": "logreg_v1"}')  # object, not a list
    with caplog.at_level(logging.WARNING):
        entries, malformed, dropped = shadow.load_roster(path)
    assert entries == [] and malformed == 0
    assert "not a JSON list" in caplog.text


def test_roster_unreadable_json_is_empty_with_warn(tmp_path, caplog):
    path = tmp_path / "shadow.json"
    path.write_text("{not valid json")
    with caplog.at_level(logging.WARNING):
        entries, malformed, dropped = shadow.load_roster(path)
    assert entries == [] and malformed == 0
    assert "unreadable" in caplog.text


def test_challenger_identity_distinguishes_params():
    a = shadow.ShadowEntry("baseline_gbm_v1", {"max_iter": 200})
    b = shadow.ShadowEntry("baseline_gbm_v1", {"max_iter": 200.0})  # numeric-equal
    c = shadow.ShadowEntry("baseline_gbm_v1", {"max_iter": 400})
    assert a.challenger() == b.challenger()  # 200 == 200.0 canonicalized
    assert a.challenger() != c.challenger()
    assert a.challenger() == "baseline_gbm_v1 " + '{"max_iter": 200}'


# --- run_shadow ---------------------------------------------------------------


def test_run_shadow_saves_each_challenger(con, tmp_path):
    symbols = seed_planted(con, n_each=8)
    result = predict.predict_for(con, champion.get_champion(), save=True)
    path = _roster(tmp_path, [{"strategy": "logreg_v1", "params": {}}])

    report = shadow.run_shadow(con, result.signal_date, roster_path=path)
    assert report.ran == 1 and report.failed == 0 and report.rostered == 1
    assert report.challengers == [CHALLENGER]
    rows = con.execute(
        "SELECT count(*) FROM shadow_predictions WHERE challenger = ?", [CHALLENGER]
    ).fetchone()[0]
    assert rows > 0
    # The saved rows are for the champion's signal date, tagged with the identity.
    stored = con.execute(
        "SELECT DISTINCT signal_date, strategy, params FROM shadow_predictions"
    ).fetchall()
    assert stored == [(result.signal_date, "logreg_v1", "{}")]
    assert set(symbols)  # sanity: fixture produced a universe


def test_run_shadow_per_challenger_crash_isolation(con, tmp_path, caplog):
    seed_planted(con, n_each=8)
    result = predict.predict_for(con, champion.get_champion(), save=True)
    # First entry has an unknown strategy → predict_for raises inside the loop;
    # the second (valid) must still run, and run_shadow must NOT raise.
    path = _roster(
        tmp_path,
        [
            {"strategy": "no_such_strategy", "params": {}},
            {"strategy": "logreg_v1", "params": {}},
        ],
    )
    with caplog.at_level(logging.WARNING):
        report = shadow.run_shadow(con, result.signal_date, roster_path=path)
    assert report.ran == 1 and report.failed == 1
    assert report.challengers == [CHALLENGER]
    assert "crashed" in caplog.text and "SKIPPED" in caplog.text
    # The good challenger's picks landed despite the crash.
    assert (
        con.execute(
            "SELECT count(*) FROM shadow_predictions WHERE challenger = ?", [CHALLENGER]
        ).fetchone()[0]
        > 0
    )


def test_run_shadow_writes_nothing_to_predictions_or_champion(con, tmp_path):
    seed_planted(con, n_each=8)
    result = predict.predict_for(con, champion.get_champion(), save=True)
    before = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    before_rows = con.execute("SELECT * FROM predictions ORDER BY symbol").fetchall()

    path = _roster(tmp_path, [{"strategy": "logreg_v1", "params": {}}])
    shadow.run_shadow(con, result.signal_date, roster_path=path)

    after = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    after_rows = con.execute("SELECT * FROM predictions ORDER BY symbol").fetchall()
    assert after == before
    assert after_rows == before_rows
    # No shadow rows leaked into the champion's table.
    assert (
        con.execute("SELECT count(*) FROM predictions WHERE strategy = 'logreg_v1'").fetchone()[0]
        == 0
    )


# --- THE isolation invariant --------------------------------------------------


def _save_champion_track(con, dates) -> None:
    """Log champion predictions on a mid-history day so score_predictions and
    the money tiles have a real (non-empty) record to compare."""
    store.save_predictions(
        con,
        champion.get_champion(),
        dates[50],
        pd.DataFrame({"symbol": ["RUN00", "FLT00"], "prob": [0.9, 0.1], "rank": [1, 2]}),
    )


def test_shadow_picks_never_affect_champion_record_or_dashboard(con, tmp_path):
    from twopercent import dashboard

    seed_planted(con, n_each=12)
    dates = sorted(pd.bdate_range("2026-01-05", periods=100).date)
    name = champion.get_champion()
    _save_champion_track(con, dates)

    # A precomputed result so the two dashboard renders never retrain (identical
    # by construction if isolation holds); predict_for reads prices/features,
    # never predictions or shadow_predictions.
    result = predict.predict_for(con, name, save=False)

    base_track = track.score_predictions(con, name, top_n=20).scored
    base_perf = track.daily_pick_performance(con, name).daily
    file_a = tmp_path / "a.html"
    dashboard.render(con, name, str(file_a), top=20, result=result)

    # Now shadow a challenger whose picks OVERLAP the champion's day + symbols.
    store.save_shadow_predictions(
        con,
        CHALLENGER,
        "logreg_v1",
        "{}",
        dates[50],
        pd.DataFrame(
            {"symbol": ["RUN00", "FLT00", "RUN01"], "prob": [0.8, 0.7, 0.6], "rank": [1, 2, 3]}
        ),
    )
    path = _roster(tmp_path, [{"strategy": "logreg_v1", "params": {}}])
    shadow.run_shadow(con, result.signal_date, roster_path=path)
    assert con.execute("SELECT count(*) FROM shadow_predictions").fetchone()[0] > 0

    after_track = track.score_predictions(con, name, top_n=20).scored
    after_perf = track.daily_pick_performance(con, name).daily
    file_b = tmp_path / "b.html"
    dashboard.render(con, name, str(file_b), top=20, result=result)

    # Byte-for-byte identical with vs without shadow rows present.
    pd.testing.assert_frame_equal(base_track, after_track)
    pd.testing.assert_frame_equal(base_perf, after_perf)
    assert file_a.read_bytes() == file_b.read_bytes()


# --- score_shadow forward (live-only) record ----------------------------------

OC = {"AAA": [0.001] * 23 + [0.03, 0.05], "BBB": [0.001] * 25}


def _save_shadow_last_signal(con) -> dt.date:
    seed_history(con, OC)
    dates = sorted(pd.bdate_range("2026-01-05", periods=25).date)
    store.save_shadow_predictions(
        con,
        CHALLENGER,
        "logreg_v1",
        "{}",
        dates[-2],
        pd.DataFrame({"symbol": ["AAA", "BBB"], "prob": [0.9, 0.1], "rank": [1, 2]}),
    )
    return dates[-1]


def _set_shadow_created(con, when_et: dt.datetime) -> None:
    local = dt.datetime.now().astimezone().tzinfo
    con.execute(
        "UPDATE shadow_predictions SET created_ts = ?",
        [when_et.astimezone(local).replace(tzinfo=None)],
    )


def test_score_shadow_backfilled_pick_excluded_from_live_record(con):
    # Saved NOW for a long-past signal date = a backfill (late): the same
    # 09:30-ET rule the champion uses must exclude it from the FORWARD record.
    _save_shadow_last_signal(con)
    report = shadow.score_shadow(con)
    assert len(report.scores) == 1
    score = report.scores[0]
    assert score.challenger == CHALLENGER
    assert score.late_days == 1
    assert score.live_days == 0
    assert score.mean_lift is None  # no live day contributes


def test_score_shadow_live_pick_forms_forward_record(con):
    target = _save_shadow_last_signal(con)
    # Created 09:00 ET on the target day — before the open, outcome unknown: live.
    _set_shadow_created(con, dt.datetime.combine(target, dt.time(9, 0), tzinfo=_EASTERN))
    report = shadow.score_shadow(con)
    score = report.scores[0]
    assert score.late_days == 0
    assert score.live_days == 1
    assert score.mean_precision is not None
    assert score.mean_base_rate is not None


def test_score_shadow_no_challengers_is_empty(con):
    seed_history(con, OC)
    report = shadow.score_shadow(con)
    assert report.scores == [] and report.failed == 0


# --- routine wiring: non-gating, after notify ---------------------------------


def test_shadow_run_step_after_notify_and_non_gating(con, tmp_path, monkeypatch):
    """The predict-mode shadow step runs AFTER notify and, if run_shadow raises
    (defense in depth — it normally never does), the step WARNs and the routine
    does not FAIL/abort."""
    from twopercent import routine

    report = routine.RoutineReport()

    class _Pred:
        signal_date = dt.date(2026, 3, 2)

    # Boom: even a hard crash must be caught and downgraded to WARN, never FAIL.
    monkeypatch.setattr(
        routine.shadow, "run_shadow", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    routine._shadow_run_step(report, con, _Pred.signal_date)
    step = report.steps[-1]
    assert step.name == "shadow"
    assert step.status == "warn"  # WARN, not FAIL
    assert report.status != "fail"


def test_shadow_run_step_empty_roster_is_ok(con, tmp_path, monkeypatch):
    from twopercent import routine

    monkeypatch.setattr(shadow, "SHADOW_PATH", tmp_path / "absent.json")
    report = routine.RoutineReport()
    routine._shadow_run_step(report, con, dt.date(2026, 3, 2))
    step = report.steps[-1]
    assert step.name == "shadow" and step.status == "ok"
    assert "no challengers" in step.detail


def test_shadow_score_step_non_gating(con, monkeypatch):
    from twopercent import routine

    report = routine.RoutineReport()
    monkeypatch.setattr(
        routine.shadow, "score_shadow", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    routine._shadow_score_step(report, con)
    step = report.steps[-1]
    assert step.name == "shadow" and step.status == "warn"
    assert report.status != "fail"
