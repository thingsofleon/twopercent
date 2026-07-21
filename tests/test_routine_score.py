"""Score mode: post-close gating, scoring, degradation detector, issue filing.

Offline per the routine-test pattern: pinned ET clocks, ingest/universe/gh
monkeypatched, never the real store."""

import datetime as dt
import subprocess

import pandas as pd
import pytest

from tests.conftest import seed_history, seed_planted
from twopercent import champion, ingest, routine, store, track

POST_CLOSE = dt.datetime(2026, 7, 17, 17, 0, tzinfo=routine._EASTERN)  # Friday 17:00 ET


def _db(con) -> str:
    return con.execute("PRAGMA database_list").fetchone()[2]


@pytest.fixture
def ready(con, monkeypatch):
    """Healthy seeded store, post-close clock, network steps stubbed out."""
    seed_planted(con, n_each=10)
    con.execute("UPDATE universe SET as_of = ?", [POST_CLOSE.date()])
    # Newest bar lands 3 days before the pinned clock — deterministic forever.
    span = (POST_CLOSE.date() - pd.Timestamp("2026-01-05").date()).days
    con.execute(f"UPDATE prices SET date = date + INTERVAL {span - 137} DAY")
    monkeypatch.setattr(routine, "_now_eastern", lambda: POST_CLOSE)
    monkeypatch.setattr(
        routine.ingest,
        "ingest",
        lambda con_, symbols, **kw: ingest.IngestResult(symbols_skipped=list(symbols)),
    )
    monkeypatch.setattr(
        routine.universe,
        "refresh_universe",
        lambda **kw: pytest.fail("universe refresh touched in score mode"),
    )
    return con


def _seed_predictions(con, strategy: str, n_days: int = 3) -> list[dt.date]:
    """Log top-5 RUN* picks for the last n_days signal dates that already have
    a following trading day in the store (so they score immediately)."""
    dates = [d for (d,) in con.execute("SELECT DISTINCT date FROM prices ORDER BY date").fetchall()]
    frame = pd.DataFrame(
        {
            "symbol": [f"RUN{i:02d}" for i in range(5)],
            "prob": [0.9, 0.8, 0.7, 0.6, 0.5],
            "rank": [1, 2, 3, 4, 5],
        }
    )
    for day in dates[-(n_days + 1) : -1]:
        store.save_predictions(con, strategy, day, frame)
    return dates


# --- clock gate ---------------------------------------------------------------


@pytest.mark.parametrize(
    "when, allowed",
    [
        (dt.datetime(2026, 7, 17, 10, 0, tzinfo=routine._EASTERN), False),  # Fri mid-session
        (dt.datetime(2026, 7, 17, 6, 0, tzinfo=routine._EASTERN), False),  # Fri pre-open
        (dt.datetime(2026, 7, 17, 16, 14, tzinfo=routine._EASTERN), False),  # Fri 16:14
        (dt.datetime(2026, 7, 17, 16, 15, tzinfo=routine._EASTERN), True),  # boundary opens
        (dt.datetime(2026, 7, 17, 17, 0, tzinfo=routine._EASTERN), True),  # Fri post-close
        (dt.datetime(2026, 7, 18, 11, 0, tzinfo=routine._EASTERN), True),  # Saturday midday
    ],
)
def test_score_mode_clock_gate(ready, monkeypatch, when, allowed):
    monkeypatch.setattr(routine, "_now_eastern", lambda: when)
    # Gate test only — skip the expensive real model/dashboard tail.
    monkeypatch.setattr(routine, "predict_for", lambda *a, **kw: None)
    monkeypatch.setattr(routine.dashboard, "render", lambda *a, **kw: "x")
    report = routine.run(db_path=_db(ready), mode="score")
    if allowed:
        assert [s.name for s in report.steps] != ["clock"]
        assert report.steps[0].status == "ok"
    else:
        assert report.status == "fail"
        assert [s.name for s in report.steps] == ["clock"]
        assert "post-close" in report.steps[0].detail


# --- score flow ---------------------------------------------------------------


def test_score_mode_happy_path_no_predict_no_universe_refresh(ready, monkeypatch):
    name = champion.get_champion()
    _seed_predictions(ready, name, n_days=3)
    before = ready.execute("SELECT count(*) FROM predictions").fetchone()[0]

    calls = []
    real_predict = routine.predict_for

    def spy(con_, strat, **kw):
        calls.append(kw)
        return real_predict(con_, strat, **kw)

    monkeypatch.setattr(routine, "predict_for", spy)
    report = routine.run(db_path=_db(ready), mode="score")

    names = [s.name for s in report.steps]
    assert names == [
        "clock",
        "doctor",
        "ingest",
        "freshness",
        "recheck",
        "score",
        "detector",
        "dashboard",
    ]
    assert "predict" not in names and "universe" not in names
    assert "notify" not in names  # score-mode runs never email the signal
    # The dashboard's display-only rescore never logs predictions.
    assert calls and all(kw.get("save") is False for kw in calls)
    after = ready.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert after == before
    # 3 seeded days scored; all late (created after target) => detector unarmed, loudly.
    score_step = next(s for s in report.steps if s.name == "score")
    assert "3 total (3 late)" in score_step.detail
    detector = next(s for s in report.steps if s.name == "detector")
    assert detector.status == "warn" and "armed after" in detector.detail
    assert report.last_scored and "lift" in report.last_scored


def test_score_mode_never_refreshes_universe_even_when_stale(ready, monkeypatch):
    ready.execute("UPDATE universe SET as_of = ?", [POST_CLOSE.date() - dt.timedelta(days=30)])
    monkeypatch.setattr(routine, "predict_for", lambda *a, **kw: None)
    monkeypatch.setattr(routine.dashboard, "render", lambda *a, **kw: "x")
    report = routine.run(db_path=_db(ready), mode="score")
    assert "universe" not in [s.name for s in report.steps]  # refresh stub would pytest.fail


def test_score_mode_counts_newly_resolved_days(ready, monkeypatch):
    name = champion.get_champion()
    dates = _seed_predictions(ready, name, n_days=2)
    newest = dates[-1]
    # Also predict on the newest bar: pending until today's ingest lands the next day.
    store.save_predictions(
        ready,
        name,
        newest,
        pd.DataFrame({"symbol": ["RUN00", "RUN01"], "prob": [0.9, 0.8], "rank": [1, 2]}),
    )

    def resolving_ingest(con_, symbols, **kw):
        nxt = (pd.Timestamp(newest) + pd.tseries.offsets.BDay(1)).date()
        oc = {s: [0.03 if s.startswith("RUN") else 0.002] for s in symbols}
        seed_history(con_, oc, start=str(nxt))
        return ingest.IngestResult(symbols_ok=list(symbols))

    monkeypatch.setattr(routine.ingest, "ingest", resolving_ingest)
    report = routine.run(db_path=_db(ready), mode="score")
    score_step = next(s for s in report.steps if s.name == "score")
    assert score_step.status == "ok"
    assert "1 new day(s) scored" in score_step.detail
    resolved_day = (pd.Timestamp(newest) + pd.tseries.offsets.BDay(1)).date()
    assert report.last_scored.startswith(str(resolved_day))


def test_score_mode_zero_new_days_warns(ready, monkeypatch):
    # No predictions at all: nothing scoreable — the run says so and exits 1.
    monkeypatch.setattr(routine, "predict_for", lambda *a, **kw: None)
    monkeypatch.setattr(routine.dashboard, "render", lambda *a, **kw: "x")
    report = routine.run(db_path=_db(ready), mode="score")
    score_step = next(s for s in report.steps if s.name == "score")
    assert score_step.status == "warn"
    assert "0 new day(s) scored" in score_step.detail
    assert report.exit_code == 1


# --- degradation -> issue filing ----------------------------------------------


def _degraded_record(n_live: int = 5, lift: float = 0.5) -> track.TrackRecord:
    dates = pd.bdate_range("2026-07-06", periods=n_live)
    scored = pd.DataFrame(
        {
            "signal_date": dates - pd.tseries.offsets.BDay(1),
            "target_date": dates,
            "hits": [1] * n_live,
            "n_scored": [20] * n_live,
            "precision": [0.05] * n_live,
            "base_rate": [0.10] * n_live,
            "lift": [lift] * n_live,
            "late": [False] * n_live,
        }
    )
    return track.TrackRecord(scored=scored, pending=[])


@pytest.fixture
def degraded(ready, monkeypatch):
    """Score run whose track record shows 5 live days at lift 0.5."""
    monkeypatch.setattr(
        routine.track, "score_predictions", lambda con_, s, top_n=20: _degraded_record()
    )
    monkeypatch.setattr(
        routine.track,
        "daily_pick_performance",
        lambda con_, s, top_n=5: track.PickPerformance(daily=pd.DataFrame()),
    )
    monkeypatch.setattr(routine, "predict_for", lambda *a, **kw: None)
    monkeypatch.setattr(routine.dashboard, "render", lambda *a, **kw: "x")
    return ready


def _gh_spy(
    monkeypatch,
    list_stdout: str = "[]",
    fail: Exception | None = None,
    fail_on: list | None = None,
) -> list:
    """Monkeypatch subprocess.run with a gh fake. `fail` raises on every call,
    or only on calls whose args start with `fail_on` when that is given."""
    calls = []

    def fake_run(args, **kw):
        calls.append((args, kw))
        # Security posture: argument LISTS only, never a shell string.
        assert isinstance(args, list)
        assert not kw.get("shell")
        if fail is not None and (fail_on is None or args[: len(fail_on)] == fail_on):
            raise fail
        if args[:3] == ["gh", "issue", "list"]:
            return subprocess.CompletedProcess(args, 0, stdout=list_stdout, stderr="")
        if args[:3] == ["gh", "label", "create"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[:3] == ["gh", "issue", "create"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="https://github.com/x/twopercent/issues/99\n", stderr=""
            )
        if args[:3] == ["gh", "issue", "lock"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected subprocess call: {args}")

    monkeypatch.setattr(routine.subprocess, "run", fake_run)
    return calls


def test_degraded_files_issue_and_exits_2(degraded, monkeypatch):
    calls = _gh_spy(monkeypatch)
    report = routine.run(db_path=_db(degraded), mode="score")

    assert report.exit_code == 2
    detector = next(s for s in report.steps if s.name == "detector")
    assert detector.status == "fail" and "DEGRADED" in detector.detail
    issue = next(s for s in report.steps if s.name == "issue")
    assert issue.status == "ok" and "issues/99" in issue.detail

    create, create_kw = next(
        (args, kw) for args, kw in calls if args[:3] == ["gh", "issue", "create"]
    )
    assert create[create.index("--title") + 1] == (
        "Auto: champion underperforming baseline (trailing-5 live lift 0.50)"
    )
    assert create[create.index("--body-file") + 1] == "-"  # body via stdin, not the shell
    assert create[create.index("--label") + 1] == "auto-degradation"
    body = create_kw["input"]
    assert "Last 10 scored days" in body
    assert ".claude/agents/investigator.md" in body
    assert champion.get_champion() in body
    assert "2026-07-10" in body  # last of the 5 degraded target dates in the table
    assert "| * |" in body  # trailing-window rows are marked in the table
    assert "5 of 5 window day(s) individually below 1.0" in body
    # Label ensured idempotently before create.
    label = next(args for args, kw in calls if args[:3] == ["gh", "label", "create"])
    assert "--force" in label and "auto-degradation" in label
    # Conversation locked at creation (public repo: comments are untrusted).
    lock = next(args for args, kw in calls if args[:3] == ["gh", "issue", "lock"])
    assert lock[3] == "99" and "--reason" in lock
    assert "locked" in issue.detail


def test_degraded_dedups_existing_open_issue(degraded, monkeypatch):
    calls = _gh_spy(monkeypatch, list_stdout='[{"number": 42, "title": "Auto: earlier"}]')
    report = routine.run(db_path=_db(degraded), mode="score")
    assert report.exit_code == 2  # detector FAIL stands even without a new issue
    issue = next(s for s in report.steps if s.name == "issue")
    assert issue.status == "warn" and "#42" in issue.detail
    assert not any(args[:3] == ["gh", "issue", "create"] for args, _ in calls)


def test_degraded_lock_failure_warns_but_issue_stands(degraded, monkeypatch):
    # Lock is best-effort: its failure must not unfile the issue or be silent.
    err = subprocess.CalledProcessError(1, ["gh"], stderr="lock forbidden")
    calls = _gh_spy(monkeypatch, fail=err, fail_on=["gh", "issue", "lock"])
    report = routine.run(db_path=_db(degraded), mode="score")
    assert report.exit_code == 2
    assert any(args[:3] == ["gh", "issue", "create"] for args, _ in calls)
    issue = next(s for s in report.steps if s.name == "issue")
    assert issue.status == "warn"
    assert "issues/99" in issue.detail and "could not lock" in issue.detail


def test_detector_pinned_to_top_20_regardless_of_cli_top(ready, monkeypatch):
    # The lift series is locked to the top-20 basket (ROADMAP): a --top 5 run
    # may restyle the dashboard but must never change what the detector sees.
    seen = []

    def fake_scores(con_, s, top_n=20):
        seen.append(top_n)
        return track.TrackRecord(scored=pd.DataFrame(), pending=[])

    monkeypatch.setattr(routine.track, "score_predictions", fake_scores)
    monkeypatch.setattr(
        routine.track,
        "daily_pick_performance",
        lambda con_, s, top_n=5: track.PickPerformance(daily=pd.DataFrame()),
    )
    monkeypatch.setattr(routine, "predict_for", lambda *a, **kw: None)
    monkeypatch.setattr(routine.dashboard, "render", lambda *a, **kw: "x")
    routine.run(db_path=_db(ready), top=5, mode="score")
    assert seen and all(n == 20 for n in seen)


def test_score_mode_malformed_champion_fails_with_report(ready, monkeypatch):
    # A partial champion.json write at the 14:45 timer must produce a FAIL
    # summary line and exit 2 — never an uncaught traceback (exit 1, no report).
    def boom(*a, **kw):
        raise ValueError("champion.json is malformed")

    monkeypatch.setattr(routine.champion, "get_champion", boom)
    report = routine.run(db_path=_db(ready), mode="score")
    assert report.exit_code == 2
    assert report.steps[-1].name == "score" and report.steps[-1].status == "fail"
    assert "champion" in report.steps[-1].detail


def test_pre_ingest_snapshot_failure_reports_unknown_new_days(ready, monkeypatch):
    # (a) The pre-ingest capture fails but post-ingest scoring works: the
    # summary must say the new-day count is UNKNOWN and warn — never report
    # a confident "0 new days" over a count it could not compute.
    calls = {"n": 0}

    def flaky_scores(con_, s, top_n=20):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("pre-ingest boom")
        return _degraded_record(lift=1.5)  # healthy: detector must not fire

    monkeypatch.setattr(routine.track, "score_predictions", flaky_scores)
    monkeypatch.setattr(
        routine.track,
        "daily_pick_performance",
        lambda con_, s, top_n=5: track.PickPerformance(daily=pd.DataFrame()),
    )
    monkeypatch.setattr(routine, "predict_for", lambda *a, **kw: None)
    monkeypatch.setattr(routine.dashboard, "render", lambda *a, **kw: "x")
    report = routine.run(db_path=_db(ready), mode="score")
    score_step = next(s for s in report.steps if s.name == "score")
    assert score_step.status == "warn"
    assert "unknown new days" in score_step.detail
    assert "0 new day(s)" not in score_step.detail


def test_post_ingest_scoring_crash_fails_run(ready, monkeypatch):
    # (b) Scoring itself crashes after ingest: score step FAIL, exit 2.
    def broken(con_, s, top_n=20):
        raise RuntimeError("scoring exploded")

    monkeypatch.setattr(routine.track, "score_predictions", broken)
    report = routine.run(db_path=_db(ready), mode="score")
    score_step = next(s for s in report.steps if s.name == "score")
    assert score_step.status == "fail" and "scoring exploded" in score_step.detail
    assert report.exit_code == 2


def test_degraded_gh_missing_warns_and_still_exits_2(degraded, monkeypatch):
    _gh_spy(monkeypatch, fail=FileNotFoundError("gh"))
    report = routine.run(db_path=_db(degraded), mode="score")
    assert report.exit_code == 2
    issue = next(s for s in report.steps if s.name == "issue")
    assert issue.status == "warn" and "NO issue was filed" in issue.detail


def test_degraded_gh_error_warns_and_still_exits_2(degraded, monkeypatch):
    err = subprocess.CalledProcessError(1, ["gh"], stderr="auth required")
    _gh_spy(monkeypatch, fail=err)
    report = routine.run(db_path=_db(degraded), mode="score")
    assert report.exit_code == 2
    issue = next(s for s in report.steps if s.name == "issue")
    assert issue.status == "warn" and "auth required" in issue.detail


# --- mode plumbing ------------------------------------------------------------


def test_unknown_mode_rejected(ready):
    with pytest.raises(ValueError, match="unknown routine mode"):
        routine.run(db_path=_db(ready), mode="bogus")


def test_issue_body_quotes_default_config_benchmark_not_variant(degraded, monkeypatch):
    """Research variants record under the champion's strategy NAME — the
    evidence bundle must quote the champion's own (params-free) benchmark,
    never a newer parameterized sweep row."""
    name = champion.get_champion()
    store.record_experiment(
        degraded,
        strategy=name,
        params={"months": 12, "top_n": 20, "dropped_columns": []},  # pre-research shape
        train_start=dt.date(2021, 7, 1),
        test_start=dt.date(2025, 7, 1),
        test_end=dt.date(2026, 6, 30),
        metrics={"lift": 2.0, "auc": 0.69},
    )
    store.record_experiment(  # NEWER variant row with a flattering lift
        degraded,
        strategy=name,
        params={"months": 12, "top_n": 20, "strategy_params": {"max_iter": 300}},
        train_start=dt.date(2021, 7, 1),
        test_start=dt.date(2025, 7, 1),
        test_end=dt.date(2026, 6, 30),
        metrics={"lift": 9.9, "auc": 0.99},
    )
    calls = _gh_spy(monkeypatch)
    routine.run(db_path=_db(degraded), mode="score")

    _, create_kw = next((args, kw) for args, kw in calls if args[:3] == ["gh", "issue", "create"])
    body = create_kw["input"]
    assert '"lift": 2.0' in body  # the champion's own benchmark
    assert "9.9" not in body  # the variant must never be quoted as the champion
