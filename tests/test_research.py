"""Research runner: clock gate, queue hygiene, ledger-based idempotency,
budget cap, crash isolation, promotion gating (band + disjoint halves),
issue filing, and the write-nothing guarantee. Offline: benchmark and gh are
monkeypatched; clocks are pinned."""

import datetime as dt
import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from twopercent import champion, issues, research, store

IN_WINDOW = dt.datetime(2026, 7, 17, 22, 0, tzinfo=research._DENVER)  # Friday 22:00 Denver

METRICS = {
    "precision_at_n": 0.3,
    "top_n": 20,
    "base_rate": 0.15,
    "lift": 2.5,
    "auc": 0.71,
    "brier": 0.12,
    "sim_top1_growth": 1.5,
    "test_rows": 100_000,
    "test_days": 250,
    "folds": 12,
}
# Shared test days for the disjoint-halves confirmation (champion + fakes).
DAILY_DATES = [d.date() for d in pd.bdate_range("2025-07-01", periods=10)]


def _db(con) -> str:
    return con.execute("PRAGMA database_list").fetchone()[2]


def _write_queue(tmp_path, configs) -> Path:
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(configs))
    return path


def _config(strategy="xgb_gbm_v1", **params):
    return {"strategy": strategy, "params": params, "note": "test config"}


def _daily_rows(prec_per_day) -> pd.DataFrame:
    """Top-20 rank rows whose per-day mean(hit) equals each day's precision."""
    rows = []
    for day, prec in zip(DAILY_DATES, prec_per_day, strict=True):
        hits = int(round(prec * 20))
        for rank in range(1, 21):
            rows.append(
                {"target_date": day, "rank": rank, "ret": 0.01, "hit": 1 if rank <= hits else 0}
            )
    return pd.DataFrame(rows)


def _seed_champion_experiment(con, lift=2.0, test_end=dt.date(2026, 6, 30), daily_prec=0.5) -> int:
    """The champion's recorded standard 12-month benchmark WITH daily rows."""
    seq = store.record_experiment(
        con,
        strategy=champion.get_champion(),
        params={"months": 12, "top_n": 20, "dropped_columns": [], "strategy_params": {}},
        train_start=dt.date(2021, 7, 1),
        test_start=dt.date(2025, 7, 1),
        test_end=test_end,
        metrics={**METRICS, "lift": lift, "auc": 0.69},
    )
    store.record_experiment_daily(con, seq, _daily_rows([daily_prec] * len(DAILY_DATES)))
    return seq


@pytest.fixture(autouse=True)
def no_real_gh(monkeypatch):
    """No research test may ever reach the real gh CLI: any subprocess call not
    intercepted by an explicit _gh_spy is a test bug and must blow up."""

    def forbidden(args, **kw):
        raise AssertionError(f"test reached real subprocess: {args}")

    monkeypatch.setattr(issues.subprocess, "run", forbidden)


@pytest.fixture(autouse=True)
def no_autogen(monkeypatch):
    """Disable the auto-search generator by default: the curated-queue tests
    below exercise budget/window/dedup/promotion mechanics and must not have
    their spare budget filled by generated configs. Generation is tested
    explicitly (test_autogen_*) by re-patching grid_configs to a small grid."""
    monkeypatch.setattr(research.generate, "grid_configs", lambda: [])


@pytest.fixture
def night(monkeypatch):
    monkeypatch.setattr(research, "_now_denver", lambda: IN_WINDOW)


@pytest.fixture
def bench_spy(monkeypatch):
    """Fake referee: records a ledger row + daily rows like the real one and
    returns METRICS. Overrides per config key: lifts (lift value), daily_prec
    (scalar or per-day list), boom (raise), device (recorded params device)."""
    state = {"calls": [], "lifts": {}, "daily_prec": {}, "boom": set(), "device": None}

    def fake(con, name, months=12, top_n=20, record=True, strategy_params=None):
        key = json.dumps(strategy_params or {}, sort_keys=True)
        state["calls"].append({"strategy": name, "months": months, "top_n": top_n, "params": key})
        if key in state["boom"]:
            raise RuntimeError("synthetic benchmark crash")
        metrics = {**METRICS, "lift": state["lifts"].get(key, METRICS["lift"])}
        if record:
            params = {"months": months, "top_n": top_n, "strategy_params": strategy_params or {}}
            if state["device"] is not None:
                params["device"] = state["device"]
            seq = store.record_experiment(
                con,
                strategy=name,
                params=params,
                train_start=dt.date(2021, 7, 1),
                test_start=dt.date(2025, 7, 1),
                test_end=dt.date(2026, 6, 30),
                metrics=metrics,
            )
            prec = state["daily_prec"].get(key, 0.8)
            precs = [prec] * len(DAILY_DATES) if isinstance(prec, int | float) else prec
            store.record_experiment_daily(con, seq, _daily_rows(precs))
        return metrics

    monkeypatch.setattr(research.backtest, "run_benchmark", fake)
    return state


def _gh_spy(monkeypatch, list_stdout: str = "[]") -> list:
    calls = []

    def fake_run(args, **kw):
        calls.append((args, kw))
        assert isinstance(args, list)  # arg LISTS only, never a shell string
        assert not kw.get("shell")
        if args[:3] == ["gh", "issue", "list"]:
            return subprocess.CompletedProcess(args, 0, stdout=list_stdout, stderr="")
        if args[:3] == ["gh", "label", "create"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[:3] == ["gh", "issue", "create"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="https://github.com/x/twopercent/issues/77\n", stderr=""
            )
        if args[:3] == ["gh", "issue", "lock"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected subprocess call: {args}")

    monkeypatch.setattr(issues.subprocess, "run", fake_run)
    return calls


# --- clock gate ---------------------------------------------------------------


@pytest.mark.parametrize(
    "when, allowed",
    [
        (dt.datetime(2026, 7, 17, 22, 0), True),  # Friday 22:00 (the timer slot)
        (dt.datetime(2026, 7, 18, 22, 0), True),  # Saturday night: offline compute is fine
        (dt.datetime(2026, 7, 17, 16, 30), True),  # window opens
        (dt.datetime(2026, 7, 17, 16, 29), False),  # one minute early
        (dt.datetime(2026, 7, 18, 4, 59), True),  # small hours
        (dt.datetime(2026, 7, 18, 5, 0), False),  # window closes
        (dt.datetime(2026, 7, 17, 6, 0), False),  # predict-run slot
        (dt.datetime(2026, 7, 17, 12, 0), False),  # market hours
        (dt.datetime(2026, 7, 17, 14, 45), False),  # score-run slot
    ],
)
def test_clock_gate_matrix(con, tmp_path, monkeypatch, when, allowed, bench_spy):
    monkeypatch.setattr(research, "_now_denver", lambda: when.replace(tzinfo=research._DENVER))
    queue = _write_queue(tmp_path, [_config(max_depth=4)])  # pending work: no queue-empty path
    report = research.run(db_path=_db(con), queue_path=queue)
    if allowed:
        assert report.steps[0].name == "clock" and report.steps[0].status == "ok"
        assert report.exit_code == 0
    else:
        assert [s.name for s in report.steps] == ["clock"]
        assert report.exit_code == 2
        assert "research window" in report.steps[0].detail


def test_window_rechecked_before_each_experiment(con, tmp_path, monkeypatch, bench_spy):
    """Entry gating is not enough: a slow night must stop starting new configs
    once past the 04:00 last-start margin, releasing the store well before the
    06:00 predict run."""
    times = iter(
        [
            dt.datetime(2026, 7, 18, 3, 0, tzinfo=research._DENVER),  # entry gate
            dt.datetime(2026, 7, 18, 3, 30, tzinfo=research._DENVER),  # config 1 may start
            dt.datetime(2026, 7, 18, 4, 10, tzinfo=research._DENVER),  # past the margin
        ]
    )
    monkeypatch.setattr(research, "_now_denver", lambda: next(times))
    queue = _write_queue(tmp_path, [_config(max_depth=4), _config(max_depth=6)])
    report = research.run(db_path=_db(con), queue_path=queue)

    assert len(bench_spy["calls"]) == 1  # second config never started
    assert report.n_ran == 1
    window = next(s for s in report.steps if s.name == "window")
    assert window.status == "warn"
    assert "stopped after 1 of 2" in window.detail and "window closing" in window.detail
    assert report.exit_code == 0  # partial completion is not failure


def test_budget_below_one_is_fatal(con, tmp_path, night, bench_spy):
    report = research.run(db_path=_db(con), budget=0, queue_path=_write_queue(tmp_path, []))
    assert report.exit_code == 2
    assert [s.name for s in report.steps] == ["budget"]
    assert not bench_spy["calls"]


# --- queue hygiene ------------------------------------------------------------


def test_empty_queue_is_loud_and_files_refill_issue(con, tmp_path, night, bench_spy, monkeypatch):
    """A truly-empty queue file is an attention state (WARN, exit 1), not a
    clean night: the runner would no-op forever, so it files a refill issue."""
    gh = _gh_spy(monkeypatch)
    report = research.run(db_path=_db(con), queue_path=_write_queue(tmp_path, []))

    assert report.exit_code == 1
    assert report.queue_exhausted
    queue_step = next(s for s in report.steps if s.name == "queue")
    # No curated work and (autouse) an empty grid → nothing to run, refill signal.
    assert queue_step.status == "warn" and "0 pending" in queue_step.detail
    assert not bench_spy["calls"]

    create = next(a for a, _ in gh if a[:3] == ["gh", "issue", "create"])
    assert create[create.index("--label") + 1] == "research-queue-empty"
    issue_step = next(s for s in report.steps if s.name == "issue")
    assert issue_step.status == "ok" and "issues/77" in issue_step.detail


def test_all_malformed_queue_files_refill_issue(con, tmp_path, night, bench_spy, monkeypatch):
    """A queue whose entries are ALL malformed leaves zero valid entries — the
    `not entries` branch with n_malformed > 0. It is still exhausted (WARN, exit
    1) and files the refill issue, and the queue detail names the drop."""
    gh = _gh_spy(monkeypatch)
    queue = _write_queue(
        tmp_path,
        [
            {"params": {"max_depth": 4}},  # no strategy
            "not even a dict",  # not a dict at all
        ],
    )
    report = research.run(db_path=_db(con), queue_path=queue)

    assert report.n_malformed == 2
    assert report.queue_exhausted and report.exit_code == 1
    assert not bench_spy["calls"]
    queue_step = next(s for s in report.steps if s.name == "queue")
    assert queue_step.status == "warn"
    assert "2 malformed" in queue_step.detail and "0 pending" in queue_step.detail

    create = next(a for a, _ in gh if a[:3] == ["gh", "issue", "create"])
    assert create[create.index("--label") + 1] == "research-queue-empty"
    issue_step = next(s for s in report.steps if s.name == "issue")
    assert issue_step.status == "ok" and "issues/77" in issue_step.detail


def test_all_recorded_queue_files_refill_issue(con, tmp_path, night, bench_spy, monkeypatch):
    """Every config already recorded (pending == []) is the state that fires
    every night once the seeded sweep is done: WARN + one refill issue."""
    gh = _gh_spy(monkeypatch)
    configs = [_config(max_depth=4), _config(max_depth=6)]
    for cfg in configs:  # seed the ledger so every queue key is already done
        store.record_experiment(
            con,
            strategy=cfg["strategy"],
            params={"months": 12, "top_n": 20, "strategy_params": cfg["params"]},
            train_start=dt.date(2021, 7, 1),
            test_start=dt.date(2025, 7, 1),
            test_end=dt.date(2026, 6, 30),
            metrics=METRICS,
        )
    report = research.run(db_path=_db(con), queue_path=_write_queue(tmp_path, configs))

    assert not bench_spy["calls"]  # nothing pending, nothing ran
    assert report.n_skipped_done == 2
    assert report.queue_exhausted and report.exit_code == 1
    queue_step = next(s for s in report.steps if s.name == "queue")
    assert queue_step.status == "warn"

    creates = [a for a, _ in gh if a[:3] == ["gh", "issue", "create"]]
    assert len(creates) == 1  # filed exactly once
    assert creates[0][creates[0].index("--label") + 1] == "research-queue-empty"
    body = next(kw["input"] for a, kw in gh if a[:3] == ["gh", "issue", "create"])
    assert "2" in body and "twopercent experiments" in body  # honest, forwardable
    assert "pull request" in body and "strategy-researcher" in body


def test_pending_work_does_not_file_refill_issue(con, tmp_path, night, bench_spy, monkeypatch):
    """A normal night with pending work must be provably unaffected — no
    research-queue-empty issue is filed."""
    _seed_champion_experiment(con, lift=2.5)  # ties survivor: no promotion path either
    gh = _gh_spy(monkeypatch)
    report = research.run(
        db_path=_db(con), queue_path=_write_queue(tmp_path, [_config(max_depth=4)])
    )

    assert report.n_ran == 1
    assert not report.queue_exhausted
    assert not any(s.name == "issue" for s in report.steps)
    assert not any(
        "research-queue-empty" in a for args, _ in gh for a in args
    )  # never the refill label
    assert not gh  # in fact no gh call at all (champion tie files nothing)


# --- auto-search generation (tier 1) -----------------------------------------

FAKE_GRID = [
    {
        "strategy": "xgb_gbm_v1",
        "params": {"reg_lambda": 5.0},
        "note": "auto-search: reg_lambda=5.0",
    },
    {"strategy": "xgb_gbm_v1", "params": {"reg_lambda": 10.0}, "note": "auto: reg_lambda=10.0"},
    {"strategy": "baseline_gbm_v1", "params": {"l2_regularization": 1.0}, "note": "auto: l2=1.0"},
]


def _seed_recorded(con, strategy, params):
    store.record_experiment(
        con,
        strategy=strategy,
        params={"months": 12, "top_n": 20, "strategy_params": params},
        train_start=dt.date(2021, 7, 1),
        test_start=dt.date(2025, 7, 1),
        test_end=dt.date(2026, 6, 30),
        metrics=METRICS,
    )


def test_autogen_fills_when_curated_all_recorded(con, tmp_path, night, bench_spy, monkeypatch):
    """The #54 state — every curated config already recorded — no longer no-ops:
    the runner tops the batch up from the auto-search grid so the GPU works, and
    files NO refill issue because there is real work to do."""
    monkeypatch.setattr(research.generate, "grid_configs", lambda: list(FAKE_GRID))
    _seed_recorded(con, "xgb_gbm_v1", {"max_depth": 4})  # the sole curated config is done
    report = research.run(
        db_path=_db(con), budget=8, queue_path=_write_queue(tmp_path, [_config(max_depth=4)])
    )

    assert report.n_generated == 3 and report.n_ran == 3
    assert not report.queue_exhausted and report.exit_code == 0
    assert not any(s.name == "issue" for s in report.steps)  # no refill signal — there was work
    ran = {c["params"] for c in bench_spy["calls"]}
    assert ran == {'{"reg_lambda": 5.0}', '{"reg_lambda": 10.0}', '{"l2_regularization": 1.0}'}


def test_autogen_respects_budget(con, tmp_path, night, bench_spy, monkeypatch):
    monkeypatch.setattr(research.generate, "grid_configs", lambda: list(FAKE_GRID))
    report = research.run(db_path=_db(con), budget=2, queue_path=_write_queue(tmp_path, []))
    assert report.n_generated == 2 and len(bench_spy["calls"]) == 2  # never exceeds budget


def test_autogen_skips_grid_config_already_in_ledger(con, tmp_path, night, bench_spy, monkeypatch):
    """A grid config already recorded (canonical key, 5.0 == 5) is never re-run —
    generation is idempotent against the ledger, like the curated queue."""
    monkeypatch.setattr(research.generate, "grid_configs", lambda: list(FAKE_GRID))
    _seed_recorded(con, "xgb_gbm_v1", {"reg_lambda": 5.0})
    report = research.run(db_path=_db(con), budget=8, queue_path=_write_queue(tmp_path, []))

    assert report.n_generated == 2  # the recorded one dropped
    ran = {c["params"] for c in bench_spy["calls"]}
    assert '{"reg_lambda": 5.0}' not in ran and '{"reg_lambda": 10.0}' in ran


def test_autogen_tops_up_alongside_curated_pending(con, tmp_path, night, bench_spy, monkeypatch):
    """Curated pending work runs first; the generator fills only the spare
    budget — curated priority is preserved, the GPU is still saturated."""
    monkeypatch.setattr(research.generate, "grid_configs", lambda: list(FAKE_GRID))
    report = research.run(
        db_path=_db(con), budget=8, queue_path=_write_queue(tmp_path, [_config(max_depth=99)])
    )
    assert report.n_ran == 4 and report.n_generated == 3  # 1 curated + 3 generated
    assert '{"max_depth": 99}' in {c["params"] for c in bench_spy["calls"]}


def test_autogen_exhausted_grid_still_files_refill_issue(
    con, tmp_path, night, bench_spy, monkeypatch
):
    """When curated AND the grid are both dry, the refill signal fires as before
    — now meaning 'even auto-search is tapped out; add new kinds of work.'"""
    gh = _gh_spy(monkeypatch)
    monkeypatch.setattr(research.generate, "grid_configs", lambda: list(FAKE_GRID))
    for cfg in FAKE_GRID:  # record every grid config too
        _seed_recorded(con, cfg["strategy"], cfg["params"])
    report = research.run(db_path=_db(con), budget=8, queue_path=_write_queue(tmp_path, []))

    assert report.n_generated == 0 and report.queue_exhausted and report.exit_code == 1
    creates = [a for a, _ in gh if a[:3] == ["gh", "issue", "create"]]
    assert (
        len(creates) == 1 and creates[0][creates[0].index("--label") + 1] == "research-queue-empty"
    )


def test_refill_issue_dedup_does_not_spam(con, tmp_path, night, bench_spy, monkeypatch):
    """A refill issue already open → WARN 'already filed', no create, no crash."""
    gh = _gh_spy(monkeypatch, list_stdout='[{"number": 88, "title": "Research queue exhausted"}]')
    report = research.run(db_path=_db(con), queue_path=_write_queue(tmp_path, []))

    assert not any(a[:3] == ["gh", "issue", "create"] for a, _ in gh)  # deduped
    issue_step = next(s for s in report.steps if s.name == "issue")
    assert issue_step.status == "warn" and "#88" in issue_step.detail
    assert report.exit_code == 1  # still WARN: the queue is still exhausted


def test_refill_issue_gh_missing_is_loud(con, tmp_path, night, bench_spy, monkeypatch):
    """gh CLI missing → loud WARN, run still completes and still reports exit 1."""

    def gh_gone(args, **kw):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(issues.subprocess, "run", gh_gone)
    report = research.run(db_path=_db(con), queue_path=_write_queue(tmp_path, []))

    issue_step = next(s for s in report.steps if s.name == "issue")
    assert issue_step.status == "warn"
    assert "NO issue was filed" in issue_step.detail and "refill signal" in issue_step.detail
    assert report.queue_exhausted and report.exit_code == 1


def test_missing_queue_file_fails_run(con, tmp_path, night, bench_spy):
    report = research.run(db_path=_db(con), queue_path=tmp_path / "nope.json")
    assert report.exit_code == 2
    assert any(s.name == "queue" and s.status == "fail" for s in report.steps)
    assert not bench_spy["calls"]


def test_malformed_entries_skipped_loudly_not_crash(con, tmp_path, night, bench_spy, caplog):
    queue = _write_queue(
        tmp_path,
        [
            {"params": {"max_depth": 4}},  # no strategy
            "not even a dict",
            {"strategy": "xgb_gbm_v1", "params": "not a dict"},
            _config(max_depth=4),  # the one valid entry
        ],
    )
    with caplog.at_level("WARNING", logger="twopercent.research"):
        report = research.run(db_path=_db(con), queue_path=queue)

    assert len(bench_spy["calls"]) == 1  # valid entry still ran
    assert report.n_malformed == 3
    assert report.exit_code == 1  # a dropped config is never a clean night
    skipped = [r.message for r in caplog.records if "malformed" in r.message]
    assert len(skipped) == 3
    assert any("3 malformed" in s.detail for s in report.steps if s.name == "queue")


def test_already_recorded_config_skipped_via_ledger(con, tmp_path, night, bench_spy):
    """Queue state = the experiments ledger: a recorded STANDARD run satisfies
    its config; a non-standard (2-month) run of another config does not."""
    store.record_experiment(
        con,
        strategy="xgb_gbm_v1",
        params={"months": 12, "top_n": 20, "strategy_params": {"max_depth": 4}},
        train_start=dt.date(2021, 7, 1),
        test_start=dt.date(2025, 7, 1),
        test_end=dt.date(2026, 6, 30),
        metrics=METRICS,
    )
    store.record_experiment(  # diagnostic short run must NOT count as done
        con,
        strategy="xgb_gbm_v1",
        params={"months": 2, "top_n": 20, "strategy_params": {"max_depth": 6}},
        train_start=dt.date(2026, 4, 1),
        test_start=dt.date(2026, 5, 1),
        test_end=dt.date(2026, 6, 30),
        metrics=METRICS,
    )
    queue = _write_queue(tmp_path, [_config(max_depth=4), _config(max_depth=6)])
    report = research.run(db_path=_db(con), queue_path=queue)

    assert report.n_skipped_done == 1
    assert [c["params"] for c in bench_spy["calls"]] == ['{"max_depth": 6}']
    assert any("1 already recorded" in s.detail for s in report.steps if s.name == "queue")


def test_done_matching_canonicalizes_int_vs_float(con, tmp_path, night, bench_spy, monkeypatch):
    """A ledger row recorded with n_estimators 200.0 must satisfy the queue's
    200 — JSON round-trips can float-ify ints, and a re-run every night would
    silently burn the budget."""
    _gh_spy(monkeypatch)  # the sole config is already recorded → exhausted, files a refill issue
    store.record_experiment(
        con,
        strategy="xgb_gbm_v1",
        params={"months": 12, "top_n": 20, "strategy_params": {"n_estimators": 200.0}},
        train_start=dt.date(2021, 7, 1),
        test_start=dt.date(2025, 7, 1),
        test_end=dt.date(2026, 6, 30),
        metrics=METRICS,
    )
    queue = _write_queue(tmp_path, [_config(n_estimators=200)])
    report = research.run(db_path=_db(con), queue_path=queue)
    assert report.n_skipped_done == 1
    assert not bench_spy["calls"]


def test_rerun_is_idempotent(con, tmp_path, night, bench_spy, monkeypatch):
    """Night two of the same queue re-runs nothing — no state file needed. With
    the whole queue now recorded it is an exhausted night: WARN + refill issue."""
    gh = _gh_spy(monkeypatch)
    queue = _write_queue(tmp_path, [_config(max_depth=4), _config(max_depth=6)])
    first = research.run(db_path=_db(con), queue_path=queue)
    assert first.n_ran == 2
    assert not first.queue_exhausted  # night one had pending work
    second = research.run(db_path=_db(con), queue_path=queue)
    assert second.n_ran == 0
    assert second.n_skipped_done == 2
    assert second.queue_exhausted and second.exit_code == 1
    assert len(bench_spy["calls"]) == 2  # nothing re-ran
    assert any(a[:3] == ["gh", "issue", "create"] for a, _ in gh)  # refill signal filed


def test_budget_cap_respected(con, tmp_path, night, bench_spy):
    queue = _write_queue(tmp_path, [_config(max_depth=d) for d in range(2, 9)])
    report = research.run(db_path=_db(con), budget=3, queue_path=queue)
    assert len(bench_spy["calls"]) == 3
    assert report.n_ran == 3
    assert any("budget 3" in line for line in report.summary_lines())


# --- crash isolation ----------------------------------------------------------


def test_one_crash_does_not_kill_the_night(con, tmp_path, night, bench_spy):
    _seed_champion_experiment(con, lift=2.5)  # ties the survivor: no issue path
    bench_spy["boom"].add('{"max_depth": 4}')
    queue = _write_queue(tmp_path, [_config(max_depth=4), _config(max_depth=6)])
    report = research.run(db_path=_db(con), queue_path=queue)

    assert len(bench_spy["calls"]) == 2  # second config still ran
    assert report.n_failed == 1 and report.n_ran == 1
    assert report.exit_code == 1
    crashed = next(s for s in report.steps if s.name == "experiment" and s.status == "warn")
    assert "retries next night" in crashed.detail
    assert any("NOT recorded" in line for line in report.summary_lines())
    # The crashed config left no ledger row, so it is NOT considered done.
    assert ("xgb_gbm_v1", '{"max_depth": 4}') not in research.recorded_configs(con)


# --- champion identity --------------------------------------------------------


def test_champion_reference_ignores_parameterized_variants(con):
    """A queue variant of the champion strategy records under the champion's
    NAME — the reference lookup must skip it, even when it is newer and shinier
    (pre-research rows without the strategy_params key count as default)."""
    champ_id = store.record_experiment(  # pre-research row: no strategy_params key at all
        con,
        strategy="baseline_gbm_v1",
        params={"months": 12, "top_n": 20, "dropped_columns": []},
        train_start=dt.date(2021, 7, 1),
        test_start=dt.date(2025, 7, 1),
        test_end=dt.date(2026, 6, 30),
        metrics={**METRICS, "lift": 2.0},
    )
    store.record_experiment(  # newer variant with a flattering lift
        con,
        strategy="baseline_gbm_v1",
        params={"months": 12, "top_n": 20, "strategy_params": {"max_iter": 300}},
        train_start=dt.date(2021, 7, 1),
        test_start=dt.date(2025, 7, 1),
        test_end=dt.date(2026, 6, 30),
        metrics={**METRICS, "lift": 9.9},
    )
    result = research._latest_standard_experiment(con, "baseline_gbm_v1")
    assert result is not None
    exp_id, metrics, test_end = result
    assert exp_id == champ_id
    assert metrics["lift"] == 2.0  # never the variant's 9.9
    assert test_end == dt.date(2026, 6, 30)


def test_stale_champion_reference_warns(con, tmp_path, night, bench_spy):
    _seed_champion_experiment(con, lift=2.0, test_end=dt.date(2026, 3, 31))
    bench_spy["lifts"]['{"max_depth": 4}'] = 2.05  # no promotion path
    report = research.run(
        db_path=_db(con), queue_path=_write_queue(tmp_path, [_config(max_depth=4)])
    )
    stale = [s for s in report.steps if s.name == "champion" and "stale" in s.detail]
    assert len(stale) == 1 and stale[0].status == "warn"
    assert "2026-03-31" in stale[0].detail


# --- champion comparison and promotion issue ----------------------------------


def test_winner_beyond_promotion_band_files_promotion_issue(
    con, tmp_path, night, bench_spy, monkeypatch
):
    champ_id = _seed_champion_experiment(con, lift=2.0)
    gh = _gh_spy(monkeypatch)
    bench_spy["lifts"]['{"max_depth": 4}'] = 2.5  # margin 0.5 >= 0.25 band
    champion_before = Path("champion.json").read_bytes()
    prices_before = store.price_row_count(con)

    report = research.run(
        db_path=_db(con), queue_path=_write_queue(tmp_path, [_config(max_depth=4)])
    )

    assert report.exit_code == 0
    issue_step = next(s for s in report.steps if s.name == "issue")
    assert issue_step.status == "ok" and "issues/77" in issue_step.detail

    create, create_kw = next((a, kw) for a, kw in gh if a[:3] == ["gh", "issue", "create"])
    assert create[create.index("--label") + 1] == "promotion-candidate"
    assert create[create.index("--body-file") + 1] == "-"  # body via stdin
    title = create[create.index("--title") + 1]
    assert "xgb_gbm_v1" in title and "2.5 vs 2.0" in title
    body = create_kw["input"]
    assert '{"max_depth": 4}' in body  # the winning config
    assert f"#{champ_id}" in body  # champion's ledger id
    assert f"#{champ_id + 1}" in body  # challenger's ledger id (recorded after)
    assert "| lift | 2.5 | 2.0 |" in body  # full metrics vs champion's
    assert "promotion band (0.25)" in body
    assert "Disjoint-halves confirmation" in body and "+0.3000" in body  # 0.8 vs 0.5
    assert "AFTER this candidate's test_end" in body  # wall-clock holdout demand
    assert "#45" in body
    assert "quant-skeptic" in body and "champion.json" in body  # promotion rules
    assert "NEVER on sim growth" in body
    assert "hypothesis" in body and "Multiple-comparisons" in body
    label = next(a for a, _ in gh if a[:3] == ["gh", "label", "create"])
    assert "--force" in label and "promotion-candidate" in label
    lock = next(a for a, _ in gh if a[:3] == ["gh", "issue", "lock"])
    assert lock[3] == "77" and "--reason" in lock

    # The runner wrote NOTHING beyond ledger rows and the issue.
    assert Path("champion.json").read_bytes() == champion_before
    assert con.execute("SELECT count(*) FROM predictions").fetchone()[0] == 0
    assert store.price_row_count(con) == prices_before


def test_within_promotion_band_files_nothing(con, tmp_path, night, bench_spy, monkeypatch):
    """2.2 vs 2.0 beats compare's 0.1 noise band but NOT the 0.25 promotion
    band — a sweep's best-of-many needs the family-wise bar, not the
    single-comparison one."""
    _seed_champion_experiment(con, lift=2.0)
    gh = _gh_spy(monkeypatch)
    bench_spy["lifts"]['{"max_depth": 4}'] = 2.2
    report = research.run(
        db_path=_db(con), queue_path=_write_queue(tmp_path, [_config(max_depth=4)])
    )

    assert not gh  # no gh subprocess call of any kind
    assert not any(s.name == "issue" for s in report.steps)
    assert report.exit_code == 0
    candidate = next(s for s in report.steps if s.name == "candidate")
    assert candidate.status == "ok" and "within the promotion band (0.25)" in candidate.detail


def test_half_window_loser_files_nothing(con, tmp_path, night, bench_spy, monkeypatch):
    """Overall winner carried by one hot half: margin +0.5 then -0.2 — the
    disjoint-halves confirmation must reject it and say so."""
    _seed_champion_experiment(con, lift=2.0)  # daily precision 0.5 every day
    gh = _gh_spy(monkeypatch)
    bench_spy["lifts"]['{"max_depth": 4}'] = 2.5  # beyond the promotion band overall
    bench_spy["daily_prec"]['{"max_depth": 4}'] = [1.0] * 5 + [0.3] * 5
    report = research.run(
        db_path=_db(con), queue_path=_write_queue(tmp_path, [_config(max_depth=4)])
    )

    assert not gh  # no issue of any kind was attempted
    candidate = next(s for s in report.steps if s.name == "candidate")
    assert candidate.status == "warn"
    assert "FAILED the disjoint-halves confirmation" in candidate.detail
    assert "one-hot-window" in candidate.detail
    assert report.exit_code == 0


def test_cpu_fallback_recording_warns_in_digest(con, tmp_path, night, bench_spy):
    bench_spy["device"] = "cpu"  # referee recorded a CPU-fallback training
    report = research.run(
        db_path=_db(con), queue_path=_write_queue(tmp_path, [_config(max_depth=4)])
    )
    device = next(s for s in report.steps if s.name == "device")
    assert device.status == "warn"
    assert "CPU FALLBACK" in device.detail and "delete their ledger rows" in device.detail


def test_winner_dedups_existing_open_issue(con, tmp_path, night, bench_spy, monkeypatch):
    _seed_champion_experiment(con, lift=2.0)
    gh = _gh_spy(monkeypatch, list_stdout='[{"number": 55, "title": "Research: earlier"}]')
    bench_spy["lifts"]['{"max_depth": 4}'] = 2.5
    report = research.run(
        db_path=_db(con), queue_path=_write_queue(tmp_path, [_config(max_depth=4)])
    )

    assert not any(a[:3] == ["gh", "issue", "create"] for a, _ in gh)
    issue_step = next(s for s in report.steps if s.name == "issue")
    assert issue_step.status == "warn" and "#55" in issue_step.detail


def test_no_champion_benchmark_disables_promotion(con, tmp_path, night, bench_spy, monkeypatch):
    # No champion experiment seeded: even a huge lift must not file an issue —
    # there is nothing sound to compare against.
    gh = _gh_spy(monkeypatch)
    bench_spy["lifts"]['{"max_depth": 4}'] = 99.0
    report = research.run(
        db_path=_db(con), queue_path=_write_queue(tmp_path, [_config(max_depth=4)])
    )

    assert not gh
    champ_step = next(s for s in report.steps if s.name == "champion")
    assert champ_step.status == "warn" and "OFF" in champ_step.detail


def test_digest_reports_night_summary(con, tmp_path, night, bench_spy):
    _seed_champion_experiment(con, lift=2.35)  # best of night lands within the band: no issue
    bench_spy["lifts"]['{"max_depth": 4}'] = 2.1
    bench_spy["lifts"]['{"max_depth": 6}'] = 2.4
    queue = _write_queue(tmp_path, [_config(max_depth=4), _config(max_depth=6)])
    report = research.run(db_path=_db(con), budget=8, queue_path=queue)

    lines = report.summary_lines()
    assert any(
        "night: 2 run (0 auto-generated), 0 skipped (already recorded), 0 failed" in line
        for line in lines
    )
    best = next(line for line in lines if line.strip().startswith("best:"))
    assert "xgb_gbm_v1" in best and "2.4" in best  # highest lift of the night
    assert "vs champion lift 2.35" in best
