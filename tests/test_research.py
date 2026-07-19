"""Research runner: clock gate, queue hygiene, ledger-based idempotency,
budget cap, crash isolation, promotion-issue filing, and the write-nothing
guarantee. Offline: benchmark and gh are monkeypatched; clocks are pinned."""

import datetime as dt
import json
import subprocess
from pathlib import Path

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


def _db(con) -> str:
    return con.execute("PRAGMA database_list").fetchone()[2]


def _write_queue(tmp_path, configs) -> Path:
    path = tmp_path / "queue.json"
    path.write_text(json.dumps(configs))
    return path


def _config(strategy="xgb_gbm_v1", **params):
    return {"strategy": strategy, "params": params, "note": "test config"}


def _seed_champion_experiment(con, lift=2.0) -> int:
    """The champion's recorded standard 12-month benchmark; returns its id."""
    return store.record_experiment(
        con,
        strategy=champion.get_champion(),
        params={"months": 12, "top_n": 20, "dropped_columns": [], "strategy_params": {}},
        train_start=dt.date(2021, 7, 1),
        test_start=dt.date(2025, 7, 1),
        test_end=dt.date(2026, 6, 30),
        metrics={**METRICS, "lift": lift, "auc": 0.69},
    )


@pytest.fixture(autouse=True)
def no_real_gh(monkeypatch):
    """No research test may ever reach the real gh CLI: any subprocess call not
    intercepted by an explicit _gh_spy is a test bug and must blow up."""

    def forbidden(args, **kw):
        raise AssertionError(f"test reached real subprocess: {args}")

    monkeypatch.setattr(issues.subprocess, "run", forbidden)


@pytest.fixture
def night(monkeypatch):
    monkeypatch.setattr(research, "_now_denver", lambda: IN_WINDOW)


@pytest.fixture
def bench_spy(monkeypatch):
    """Fake referee: records a ledger row like the real one, returns METRICS
    (override per-config via lifts dict; raise via boom set)."""
    state = {"calls": [], "lifts": {}, "boom": set()}

    def fake(con, name, months=12, top_n=20, record=True, strategy_params=None):
        key = json.dumps(strategy_params or {}, sort_keys=True)
        state["calls"].append({"strategy": name, "months": months, "top_n": top_n, "params": key})
        if key in state["boom"]:
            raise RuntimeError("synthetic benchmark crash")
        metrics = {**METRICS, "lift": state["lifts"].get(key, METRICS["lift"])}
        if record:
            store.record_experiment(
                con,
                strategy=name,
                params={"months": months, "top_n": top_n, "strategy_params": strategy_params or {}},
                train_start=dt.date(2021, 7, 1),
                test_start=dt.date(2025, 7, 1),
                test_end=dt.date(2026, 6, 30),
                metrics=metrics,
            )
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
def test_clock_gate_matrix(con, tmp_path, monkeypatch, when, allowed):
    monkeypatch.setattr(research, "_now_denver", lambda: when.replace(tzinfo=research._DENVER))
    queue = _write_queue(tmp_path, [])
    report = research.run(db_path=_db(con), queue_path=queue)
    if allowed:
        assert report.steps[0].name == "clock" and report.steps[0].status == "ok"
        assert report.exit_code == 0
    else:
        assert [s.name for s in report.steps] == ["clock"]
        assert report.exit_code == 2
        assert "research window" in report.steps[0].detail


# --- queue hygiene ------------------------------------------------------------


def test_empty_queue_is_loud_noop_exit_0(con, tmp_path, night, bench_spy):
    report = research.run(db_path=_db(con), queue_path=_write_queue(tmp_path, []))
    assert report.exit_code == 0
    queue_step = next(s for s in report.steps if s.name == "queue")
    assert "empty" in queue_step.detail
    assert not bench_spy["calls"]


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


def test_rerun_is_idempotent(con, tmp_path, night, bench_spy):
    """Night two of the same queue re-runs nothing — no state file needed."""
    queue = _write_queue(tmp_path, [_config(max_depth=4), _config(max_depth=6)])
    first = research.run(db_path=_db(con), queue_path=queue)
    assert first.n_ran == 2
    second = research.run(db_path=_db(con), queue_path=queue)
    assert second.n_ran == 0
    assert second.n_skipped_done == 2
    assert second.exit_code == 0
    assert len(bench_spy["calls"]) == 2  # nothing re-ran


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


# --- champion comparison and promotion issue ----------------------------------


def test_winner_outside_noise_band_files_promotion_issue(
    con, tmp_path, night, bench_spy, monkeypatch
):
    champ_id = _seed_champion_experiment(con, lift=2.0)
    gh = _gh_spy(monkeypatch)
    bench_spy["lifts"]['{"max_depth": 4}'] = 2.5
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


def test_within_noise_band_files_nothing(con, tmp_path, night, bench_spy, monkeypatch):
    _seed_champion_experiment(con, lift=2.0)
    gh = _gh_spy(monkeypatch)
    bench_spy["lifts"]['{"max_depth": 4}'] = 2.05  # inside the 0.1 band
    report = research.run(
        db_path=_db(con), queue_path=_write_queue(tmp_path, [_config(max_depth=4)])
    )

    assert not gh  # no gh subprocess call of any kind
    assert not any(s.name == "issue" for s in report.steps)
    assert report.exit_code == 0
    assert any(
        "challenger" not in s.detail or "within noise" in s.detail
        for s in report.steps
        if s.name == "experiment"
    )


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
    _seed_champion_experiment(con, lift=2.35)  # best of night lands within noise: no issue
    bench_spy["lifts"]['{"max_depth": 4}'] = 2.1
    bench_spy["lifts"]['{"max_depth": 6}'] = 2.4
    queue = _write_queue(tmp_path, [_config(max_depth=4), _config(max_depth=6)])
    report = research.run(db_path=_db(con), budget=8, queue_path=queue)

    lines = report.summary_lines()
    assert any("night: 2 run, 0 skipped (already recorded), 0 failed" in line for line in lines)
    best = next(line for line in lines if line.strip().startswith("best:"))
    assert "xgb_gbm_v1" in best and "2.4" in best  # highest lift of the night
    assert "vs champion lift 2.35" in best
