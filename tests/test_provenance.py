"""Git provenance: the code step's source of truth (#114)."""

from __future__ import annotations

import subprocess

import pytest

from tests import conftest
from twopercent import provenance


@pytest.fixture(autouse=True)
def real_git(monkeypatch):
    """This module tests the REAL git_state, not conftest's clean-main fake."""
    monkeypatch.setattr(provenance, "git_state", conftest.REAL_GIT_STATE)


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    """A real throwaway git repo as the process cwd; None if git is absent."""
    monkeypatch.chdir(tmp_path)
    try:
        for cmd in (
            ["git", "init", "-q", "-b", "main"],
            ["git", "config", "user.email", "t@t"],
            ["git", "config", "user.name", "t"],
        ):
            subprocess.run(cmd, check=True, capture_output=True)
        (tmp_path / "f.txt").write_text("x")
        subprocess.run(["git", "add", "f.txt"], check=True, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", "init"], check=True, capture_output=True)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git unavailable")
    return tmp_path


def test_clean_main_is_production(git_repo):
    state = provenance.git_state()
    assert state.branch == "main"
    assert state.dirty is False
    assert len(state.commit) == 40
    assert state.is_clean_production
    assert "main @" in state.describe() and "clean" in state.describe()


def test_dirty_tree_is_not_production(git_repo):
    (git_repo / "f.txt").write_text("changed")
    state = provenance.git_state()
    assert state.dirty is True
    assert not state.is_clean_production
    assert "dirty tree" in state.describe()


def test_feature_branch_is_not_production(git_repo):
    subprocess.run(["git", "checkout", "-q", "-b", "feat/x"], check=True, capture_output=True)
    state = provenance.git_state()
    assert state.branch == "feat/x"
    assert not state.is_clean_production


def test_outside_a_repo_is_unknowable_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = provenance.git_state()
    assert state.commit is None and state.branch is None and state.dirty is None
    # "Cannot verify" must never render as "verified".
    assert not state.is_clean_production
    assert "unknowable" in state.describe()


def test_git_binary_missing_degrades_to_unknowable(monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda *a: None)
    state = provenance.git_state()
    assert state == provenance.GitState(None, None, None)


def _states():
    G = provenance.GitState
    return {
        "clean_main": G("a" * 40, "main", False),
        "feature": G("b" * 40, "feat/x", False),
        "dirty_main": G("c" * 40, "main", True),
        "unknown": G(None, None, None),
    }


@pytest.mark.parametrize(
    ("state_key", "expect"),
    [
        ("clean_main", "ok"),
        ("feature", "warn"),
        ("dirty_main", "warn"),
        ("unknown", "warn"),
    ],
)
def test_code_step_warns_on_everything_but_clean_main(monkeypatch, state_key, expect):
    from twopercent import routine

    monkeypatch.setattr(provenance, "git_state", lambda: _states()[state_key])
    report = routine.RoutineReport()
    routine._code_step(report)
    (step,) = report.steps
    assert step.name == "code"
    assert step.status == expect
    if expect == "warn":
        assert "#114" in step.detail


def test_benchmark_records_the_code_that_produced_the_row(con, monkeypatch):
    """#114's cheapest fix: a ledger row is always traceable to its commit, so
    'these 56 came from an unmerged branch' becomes one query, not an autopsy."""
    import json

    from tests.conftest import seed_planted
    from twopercent import backtest, store

    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    monkeypatch.setattr(
        provenance, "git_state", lambda: provenance.GitState("d" * 40, "feat/live", True)
    )
    seed_planted(con)
    backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5)
    params = json.loads(store.list_experiments(con)["params"].iloc[0])
    assert params["code"] == {"commit": "d" * 40, "branch": "feat/live", "dirty": True}
