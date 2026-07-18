import datetime as dt

import pandas as pd
import pytest

from tests.conftest import seed_history, seed_planted
from twopercent import ingest, routine, store

CLOSED = dt.datetime(2026, 7, 17, 7, 0, tzinfo=routine._EASTERN)  # Friday pre-open


def _db(con) -> str:
    return con.execute("PRAGMA database_list").fetchone()[2]


@pytest.fixture
def ready(con, monkeypatch):
    """A healthy seeded store, clock pre-open, network steps stubbed out."""
    seed_planted(con, n_each=10)
    con.execute("UPDATE universe SET as_of = ?", [dt.date.today()])
    # Keep the store's newest bar inside the freshness window.
    span = (dt.date.today() - pd.Timestamp("2026-01-05").date()).days
    con.execute(f"UPDATE prices SET date = date + INTERVAL {span - 141} DAY")
    monkeypatch.setattr(routine, "_now_eastern", lambda: CLOSED)
    monkeypatch.setattr(
        routine.ingest,
        "ingest",
        lambda con_, symbols, **kw: ingest.IngestResult(symbols_skipped=list(symbols)),
    )
    monkeypatch.setattr(
        routine.universe, "refresh_universe", lambda **kw: pytest.fail("network touched")
    )
    return con


def test_routine_happy_path_completes_all_steps(ready):
    report = routine.run(db_path=_db(ready))
    names = [s.name for s in report.steps]
    assert names == [
        "clock",
        "doctor",
        "universe",
        "ingest",
        "freshness",
        "recheck",
        "predict",
        "dashboard",
        "scoring",
    ]
    assert report.status in ("ok", "warn")
    assert report.top_candidates
    assert all(sym.startswith("RUN") for sym, _ in report.top_candidates)


def test_routine_refuses_to_run_while_market_open(ready, monkeypatch):
    midday = dt.datetime(2026, 7, 17, 11, 0, tzinfo=routine._EASTERN)  # Friday 11:00 ET
    monkeypatch.setattr(routine, "_now_eastern", lambda: midday)
    report = routine.run(db_path=_db(ready))
    assert report.status == "fail"
    assert [s.name for s in report.steps] == ["clock"]
    assert "partial" in report.steps[0].detail
    n = ready.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert n == 0  # nothing ran


def test_market_hours_boundaries():
    friday = dt.date(2026, 7, 17)
    saturday = dt.date(2026, 7, 18)

    def mk(d, h, m):
        return dt.datetime(d.year, d.month, d.day, h, m, tzinfo=routine._EASTERN)

    assert routine._market_is_open(mk(friday, 9, 25))
    assert routine._market_is_open(mk(friday, 16, 14))
    assert not routine._market_is_open(mk(friday, 9, 24))
    assert not routine._market_is_open(mk(friday, 16, 15))
    assert not routine._market_is_open(mk(saturday, 11, 0))


def test_routine_aborts_on_ingest_introduced_corruption(ready, monkeypatch):
    def corrupting_ingest(con_, symbols, **kw):
        # Ingest "succeeds" but writes a fresh split-artifact-scale bar.
        seed_history(con_, {"EVIL": [0.0] * 3 + [0.9]}, start="2026-07-06")
        return ingest.IngestResult(symbols_skipped=list(symbols))

    monkeypatch.setattr(routine.ingest, "ingest", corrupting_ingest)
    report = routine.run(db_path=_db(ready))
    assert report.status == "fail"
    assert report.steps[-1].name == "recheck"
    assert "EVIL" in report.steps[-1].detail
    n = ready.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert n == 0  # the model never trained on the corrupted store


def test_routine_preexisting_problems_warn_but_run(ready):
    # A pre-existing extreme bar (present in the pre-ingest baseline) must not
    # abort the run — it warns via the doctor step and the model proceeds.
    seed_history(ready, {"OLDX": [0.0] * 3 + [0.9]}, start="2026-06-01")
    report = routine.run(db_path=_db(ready))
    assert report.status == "warn"
    assert any(s.name == "predict" for s in report.steps)  # ran to completion
    doctor_step = next(s for s in report.steps if s.name == "doctor")
    assert doctor_step.status == "warn"


def test_routine_staleness_gate(ready):
    # Age the store by dropping everything newer than 60 days ago.
    ready.execute("DELETE FROM prices WHERE date > current_date - INTERVAL 60 DAY")
    report = routine.run(db_path=_db(ready))
    assert report.status == "fail"
    assert report.steps[-1].name == "freshness"
    assert "track record" in report.steps[-1].detail


def test_routine_ingest_failure_rate_boundary(ready, monkeypatch):
    # 20 symbols in the seeded universe: 1/20 = exactly 5% must pass (strict >),
    # 2/20 must abort.
    def fail_n(n):
        def fake(con_, symbols, **kw):
            symbols = list(symbols)
            return ingest.IngestResult(symbols_failed=symbols[:n], symbols_skipped=symbols[n:])

        return fake

    monkeypatch.setattr(routine.ingest, "ingest", fail_n(1))
    assert routine.run(db_path=_db(ready)).status in ("ok", "warn")

    monkeypatch.setattr(routine.ingest, "ingest", fail_n(2))
    report = routine.run(db_path=_db(ready))
    assert report.status == "fail"
    assert report.steps[-1].name == "ingest"


def test_routine_unaccounted_symbols_fail(ready, monkeypatch):
    def lossy_ingest(con_, symbols, **kw):
        return ingest.IngestResult(symbols_skipped=list(symbols)[:-3])  # drops 3

    monkeypatch.setattr(routine.ingest, "ingest", lossy_ingest)
    report = routine.run(db_path=_db(ready))
    assert report.status == "fail"
    assert "unaccounted" in report.steps[-1].detail


def test_routine_universe_refresh_failure_degrades_not_aborts(ready, monkeypatch):
    ready.execute("UPDATE universe SET as_of = ?", [dt.date.today() - dt.timedelta(days=30)])

    def broken_refresh(**kw):
        raise RuntimeError("screener down")

    monkeypatch.setattr(routine.universe, "refresh_universe", broken_refresh)
    report = routine.run(db_path=_db(ready))
    uni_step = next(s for s in report.steps if s.name == "universe")
    assert uni_step.status == "warn"
    assert "screener down" in uni_step.detail
    assert any(s.name == "predict" for s in report.steps)  # run continued
    assert report.exit_code == 1  # degraded, not failed


def test_routine_stale_universe_triggers_refresh(ready, monkeypatch):
    ready.execute("UPDATE universe SET as_of = ?", [dt.date.today() - dt.timedelta(days=30)])
    called = {}

    def fake_refresh(**kw):
        called["yes"] = True
        return pd.DataFrame(
            {"symbol": ["RUN00"], "name": ["RUN00"], "market_cap": [1e9], "sector": ["Tech"]}
        )

    monkeypatch.setattr(routine.universe, "refresh_universe", fake_refresh)
    routine.run(db_path=_db(ready))
    assert called.get("yes")
    assert store.latest_universe(ready)["as_of"].iloc[0].date() == dt.date.today()


def test_predictions_record_universe_snapshot(ready):
    routine.run(db_path=_db(ready))
    as_of = ready.execute("SELECT DISTINCT universe_as_of FROM predictions").fetchall()
    assert as_of == [(dt.date.today(),)]


def test_summary_lines_shape(ready):
    report = routine.run(db_path=_db(ready))
    lines = report.summary_lines()
    assert lines[0].startswith("routine:")
    assert any("top candidates" in line for line in lines)
