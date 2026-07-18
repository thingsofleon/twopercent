import datetime as dt

import pandas as pd
import pytest

from tests.conftest import seed_planted
from twopercent import ingest, routine, store


@pytest.fixture
def ready(con, monkeypatch):
    """A healthy seeded store with network steps stubbed out."""
    seed_planted(con, n_each=10)
    # Universe is fresh (seed_planted stamps 2026-06-01; make it today).
    con.execute("UPDATE universe SET as_of = ?", [dt.date.today()])
    # Prices end 2026-05-22 (100 bdays from Jan 5) — silence staleness by
    # aligning the doctor's clock to the store's own max date is not possible,
    # so stub ingest to a no-op success and accept the stale WARN path where
    # noted per-test.
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
    report = routine.run(db_path=ready.execute("PRAGMA database_list").fetchone()[2])
    names = [s.name for s in report.steps]
    assert names == ["doctor", "universe", "ingest", "recheck", "predict", "dashboard"]
    assert report.status in ("ok", "warn")  # seeded store has no hard problems
    assert report.top_candidates  # predictions produced
    assert all(sym.startswith("RUN") for sym, _ in report.top_candidates)


def test_routine_aborts_before_model_on_invalid_bars(ready):
    ready.execute("UPDATE prices SET open = 0.0 WHERE symbol = 'RUN00'")
    db = ready.execute("PRAGMA database_list").fetchone()[2]
    report = routine.run(db_path=db)

    assert report.status == "fail"
    assert [s.name for s in report.steps] == ["doctor"]  # gate stopped everything
    n_predictions = ready.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert n_predictions == 0  # the model never ran on bad data
    assert report.exit_code == 2


def test_routine_ingest_failure_rate_aborts(ready, monkeypatch):
    def failing_ingest(con_, symbols, **kw):
        return ingest.IngestResult(symbols_failed=list(symbols))

    monkeypatch.setattr(routine.ingest, "ingest", failing_ingest)
    db = ready.execute("PRAGMA database_list").fetchone()[2]
    report = routine.run(db_path=db)
    assert report.status == "fail"
    assert report.steps[-1].name == "ingest"


def test_routine_universe_refresh_failure_degrades_not_aborts(ready, monkeypatch):
    ready.execute("UPDATE universe SET as_of = ?", [dt.date.today() - dt.timedelta(days=30)])

    def broken_refresh(**kw):
        raise RuntimeError("screener down")

    monkeypatch.setattr(routine.universe, "refresh_universe", broken_refresh)
    db = ready.execute("PRAGMA database_list").fetchone()[2]
    report = routine.run(db_path=db)

    assert report.status in ("warn", "ok") or report.status == "warn"
    uni_step = next(s for s in report.steps if s.name == "universe")
    assert uni_step.status == "warn" and "screener down" in uni_step.detail
    assert any(s.name == "predict" for s in report.steps)  # run continued


def test_routine_stale_universe_triggers_refresh(ready, monkeypatch):
    ready.execute("UPDATE universe SET as_of = ?", [dt.date.today() - dt.timedelta(days=30)])
    called = {}

    def fake_refresh(**kw):
        called["yes"] = True
        return pd.DataFrame(
            {
                "symbol": ["RUN00"],
                "name": ["RUN00"],
                "market_cap": [1e9],
                "sector": ["Tech"],
            }
        )

    monkeypatch.setattr(routine.universe, "refresh_universe", fake_refresh)
    db = ready.execute("PRAGMA database_list").fetchone()[2]
    routine.run(db_path=db)
    assert called.get("yes")
    latest = store.latest_universe(ready)
    assert latest["as_of"].iloc[0].date() == dt.date.today()


def test_summary_lines_shape(ready):
    db = ready.execute("PRAGMA database_list").fetchone()[2]
    report = routine.run(db_path=db)
    lines = report.summary_lines()
    assert lines[0].startswith("routine:")
    assert any("top candidates" in line for line in lines)
