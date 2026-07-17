import datetime as dt

import pandas as pd
import pytest
from typer.testing import CliRunner

from tests.conftest import seed_history
from twopercent import doctor, store
from twopercent.cli import app

START = dt.date(2026, 1, 5)  # a Monday; 15 business days end 2026-01-23
LAST = dt.date(2026, 1, 23)

runner = CliRunner()


@pytest.fixture
def defective(con):
    """A store seeded with exactly one instance of each defect class.

    - GAPPY: one bar (2026-01-12) deleted mid-history → gap vs the calendar
    - STALE: only 3 bars, last on 2026-01-07 → 16 days before store max
    - WILD:  +60% bar (2026-01-12) and −52% bar (2026-01-15); no ingest_meta row
    - FLAT:  zero volume on 3 consecutive bars (2026-01-07..09)
    - SHORT: zero-volume runs of 2 and 1 bars — below the run threshold
    - GHOST: in the latest universe but has no price rows
    """
    wild = [0.01] * 15
    wild[5] = 0.60
    wild[8] = -0.52
    seed_history(
        con,
        {
            "CLEAN": [0.01] * 15,
            "GAPPY": [0.01] * 15,
            "STALE": [0.01] * 3,
            "WILD": wild,
            "FLAT": [0.01] * 15,
            "SHORT": [0.01] * 15,
        },
    )
    con.execute("DELETE FROM prices WHERE symbol = 'GAPPY' AND date = DATE '2026-01-12'")
    con.execute(
        """
        UPDATE prices SET volume = 0 WHERE symbol = 'FLAT'
        AND date BETWEEN DATE '2026-01-07' AND DATE '2026-01-09'
        """
    )
    con.execute(
        """
        UPDATE prices SET volume = 0 WHERE symbol = 'SHORT'
        AND date IN (DATE '2026-01-05', DATE '2026-01-06', DATE '2026-01-08')
        """
    )
    symbols = ["CLEAN", "GAPPY", "STALE", "WILD", "FLAT", "SHORT", "GHOST"]
    store.upsert_universe(
        con,
        pd.DataFrame(
            {
                "symbol": symbols,
                "name": [f"{s} Corp" for s in symbols],
                "market_cap": [1e9] * len(symbols),
            }
        ),
        as_of=LAST,
    )
    store.record_ingest_from(con, ["CLEAN", "GAPPY", "STALE", "FLAT", "SHORT"], START)
    return con


@pytest.fixture
def clean(con):
    seed_history(con, {"AAA": [0.01] * 10, "BBB": [0.02] * 10})
    store.upsert_universe(
        con,
        pd.DataFrame(
            {"symbol": ["AAA", "BBB"], "name": ["A Corp", "B Corp"], "market_cap": [2e9, 1e9]}
        ),
        as_of=dt.date(2026, 1, 16),
    )
    store.record_ingest_from(con, ["AAA", "BBB"], START)
    return con


def test_gaps_catches_exactly_the_gap(defective):
    gaps = doctor.gap_counts(defective)
    assert gaps["symbol"].tolist() == ["GAPPY"]
    assert gaps["missing"].tolist() == [1]
    assert gaps["first_missing"].iloc[0].date() == dt.date(2026, 1, 12)
    assert gaps["last_missing"].iloc[0].date() == dt.date(2026, 1, 12)


def test_stale_catches_exactly_the_stale_symbol(defective):
    stale = doctor.stale_symbols(defective)
    assert stale["symbol"].tolist() == ["STALE"]
    assert stale["last_date"].iloc[0].date() == dt.date(2026, 1, 7)
    assert stale["age_days"].iloc[0] == 16


def test_stale_respects_custom_threshold(defective):
    assert doctor.stale_symbols(defective, stale_days=16).empty  # strictly older than N
    assert doctor.stale_symbols(defective, stale_days=15)["symbol"].tolist() == ["STALE"]


def test_extreme_bars_catches_both_directions(defective):
    ext = doctor.extreme_bars(defective)
    found = {(row.symbol, row.date.date()) for row in ext.itertuples()}
    assert found == {("WILD", dt.date(2026, 1, 12)), ("WILD", dt.date(2026, 1, 15))}
    assert ext["oc_return"].iloc[0] == pytest.approx(0.60)  # worst first


def test_zero_volume_run_caught_short_runs_ignored(defective):
    runs = doctor.zero_volume_runs(defective)
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["symbol"] == "FLAT"
    assert row["run_length"] == 3
    assert row["run_start"].date() == dt.date(2026, 1, 7)
    assert row["run_end"].date() == dt.date(2026, 1, 9)


def test_coverage_universe_symbol_without_prices(defective):
    assert doctor.universe_symbols_without_prices(defective) == ["GHOST"]


def test_coverage_price_symbol_without_meta(defective):
    assert doctor.price_symbols_without_meta(defective) == ["WILD"]


def test_run_collects_every_problem(defective):
    report = doctor.run(defective)
    assert not report.ok
    # 1 gap symbol + 1 stale + 2 extreme bars + 1 zero run + GHOST + WILD meta
    assert report.problem_count == 7
    text = "\n".join(doctor.format_report(report))
    for symbol in ["GAPPY", "STALE", "WILD", "FLAT", "GHOST"]:
        assert symbol in text
    assert "[FAIL]" in text
    assert "CLEAN" not in text
    assert "SHORT" not in text


def test_clean_store_passes_every_check(clean):
    report = doctor.run(clean)
    assert report.ok
    assert report.problem_count == 0
    assert doctor.gap_counts(clean).empty
    assert doctor.stale_symbols(clean).empty
    assert doctor.extreme_bars(clean).empty
    assert doctor.zero_volume_runs(clean).empty
    assert doctor.universe_symbols_without_prices(clean) == []
    assert doctor.price_symbols_without_meta(clean) == []
    text = "\n".join(doctor.format_report(report))
    assert "[FAIL]" not in text
    assert text.count("[ OK ]") == 4


def test_missing_universe_warns_but_does_not_fail(con):
    seed_history(con, {"AAA": [0.01] * 5})
    store.record_ingest_from(con, ["AAA"], START)
    report = doctor.run(con)
    assert report.ok
    text = "\n".join(doctor.format_report(report))
    assert "no universe stored" in text


def test_cli_exits_1_and_reports_on_defects(defective, tmp_path):
    defective.close()
    result = runner.invoke(app, ["doctor", "--db", str(tmp_path / "test.duckdb")])
    assert result.exit_code == 1
    for symbol in ["GAPPY", "STALE", "WILD", "FLAT", "GHOST"]:
        assert symbol in result.output
    assert "problems found" in result.output


def test_cli_exits_0_on_clean_store(clean, tmp_path):
    clean.close()
    result = runner.invoke(app, ["doctor", "--db", str(tmp_path / "test.duckdb")])
    assert result.exit_code == 0
    assert "all checks passed" in result.output


def test_cli_exits_1_on_empty_store(tmp_path):
    result = runner.invoke(app, ["doctor", "--db", str(tmp_path / "empty.duckdb")])
    assert result.exit_code == 1
    assert "no price data" in result.output
