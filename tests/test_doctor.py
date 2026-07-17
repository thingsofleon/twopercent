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
    - STALE: only 3 bars, last on 2026-01-07 → 12 trading days behind store max
    - TAIL:  missing its last 3 trading days (last bar 2026-01-20)
    - WILD:  +60% bar (2026-01-12) and −52% bar (2026-01-15); no ingest_meta row
    - FLAT:  zero/NULL volume on 3 consecutive bars (2026-01-07..09)
    - SHORT: zero-volume runs of 2 and 1 bars — below the run threshold
    - BADO:  open=0 on its last 3 bars and NaN close on 2026-01-20 — rows the
      daily_returns view excludes, so gap/extreme/stale checks cannot see them
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
            "TAIL": [0.01] * 12,
            "WILD": wild,
            "FLAT": [0.01] * 15,
            "SHORT": [0.01] * 15,
            "BADO": [0.01] * 15,
        },
    )
    con.execute("DELETE FROM prices WHERE symbol = 'GAPPY' AND date = DATE '2026-01-12'")
    con.execute(
        """
        UPDATE prices SET volume = 0 WHERE symbol = 'FLAT'
        AND date IN (DATE '2026-01-07', DATE '2026-01-08')
        """
    )
    con.execute(
        "UPDATE prices SET volume = NULL WHERE symbol = 'FLAT' AND date = DATE '2026-01-09'"
    )
    con.execute(
        """
        UPDATE prices SET volume = 0 WHERE symbol = 'SHORT'
        AND date IN (DATE '2026-01-05', DATE '2026-01-06', DATE '2026-01-08')
        """
    )
    con.execute("UPDATE prices SET open = 0 WHERE symbol = 'BADO' AND date >= DATE '2026-01-21'")
    con.execute(
        """
        UPDATE prices SET close = CAST('nan' AS DOUBLE)
        WHERE symbol = 'BADO' AND date = DATE '2026-01-20'
        """
    )
    symbols = ["CLEAN", "GAPPY", "STALE", "TAIL", "WILD", "FLAT", "SHORT", "BADO", "GHOST"]
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
    store.record_ingest_from(
        con, ["CLEAN", "GAPPY", "STALE", "TAIL", "FLAT", "SHORT", "BADO"], START
    )
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


def test_stale_measured_in_trading_days_behind_store_max(defective):
    stale = doctor.stale_symbols(defective)
    assert stale["symbol"].tolist() == ["STALE", "TAIL"]  # worst first
    assert stale["last_date"].iloc[0].date() == dt.date(2026, 1, 7)
    assert stale["trading_days_behind"].tolist() == [12, 3]


def test_stale_respects_custom_threshold(defective):
    assert doctor.stale_symbols(defective, stale_days=12).empty  # strictly more than N
    assert doctor.stale_symbols(defective, stale_days=11)["symbol"].tolist() == ["STALE"]
    assert doctor.stale_symbols(defective, stale_days=3).empty is False


def test_tail_missing_trading_days_flagged_despite_interior_only_gaps(con):
    # A symbol missing its last 3 trading days: the gap check is interior-only
    # and calendar-day staleness (< a week) would pass this silently.
    seed_history(con, {"FULL": [0.01] * 10, "TAILGAP": [0.01] * 7})
    stale = doctor.stale_symbols(con)
    assert stale["symbol"].tolist() == ["TAILGAP"]
    assert stale["trading_days_behind"].iloc[0] == 3
    assert doctor.gap_counts(con).empty


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


def test_extreme_threshold_boundary_at_non_round_open(con):
    rows = [
        # symbol, open, close → oc_return; exactly ±50% is not suspicious (strict >)
        ("UPEDGE", 5.00, 7.50),  # exactly +50%
        ("UPOVER", 5.00, 7.51),  # +50.2% — flagged
        ("DOWNEDGE", 5.00, 2.50),  # exactly −50%
        ("DOWNOVER", 5.00, 2.49),  # −50.2% — flagged
    ]
    store.upsert_prices(
        con,
        pd.DataFrame(
            {
                "symbol": [r[0] for r in rows],
                "date": [START] * len(rows),
                "open": [r[1] for r in rows],
                "high": [max(r[1], r[2]) for r in rows],
                "low": [min(r[1], r[2]) for r in rows],
                "close": [r[2] for r in rows],
                "adj_close": [r[2] for r in rows],
                "volume": [1_000_000] * len(rows),
            }
        ),
    )
    ext = doctor.extreme_bars(con)
    assert set(ext["symbol"]) == {"UPOVER", "DOWNOVER"}


def test_invalid_bars_catches_rows_the_returns_view_excludes(defective):
    inv = doctor.invalid_bars(defective)
    assert inv["symbol"].tolist() == ["BADO"]
    assert inv["invalid"].tolist() == [4]  # 3 open=0 bars + 1 NaN close
    assert inv["first_invalid"].iloc[0].date() == dt.date(2026, 1, 20)
    assert inv["last_invalid"].iloc[0].date() == LAST
    # the blindness this check exists for: BADO's raw bars look fresh and its
    # valid span has no interior hole, so neither stale nor gaps flag it
    assert "BADO" not in doctor.stale_symbols(defective)["symbol"].tolist()
    assert "BADO" not in doctor.gap_counts(defective)["symbol"].tolist()


def test_recent_invalid_bars_fail_the_report(con):
    # Reviewer reproduction: open=0.0 on recent bars used to report a healthy store.
    seed_history(con, {"AAA": [0.01] * 10, "BBB": [0.01] * 10})
    store.record_ingest_from(con, ["AAA", "BBB"], START)
    con.execute("UPDATE prices SET open = 0 WHERE symbol = 'BBB' AND date >= DATE '2026-01-14'")
    report = doctor.run(con)
    assert not report.ok
    assert report.problem_count == 1  # gaps, stale, extreme, runs all blind to this
    assert report.invalid["symbol"].tolist() == ["BBB"]
    assert report.invalid["invalid"].tolist() == [3]
    text = "\n".join(doctor.format_report(report))
    assert "BBB" in text


def test_invalid_bars_null_open_and_close(con):
    seed_history(con, {"AAA": [0.01] * 5, "NULLY": [0.01] * 5})
    con.execute("UPDATE prices SET open = NULL WHERE symbol = 'NULLY' AND date = DATE '2026-01-06'")
    con.execute(
        "UPDATE prices SET close = NULL WHERE symbol = 'NULLY' AND date = DATE '2026-01-08'"
    )
    inv = doctor.invalid_bars(con)
    assert inv["symbol"].tolist() == ["NULLY"]
    assert inv["invalid"].tolist() == [2]


def test_coverage_universe_symbol_without_prices(defective):
    assert doctor.universe_symbols_without_prices(defective) == ["GHOST"]


def test_coverage_price_symbol_without_meta(defective):
    assert doctor.price_symbols_without_meta(defective) == ["WILD"]


def test_run_collects_every_problem(defective):
    report = doctor.run(defective)
    assert not report.ok
    # 1 gap symbol + 2 stale + 2 extreme bars + 1 zero run + 1 invalid symbol
    # + GHOST (no prices) + WILD (no meta)
    assert report.problem_count == 9
    text = "\n".join(doctor.format_report(report))
    for symbol in ["GAPPY", "STALE", "TAIL", "WILD", "FLAT", "BADO", "GHOST"]:
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
    assert doctor.invalid_bars(clean).empty
    assert doctor.universe_symbols_without_prices(clean) == []
    assert doctor.price_symbols_without_meta(clean) == []
    text = "\n".join(doctor.format_report(report))
    assert "[FAIL]" not in text
    assert text.count("[ OK ]") == 5


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
    for symbol in ["GAPPY", "STALE", "TAIL", "WILD", "FLAT", "BADO", "GHOST"]:
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
