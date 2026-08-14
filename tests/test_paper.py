"""Forward-only paper trading: the ledger, the guard, and the cost sensitivity."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from tests.conftest import seed_history
from twopercent import paper, store
from twopercent.predict import predict_for


@pytest.fixture
def traded(con):
    """A seeded store with two logged, scored prediction days."""
    data = {f"RUN{i}": [0.03 + 0.001 * (i % 5)] * 60 for i in range(4)}
    data |= {f"FLT{i}": [0.002 + 0.001 * (i % 3)] * 60 for i in range(4)}
    seed_history(con, data, vary_volume=True)
    store.upsert_universe(
        con,
        pd.DataFrame(
            {
                "symbol": list(data),
                "name": list(data),
                "market_cap": [1e9] * len(data),
                "sector": ["Tech"] * len(data),
            }
        ),
        as_of=dt.date(2026, 3, 1),
    )
    dates = sorted(pd.bdate_range("2026-01-05", periods=60).date)
    predict_for(con, "baseline_gbm_v1", signal_date=dates[-3], save=True)
    return con, dates[-2]  # the target day of that prediction


def test_records_a_days_basket(traded):
    con, target = traded
    n = paper.record_day(con, "baseline_gbm_v1", target, today=target)

    assert n > 0
    rows = con.execute(
        "SELECT symbol, gross_return, exit_reason, rule FROM paper_trades"
    ).fetchall()
    assert len(rows) == n
    assert {r[3] for r in rows} == {paper.RULE}
    # The rule: a filled limit books EXACTLY +2%; everything else books the close.
    for _sym, gross, reason, _rule in rows:
        if reason == "limit":
            assert abs(gross - 0.02) < 1e-12
        else:
            assert reason == "close"


def test_refuses_to_backfill(traded):
    """THE property that makes this evidence rather than another backtest.

    A ledger that can be backfilled inherits the backtest's survivorship and the
    hindsight in every feature and threshold choice already made. Nothing in
    this table may be chosen after the fact.
    """
    con, target = traded
    stale = target + dt.timedelta(days=paper.MAX_BACKFILL_DAYS + 1)

    with pytest.raises(ValueError, match="FORWARD-ONLY"):
        paper.record_day(con, "baseline_gbm_v1", target, today=stale)

    assert con.execute("SELECT count(*) FROM paper_trades").fetchone()[0] == 0


def test_recording_a_day_twice_replaces_it(traded):
    con, target = traded
    first = paper.record_day(con, "baseline_gbm_v1", target, today=target)
    again = paper.record_day(con, "baseline_gbm_v1", target, today=target)

    assert first == again
    assert con.execute("SELECT count(*) FROM paper_trades").fetchone()[0] == first


def test_report_charges_costs_every_day_and_spans_the_grid(traded):
    """The rule closes daily, so there is no holding period to amortise costs
    over — a strategy that trades every name every day pays the spread every
    day, which is exactly what a gross backtest hides."""
    con, target = traded
    paper.record_day(con, "baseline_gbm_v1", target, today=target)

    table = paper.report(con, "baseline_gbm_v1", basket=5)

    assert list(table["cost_bps"]) == list(paper.COST_GRID_BPS)
    # Growth is monotonically WORSE as costs rise — if it is not, costs are not
    # actually being charged.
    assert list(table["growth"]) == sorted(table["growth"], reverse=True)
    assert table["growth"].iloc[0] > table["growth"].iloc[-1]
    # One day recorded, so mean_daily at zero cost is that day's basket return.
    assert (table["days"] == 1).all()


def test_costs_are_not_stored_so_the_model_can_be_corrected(traded):
    """Costs are applied at REPORT time. The ledger keeps observed gross
    returns, so a corrected cost model never requires rewriting history."""
    con, target = traded
    paper.record_day(con, "baseline_gbm_v1", target, today=target)

    cols = {r[0] for r in con.execute("DESCRIBE paper_trades").fetchall()}
    assert "gross_return" in cols
    assert not {c for c in cols if "cost" in c or "net" in c}


def test_breakeven_is_zero_when_the_gross_edge_is_negative(con):
    """ "No cost makes this profitable" must be sayable, not rounded away."""
    paper.ensure_schema(con)
    con.execute(
        "INSERT INTO paper_trades VALUES "
        "('s', DATE '2026-02-02', 'AAA', 1, ?, -0.01, 'close', now())",
        [paper.RULE],
    )
    assert paper.breakeven_bps(con, "s", basket=5) == 0.0


def test_breakeven_converts_the_edge_into_a_spread_question(con):
    """The single most useful number: it turns "is the edge real?" into "is the
    edge bigger than the spread?" — a question about the market, not the model."""
    paper.ensure_schema(con)
    con.execute(
        "INSERT INTO paper_trades VALUES "
        "('s', DATE '2026-02-02', 'AAA', 1, ?, 0.004, 'limit', now())",
        [paper.RULE],
    )
    assert paper.breakeven_bps(con, "s", basket=5) == 40.0  # 0.4% = 40 bps


def test_empty_ledger_reports_nothing_rather_than_zero(con):
    """An empty forward record must not render as a result."""
    assert paper.report(con, "baseline_gbm_v1").empty
    assert paper.breakeven_bps(con, "baseline_gbm_v1") is None
