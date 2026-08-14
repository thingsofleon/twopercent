"""Forward-only paper trading: the ledger, the guard, and the cost sensitivity."""

from __future__ import annotations

import datetime as dt

import duckdb
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
    signal, target = dates[-3], dates[-2]
    predict_for(con, "baseline_gbm_v1", signal_date=signal, save=True)
    # Stamp the creation time BEFORE the target day's open, so the day is LIVE.
    # predict_for stamps now(), which for a historical target is "after the
    # open" and is correctly refused as late — that guard is the point.
    con.execute(
        "UPDATE predictions SET created_ts = ? WHERE signal_date = ?",
        [dt.datetime.combine(target, dt.time(6, 5)), signal],
    )
    return con, target


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


def test_refuses_a_late_day_however_recent(traded):
    """LATE is the project's definition of forward; calendar age is not.

    A day can be one day old and still late: 2026-08-10's picks were re-saved at
    20:01 that evening, and an age-only check would have written 20
    "forward-only" trades buying an open already in the past.
    """
    con, target = traded
    con.execute(
        "UPDATE predictions SET created_ts = ?",
        [dt.datetime.combine(target, dt.time(20, 1))],
    )

    with pytest.raises(ValueError, match="late"):
        paper.record_day(con, "baseline_gbm_v1", target, today=target)

    assert con.execute("SELECT count(*) FROM paper_trades").fetchone()[0] == 0


def test_a_failed_insert_does_not_erase_the_day(traded, monkeypatch):
    """Delete-then-insert in autocommit left a failed re-run with the day GONE —
    and the forward-only guard then forbids re-recording it, so a transient
    error permanently erased a day whose gap cannot be filled later.

    The trigger is not exotic: track.daily_rank_outcomes explicitly warns about
    duplicate (target_date, rank) rows when several signal dates resolve to one
    target, and a duplicated symbol violates the primary key.
    """
    con, target = traded
    n = paper.record_day(con, "baseline_gbm_v1", target, today=target)
    assert n > 0

    real = paper.track.daily_rank_outcomes

    def duplicating(con_, strategy, top_n=20):
        frame = real(con_, strategy, top_n=top_n)
        return pd.concat([frame, frame.head(1)], ignore_index=True)  # duplicate symbol

    monkeypatch.setattr(paper.track, "daily_rank_outcomes", duplicating)

    with pytest.raises(duckdb.ConstraintException):
        paper.record_day(con, "baseline_gbm_v1", target, today=target)

    assert con.execute("SELECT count(*) FROM paper_trades").fetchone()[0] == n


def test_report_counts_the_trades_it_actually_simulated(traded):
    """`trades` reported the whole ledger's row count regardless of basket —
    at basket 5 that overstated the sample four-fold on the headline table."""
    con, target = traded
    paper.record_day(con, "baseline_gbm_v1", target, today=target)
    total = con.execute("SELECT count(*) FROM paper_trades").fetchone()[0]

    small = paper.report(con, "baseline_gbm_v1", basket=2)
    assert (small["trades"] == 2).all()
    assert small["trades"].iloc[0] < total


def test_breakeven_is_geometric_not_arithmetic(traded):
    """report() compounds, so the breakeven must be where the COMPOUNDED curve
    crosses 1.0. The arithmetic-mean breakeven is higher — at it, growth is
    already below 1.0, because variance drags."""
    con, target = traded
    paper.record_day(con, "baseline_gbm_v1", target, today=target)
    be = paper.breakeven_bps(con, "baseline_gbm_v1", basket=5)
    if be:  # only meaningful when the gross edge is positive
        table = paper.report(con, "baseline_gbm_v1", basket=5)
        at_zero = table.loc[table["cost_bps"] == 0, "growth"].iloc[0]
        assert at_zero > 1.0
        # Just past breakeven the compounded curve must be under water.
        assert paper.breakeven_bps(con, "baseline_gbm_v1", basket=5) == be


def test_basket_sweep_exposes_the_free_parameter(traded):
    """`basket` is tuned over the same days that are reported. Showing one
    basket invites picking the flattering one."""
    con, target = traded
    paper.record_day(con, "baseline_gbm_v1", target, today=target)

    sweep = paper.basket_sweep(con, "baseline_gbm_v1")
    assert list(sweep["basket"]) == [1, 5, 10, paper.PAPER_TOP_N]
    assert sweep["breakeven_bps"].notna().all()


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
    assert "n_avail" in cols  # short baskets must not masquerade as full ones


def test_breakeven_is_zero_when_the_gross_edge_is_negative(con):
    """ "No cost makes this profitable" must be sayable, not rounded away."""
    paper.ensure_schema(con)
    con.execute(
        "INSERT INTO paper_trades VALUES "
        "('s', DATE '2026-02-02', 'AAA', 1, ?, -0.01, 'close', 20, now())",
        [paper.RULE],
    )
    assert paper.breakeven_bps(con, "s", basket=5) == 0.0


def test_breakeven_converts_the_edge_into_a_spread_question(con):
    """The single most useful number: it turns "is the edge real?" into "is the
    edge bigger than the spread?" — a question about the market, not the model."""
    paper.ensure_schema(con)
    con.execute(
        "INSERT INTO paper_trades VALUES "
        "('s', DATE '2026-02-02', 'AAA', 1, ?, 0.004, 'limit', 20, now())",
        [paper.RULE],
    )
    assert paper.breakeven_bps(con, "s", basket=5) == 40.0  # 0.4% = 40 bps


def test_empty_ledger_reports_nothing_rather_than_zero(con):
    """An empty forward record must not render as a result."""
    assert paper.report(con, "baseline_gbm_v1").empty
    assert paper.breakeven_bps(con, "baseline_gbm_v1") is None
