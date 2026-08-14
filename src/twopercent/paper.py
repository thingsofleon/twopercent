"""Forward-only paper trading of ONE exit rule, with costs.

Everything else in this project measures PREDICTION. This measures whether the
prediction is tradeable, which is a different question and the one the dashboard
cannot answer: its exit-rule explorer is a what-if over history, gross of costs,
and every version of it examined so far turned out to be flattered in some
direction. The only way to find out is to fix a rule, price it honestly, and
watch it forward.

THE RULE: `limit_2pct`. Buy at the open, sell at +2% if the day touches it,
otherwise sell at the close. Chosen because it is the rule the TARGET was
designed around — ROADMAP's locked-in definition is "a pre-placed +2% limit
would have filled, deterministic on the day's high" — so its fill assumption is
one the project already committed to and documented, rather than a new one
invented here. The alternatives were rejected for adding assumptions:
`limit_stop` needs an intra-day ordering that is unresolvable ~30% of the time,
and a stop fill that measurement showed is not the trigger price; `trailing`
needs a full intraday path and is currently withdrawn for exactly that reason.
`hold_close` needs nothing extra but does not match the signal — the model
predicts a TOUCH, not a close.

FORWARD ONLY. record_day() refuses a target day older than MAX_BACKFILL_DAYS.
A ledger that can be backfilled is just a backtest with extra steps, and would
inherit every bias the backtest has (survivorship, hindsight in the feature and
threshold choices). The point of this table is that nothing in it was chosen
after the fact.

COSTS ARE APPLIED AT REPORT TIME, never stored. The ledger keeps observed
prices and gross returns; net is computed on the way out. So the cost model can
be corrected — and it will be, it is currently an estimate — without rewriting
history, and the same trades can be shown at several cost levels. report()
returns a SENSITIVITY table rather than one number, because a single net figure
invites trusting a cost assumption that has not been measured.
"""

from __future__ import annotations

import datetime as dt
import logging

import duckdb
import pandas as pd

from twopercent import scan, track

logger = logging.getLogger(__name__)

RULE = "limit_2pct"
# The basket actually traded. Matches the detector's pinned top-20 so the paper
# record and the degradation signal describe the same picks.
PAPER_TOP_N = 20
# Refuse to record a target day older than this. Forward-only is the entire
# value of the ledger; see the module docstring.
MAX_BACKFILL_DAYS = 5

# Round-trip cost levels, in basis points of notional, that report() evaluates.
# NOT a measurement — a sensitivity grid. Commission on US equities is often
# zero, so this is dominated by the bid-ask spread and by slippage against the
# open/close prints. The model picks median-$13.80 stocks whose bottom quartile
# trades under $7M a day, where a 25-50bp round trip is unremarkable, so the
# grid deliberately spans "free" to "expensive" and the reader picks.
COST_GRID_BPS = (0, 10, 25, 50, 100)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    strategy TEXT NOT NULL,
    target_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    rank INTEGER NOT NULL,
    rule TEXT NOT NULL,
    gross_return DOUBLE NOT NULL,
    exit_reason TEXT NOT NULL,
    recorded_ts TIMESTAMP NOT NULL,
    PRIMARY KEY (strategy, target_date, symbol)
);
"""


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(_SCHEMA)


def record_day(
    con: duckdb.DuckDBPyConnection,
    strategy: str,
    target_date: dt.date,
    today: dt.date | None = None,
) -> int:
    """Record one completed trading day's basket. Returns rows written.

    Idempotent per (strategy, target_date): re-running a day replaces it, which
    keeps a re-run of the score routine from double-counting.
    """
    ensure_schema(con)
    today = today or dt.date.today()
    age = (today - target_date).days
    if age > MAX_BACKFILL_DAYS:
        raise ValueError(
            f"refusing to paper-trade {target_date}: {age} days old, past the "
            f"{MAX_BACKFILL_DAYS}-day limit. This ledger is FORWARD-ONLY — "
            "backfilling it would inherit the backtest's survivorship and "
            "hindsight and destroy the only untainted evidence it holds"
        )
    outcomes = track.daily_rank_outcomes(con, strategy, top_n=PAPER_TOP_N)
    if outcomes.empty:
        return 0
    day = outcomes[pd.to_datetime(outcomes["target_date"]).dt.date == target_date]
    if day.empty:
        return 0
    rows = day.nsmallest(PAPER_TOP_N, "rank").copy()
    # The rule: sell at +2% if the day touched it (the guarded touch event —
    # a glitch-suspect high never counts as a fill), else sell at the close.
    filled = rows["hit"].astype(bool)
    rows["gross_return"] = filled.map({True: scan.DEFAULT_THRESHOLD}).fillna(rows["oc_return"])
    rows["exit_reason"] = filled.map({True: "limit", False: "close"})
    rows["rule"] = RULE
    rows["strategy"] = strategy
    rows["target_date"] = target_date
    con.execute(
        "DELETE FROM paper_trades WHERE strategy = ? AND target_date = ?",
        [strategy, target_date],
    )
    con.register(
        "_paper_in",
        rows[["strategy", "target_date", "symbol", "rank", "rule", "gross_return", "exit_reason"]],
    )
    con.execute(
        "INSERT INTO paper_trades SELECT strategy, target_date, symbol, rank, rule, "
        "gross_return, exit_reason, now() FROM _paper_in"
    )
    con.unregister("_paper_in")
    logger.info(
        "paper: recorded %d trade(s) for %s (%s), %d filled at the limit",
        len(rows),
        target_date,
        RULE,
        int(filled.sum()),
    )
    return len(rows)


def report(con: duckdb.DuckDBPyConnection, strategy: str, basket: int = 5) -> pd.DataFrame:
    """Net P&L of the paper record at each cost level. One row per level.

    Equal-weight `basket` names per day, compounded. Costs are charged as a
    round trip on the full notional of every position, every day — this rule
    closes daily, so there is no holding period to amortise them over. That is
    the point: a strategy that trades every name every day pays the spread every
    day, which is exactly the effect a gross backtest hides.
    """
    ensure_schema(con)
    trades = con.execute(
        "SELECT target_date, rank, gross_return, exit_reason FROM paper_trades "
        "WHERE strategy = ? AND rule = ? ORDER BY target_date, rank",
        [strategy, RULE],
    ).df()
    if trades.empty:
        return pd.DataFrame(
            columns=["cost_bps", "growth", "days", "trades", "win_rate", "mean_daily"]
        )
    rows = []
    for bps in COST_GRID_BPS:
        cost = bps / 10_000.0
        growth, day_returns = 1.0, []
        for _day, grp in trades.groupby("target_date"):
            picks = grp.nsmallest(basket, "rank")
            net = picks["gross_return"] - cost
            day_ret = float(net.mean())
            day_returns.append(day_ret)
            growth *= 1 + day_ret
        rows.append(
            {
                "cost_bps": bps,
                "growth": round(growth, 4),
                "days": len(day_returns),
                "trades": int(len(trades)),
                "win_rate": round(sum(1 for r in day_returns if r > 0) / len(day_returns), 4),
                "mean_daily": round(sum(day_returns) / len(day_returns), 6),
            }
        )
    return pd.DataFrame(rows)


def breakeven_bps(con: duckdb.DuckDBPyConnection, strategy: str, basket: int = 5) -> float | None:
    """Round-trip cost, in bps, at which the strategy stops making money.

    The single most useful number here: it converts "is the edge real?" into
    "is the edge bigger than the spread?", which is a question about the market
    rather than about the model. Below ~10bps the answer is almost certainly no
    for names this thin.
    """
    ensure_schema(con)
    trades = con.execute(
        "SELECT target_date, rank, gross_return FROM paper_trades WHERE strategy = ? AND rule = ?",
        [strategy, RULE],
    ).df()
    if trades.empty:
        return None
    daily = [
        float(grp.nsmallest(basket, "rank")["gross_return"].mean())
        for _d, grp in trades.groupby("target_date")
    ]
    mean_daily = sum(daily) / len(daily)
    return round(mean_daily * 10_000.0, 2) if mean_daily > 0 else 0.0
