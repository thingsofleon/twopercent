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

# Days of forward record below which growth and breakeven are NOT a result.
# One trading month. A single day with a 100% win rate produces "growth 1.02,
# breakeven 200bps", which reads as a finding and is noise — and this ledger is
# deliberately slow to fill, so the misleading window is weeks long, not hours.
MIN_REPORT_DAYS = 20

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_trades (
    strategy TEXT NOT NULL,
    target_date DATE NOT NULL,
    symbol TEXT NOT NULL,
    rank INTEGER NOT NULL,
    rule TEXT NOT NULL,
    gross_return DOUBLE NOT NULL,
    exit_reason TEXT NOT NULL,
    n_avail INTEGER,
    recorded_ts TIMESTAMP NOT NULL,
    PRIMARY KEY (strategy, target_date, symbol)
);
-- Nullable and ALTER-added: a store that created this table before n_avail
-- existed keeps its rows, with NULL meaning "recorded before the basket size
-- was tracked". CREATE TABLE IF NOT EXISTS silently keeps the old shape, which
-- is how the column went missing on a live store in the first place.
ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS n_avail INTEGER;
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
    # LATE is the project's definition of forward, and calendar arithmetic is
    # not a substitute for it. track.daily_rank_outcomes flags a day late when
    # its predictions were CREATED after the target day's 09:30 open — outcome
    # already knowable. A day can be one day old and still late: 2026-08-10's
    # picks were re-saved at 20:01 that evening, and the age check alone would
    # have written 20 "forward-only" trades buying an open that was already in
    # the past. Every other consumer honours this flag; this table, whose entire
    # value is that nothing in it was chosen after the fact, must not be the
    # exception.
    if bool(day["late"].iloc[0]):
        raise ValueError(
            f"refusing to paper-trade {target_date}: its predictions were created "
            "AFTER the target day's open (late), so the outcome was knowable when "
            "they were made. Forward-only means live-only, not merely recent"
        )
    rows = day.nsmallest(PAPER_TOP_N, "rank").copy()
    n_avail = len(day)
    # The rule: sell at +2% if the day touched it (the guarded touch event —
    # a glitch-suspect high never counts as a fill), else sell at the close.
    filled = rows["hit"].astype(bool)
    rows["gross_return"] = filled.map({True: scan.DEFAULT_THRESHOLD}).fillna(rows["oc_return"])
    rows["exit_reason"] = filled.map({True: "limit", False: "close"})
    rows["rule"] = RULE
    rows["strategy"] = strategy
    rows["target_date"] = target_date
    # How many of the intended basket actually traded. A pick absent because it
    # was halted or delisted at the open is exactly the catastrophic case a real
    # trader cares about, and averaging the survivors at full weight would
    # silently substitute it away — survivorship inside a table sold as
    # untainted.
    rows["n_avail"] = n_avail
    if n_avail < PAPER_TOP_N:
        logger.warning(
            "paper: %s traded only %d of the intended top-%d — the missing picks "
            "are NOT substituted; the day carries a smaller basket",
            target_date,
            n_avail,
            PAPER_TOP_N,
        )
    cols = [
        "strategy",
        "target_date",
        "symbol",
        "rank",
        "rule",
        "gross_return",
        "exit_reason",
        "n_avail",
    ]
    con.register("_paper_in", rows[cols])
    try:
        # ONE transaction. Delete-then-insert in autocommit meant a failed
        # insert left the day DELETED — and the forward-only guard then forbids
        # re-recording it, so a transient error permanently erased a day from a
        # ledger whose gaps cannot be filled later.
        con.execute("BEGIN TRANSACTION")
        con.execute(
            "DELETE FROM paper_trades WHERE strategy = ? AND target_date = ? AND rule = ?",
            [strategy, target_date, RULE],
        )
        # Columns NAMED, never positional: an ALTER-added column lands at the
        # END of the table, so a positional INSERT silently maps n_avail onto
        # recorded_ts on any store that predates it.
        con.execute(
            "INSERT INTO paper_trades "
            "(strategy, target_date, symbol, rank, rule, gross_return, exit_reason, "
            " n_avail, recorded_ts) "
            "SELECT strategy, target_date, symbol, rank, rule, gross_return, "
            "exit_reason, n_avail, now() FROM _paper_in"
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("_paper_in")
    logger.info(
        "paper: recorded %d trade(s) for %s (%s), %d filled at the limit",
        len(rows),
        target_date,
        RULE,
        int(filled.sum()),
    )
    return len(rows)


def report(
    con: duckdb.DuckDBPyConnection, strategy: str, basket: int = PAPER_TOP_N
) -> pd.DataFrame:
    """Net P&L of the paper record at each cost level. One row per level.

    Equal-weight `basket` names per day, compounded. Costs are charged as a
    round trip on the full notional of every position, every day — this rule
    closes daily, so there is no holding period to amortise them over. That is
    the point: a strategy that trades every name every day pays the spread every
    day, which is exactly the effect a gross backtest hides.

    Defaults to PAPER_TOP_N, the basket actually recorded and the one the
    detector, dashboard and live track record all report. It defaulted to 5,
    which is the SMALLEST, highest-variance, lowest-capacity basket — and the
    only one whose edge survives realistic costs. `basket` is tunable over the
    same days that are reported, so a default that happens to flatter is a
    multiple comparison presented as a headline.
    """
    ensure_schema(con)
    trades = con.execute(
        "SELECT target_date, rank, gross_return, n_avail FROM paper_trades "
        "WHERE strategy = ? AND rule = ? ORDER BY target_date, rank",
        [strategy, RULE],
    ).df()
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "cost_bps",
                "growth",
                "days",
                "trades",
                "day_win_rate",
                "mean_daily",
                "sd_daily",
                "t_stat",
            ]
        )
    rows = []
    for bps in COST_GRID_BPS:
        cost = bps / 10_000.0
        growth, day_returns, n_trades = 1.0, [], 0
        for _day, grp in trades.groupby("target_date"):
            picks = grp.nsmallest(basket, "rank")
            n_trades += len(picks)
            day_ret = float((picks["gross_return"] - cost).mean())
            day_returns.append(day_ret)
            growth *= 1 + day_ret
        n = len(day_returns)
        mean = sum(day_returns) / n
        sd = (sum((r - mean) ** 2 for r in day_returns) / (n - 1)) ** 0.5 if n > 1 else float("nan")
        rows.append(
            {
                "cost_bps": bps,
                "growth": round(growth, 4),
                "days": n,
                # The trades ACTUALLY SIMULATED at this basket. This reported the
                # whole ledger's row count, which at basket 5 overstated the
                # sample four-fold on the headline table.
                "trades": n_trades,
                "day_win_rate": round(sum(1 for r in day_returns if r > 0) / n, 4),
                "mean_daily": round(mean, 6),
                "sd_daily": round(sd, 6) if sd == sd else None,
                # n is small and will stay small for months. A growth figure
                # without its dispersion is the single most likely way this
                # ledger gets mistaken for evidence of profitability.
                "t_stat": round(mean / (sd / n**0.5), 2) if sd == sd and sd > 0 else None,
            }
        )
    return pd.DataFrame(rows)


def basket_sweep(con: duckdb.DuckDBPyConnection, strategy: str) -> pd.DataFrame:
    """Breakeven cost at every basket size, so the choice cannot hide.

    `basket` is a free parameter tuned over the same days that are reported.
    Showing one basket invites picking the flattering one; showing the sweep
    makes the sensitivity the reader's to judge.
    """
    return pd.DataFrame(
        [
            {"basket": b, "breakeven_bps": breakeven_bps(con, strategy, basket=b)}
            for b in (1, 5, 10, PAPER_TOP_N)
        ]
    )


def breakeven_bps(
    con: duckdb.DuckDBPyConnection, strategy: str, basket: int = PAPER_TOP_N
) -> float | None:
    """Round-trip cost, in bps, at which the strategy stops compounding upward.

    The single most useful number here: it converts "is the edge real?" into
    "is the edge bigger than the spread?", which is a question about the market
    rather than about the model.

    GEOMETRIC, matching report()'s compounded `growth`. The arithmetic-mean
    breakeven overstates it — at the arithmetic answer the compounded curve is
    already below 1.0, because variance drags. Solved by bisection rather than
    closed form so it tracks whatever report() actually computes.
    """
    ensure_schema(con)
    trades = con.execute(
        "SELECT target_date, rank, gross_return FROM paper_trades WHERE strategy = ? AND rule = ?",
        [strategy, RULE],
    ).df()
    if trades.empty:
        return None
    daily_gross = [
        list(grp.nsmallest(basket, "rank")["gross_return"])
        for _d, grp in trades.groupby("target_date")
    ]

    def growth_at(cost: float) -> float:
        g = 1.0
        for day in daily_gross:
            g *= 1 + (sum(r - cost for r in day) / len(day))
        return g

    if growth_at(0.0) <= 1.0:
        return 0.0  # no cost makes this profitable
    lo, hi = 0.0, 0.05  # 0 to 500bps brackets any plausible answer
    for _ in range(60):
        mid = (lo + hi) / 2
        if growth_at(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    return round(lo * 10_000.0, 2)
