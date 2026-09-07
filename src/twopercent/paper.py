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

OUTCOMES ARE FROZEN AT FIRST RECORDING — deliberately, and it costs accuracy.
The score run records ~50 minutes after the close, and the provider's bars keep
revising after that: measured 2026-09-08 on the live store, 15 recorded days
disagreed with the then-current tape by ~30pp of top-1 compounded growth (4
fill verdicts flipped by revised opens/highs; on one day the eventual rank-1
symbol's bar had not arrived at recording time at all). The ledger keeps the
FIRST post-close observation anyway, because re-recording would let a table
sold as immutable rewrite itself every time the feed moved — but that makes it
a record of what was SEEN, not of the final tape, and the two can disagree.
drift() measures that disagreement instead of leaving it to be discovered;
--grid prints it. The revisions observed so far were not systematically
conservative (three hurt the ledger, one helped).
"""

from __future__ import annotations

import datetime as dt
import logging

import duckdb
import pandas as pd

from twopercent import scan, track
from twopercent.strategy import pick_return_band

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
-- Per-pick outcome returns of the target day (open-to-high/low/close), the
-- same trio experiment_daily stores: with them, EVERY daily exit rule replays
-- deterministically from the ledger, forward-only, without re-recording
-- anything (#118). Nullable and ALTER-added like n_avail: rows recorded
-- before these existed keep NULL = "outcomes not tracked", and consumers must
-- EXCLUDE and count such days, never average around them.
ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS oh DOUBLE;
ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS ol DOUBLE;
ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS oc DOUBLE;
"""

# The replay grid (#118). Rules are the two an oh/ol/oc row prices EXACTLY:
# limit_stop needs an intraday ordering that daily bars cannot supply (its
# honest answer is a band, and a band is never a winning cell) and trailing is
# withdrawn (#105). Baskets are nested prefixes of the recorded top-20; sizes
# beyond PAPER_TOP_N would need recording more ranks and are out of scope.
GRID_RULES = ("hold_close", "limit_2pct")
GRID_BASKETS = (1, 5, 10, PAPER_TOP_N)
# Per-basket day floors, replacing the one-size MIN_REPORT_DAYS for grid cells:
# a top-1 cell is ONE pick per day, so at 20 days its growth is a coin path.
# HONESTY (quant-skeptic, PR #119): these floors do NOT equalize evidence.
# Measured daily sd runs ~0.052 (top-1) to ~0.011 (top-20), so top-1 at its
# 60-day floor still carries ~2.7x the standard error of top-20 at 20 days —
# SE parity would need ~450 top-1 days, which is not a floor, it is a career.
# The floors only remove the absurdest reads; the per-cell mde80_daily column
# is the real power statement, and the reader must use it.
GRID_MIN_DAYS = {1: 60, 5: 40, 10: MIN_REPORT_DAYS, PAPER_TOP_N: MIN_REPORT_DAYS}


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
    # The target day's outcome returns ride along (#118) so every exit rule —
    # not just the one traded — replays from this row later. Same-day data,
    # written at scoring time: nothing here was knowable before the open.
    rows["oh"] = rows["high_return"]
    rows["ol"] = rows["low_return"]
    rows["oc"] = rows["oc_return"]
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
        "oh",
        "ol",
        "oc",
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
            " n_avail, oh, ol, oc, recorded_ts) "
            "SELECT strategy, target_date, symbol, rank, rule, gross_return, "
            "exit_reason, n_avail, oh, ol, oc, now() FROM _paper_in"
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


def pending_days(
    con: duckdb.DuckDBPyConnection, strategy: str, today: dt.date | None = None
) -> list[dt.date]:
    """Scored days still eligible for the ledger and not yet in it.

    The step used to record only days scored by THAT run, which silently lost
    days: scoring also happens in the PREDICT routine, so any day first scored
    there was invisible to the ledger forever. It also lost days whenever a
    score run aborted (a real +50% biotech move tripped the corruption gate on
    2026-08-19) or was held back as incomplete (2026-08-18). Three of five days
    went missing that way, in the one table that is supposed to be the record.

    Asking "what is eligible and absent?" instead is SELF-HEALING: a missed run
    costs nothing as long as the day is still inside the forward window. The
    guards are unchanged — record_day still refuses anything late or older than
    MAX_BACKFILL_DAYS — so completeness is bought without loosening the
    forward-only property that makes the ledger evidence.
    """
    today = today or dt.date.today()
    ensure_schema(con)
    scored = track.score_predictions(con, strategy, top_n=PAPER_TOP_N).scored
    if scored.empty:
        return []
    have = {
        r[0]
        for r in con.execute(
            "SELECT DISTINCT target_date FROM paper_trades WHERE strategy = ? AND rule = ?",
            [strategy, RULE],
        ).fetchall()
    }
    out = []
    for row in scored.itertuples():
        day = pd.Timestamp(row.target_date).date()
        if day in have or bool(row.late):
            continue
        if (today - day).days > MAX_BACKFILL_DAYS:
            continue
        out.append(day)
    return sorted(out)


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


def _pick_returns(
    con: duckdb.DuckDBPyConnection, strategy: str, rule: str
) -> tuple[pd.DataFrame, int]:
    """Per-pick gross return of `rule` over the ledger; (frame, excluded_days).

    For the TRADED rule (limit_2pct) the recorded `gross_return` is the answer —
    the ledger IS the record, and works for every row ever written. Any other
    rule replays from the stored oh/ol/oc through strategy.pick_return_band,
    the single definition of exit rules (its band endpoints are equal for every
    rule allowed in GRID_RULES, so `worst` is exact, not a bound). `filled`
    is exit_reason == 'limit': the guarded touch event as recorded at scoring
    time, never a raw high comparison re-derived here.

    A day where ANY recorded pick lacks outcomes (written before the oh/ol/oc
    columns existed) cannot replay a basket honestly and is EXCLUDED whole and
    counted — the alternative is a growth curve silently computed over easier
    days than the ones on screen. In practice such days are all-or-nothing:
    the columns arrived together in one migration.

    Only GRID_RULES are accepted. This is a hard error, not a convenience:
    `limit_stop` on a both-touched day returns a (worst, best) BAND, and taking
    [0] would silently present the lower bound as the number — the repo's
    signature failure mode. If a band rule ever joins the grid it must return
    both endpoints, like the explorer does; `trailing` is withdrawn (#105).
    """
    if rule not in GRID_RULES:
        raise ValueError(
            f"unsupported replay rule {rule!r}: only {GRID_RULES} price a ledger row "
            "exactly (limit_stop is a band, trailing is withdrawn — #118/#105)"
        )
    ensure_schema(con)
    trades = con.execute(
        "SELECT target_date, symbol, rank, gross_return, exit_reason, oh, ol, oc "
        "FROM paper_trades WHERE strategy = ? AND rule = ? ORDER BY target_date, rank",
        [strategy, RULE],
    ).df()
    if trades.empty:
        return trades.assign(ret=pd.Series(dtype=float)), 0
    if rule == RULE:
        return trades.assign(ret=trades["gross_return"]), 0
    row_ok = trades[["oh", "ol", "oc"]].notna().all(axis=1)
    complete_days = trades.assign(_ok=row_ok).groupby("target_date")["_ok"].transform("all")
    excluded = int(trades.loc[~complete_days, "target_date"].nunique())
    usable = trades[complete_days].copy()
    if excluded:
        logger.warning(
            "paper grid: %d day(s) excluded from %s replay — recorded before per-pick "
            "outcomes were stored, so the rule cannot be priced on them",
            excluded,
            rule,
        )
    if usable.empty:
        return usable.assign(ret=pd.Series(dtype=float)), excluded
    usable["ret"] = [
        pick_return_band(rule, row.ol, row.oc, row.exit_reason == "limit")[0]
        for row in usable.itertuples()
    ]
    return usable, excluded


def grid(con: duckdb.DuckDBPyConnection, strategy: str) -> pd.DataFrame:
    """Every replayable rule x basket cell of the forward record — the WHOLE grid.

    One row per (rule, basket): gross growth, day/trade counts, t-stat and
    breakeven bps, plus the cell's day floor (`min_days`) and how many days its
    replay had to exclude. Always the full grid, never one cell — a single cell
    is a multiple comparison presented as a headline (same reason basket_sweep
    shows the curve). The grid is a FAMILY of len(GRID_RULES)*len(GRID_BASKETS)
    cells: per #118 a "winning" cell must be named first and then confirmed on
    days arriving AFTER it was named; nothing here does that naming.
    """
    rows = []
    for rule in GRID_RULES:
        # ONE replay per rule: grid cells and their breakevens share it, so the
        # exclusion warning fires once per rule, not once per cell (reviewer
        # finding, PR #119 — five identical warnings read like five problems).
        picks, excluded = _pick_returns(con, strategy, rule)
        for basket in GRID_BASKETS:
            per_day = [grp.nsmallest(basket, "rank") for _d, grp in picks.groupby("target_date")]
            daily = [float(day["ret"].mean()) for day in per_day]
            n = len(daily)
            n_trades = sum(len(day) for day in per_day)
            growth = 1.0
            for r in daily:
                growth *= 1 + r
            mean = sum(daily) / n if n else None
            sd = (sum((r - mean) ** 2 for r in daily) / (n - 1)) ** 0.5 if n > 1 else None
            # Same guard convention as report(): NaN sd must yield None, never a
            # printed NaN t (the pin test keeps these two surfaces one number).
            has_sd = sd is not None and sd == sd and sd > 0
            rows.append(
                {
                    "rule": rule,
                    "basket": basket,
                    "days": n,
                    "excluded_days": excluded,
                    "trades": n_trades,
                    "growth": round(growth, 4) if n else None,
                    "mean_daily": round(mean, 6) if n else None,
                    "t_stat": round(mean / (sd / n**0.5), 2) if has_sd else None,
                    # The smallest daily mean this cell could detect at 80% power
                    # (two-sided 0.05, normal approx: (z_.975 + z_.80)·se — the
                    # ab.py idea, constants inlined to stay stdlib). UNCORRECTED
                    # for the 8-cell family; it is a per-cell power statement,
                    # not a promotion threshold. It is what stops "not
                    # significant on 20 days" being read as "no edge".
                    "mde80_daily": round(2.801585 * sd / n**0.5, 6) if has_sd else None,
                    "breakeven_bps": _breakeven_from_daily([list(day["ret"]) for day in per_day]),
                    "min_days": GRID_MIN_DAYS[basket],
                }
            )
    return pd.DataFrame(rows)


def _breakeven_from_daily(daily_gross: list[list[float]]) -> float | None:
    """Bisection for the round-trip cost at which compounding stops; see
    breakeven_bps for why geometric."""
    if not daily_gross:
        return None

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


def breakeven_bps(
    con: duckdb.DuckDBPyConnection,
    strategy: str,
    basket: int = PAPER_TOP_N,
    rule: str = RULE,
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
    picks, _excluded = _pick_returns(con, strategy, rule)
    if picks.empty:
        return None
    return _breakeven_from_daily(
        [list(grp.nsmallest(basket, "rank")["ret"]) for _d, grp in picks.groupby("target_date")]
    )


def drift(con: duckdb.DuckDBPyConnection, strategy: str) -> dict:
    """How far the frozen ledger has drifted from today's bars. Measured, not guessed.

    Recomputes each recorded pick's outcome through the SAME production path the
    explorer's LIVE row uses (track.daily_rank_outcomes) and counts
    disagreements: rows whose oh/ol/oc moved, fill verdicts that flipped, and
    recorded rows whose symbol no longer appears in today's outcome frame (or
    appeared only after recording). Zero everywhere means the tape has not
    moved; anything else quantifies the freeze documented in the module
    docstring. Read-only; nothing is ever rewritten.
    """
    ensure_schema(con)
    stored = con.execute(
        "SELECT target_date, symbol, gross_return, exit_reason, oh, ol, oc "
        "FROM paper_trades WHERE strategy = ? AND rule = ?",
        [strategy, RULE],
    ).df()
    empty = {
        "rows": 0,
        "rows_compared": 0,
        "rows_with_outcomes": 0,
        "outcome_moved": 0,
        "fill_flipped": 0,
        "unmatched": 0,
    }
    if stored.empty:
        return empty
    current = track.daily_rank_outcomes(con, strategy, top_n=PAPER_TOP_N)
    if current.empty:
        return {**empty, "rows": len(stored), "unmatched": len(stored)}
    stored["target_date"] = pd.to_datetime(stored["target_date"]).dt.date
    current = current.copy()
    current["target_date"] = pd.to_datetime(current["target_date"]).dt.date
    merged = stored.merge(
        current[["target_date", "symbol", "high_return", "low_return", "oc_return", "hit"]],
        on=["target_date", "symbol"],
        how="left",
    )
    matched = merged["hit"].notna()
    moved = fill_flipped = with_outcomes = 0
    for row in merged[matched].itertuples():
        was_filled = row.exit_reason == "limit"
        now_filled = bool(row.hit)
        if was_filled != now_filled:
            fill_flipped += 1
        stored_out = (row.oh, row.ol, row.oc)
        current_out = (row.high_return, row.low_return, row.oc_return)
        # Rows recorded before the outcome columns existed have nothing to
        # compare — they must shrink the DENOMINATOR, not read as "unmoved".
        if all(a is not None and a == a for a in stored_out):
            with_outcomes += 1
            if any(abs(a - b) > 1e-9 for a, b in zip(stored_out, current_out, strict=True)):
                moved += 1
    result = {
        "rows": len(stored),
        "rows_compared": int(matched.sum()),
        "rows_with_outcomes": with_outcomes,
        "outcome_moved": moved,
        "fill_flipped": fill_flipped,
        # Recorded symbols absent from today's outcome frame for that day —
        # the bar vanished or the whole day fell out of the frame.
        "unmatched": int((~matched).sum()),
    }
    if moved or fill_flipped or result["unmatched"]:
        logger.warning(
            "paper drift: of %d recorded rows, %d outcome(s) moved vs today's bars, "
            "%d fill verdict(s) flipped, %d no longer match a current outcome row — the "
            "ledger keeps its first observation (see module docstring); this is the size "
            "of that choice",
            result["rows_with_outcomes"],
            moved,
            fill_flipped,
            result["unmatched"],
        )
    return result
