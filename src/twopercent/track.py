"""Track record: score logged predictions against what actually happened."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from twopercent import scan, store
from twopercent.scan import _THRESHOLD_EPSILON, DEFAULT_THRESHOLD, touch_event_predicate

logger = logging.getLogger(__name__)

# Completeness gate (issue #65): a trading day is COMPLETE (trustworthy to score
# or predict from) iff its valid-bar coverage is not materially short of the
# recent norm — count(valid bars on D) >= COMPLETENESS_MIN_FRACTION * median(count
# over up to the COMPLETENESS_MEDIAN_WINDOW dates STRICTLY BEFORE D). This module
# is the SOURCE OF TRUTH for these constants; the store.complete_trading_days view
# (which the scoring queries below and daily_base_rates SELECT from) hardcodes the
# same 0.9 / 20 / 5 with a comment pointing back here. Never score or predict from
# an incomplete day: an in-progress day carries only a provisional pre-market/
# partial bar covering a fraction of the universe, and scoring it counts a day
# that has not finished trading. The window is trailing-only, so the verdict never
# uses future data (a later date can never change an earlier date's verdict).
#
# A date is JUDGED as soon as it has >= COMPLETENESS_MIN_PRIOR_DATES prior trading
# dates, using the median over up to COMPLETENESS_MEDIAN_WINDOW trailing dates
# (whatever exists, >= the minimum). A date with fewer priors (a from-scratch
# backfill / brand-new store younger than the minimum) is treated complete — too
# little history to judge a baseline — but that regime is LOUD, never silent:
# predict._default_signal_date and the routine completeness step WARN when the
# latest day is USED yet unjudgeable, so a young store never silently scores or
# forecasts from a genuinely-provisional edge day (the whole point of #65).
COMPLETENESS_MIN_FRACTION = 0.9
COMPLETENESS_MEDIAN_WINDOW = 20
COMPLETENESS_MIN_PRIOR_DATES = 5


@dataclass
class PickPerformance:
    """Per-day realized outcomes of the top-ranked picks, plus summaries.

    Reach-predictor (Stage B): this app reports prediction QUALITY, never
    trading P&L, so the summaries are reach-rates — no dollar growth. `late`
    marks days whose predictions were NOT created before the target day's
    market open (backfills, and evening-of-target saves); the live reach-rate
    excludes them by default, since an outcome that was already known when the
    prediction was saved is not forecasting skill.
    """

    daily: pd.DataFrame  # target_date, top1_symbol, top1_rank, top1_return,
    # top1_hit, topn_return, topn_hits, n_avail, late

    @property
    def days(self) -> int:
        return len(self.daily)

    @property
    def live(self) -> pd.DataFrame:
        return self.daily[~self.daily["late"]] if self.days else self.daily

    @property
    def late_days(self) -> int:
        return int(self.daily["late"].sum()) if self.days else 0

    def precision_at_1(self, include_late: bool = False) -> float | None:
        frame = self.daily if include_late else self.live
        return float(frame["top1_hit"].mean()) if len(frame) else None


# Trailing windows for the walk-forward reach record, in TRADING days. Shared
# by the dashboard's trailing-window explorer (reach-rate / base rate / lift).
SIM_WINDOW_SPECS = [
    ("1 week", 5),
    ("1 month", 21),
    ("3 months", 63),
    ("6 months", 126),
    ("1 year", 252),
]


def daily_base_rates(con: duckdb.DuckDBPyConnection, dates: list[dt.date]) -> dict[dt.date, float]:
    """Share of ALL stored names REACHING >= 2% intraday (touch) on each COMPLETE date.

    The touch event (scan.touch_event_predicate), epsilon-guarded, same as the
    scoring queries — so the base rate the lift divides by means the same thing
    the picks are scored against. Touching +2% is far more common than closing
    +2%, so this base rate is markedly higher than the close-era one (M1). Dates
    missing from daily_returns are absent from the result — the caller must render
    that as unknown, never as zero. INCOMPLETE dates (a provisional in-progress
    day, per store.complete_trading_days) are likewise absent: their base rate
    is computed over a fraction of the universe and is unreliable, so it is
    returned as unknown, never as a number — consistent with the scoring queries
    refusing to resolve targets onto those days."""
    if not dates:
        return {}
    threshold = DEFAULT_THRESHOLD - _THRESHOLD_EPSILON
    frame = pd.DataFrame({"date": pd.to_datetime(sorted(set(dates)))})
    con.register("base_dates_in", frame)
    rows = con.execute(
        f"""
        SELECT dr.date,
               avg(CASE WHEN {touch_event_predicate("dr.high_return", "dr.high_glitch_suspect")}
                        THEN 1.0 ELSE 0.0 END)
        FROM daily_returns dr
        JOIN complete_trading_days c ON c.date = dr.date
        WHERE dr.date IN (SELECT CAST(date AS DATE) FROM base_dates_in)
        GROUP BY dr.date
        """,
        [threshold],
    ).fetchall()
    con.unregister("base_dates_in")
    return {d: float(rate) for d, rate in rows}


def _created_by_target(
    con: duckdb.DuckDBPyConnection, table: str, id_col: str, id_val: str
) -> dict:
    """target_date -> max(created_ts) for a prediction-shaped table.

    `table`/`id_col` are internal constants ('predictions'/'strategy' or
    'shadow_predictions'/'challenger'), never user input — no injection surface.
    When multiple signal dates resolve to ONE target date (missing intermediate
    bars), the day takes the max created_ts across all of them: any-late means
    late, so a half-backfilled day can never pass as live.

    Target resolution uses complete_trading_days (NOT raw daily_returns), so the
    late-flag lookup keys on the SAME target date the scoring queries resolve —
    otherwise a live day whose next bar is provisional would resolve to a later
    complete day here and get a None (→ wrongly late) created_ts.

    Touch era ONLY (M1): filters event = TOUCH_EVENT so a pre-cutover (NULL-event)
    prediction resolving to the same target date can never bleed its created_ts
    into the touch record's late flags."""
    return dict(
        con.execute(
            f"SELECT target_date, max(created_ts) FROM ("
            f"  SELECT min(d.date) AS target_date, max(pr.created_ts) AS created_ts"
            f"  FROM {table} pr"
            f"  JOIN complete_trading_days d ON d.date > pr.signal_date"
            f"  WHERE pr.{id_col} = ? AND pr.event = '{scan.TOUCH_EVENT}' GROUP BY pr.signal_date"
            f") GROUP BY target_date",
            [id_val],
        ).fetchall()
    )


def _late_from_created(created: dict):
    """THE one live/late definition in the codebase: a prediction is LIVE only
    if created before the target day's open (09:30 ET). Date-granularity
    comparison would count an evening-of-target save — outcome fully known — as
    live. Unknown target dates are late. Both the champion (predictions) and the
    shadow challengers (shadow_predictions) build their `created` map differently
    but score liveness through THIS closure, so the forward record means the same
    thing for both."""
    eastern = ZoneInfo("America/New_York")
    local = dt.datetime.now().astimezone().tzinfo

    def _is_late(td) -> bool:
        created_ts = created.get(pd.Timestamp(td).date())
        if created_ts is None:
            return True
        open_et = dt.datetime.combine(pd.Timestamp(td).date(), dt.time(9, 30), tzinfo=eastern)
        return created_ts.replace(tzinfo=local) > open_et

    return _is_late


def _late_lookup(con: duckdb.DuckDBPyConnection, strategy: str):
    """Callable target_date -> late flag for the champion's predictions table."""
    return _late_from_created(_created_by_target(con, "predictions", "strategy", strategy))


def _shadow_late_lookup(con: duckdb.DuckDBPyConnection, challenger: str):
    """Callable target_date -> late flag for a shadow challenger's picks — the
    SAME 09:30-ET rule as the champion, so backfilled shadow picks are excluded
    from the forward record exactly like the champion's live reach-rate."""
    return _late_from_created(
        _created_by_target(con, "shadow_predictions", "challenger", challenger)
    )


def daily_rank_outcomes(
    con: duckdb.DuckDBPyConnection, strategy: str, top_n: int = 20
) -> pd.DataFrame:
    """Per-day per-rank realized outcomes of the logged predictions (rank <= top_n).

    One row per pick that ACTUALLY TRADED on its target day (a pick with no
    target-day bar — delisted/halted/repaired — is simply absent). Basket
    substitution semantics live with the consumer: taking the first N
    available rows of a day in rank order IS the "trader takes the next name"
    rule — a missing rank 1 makes rank 2 the traded top pick.

    Columns: target_date, rank, oc_return, hit, late (the whole day's
    predictions created at/after that day's 09:30 ET open). hit is the touch
    event; oc_return is kept as the (open-to-close) return magnitude. Touch era
    only (event = TOUCH_EVENT): pre-cutover predictions are archived out (M1)."""
    threshold = DEFAULT_THRESHOLD - _THRESHOLD_EPSILON
    frame = con.execute(
        f"""
        WITH days AS (SELECT date FROM complete_trading_days),  -- gate: complete days only (#65)
        resolved AS (
            SELECT p.signal_date, min(d.date) AS target_date
            FROM (
                SELECT DISTINCT signal_date FROM predictions
                WHERE strategy = ? AND event = '{scan.TOUCH_EVENT}'
            ) p
            JOIN days d ON d.date > p.signal_date GROUP BY p.signal_date
        )
        SELECT r.target_date, pr.rank, dr.oc_return,
               CASE WHEN {touch_event_predicate("dr.high_return", "dr.high_glitch_suspect")}
                    THEN 1 ELSE 0 END AS hit
        FROM predictions pr
        JOIN resolved r ON pr.signal_date = r.signal_date
        JOIN daily_returns dr ON dr.symbol = pr.symbol AND dr.date = r.target_date
        WHERE pr.strategy = ? AND pr.rank <= ? AND pr.event = '{scan.TOUCH_EVENT}'
        ORDER BY r.target_date, pr.rank
        """,
        [strategy, threshold, strategy, top_n],
    ).df()
    if not frame.empty:
        merged = int(frame.duplicated(subset=["target_date", "rank"]).sum())
        if merged:
            logger.warning(
                "rank outcomes for %s: %d duplicate (day, rank) row(s) — multiple "
                "signal dates resolved to one target (missing intermediate bars)",
                strategy,
                merged,
            )
        is_late = _late_lookup(con, strategy)
        frame["late"] = [is_late(td) for td in frame["target_date"]]
    return frame


def daily_pick_performance(
    con: duckdb.DuckDBPyConnection, strategy: str, top_n: int = 5
) -> PickPerformance:
    """Realized open-to-close returns of the logged rank-1 and top-N picks.

    A pick whose target-day bar is missing (delisted/halted/repaired) is
    excluded from that day's basket; `top1` means the best-ranked pick that
    ACTUALLY TRADED (a trader finding rank 1 halted at the open takes the
    next name). n_avail records how many of the top-N traded; days with no
    tradeable picks are dropped from the simulation."""
    threshold = DEFAULT_THRESHOLD - _THRESHOLD_EPSILON
    daily = con.execute(
        f"""
        WITH days AS (SELECT date FROM complete_trading_days),  -- gate: complete days only (#65)
        resolved AS (
            SELECT p.signal_date, min(d.date) AS target_date
            FROM (
                SELECT DISTINCT signal_date FROM predictions
                WHERE strategy = ? AND event = '{scan.TOUCH_EVENT}'
            ) p
            JOIN days d ON d.date > p.signal_date GROUP BY p.signal_date
        ),
        picks AS (
            -- hit is the TOUCH event, computed per row; top1_return / topn_return
            -- top1_return / topn_return keep the open-to-close magnitude as an
            -- archival return; the dashboard shows reach ✓/✗, not a $ figure.
            SELECT r.target_date, pr.rank, pr.symbol, dr.oc_return,
                   CASE WHEN {touch_event_predicate("dr.high_return", "dr.high_glitch_suspect")}
                        THEN 1 ELSE 0 END AS hit
            FROM predictions pr
            JOIN resolved r ON pr.signal_date = r.signal_date
            JOIN daily_returns dr ON dr.symbol = pr.symbol AND dr.date = r.target_date
            WHERE pr.strategy = ? AND pr.rank <= ? AND pr.event = '{scan.TOUCH_EVENT}'
        )
        SELECT
            target_date,
            arg_min(symbol, rank) AS top1_symbol,
            min(rank) AS top1_rank,
            arg_min(oc_return, rank) AS top1_return,
            arg_min(hit, rank) AS top1_hit,
            avg(oc_return) AS topn_return,
            sum(hit) AS topn_hits,
            count(*) AS n_avail
        FROM picks
        GROUP BY target_date
        ORDER BY target_date
        """,
        [strategy, threshold, strategy, top_n],
    ).df()
    if not daily.empty:
        is_late = _late_lookup(con, strategy)
        daily["late"] = [is_late(td) for td in daily["target_date"]]

        substituted = int((daily["top1_rank"] > 1).sum())
        short_days = int((daily["n_avail"] < top_n).sum())
        merged_days = int((daily["n_avail"] > top_n).sum())
        if substituted or short_days:
            logger.warning(
                "pick performance for %s: %d day(s) with rank-1 pick untradeable "
                "(substituted next available — a halted/delisted rank-1 may hide "
                "exactly the catastrophic day), %d day(s) with fewer than %d "
                "tradeable picks",
                strategy,
                substituted,
                short_days,
                top_n,
            )
        if merged_days:
            logger.warning(
                "pick performance for %s: %d day(s) merged picks from multiple "
                "signal dates resolving to one target (missing intermediate bars)",
                strategy,
                merged_days,
            )
    return PickPerformance(daily=daily)


# Trailing window (in LIVE scored days) for the degradation detector.
DEGRADATION_WINDOW = 5
# Comparison rule (M3): DEGRADED iff POOLED EXCESS PRECISION < 0 − 1e-9 — the
# window's pooled hit rate (Σhits / Σpicks) is below its pooled base rate by more
# than FP noise. This REPLACES the old trailing-5 mean-of-daily-lift-RATIOS < 1.0
# rule, which was calibrated for close base rates and false-fires under touch:
# lift is capped at 1/base_rate, so a hot high-base-rate streak (base rate near
# the ceiling) drags a mean of ratios toward 1.0 even while the picks beat the
# base rate in absolute, pooled terms. Averaging ratios across heterogeneous
# base rates is the bug; pooling before dividing is base-rate-invariant.
_EXCESS_DEGRADE_EPSILON = 1e-9


@dataclass
class DegradationVerdict:
    """Outcome of the post-close degradation check over the scored track record."""

    degraded: bool
    armed: bool  # enough live days for the window to be meaningful
    live_days: int
    window: int
    # Pooled over the trailing window: precision = Σhits/Σpicks, base rate is
    # pick-weighted so it sits on the same footing, excess = precision − base.
    # All None until armed. DEGRADED iff pooled_excess_precision < 0 (epsilon).
    pooled_excess_precision: float | None
    pooled_precision: float | None
    pooled_base_rate: float | None
    excluded_null_lift: int
    # Window days individually below the base rate (lift < 1). Pooling has its own
    # false-negative mode (one huge low-base-rate day can lift the pooled precision
    # while most days lose), so the per-day count is always disclosed even when the
    # trigger stays quiet. Before arming, counted over the live days seen so far.
    days_below_1: int
    detail: str


def degradation_verdict(
    scored: pd.DataFrame, window: int = DEGRADATION_WINDOW
) -> DegradationVerdict:
    """Is the champion underperforming its baseline on recent LIVE days?

    Live days only (`late == False`), ordered by target_date: late/backfilled
    days have known outcomes, so including them would let a backfill mask or
    manufacture a degradation. Days with NULL lift (zero base rate — the pooled
    base rate would be undefined for that day) are excluded with a loud warning.
    Fewer than `window` live days means the detector is not yet armed and SAYS
    so — it never silently reports healthy.

    The trip is POOLED EXCESS PRECISION over the window (see _EXCESS_DEGRADE_EPSILON):
    pooled precision Σhits/Σpicks minus the pick-weighted pooled base rate. Under
    touch base rates this is robust where the old mean-of-lift-ratios false-fired.
    """
    excluded = 0
    if scored.empty:
        live = scored
    else:
        ordered = scored.sort_values("target_date")
        live = ordered[~ordered["late"].astype(bool)]
        has_lift = live["lift"].notna()
        excluded = int((~has_lift).sum())
        if excluded:
            logger.warning(
                "degradation detector: excluded %d live day(s) with NULL lift "
                "(zero base rate) from the trailing window",
                excluded,
            )
        live = live[has_lift]
    n = len(live)
    suffix = f"; {excluded} zero-base-rate day(s) excluded" if excluded else ""

    def _below(frame: pd.DataFrame) -> int:
        return (
            int((frame["lift"].astype(float) < 1.0 - _EXCESS_DEGRADE_EPSILON).sum())
            if len(frame)
            else 0
        )

    if n < window:
        return DegradationVerdict(
            degraded=False,
            armed=False,
            live_days=n,
            window=window,
            pooled_excess_precision=None,
            pooled_precision=None,
            pooled_base_rate=None,
            excluded_null_lift=excluded,
            days_below_1=_below(live),
            detail=(
                f"armed after {window - n} more live day(s) — {n}/{window} live days "
                f"scored; detector cannot fire yet{suffix}"
            ),
        )
    win = live.tail(window)
    picks = win["n_scored"].astype(float)
    total_picks = float(picks.sum())
    pooled_precision = float(win["hits"].astype(float).sum() / total_picks)
    # Pick-weighted so the base rate pools on the same footing as precision (both
    # are Σ(numerator)/Σ(picks)); an unweighted mean of daily base rates would
    # reintroduce the averaging-across-heterogeneous-days bias M3 warns about.
    pooled_base_rate = float((win["base_rate"].astype(float) * picks).sum() / total_picks)
    excess = pooled_precision - pooled_base_rate
    below = _below(win)
    degraded = excess < -_EXCESS_DEGRADE_EPSILON
    state = "DEGRADED" if degraded else "not degraded"
    return DegradationVerdict(
        degraded=degraded,
        armed=True,
        live_days=n,
        window=window,
        pooled_excess_precision=excess,
        pooled_precision=pooled_precision,
        pooled_base_rate=pooled_base_rate,
        excluded_null_lift=excluded,
        days_below_1=below,
        detail=(
            f"{state}: trailing-{window} live pooled precision {pooled_precision:.4g} vs "
            f"pooled base {pooled_base_rate:.4g} (excess {excess:+.4g}; DEGRADED when < 0; "
            f"{below} of {window} window day(s) below the base rate){suffix}"
        ),
    )


@dataclass
class TrackRecord:
    scored: pd.DataFrame  # one row per scoreable signal_date; `late` column marks
    # days whose predictions were created at/after the target day's 09:30 ET
    # open (backfills AND evening-of-target saves, per _late_lookup) — they
    # are shown, but must never be read as live forecasting skill
    pending: list  # signal_dates predicted but with no next trading day ingested yet

    @property
    def late_days(self) -> int:
        return int(self.scored["late"].sum()) if not self.scored.empty else 0


def _score_table(
    con: duckdb.DuckDBPyConnection,
    table: str,
    id_col: str,
    id_val: str,
    top_n: int,
    late_lookup,
    signal_dates: list[dt.date],
) -> TrackRecord:
    """Per-day top-N outcome for a prediction-shaped table.

    Shared by the champion (predictions/strategy) and shadow challengers
    (shadow_predictions/challenger) so both compute precision/base-rate/lift
    identically. `table`/`id_col` are internal constants, never user input.

    ONE definition of live in the codebase: `late_lookup` is the 09:30-ET
    any-late rule (_late_from_created), the same flag the live reach-rate uses.
    A plain date-granularity comparison here would count an evening-of-target
    save (outcome fully known) as live and feed the degradation detector weaker
    evidence than the dashboard shows."""
    threshold = DEFAULT_THRESHOLD - _THRESHOLD_EPSILON
    scored = con.execute(
        f"""
        WITH days AS (SELECT date FROM complete_trading_days),  -- gate: complete days only (#65)
        pred_days AS (
            -- Touch era only (M1): pre-cutover NULL-event predictions are archived
            -- out of the live record — a clean reset, never a silent re-score.
            SELECT DISTINCT signal_date FROM {table}
            WHERE {id_col} = ? AND event = '{scan.TOUCH_EVENT}'
        ),
        resolved AS (
            SELECT p.signal_date, min(d.date) AS target_date
            FROM pred_days p JOIN days d ON d.date > p.signal_date
            GROUP BY p.signal_date
        ),
        top_hits AS (
            SELECT r.signal_date, r.target_date,
                   count(*) AS n_scored,
                   sum(CASE WHEN {touch_event_predicate("dr.high_return", "dr.high_glitch_suspect")}
                            THEN 1 ELSE 0 END) AS hits
            FROM {table} pr
            JOIN resolved r ON pr.signal_date = r.signal_date
            JOIN daily_returns dr ON dr.symbol = pr.symbol AND dr.date = r.target_date
            WHERE pr.{id_col} = ? AND pr.rank <= ? AND pr.event = '{scan.TOUCH_EVENT}'
            GROUP BY r.signal_date, r.target_date
        ),
        base AS (
            -- touch base rate over ALL names (the market bar the lift divides by).
            SELECT date,
                   avg(CASE WHEN {touch_event_predicate("high_return", "high_glitch_suspect")}
                            THEN 1.0 ELSE 0.0 END) AS base_rate
            FROM daily_returns GROUP BY date
        )
        SELECT t.signal_date, t.target_date, t.hits, t.n_scored,
               t.hits / t.n_scored AS precision,
               b.base_rate,
               CASE WHEN b.base_rate > 0 THEN (t.hits / t.n_scored) / b.base_rate END AS lift
        FROM top_hits t JOIN base b ON b.date = t.target_date
        ORDER BY t.signal_date
        """,
        [id_val, threshold, id_val, top_n, threshold],
    ).df()

    if not scored.empty:
        scored["late"] = [late_lookup(td) for td in scored["target_date"]]
    resolved = set(pd.to_datetime(scored["signal_date"]).dt.date) if not scored.empty else set()
    pending = [d for d in signal_dates if d not in resolved]
    return TrackRecord(scored=scored, pending=pending)


def score_predictions(
    con: duckdb.DuckDBPyConnection, strategy: str, top_n: int = 20
) -> TrackRecord:
    """Per-day outcome of the top-N logged predictions.

    The target day is the first COMPLETE trading date in the store after each
    signal_date (store.complete_trading_days — a provisional in-progress day is
    NOT a resolvable target, #65). Days whose target isn't a complete trading
    day yet are returned in `pending`, never silently dropped — exactly like a
    signal date with no next day ingested at all: the dashboard shows "Awaiting
    outcomes", never a row scored against a partial pre-market bar.
    """
    return _score_table(
        con,
        "predictions",
        "strategy",
        strategy,
        top_n,
        _late_lookup(con, strategy),
        store.predicted_signal_dates(con, strategy),
    )


def score_shadow_predictions(
    con: duckdb.DuckDBPyConnection, challenger: str, top_n: int = 20
) -> TrackRecord:
    """Per-day outcome of ONE shadow challenger's top-N logged picks.

    The shadow twin of score_predictions, reading shadow_predictions keyed by
    challenger identity and using the same 09:30-ET live/late rule — so the
    challenger's forward record means the same thing as the champion's. Reads
    ONLY shadow_predictions; the champion's track record is untouched."""
    return _score_table(
        con,
        "shadow_predictions",
        "challenger",
        challenger,
        top_n,
        _shadow_late_lookup(con, challenger),
        store.shadow_signal_dates(con, challenger),
    )
