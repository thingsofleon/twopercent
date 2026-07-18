"""Track record: score logged predictions against what actually happened."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd

from twopercent import store
from twopercent.scan import _THRESHOLD_EPSILON, DEFAULT_THRESHOLD

logger = logging.getLogger(__name__)

# Assumed round-trip trading cost (entry + exit) per daily position, applied
# to every simulated day. 30 bps is a deliberate, documented GUESS pitched for
# liquid-ish small caps at open/close; real spreads on thin names can be far
# worse. The simulation is an upper bound on execution quality, not a promise.
COST_ROUND_TRIP = 0.003


@dataclass
class PickPerformance:
    """Per-day realized outcomes of the top-ranked picks, plus summaries.

    `late` marks days whose predictions were NOT created before the target
    day's market open (backfills, and evening-of-target saves) — those days
    are excluded from the headline money metrics by default: a compounded
    dollar figure that includes known outcomes is not forecasting skill.
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

    def growth(self, column: str = "top1_return", include_late: bool = False) -> float | None:
        """Growth of $1 compounding `column` daily, net of COST_ROUND_TRIP.

        Live days only by default — see class docstring."""
        frame = self.daily if include_late else self.live
        if not len(frame):
            return None
        return float((1 + frame[column] - COST_ROUND_TRIP).prod())


# Trailing windows for the simulated walk-forward record, in TRADING days.
SIM_WINDOW_SPECS = [
    ("1 week", 5),
    ("1 month", 21),
    ("3 months", 63),
    ("6 months", 126),
    ("1 year", 252),
]


@dataclass
class SimWindows:
    """Trailing-window summary of a benchmark's per-rank sim record for one basket.

    `days_available` is always reported, even when every window is omitted,
    so callers can say "N sim days available" instead of silently showing
    nothing (a window is NEVER computed on a shorter span than requested).
    """

    days_available: int
    basket: int
    windows: list[dict]  # label, days, growth, hit_rate, short_days


def sim_windows(daily: pd.DataFrame, n: int) -> SimWindows:
    """Growth of $1 and hit rates of the top-`n` basket over trailing windows.

    CONTIGUOUS RANKS ONLY: baskets by `rank <= n`, which equals the
    dashboard's first-n-available rule only when each day's ranks are
    1..k with no gaps — true for experiment_daily rows by construction.
    Never feed live-style frames (missing ranks) here; use the dashboard
    summarizer, whose first-n rule IS the substitution semantics.

    `daily` is a per-rank experiment_daily frame (ordered by target_date,
    rank) with columns target_date, rank, ret, hit. Each day's basket return
    is the mean ret of ranks <= n present that day; days with fewer than n
    ranks use what exists and are counted in the window's `short_days` (the
    caller must disclose them — partial coverage is never silent). Growth
    compounds daily net of COST_ROUND_TRIP, same formula as
    PickPerformance.growth. Hits were computed at benchmark time with the
    epsilon-guarded threshold — no re-derivation here. Windows longer than
    the available history are omitted. A non-finite ret/hit raises: skipna
    aggregation would silently compound a shorter window than claimed.
    """
    if len(daily):
        corrupt = int(
            (~np.isfinite(daily["ret"].astype(float))).sum()
            + (~np.isfinite(daily["hit"].astype(float))).sum()
        )
        if corrupt:
            raise ValueError(
                f"sim daily frame has {corrupt} non-finite ret/hit value(s) — "
                "refusing to summarize around corrupt rows"
            )
    basket = daily[daily["rank"] <= n]
    per_day = basket.groupby("target_date").agg(
        ret=("ret", "mean"), hit=("hit", "mean"), picks=("rank", "count")
    )
    windows: list[dict] = []
    for label, w in SIM_WINDOW_SPECS:
        if len(per_day) < w:
            continue
        tail = per_day.iloc[-w:]
        windows.append(
            {
                "label": label,
                "days": w,
                "growth": float((1 + tail["ret"] - COST_ROUND_TRIP).prod()),
                "hit_rate": float(tail["hit"].mean()),
                "short_days": int((tail["picks"] < n).sum()),
            }
        )
    return SimWindows(days_available=len(per_day), basket=n, windows=windows)


def daily_base_rates(con: duckdb.DuckDBPyConnection, dates: list[dt.date]) -> dict[dt.date, float]:
    """Share of ALL stored names doing >= 2% open-to-close on each date.

    The epsilon-guarded threshold, same as the scoring queries. Dates missing
    from daily_returns are absent from the result — the caller must render
    that as unknown, never as zero."""
    if not dates:
        return {}
    threshold = DEFAULT_THRESHOLD - _THRESHOLD_EPSILON
    frame = pd.DataFrame({"date": pd.to_datetime(sorted(set(dates)))})
    con.register("base_dates_in", frame)
    rows = con.execute(
        """
        SELECT date, avg(CASE WHEN oc_return >= ? THEN 1.0 ELSE 0.0 END)
        FROM daily_returns
        WHERE date IN (SELECT CAST(date AS DATE) FROM base_dates_in)
        GROUP BY date
        """,
        [threshold],
    ).fetchall()
    con.unregister("base_dates_in")
    return {d: float(rate) for d, rate in rows}


def _late_lookup(con: duckdb.DuckDBPyConnection, strategy: str):
    """Callable target_date -> late flag, shared by every scorer.

    A prediction is LIVE only if created before the target day's open
    (09:30 ET). Date-granularity comparison would count an evening-of-target
    save — outcome fully known — as live. When multiple signal dates resolve
    to ONE target date (missing intermediate bars), the day takes the max
    created_ts across all of them: any-late means late, so a half-backfilled
    day can never pass as live. Unknown target dates are late."""
    created = dict(
        con.execute(
            "SELECT target_date, max(created_ts) FROM ("
            "  SELECT min(d.date) AS target_date, max(pr.created_ts) AS created_ts"
            "  FROM predictions pr"
            "  JOIN (SELECT DISTINCT date FROM daily_returns) d ON d.date > pr.signal_date"
            "  WHERE pr.strategy = ? GROUP BY pr.signal_date"
            ") GROUP BY target_date",
            [strategy],
        ).fetchall()
    )
    eastern = ZoneInfo("America/New_York")
    local = dt.datetime.now().astimezone().tzinfo

    def _is_late(td) -> bool:
        created_ts = created.get(pd.Timestamp(td).date())
        if created_ts is None:
            return True
        open_et = dt.datetime.combine(pd.Timestamp(td).date(), dt.time(9, 30), tzinfo=eastern)
        return created_ts.replace(tzinfo=local) > open_et

    return _is_late


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
    predictions created at/after that day's 09:30 ET open)."""
    threshold = DEFAULT_THRESHOLD - _THRESHOLD_EPSILON
    frame = con.execute(
        """
        WITH days AS (SELECT DISTINCT date FROM daily_returns),
        resolved AS (
            SELECT p.signal_date, min(d.date) AS target_date
            FROM (SELECT DISTINCT signal_date FROM predictions WHERE strategy = ?) p
            JOIN days d ON d.date > p.signal_date GROUP BY p.signal_date
        )
        SELECT r.target_date, pr.rank, dr.oc_return,
               CASE WHEN dr.oc_return >= ? THEN 1 ELSE 0 END AS hit
        FROM predictions pr
        JOIN resolved r ON pr.signal_date = r.signal_date
        JOIN daily_returns dr ON dr.symbol = pr.symbol AND dr.date = r.target_date
        WHERE pr.strategy = ? AND pr.rank <= ?
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
        """
        WITH days AS (SELECT DISTINCT date FROM daily_returns),
        resolved AS (
            SELECT p.signal_date, min(d.date) AS target_date
            FROM (SELECT DISTINCT signal_date FROM predictions WHERE strategy = ?) p
            JOIN days d ON d.date > p.signal_date GROUP BY p.signal_date
        ),
        picks AS (
            SELECT r.target_date, pr.rank, pr.symbol, dr.oc_return
            FROM predictions pr
            JOIN resolved r ON pr.signal_date = r.signal_date
            JOIN daily_returns dr ON dr.symbol = pr.symbol AND dr.date = r.target_date
            WHERE pr.strategy = ? AND pr.rank <= ?
        )
        SELECT
            target_date,
            arg_min(symbol, rank) AS top1_symbol,
            min(rank) AS top1_rank,
            arg_min(oc_return, rank) AS top1_return,
            CASE WHEN arg_min(oc_return, rank) >= ? THEN 1 ELSE 0 END AS top1_hit,
            avg(oc_return) AS topn_return,
            sum(CASE WHEN oc_return >= ? THEN 1 ELSE 0 END) AS topn_hits,
            count(*) AS n_avail
        FROM picks
        GROUP BY target_date
        ORDER BY target_date
        """,
        [strategy, strategy, top_n, threshold, threshold],
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
# Comparison rule: DEGRADED iff mean lift < 1.0 − 1e-9 — strictly below the
# baseline by more than FP noise. A mean within epsilon of 1.0 (or above)
# is NOT degraded, so the detector never fires on rounding error alone.
_LIFT_DEGRADE_EPSILON = 1e-9


@dataclass
class DegradationVerdict:
    """Outcome of the post-close degradation check over the scored track record."""

    degraded: bool
    armed: bool  # enough live days for the window to be meaningful
    live_days: int
    window: int
    trailing_mean_lift: float | None  # None until armed
    excluded_null_lift: int
    detail: str


def degradation_verdict(
    scored: pd.DataFrame, window: int = DEGRADATION_WINDOW
) -> DegradationVerdict:
    """Is the champion underperforming its baseline on recent LIVE days?

    Live days only (`late == False`), ordered by target_date: late/backfilled
    days have known outcomes, so including them would let a backfill mask or
    manufacture a degradation. Days with NULL lift (zero base rate — lift is
    undefined, not zero) are excluded with a loud warning. Fewer than `window`
    live days means the detector is not yet armed and SAYS so — it never
    silently reports healthy.
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
    if n < window:
        so_far = f", mean live lift so far {float(live['lift'].mean()):.7g}" if n else ""
        return DegradationVerdict(
            degraded=False,
            armed=False,
            live_days=n,
            window=window,
            trailing_mean_lift=None,
            excluded_null_lift=excluded,
            detail=(
                f"armed after {window - n} more live day(s) — {n}/{window} live days "
                f"scored{so_far}; detector cannot fire yet{suffix}"
            ),
        )
    mean_lift = float(live["lift"].tail(window).astype(float).mean())
    degraded = mean_lift < 1.0 - _LIFT_DEGRADE_EPSILON
    state = "DEGRADED" if degraded else "not degraded"
    return DegradationVerdict(
        degraded=degraded,
        armed=True,
        live_days=n,
        window=window,
        trailing_mean_lift=mean_lift,
        excluded_null_lift=excluded,
        detail=(
            f"{state}: trailing-{window} live mean lift {mean_lift:.7g} vs baseline 1.0{suffix}"
        ),
    )


@dataclass
class TrackRecord:
    scored: pd.DataFrame  # one row per scoreable signal_date; `late` column marks
    # days whose predictions were logged AFTER the target date (backfills) —
    # they are shown, but must never be read as live forecasting skill
    pending: list  # signal_dates predicted but with no next trading day ingested yet

    @property
    def late_days(self) -> int:
        return int(self.scored["late"].sum()) if not self.scored.empty else 0


def score_predictions(
    con: duckdb.DuckDBPyConnection, strategy: str, top_n: int = 20
) -> TrackRecord:
    """Per-day outcome of the top-N logged predictions.

    The target day is the first trading date in the store after each
    signal_date. Days whose target isn't ingested yet are returned in
    `pending`, never silently dropped.
    """
    threshold = DEFAULT_THRESHOLD - _THRESHOLD_EPSILON
    scored = con.execute(
        """
        WITH days AS (SELECT DISTINCT date FROM daily_returns),
        pred_days AS (
            SELECT DISTINCT signal_date FROM predictions WHERE strategy = ?
        ),
        resolved AS (
            SELECT p.signal_date, min(d.date) AS target_date
            FROM pred_days p JOIN days d ON d.date > p.signal_date
            GROUP BY p.signal_date
        ),
        top_hits AS (
            SELECT r.signal_date, r.target_date,
                   count(*) AS n_scored,
                   sum(CASE WHEN dr.oc_return >= ? THEN 1 ELSE 0 END) AS hits
            FROM predictions pr
            JOIN resolved r ON pr.signal_date = r.signal_date
            JOIN daily_returns dr ON dr.symbol = pr.symbol AND dr.date = r.target_date
            WHERE pr.strategy = ? AND pr.rank <= ?
            GROUP BY r.signal_date, r.target_date
        ),
        base AS (
            SELECT date, avg(CASE WHEN oc_return >= ? THEN 1.0 ELSE 0.0 END) AS base_rate
            FROM daily_returns GROUP BY date
        )
        SELECT t.signal_date, t.target_date, t.hits, t.n_scored,
               t.hits / t.n_scored AS precision,
               b.base_rate,
               CASE WHEN b.base_rate > 0 THEN (t.hits / t.n_scored) / b.base_rate END AS lift
        FROM top_hits t JOIN base b ON b.date = t.target_date
        ORDER BY t.signal_date
        """,
        [strategy, threshold, strategy, top_n, threshold],
    ).df()

    if not scored.empty:
        created = dict(
            con.execute(
                "SELECT signal_date, max(created_ts) FROM predictions "
                "WHERE strategy = ? GROUP BY signal_date",
                [strategy],
            ).fetchall()
        )
        scored["late"] = [
            created[pd.Timestamp(sd).date()].date() > pd.Timestamp(td).date()
            for sd, td in zip(scored["signal_date"], scored["target_date"], strict=True)
        ]
    resolved = set(pd.to_datetime(scored["signal_date"]).dt.date) if not scored.empty else set()
    pending = [d for d in store.predicted_signal_dates(con, strategy) if d not in resolved]
    return TrackRecord(scored=scored, pending=pending)
