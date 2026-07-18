"""Track record: score logged predictions against what actually happened."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import duckdb
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
    """Trailing-window summary of a benchmark's per-day sim record.

    `days_available` is always reported, even when every window is omitted,
    so callers can say "N sim days available" instead of silently showing
    nothing (a window is NEVER computed on a shorter span than requested).
    """

    days_available: int
    windows: list[dict]  # label, days, top1_growth, top5_growth,
    # top1_hit_rate, top5_hit_rate


def sim_windows(daily: pd.DataFrame) -> SimWindows:
    """Growth of $1 and hit rates over standard trailing windows of sim days.

    `daily` is an experiment_daily frame (target_date-ordered) with columns
    top1_ret, top1_hit, top5_ret, top5_hits. Growth compounds each day net of
    COST_ROUND_TRIP, same formula as PickPerformance.growth. Hits were
    computed at benchmark time with the epsilon-guarded threshold — no
    re-derivation here. Windows longer than the available history are omitted.
    """
    windows: list[dict] = []
    for label, n in SIM_WINDOW_SPECS:
        if len(daily) < n:
            continue
        tail = daily.iloc[-n:]
        windows.append(
            {
                "label": label,
                "days": n,
                "top1_growth": float((1 + tail["top1_ret"] - COST_ROUND_TRIP).prod()),
                "top5_growth": float((1 + tail["top5_ret"] - COST_ROUND_TRIP).prod()),
                "top1_hit_rate": float(tail["top1_hit"].mean()),
                "top5_hit_rate": float(tail["top5_hits"].mean()),
            }
        )
    return SimWindows(days_available=len(daily), windows=windows)


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
        created = dict(
            con.execute(
                "SELECT min(d.date), max(pr.created_ts) FROM predictions pr "
                "JOIN (SELECT DISTINCT date FROM daily_returns) d ON d.date > pr.signal_date "
                "WHERE pr.strategy = ? GROUP BY pr.signal_date",
                [strategy],
            ).fetchall()
        )
        # A prediction is LIVE only if created before the target day's open
        # (09:30 ET). Date-granularity comparison would count an
        # evening-of-target save — outcome fully known — as live.
        eastern = ZoneInfo("America/New_York")
        local = dt.datetime.now().astimezone().tzinfo

        def _is_late(td) -> bool:
            created_ts = created.get(pd.Timestamp(td).date())
            if created_ts is None:
                return True
            open_et = dt.datetime.combine(pd.Timestamp(td).date(), dt.time(9, 30), tzinfo=eastern)
            return created_ts.replace(tzinfo=local) > open_et

        daily["late"] = [_is_late(td) for td in daily["target_date"]]

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
