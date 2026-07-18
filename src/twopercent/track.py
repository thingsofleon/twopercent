"""Track record: score logged predictions against what actually happened."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from twopercent import store
from twopercent.scan import _THRESHOLD_EPSILON, DEFAULT_THRESHOLD

# Assumed round-trip trading cost (entry + exit) per daily position, applied
# to every simulated day. 30 bps is a deliberate, documented GUESS pitched for
# liquid-ish small caps at open/close; real spreads on thin names can be far
# worse. The simulation is an upper bound on execution quality, not a promise.
COST_ROUND_TRIP = 0.003


@dataclass
class PickPerformance:
    """Per-day realized outcomes of the top-ranked picks, plus summaries."""

    daily: pd.DataFrame  # target_date, top1_symbol, top1_return, top1_hit,
    # topn_return (equal-weight over available ranks), topn_hits, n_avail, late

    @property
    def days(self) -> int:
        return len(self.daily)

    def precision_at_1(self) -> float | None:
        return float(self.daily["top1_hit"].mean()) if self.days else None

    def growth(self, column: str = "top1_return") -> float | None:
        """Growth of $1 compounding `column` daily, net of COST_ROUND_TRIP."""
        if not self.days:
            return None
        return float((1 + self.daily[column] - COST_ROUND_TRIP).prod())


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
        daily["late"] = [
            created.get(pd.Timestamp(td).date(), pd.Timestamp.min).date() > pd.Timestamp(td).date()
            for td in daily["target_date"]
        ]
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
