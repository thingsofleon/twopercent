"""Track record: score logged predictions against what actually happened."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd

from twopercent import store
from twopercent.scan import _THRESHOLD_EPSILON, DEFAULT_THRESHOLD


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
