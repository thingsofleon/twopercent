import pandas as pd

from tests.conftest import seed_history
from twopercent import store, track

OC = {"AAA": [0.001] * 23 + [0.03, 0.05], "BBB": [0.001] * 25}


def test_backfilled_predictions_flagged_late(con):
    seed_history(con, OC)
    dates = sorted(pd.bdate_range("2026-01-05", periods=25).date)
    # Saving NOW for a long-past signal date = a backfill: created_ts is after
    # the target date, so the day must carry late=True.
    store.save_predictions(
        con,
        "s",
        dates[-2],
        pd.DataFrame({"symbol": ["AAA", "BBB"], "prob": [0.9, 0.1], "rank": [1, 2]}),
    )
    record = track.score_predictions(con, "s", top_n=2)
    assert len(record.scored) == 1
    assert bool(record.scored["late"].iloc[0]) is True
    assert record.late_days == 1
