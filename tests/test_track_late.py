import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd

from tests.conftest import seed_history
from twopercent import store, track

OC = {"AAA": [0.001] * 23 + [0.03, 0.05], "BBB": [0.001] * 25}

_EASTERN = ZoneInfo("America/New_York")


def _save_last_signal(con) -> dt.date:
    """Seed history, predict on the second-to-last day, return the target day."""
    seed_history(con, OC)
    dates = sorted(pd.bdate_range("2026-01-05", periods=25).date)
    store.save_predictions(
        con,
        "s",
        dates[-2],
        pd.DataFrame({"symbol": ["AAA", "BBB"], "prob": [0.9, 0.1], "rank": [1, 2]}),
    )
    return dates[-1]


def _set_created(con, when_et: dt.datetime) -> None:
    """Pin created_ts to an ET wall time, stored the way save_predictions
    stores it: naive local (converted with the same fixed local offset
    _late_lookup uses, so the round trip is exact)."""
    local = dt.datetime.now().astimezone().tzinfo
    con.execute(
        "UPDATE predictions SET created_ts = ?",
        [when_et.astimezone(local).replace(tzinfo=None)],
    )


def test_backfilled_predictions_flagged_late(con):
    # Saving NOW for a long-past signal date = a backfill: created_ts is after
    # the target date, so the day must carry late=True.
    _save_last_signal(con)
    record = track.score_predictions(con, "s", top_n=2)
    assert len(record.scored) == 1
    assert bool(record.scored["late"].iloc[0]) is True
    assert record.late_days == 1


def test_evening_of_target_save_is_late(con):
    # Created 23:59 ET ON the target date: the outcome was fully known, so
    # the day is LATE. A date-granularity comparison (created.date() >
    # target.date()) would have called this live — score_predictions must use
    # the same 09:30-ET rule (_late_lookup) as the money metrics.
    target = _save_last_signal(con)
    _set_created(con, dt.datetime.combine(target, dt.time(23, 59), tzinfo=_EASTERN))
    record = track.score_predictions(con, "s", top_n=2)
    assert bool(record.scored["late"].iloc[0]) is True
    assert record.late_days == 1


def test_pre_open_save_on_target_day_is_live(con):
    # Created 09:00 ET on the target day — before that day's open, outcome
    # unknown: live.
    target = _save_last_signal(con)
    _set_created(con, dt.datetime.combine(target, dt.time(9, 0), tzinfo=_EASTERN))
    record = track.score_predictions(con, "s", top_n=2)
    assert bool(record.scored["late"].iloc[0]) is False
    assert record.late_days == 0
