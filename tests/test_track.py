import datetime as dt
import logging

import pandas as pd

from tests.conftest import seed_history
from twopercent import store, track

# 25 business days from 2026-01-05; last day is 2026-02-06.
OC = {
    "HIT1": [0.001] * 23 + [0.03, 0.05],  # 2% on the last two days
    "HIT2": [0.001] * 23 + [0.001, 0.021],  # 2% on the last day only
    "MISS": [0.001] * 25,  # never
}


def _seed(con):
    seed_history(con, OC)
    dates = sorted(pd.bdate_range("2026-01-05", periods=25).date)
    return dates


def _save(con, signal_date, ranked):
    store.save_predictions(
        con,
        "test_strat",
        signal_date,
        pd.DataFrame({"symbol": ranked, "prob": [0.9, 0.5, 0.1], "rank": [1, 2, 3]}),
    )


def test_scoring_math_and_target_resolution(con):
    dates = _seed(con)
    # Predict on the second-to-last day; outcome day is the last day, where
    # HIT2 (rank 1) and HIT1 (rank 2) moved 2%+ and MISS didn't.
    _save(con, dates[-2], ["HIT2", "HIT1", "MISS"])

    record = track.score_predictions(con, "test_strat", top_n=3)
    assert len(record.scored) == 1
    row = record.scored.iloc[0]
    assert pd.Timestamp(row["target_date"]).date() == dates[-1]
    assert row["hits"] == 2 and row["n_scored"] == 3
    assert abs(row["precision"] - 2 / 3) < 1e-12
    assert abs(row["base_rate"] - 2 / 3) < 1e-12  # 2 of 3 symbols moved that day
    assert record.pending == []


def test_top_n_restricts_scoring(con):
    dates = _seed(con)
    _save(con, dates[-2], ["MISS", "HIT1", "HIT2"])  # rank 1 is the miss
    record = track.score_predictions(con, "test_strat", top_n=1)
    row = record.scored.iloc[0]
    assert row["hits"] == 0 and row["n_scored"] == 1


def test_unscoreable_day_is_pending_not_dropped(con):
    dates = _seed(con)
    _save(con, dates[-1], ["HIT1", "HIT2", "MISS"])  # no next trading day ingested
    record = track.score_predictions(con, "test_strat", top_n=3)
    assert record.scored.empty
    assert record.pending == [dates[-1]]


def test_save_predictions_idempotent(con):
    _seed(con)
    day = dt.date(2026, 2, 5)
    _save(con, day, ["HIT1", "HIT2", "MISS"])
    _save(con, day, ["HIT1", "HIT2", "MISS"])
    count = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert count == 3


def _sim_daily(rets_by_day: list[list[float]], hits_by_day: list[list[int]]) -> pd.DataFrame:
    """Per-rank sim frame: day i gets ranks 1..len(rets_by_day[i])."""
    dates = sorted(pd.bdate_range("2026-01-05", periods=len(rets_by_day)).date)
    rows = [
        {"target_date": d, "rank": rank, "ret": ret, "hit": hit}
        for d, rets, hits in zip(dates, rets_by_day, hits_by_day, strict=True)
        for rank, (ret, hit) in enumerate(zip(rets, hits, strict=True), start=1)
    ]
    return pd.DataFrame(rows)


def test_sim_windows_top1_exact_week_math():
    # Adversarial (non-round) rank-1 returns; hits as stored at benchmark time.
    daily = _sim_daily(
        rets_by_day=[[0.0203, 0.9], [-0.0117, 0.9], [0.0329, 0.9], [0.004, 0.9], [-0.0009, 0.9]],
        hits_by_day=[[1, 1], [0, 1], [1, 1], [0, 1], [0, 1]],
    )
    summary = track.sim_windows(daily, n=1)
    assert summary.days_available == 5
    assert summary.basket == 1
    assert [w["label"] for w in summary.windows] == ["1 week"]
    week = summary.windows[0]
    assert week["days"] == 5
    # Hand-derived per-day multipliers: 1 + ret − COST_ROUND_TRIP (0.003).
    # Rank 2 (the loud 0.9 returns) must not leak into a top-1 basket.
    expected = 1.0173 * 0.9853 * 1.0299 * 1.001 * 0.9961
    assert abs(week["growth"] - expected) < 1e-9
    assert abs(week["hit_rate"] - 2 / 5) < 1e-12
    assert week["short_days"] == 0


def test_sim_windows_basket_mean_and_short_day():
    # 5 days of 10 ranks each — except day 3 has only 6 picks. A top-10 basket
    # on that day averages the 6 that exist and the day is counted short.
    full = [0.0203, -0.0117, 0.0329, 0.004, -0.0009, 0.0251, 0.0008, -0.0301, 0.0107, 0.0022]
    short = full[:6]
    rets = [full, full, short, full, full]
    hits = [[1 if r >= 0.02 else 0 for r in day] for day in rets]
    daily = _sim_daily(rets, hits)

    summary = track.sim_windows(daily, n=10)
    assert summary.days_available == 5
    week = summary.windows[0]
    assert week["short_days"] == 1
    full_mean = sum(full) / 10
    short_mean = sum(short) / 6
    expected = (1 + full_mean - track.COST_ROUND_TRIP) ** 4 * (
        1 + short_mean - track.COST_ROUND_TRIP
    )
    assert abs(week["growth"] - expected) < 1e-9
    # Hit rate: mean of day fractions — 3/10 on full days, 3/6 on the short day
    # (0.0203, 0.0329, 0.0251 clear 2% among its six picks).
    assert abs(week["hit_rate"] - (4 * (3 / 10) + 3 / 6) / 5) < 1e-12


def test_sim_windows_too_few_days_omits_but_reports_count():
    daily = _sim_daily(
        rets_by_day=[[0.0203], [-0.0117], [0.0329], [0.004]],
        hits_by_day=[[1], [0], [1], [0]],
    )
    summary = track.sim_windows(daily, n=1)
    assert summary.windows == []  # 4 < 5: no window computed on a shorter span
    assert summary.days_available == 4  # ...but the available count is still reported


def test_sim_windows_use_trailing_days_only():
    # 21 days: 16 loud early days, then 5 quiet ones. The 1-week window must
    # reflect ONLY the trailing 5, or it is silently a different window.
    rets = [[0.0517]] * 16 + [[0.0007]] * 5
    hits = [[1]] * 16 + [[0]] * 5
    summary = track.sim_windows(_sim_daily(rets, hits), n=1)
    assert summary.days_available == 21
    assert [w["label"] for w in summary.windows] == ["1 week", "1 month"]
    week, month = summary.windows
    assert abs(week["growth"] - (1 + 0.0007 - track.COST_ROUND_TRIP) ** 5) < 1e-9
    assert week["hit_rate"] == 0.0
    assert week["growth"] < 1 < month["growth"]
    assert abs(month["hit_rate"] - 16 / 21) < 1e-12


def test_sim_windows_refuses_non_finite_rows():
    import numpy as np
    import pytest

    daily = _sim_daily(
        rets_by_day=[[0.0203], [-0.0117], [0.0329], [0.004], [0.001]],
        hits_by_day=[[1], [0], [1], [0], [0]],
    )
    daily.loc[2, "ret"] = np.nan  # bypasses the writer guard on purpose
    # skipna math would report a clean-looking 5-day window compounding 4 days.
    with pytest.raises(ValueError, match="non-finite"):
        track.sim_windows(daily, n=1)


def test_daily_rank_outcomes_rows_and_hits(con):
    dates = _seed(con)
    _save(con, dates[-2], ["HIT2", "HIT1", "MISS"])
    con.execute(
        "UPDATE predictions SET created_ts = ? WHERE strategy = 'test_strat'",
        [dt.datetime.combine(dates[-1], dt.time(6, 0))],  # before 09:30 ET open → live
    )
    frame = track.daily_rank_outcomes(con, "test_strat")
    assert len(frame) == 3
    assert frame["rank"].tolist() == [1, 2, 3]
    assert frame["hit"].tolist() == [1, 1, 0]  # HIT2 +2.1%, HIT1 +5%, MISS +0.1%
    assert abs(frame["oc_return"].iloc[0] - 0.021) < 1e-12
    assert not frame["late"].any()
    assert all(pd.Timestamp(d).date() == dates[-1] for d in frame["target_date"])


def test_daily_rank_outcomes_missing_rank_is_absent_not_phantom(con):
    # Rank 1 never traded (no bars at all): the frame starts at rank 2, and a
    # consumer taking the first N available rows gets the substituted basket.
    dates = _seed(con)
    store.save_predictions(
        con,
        "test_strat",
        dates[-2],
        pd.DataFrame(
            {"symbol": ["GONE", "HIT1", "MISS"], "prob": [0.9, 0.5, 0.1], "rank": [1, 2, 3]}
        ),
    )
    frame = track.daily_rank_outcomes(con, "test_strat")
    assert frame["rank"].tolist() == [2, 3]
    assert frame["hit"].tolist() == [1, 0]


def test_late_flag_any_late_wins_on_merged_target_days(con):
    # Friday and Saturday signals both resolve to Monday (weekend gap). The
    # Friday save was live; the Saturday one is a backfill. A day is live only
    # if EVERY prediction for it beat the open — the merged day must be late,
    # regardless of which signal date the lookup happens to see last.
    _seed(con)
    fri, sat, mon = dt.date(2026, 1, 9), dt.date(2026, 1, 10), dt.date(2026, 1, 12)
    _save(con, fri, ["HIT1", "HIT2", "MISS"])
    _save(con, sat, ["HIT2", "HIT1", "MISS"])
    con.execute(
        "UPDATE predictions SET created_ts = ? WHERE strategy = 'test_strat' AND signal_date = ?",
        [dt.datetime.combine(mon, dt.time(6, 0)), fri],  # before Monday's 09:30 ET open
    )
    frame = track.daily_rank_outcomes(con, "test_strat")
    merged = frame[[pd.Timestamp(d).date() == mon for d in frame["target_date"]]]
    assert len(merged) == 6  # both signal dates' picks landed on Monday
    assert merged["late"].all()  # half-backfilled day can never pass as live


def test_daily_rank_outcomes_late_flag_on_backfill(con):
    dates = _seed(con)
    _save(con, dates[-2], ["HIT2", "HIT1", "MISS"])  # created now, target long past
    frame = track.daily_rank_outcomes(con, "test_strat")
    assert frame["late"].all()


def test_save_predictions_rerun_replaces_whole_slice(con):
    # A re-run that scores FEWER symbols (liquidity floor kicked one out) must
    # not leave the dropped symbol behind as a phantom rank from the first save.
    day = dt.date(2026, 2, 5)

    def ranked(symbols: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": symbols,
                "prob": [1 - i / 100 for i in range(len(symbols))],
                "rank": range(1, len(symbols) + 1),
            }
        )

    twenty = [f"S{i:02d}" for i in range(20)]
    store.save_predictions(con, "s", day, ranked(twenty))
    store.save_predictions(con, "s", day, ranked(twenty[1:]))  # S00 now excluded

    rows = con.execute(
        "SELECT symbol, rank FROM predictions WHERE strategy = 's' AND signal_date = ? "
        "ORDER BY rank",
        [day],
    ).df()
    assert len(rows) == 19
    assert "S00" not in set(rows["symbol"])
    assert rows["rank"].tolist() == list(range(1, 20))  # contiguous, no phantom rank 1


# --- degradation detector -----------------------------------------------------


def _scored_frame(lifts, late=None, start="2026-06-01"):
    """Craft a score_predictions-shaped frame with the given lifts, in
    target_date order (NaN lift = zero-base-rate day)."""
    n = len(lifts)
    dates = pd.bdate_range(start, periods=n)
    return pd.DataFrame(
        {
            "signal_date": dates - pd.tseries.offsets.BDay(1),
            "target_date": dates,
            "hits": [1] * n,
            "n_scored": [20] * n,
            "precision": [0.05] * n,
            "base_rate": [0.10] * n,
            "lift": lifts,
            "late": [False] * n if late is None else late,
        }
    )


def test_detector_fires_at_exactly_five_live_days():
    verdict = track.degradation_verdict(_scored_frame([0.99] * 5))
    assert verdict.degraded and verdict.armed
    assert verdict.live_days == 5
    assert abs(verdict.trailing_mean_lift - 0.99) < 1e-12
    assert "DEGRADED" in verdict.detail


def test_detector_epsilon_boundary_adversarial():
    # Comparison rule (track._LIFT_DEGRADE_EPSILON): DEGRADED iff
    # mean lift < 1.0 - 1e-9. A mean 1e-7 below 1.0 fires (far outside FP
    # noise); a mean above never does; exactly 1.0 or within epsilon below
    # resolves to NOT degraded — the detector never fires on rounding error.
    assert track.degradation_verdict(_scored_frame([0.9999999] * 5)).degraded
    assert not track.degradation_verdict(_scored_frame([1.0000001] * 5)).degraded
    assert not track.degradation_verdict(_scored_frame([1.0] * 5)).degraded
    assert not track.degradation_verdict(_scored_frame([1.0 - 1e-12] * 5)).degraded
    # Pin the guard band itself: 2e-9 below 1.0 is strictly outside the 1e-9
    # epsilon and fires; 0.5e-9 below is inside the band and must not.
    assert track.degradation_verdict(_scored_frame([1.0 - 2e-9] * 5)).degraded
    assert not track.degradation_verdict(_scored_frame([1.0 - 0.5e-9] * 5)).degraded


def test_detector_not_armed_below_five_live_days():
    verdict = track.degradation_verdict(_scored_frame([0.5] * 4))
    assert not verdict.degraded and not verdict.armed
    assert verdict.live_days == 4
    assert verdict.trailing_mean_lift is None
    assert "armed after 1 more live day" in verdict.detail  # loud, never silent


def test_detector_excludes_late_days_from_window():
    # 5 awful live days then 3 stellar LATE days (most recent) — a backfill
    # with known outcomes must never mask a live degradation.
    verdict = track.degradation_verdict(
        _scored_frame([0.5] * 5 + [5.0] * 3, late=[False] * 5 + [True] * 3)
    )
    assert verdict.degraded and verdict.live_days == 5

    # And late days never count TOWARD arming either: 3 live + 4 late = unarmed.
    verdict = track.degradation_verdict(
        _scored_frame(
            [0.5, 5.0, 0.5, 5.0, 0.5, 5.0, 5.0],
            late=[False, True, False, True, False, True, True],
        )
    )
    assert not verdict.armed and verdict.live_days == 3


def test_detector_discloses_days_below_one_when_mean_survives():
    # False-negative mode of the mean: one lucky low-base-rate spike day
    # (lift 5.0) carries four zero days to a mean of exactly 1.0. The locked
    # trigger stays quiet — but the per-day count must not let it hide.
    verdict = track.degradation_verdict(_scored_frame([0.0, 0.0, 0.0, 0.0, 5.0]))
    assert not verdict.degraded and verdict.armed
    assert verdict.days_below_1 == 4
    assert "4 of 5 window day(s) below 1.0" in verdict.detail


def test_detector_excludes_null_lift_days_with_warning(caplog):
    # Most recent live day has NULL lift (zero base rate): excluded loudly,
    # the window falls back to the 5 defined-lift days.
    frame = _scored_frame([0.5] * 5 + [float("nan")])
    with caplog.at_level(logging.WARNING, logger="twopercent.track"):
        verdict = track.degradation_verdict(frame)
    assert verdict.degraded and verdict.live_days == 5
    assert verdict.excluded_null_lift == 1
    assert "NULL lift" in caplog.text
    assert "zero-base-rate" in verdict.detail


def test_detector_empty_track_record_reports_unarmed():
    verdict = track.degradation_verdict(pd.DataFrame())
    assert not verdict.degraded and not verdict.armed
    assert verdict.live_days == 0
    assert "armed after 5 more live day" in verdict.detail
