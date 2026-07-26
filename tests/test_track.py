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


def test_pick_performance_excludes_ohlc_corrupt_rank1_and_substitutes(con, caplog):
    # The exact bug: a rank-1 pick whose target-day bar has high < open (the
    # ENHA 2026-07-24 shape, open=3.51 high=3.40, a -16.8% impossible move) must
    # be excluded from the basket so the dashboard's Top-1 number is NOT computed
    # from the corrupt bar; the top-1 substitutes to the next available rank and
    # the substitution is disclosed (top1_rank > 1, n_avail < top_n, warning).
    oc = {
        "BADHI": [0.001] * 25,  # target-day bar overwritten to the ENHA shape below
        "GOOD1": [0.001] * 24 + [0.021],  # +2.1% on the target day
        "GOOD2": [0.001] * 25,  # flat
        # Filler names give the target day realistic coverage: excluding ONE
        # corrupt bar (BADHI) must not trip the completeness gate (#65) — in a
        # 3-name universe it would be a 33% cliff, in production ~0.03%.
        **{f"FILL{i:02d}": [0.001] * 25 for i in range(12)},
    }
    seed_history(con, oc)
    dates = sorted(pd.bdate_range("2026-01-05", periods=25).date)
    con.execute(
        "UPDATE prices SET open = 3.51, high = 3.40, low = 2.90, close = 2.92 "
        "WHERE symbol = 'BADHI' AND date = ?",
        [dates[-1]],
    )
    store.save_predictions(
        con,
        "test_strat",
        dates[-2],
        pd.DataFrame(
            {"symbol": ["BADHI", "GOOD1", "GOOD2"], "prob": [0.9, 0.5, 0.1], "rank": [1, 2, 3]}
        ),
    )

    with caplog.at_level(logging.WARNING, logger="twopercent.track"):
        perf = track.daily_pick_performance(con, "test_strat", top_n=3)

    assert len(perf.daily) == 1
    row = perf.daily.iloc[0]
    assert pd.Timestamp(row["target_date"]).date() == dates[-1]
    assert row["top1_symbol"] == "GOOD1"  # corrupt rank-1 excluded, substituted
    assert row["top1_rank"] == 2  # substitution disclosed
    assert row["n_avail"] == 2  # BADHI's corrupt bar never counted
    assert abs(row["top1_return"] - 0.021) < 1e-9  # NOT the -16.8% corrupt move
    assert row["top1_return"] > 0
    assert "substituted next available" in caplog.text


# --- degradation detector -----------------------------------------------------


def _scored_frame(days, late=None, start="2026-06-01"):
    """score_predictions-shaped frame from per-day (hits, n_scored, base_rate)
    tuples, in target_date order.

    precision = hits/n_scored and lift = precision/base_rate are DERIVED so the
    frame is internally consistent — the recalibrated detector pools hits/
    n_scored/base_rate, and lift only gates null-day exclusion and the
    days-below-base disclosure. base_rate 0 → NaN lift (zero-base-rate day)."""
    n = len(days)
    dates = pd.bdate_range(start, periods=n)
    hits = [float(h) for h, _, _ in days]
    n_scored = [float(s) for _, s, _ in days]
    base = [float(b) for _, _, b in days]
    precision = [h / s if s else float("nan") for h, s in zip(hits, n_scored, strict=True)]
    lift = [p / b if b > 0 else float("nan") for p, b in zip(precision, base, strict=True)]
    return pd.DataFrame(
        {
            "signal_date": dates - pd.tseries.offsets.BDay(1),
            "target_date": dates,
            "hits": hits,
            "n_scored": n_scored,
            "precision": precision,
            "base_rate": base,
            "lift": lift,
            "late": [False] * n if late is None else late,
        }
    )


def test_detector_fires_at_exactly_five_live_days():
    # 5 live days pooling precision 0.05 (1/20) vs base rate 0.10 → excess -0.05.
    verdict = track.degradation_verdict(_scored_frame([(1, 20, 0.10)] * 5))
    assert verdict.degraded and verdict.armed
    assert verdict.live_days == 5
    assert abs(verdict.pooled_precision - 0.05) < 1e-12
    assert abs(verdict.pooled_base_rate - 0.10) < 1e-12
    assert abs(verdict.pooled_excess_precision - (-0.05)) < 1e-12
    assert "DEGRADED" in verdict.detail


def test_detector_epsilon_boundary_adversarial():
    # Comparison rule (track._EXCESS_DEGRADE_EPSILON): DEGRADED iff pooled excess
    # precision < 0 - 1e-9. Pooled precision here is 1/4 = 0.25 (exactly
    # representable); the base rate is nudged around it so the excess lands on
    # the guard band. Outside the band fires; inside/at/above never does — the
    # detector never fires on rounding error.
    assert track.degradation_verdict(_scored_frame([(1, 4, 0.25 + 1e-7)] * 5)).degraded
    assert not track.degradation_verdict(_scored_frame([(1, 4, 0.25 - 1e-7)] * 5)).degraded
    assert not track.degradation_verdict(_scored_frame([(1, 4, 0.25)] * 5)).degraded
    assert not track.degradation_verdict(_scored_frame([(1, 4, 0.25 - 1e-12)] * 5)).degraded
    # Pin the guard band itself: excess 2e-9 below 0 is strictly outside the
    # 1e-9 epsilon and fires; 0.5e-9 below is inside the band and must not.
    assert track.degradation_verdict(_scored_frame([(1, 4, 0.25 + 2e-9)] * 5)).degraded
    assert not track.degradation_verdict(_scored_frame([(1, 4, 0.25 + 0.5e-9)] * 5)).degraded


def test_detector_high_base_rate_streak_does_not_false_fire():
    # M3: the OLD mean-of-daily-lift-RATIOS rule false-fired on a hot
    # high-base-rate streak. Two low-base days the model narrowly missed (lift
    # 0.8) plus three high-base days it beat (lift ~1.09): the mean of ratios is
    # < 1.0, yet in POOLED absolute terms the picks beat the base rate (0.604 vs
    # 0.56). The recalibrated detector must NOT fire.
    days = [(4, 100, 0.05)] * 2 + [(98, 100, 0.90)] * 3
    frame = _scored_frame(days)
    old_mean_lift = float(frame["lift"].mean())
    assert old_mean_lift < 1.0  # the old rule would have tripped
    verdict = track.degradation_verdict(frame)
    assert not verdict.degraded and verdict.armed
    assert verdict.pooled_excess_precision > 0
    assert abs(verdict.pooled_precision - 0.604) < 1e-9
    assert abs(verdict.pooled_base_rate - 0.56) < 1e-9


def test_detector_fires_when_picks_stop_beating_base_rate():
    # M3 flip side: when the picks genuinely stop beating the base rate — pooled
    # precision 0.02 under a 0.30 base rate every day — the detector DOES fire.
    verdict = track.degradation_verdict(_scored_frame([(2, 100, 0.30)] * 5))
    assert verdict.degraded and verdict.armed
    assert verdict.pooled_excess_precision < 0


def test_detector_not_armed_below_five_live_days():
    verdict = track.degradation_verdict(_scored_frame([(1, 20, 0.30)] * 4))
    assert not verdict.degraded and not verdict.armed
    assert verdict.live_days == 4
    assert verdict.pooled_excess_precision is None
    assert verdict.pooled_precision is None and verdict.pooled_base_rate is None
    assert "armed after 1 more live day" in verdict.detail  # loud, never silent


def test_detector_excludes_late_days_from_window():
    # 5 awful live days (precision 0.05 vs base 0.20) then 3 stellar LATE days
    # (most recent) — a backfill with known outcomes must never mask a live
    # degradation.
    verdict = track.degradation_verdict(
        _scored_frame([(1, 20, 0.20)] * 5 + [(19, 20, 0.20)] * 3, late=[False] * 5 + [True] * 3)
    )
    assert verdict.degraded and verdict.live_days == 5

    # And late days never count TOWARD arming either: 3 live + 4 late = unarmed.
    verdict = track.degradation_verdict(
        _scored_frame(
            [(1, 20, 0.20), (19, 20, 0.20)] * 3 + [(19, 20, 0.20)],
            late=[False, True, False, True, False, True, True],
        )
    )
    assert not verdict.armed and verdict.live_days == 3


def test_detector_discloses_days_below_base_when_pool_survives():
    # False-negative mode of pooling: four zero-hit days plus one big day
    # (60/100 at a 0.10 base) leave pooled precision 0.12 > base 0.10, so the
    # trigger stays quiet — but the per-day below-base count must not hide.
    verdict = track.degradation_verdict(_scored_frame([(0, 100, 0.10)] * 4 + [(60, 100, 0.10)]))
    assert not verdict.degraded and verdict.armed
    assert verdict.days_below_1 == 4
    assert "4 of 5 window day(s) below the base rate" in verdict.detail


def test_detector_excludes_null_lift_days_with_warning(caplog):
    # Most recent live day has NULL lift (zero base rate): excluded loudly,
    # the window falls back to the 5 defined-lift days (each 0.05 vs 0.10).
    frame = _scored_frame([(1, 20, 0.10)] * 5 + [(0, 20, 0.0)])
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
