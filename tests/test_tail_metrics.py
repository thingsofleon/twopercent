import datetime as dt

import pandas as pd
import pytest

from tests.conftest import seed_history, seed_planted
from twopercent import backtest, store, track

DAYS = sorted(pd.bdate_range("2026-01-05", periods=25).date)

OC = {
    "WIN": [0.001] * 23 + [0.001, 0.04],  # +4% on the last day
    "LOSE": [0.001] * 23 + [0.001, -0.03],  # −3% on the last day
    "ALSO": [0.001] * 23 + [0.001, 0.025],  # +2.5% on the last day
}


def _save(con, ranked, strategy="s"):
    store.save_predictions(
        con,
        strategy,
        DAYS[-2],
        pd.DataFrame({"symbol": ranked, "prob": [0.9, 0.5, 0.2], "rank": [1, 2, 3]}),
    )


def test_pick_performance_math_and_costed_growth(con):
    seed_history(con, OC)
    _save(con, ["WIN", "LOSE", "ALSO"])

    picks = track.daily_pick_performance(con, "s", top_n=3)
    assert picks.days == 1
    row = picks.daily.iloc[0]
    assert row["top1_symbol"] == "WIN"
    assert abs(row["top1_return"] - 0.04) < 1e-12
    assert row["top1_hit"] == 1
    assert abs(row["topn_return"] - (0.04 - 0.03 + 0.025) / 3) < 1e-12
    assert row["topn_hits"] == 2  # WIN and ALSO
    assert row["n_avail"] == 3

    assert picks.precision_at_1() == 1.0
    # Growth is net of the assumed round-trip cost.
    assert abs(picks.growth("top1_return") - (1 + 0.04 - track.COST_ROUND_TRIP)) < 1e-12


def test_pick_performance_top1_is_best_available(con):
    # Rank 1's target bar missing (delisted before target) → the tradeable
    # top pick is rank 2, not a phantom.
    seed_history(con, OC)
    _save(con, ["GONE", "WIN", "LOSE"])  # GONE has no price rows at all

    picks = track.daily_pick_performance(con, "s", top_n=3)
    row = picks.daily.iloc[0]
    assert row["top1_symbol"] == "WIN"
    assert row["n_avail"] == 2


def test_pick_performance_empty(con):
    picks = track.daily_pick_performance(con, "nope")
    assert picks.days == 0
    assert picks.precision_at_1() is None
    assert picks.growth() is None


def test_benchmark_reports_tail_metrics_and_sim(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con, n_each=30)
    metrics = backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5)

    # Runners do +3.0–3.4% every day and are perfectly identifiable: the top
    # pick hits every day and $1 compounds at roughly (1 + 3.2% − cost)^days.
    assert metrics["precision_at_1"] == 1.0
    assert metrics["precision_at_5"] == 1.0
    days = metrics["test_days"]
    low = (1 + 0.030 - track.COST_ROUND_TRIP) ** days
    high = (1 + 0.034 - track.COST_ROUND_TRIP) ** days
    assert low * 0.99 <= metrics["sim_top1_growth"] <= high * 1.01
    assert metrics["sim_top5_growth"] > 1.0


def test_next_oc_return_is_label_side(con):
    # The realized-return column must exist for scoring but never be a
    # feature or metadata column (it is the future).
    from twopercent.features import FEATURE_COLUMNS, METADATA_COLUMNS, feature_frame

    seed_history(con, OC)
    frame = feature_frame(con)
    assert "next_oc_return" in frame.columns
    assert "next_oc_return" not in FEATURE_COLUMNS
    assert "next_oc_return" not in METADATA_COLUMNS
    labeled = frame[frame["did_2pct_next"].notna()]
    win_last = labeled[
        (labeled["symbol"] == "WIN") & (pd.to_datetime(labeled["signal_date"]).dt.date == DAYS[-2])
    ]
    assert abs(float(win_last["next_oc_return"].iloc[0]) - 0.04) < 1e-9


def test_late_flag_on_backfilled_pick_days(con):
    seed_history(con, OC)
    _save(con, ["WIN", "LOSE", "ALSO"])  # created now, target long past
    picks = track.daily_pick_performance(con, "s", top_n=3)
    assert bool(picks.daily["late"].iloc[0]) is True


@pytest.fixture
def con_with_universe(con):
    seed_history(con, OC)
    store.upsert_universe(
        con,
        pd.DataFrame(
            {
                "symbol": list(OC),
                "name": list(OC),
                "market_cap": [1e9] * 3,
                "sector": ["Tech"] * 3,
            }
        ),
        as_of=dt.date(2026, 2, 1),
    )
    return con


def test_dashboard_shows_pick_tiles_and_column(con_with_universe, tmp_path):
    from twopercent import dashboard
    from twopercent.predict import predict_for

    _save(con_with_universe, ["WIN", "LOSE", "ALSO"], strategy="baseline_gbm_v1")
    result = predict_for(con_with_universe, "baseline_gbm_v1", save=False)
    out = tmp_path / "d.html"
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(dashboard.build_html(con_with_universe, result, top=3))
    content = out.read_text()
    assert "Top pick hit rate" in content
    assert "$1 → top pick daily" in content
    assert "Top pick</th>" in content
    assert "WIN" in content
