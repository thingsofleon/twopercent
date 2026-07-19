import json

import pandas as pd

from tests.conftest import seed_planted
from twopercent import backtest, store
from twopercent.strategies.base import _REGISTRY, register

if "split_spy" not in _REGISTRY:

    @register("split_spy")
    class SplitSpy:
        """Records fold boundaries to prove train never touches test dates."""

        observed: list = []

        def fit(self, train):
            self._train_max = pd.to_datetime(train["target_date"]).max()

        def predict_proba(self, rows):
            test_min = pd.to_datetime(rows["target_date"]).min()
            SplitSpy.observed.append((self._train_max, test_min))
            return pd.Series(0.5, index=rows.index)


def test_walk_forward_never_trains_on_test_dates(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    _REGISTRY["split_spy"].observed.clear()
    backtest.run_benchmark(con, "split_spy", months=2, top_n=5, record=False)
    assert _REGISTRY["split_spy"].observed  # folds actually ran
    for train_max, test_min in _REGISTRY["split_spy"].observed:
        assert train_max < test_min


def test_planted_signal_is_detected_and_recorded(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    metrics = backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5)

    # Runners are perfectly identified by oc_return_today: near-perfect ranking.
    assert metrics["auc"] > 0.9
    assert metrics["lift"] > 1.5
    assert metrics["base_rate"] < 0.6

    # The run landed in the experiments table with parseable metrics.
    experiments = store.list_experiments(con)
    assert len(experiments) == 1
    assert experiments["strategy"].iloc[0] == "baseline_gbm_v1"
    assert json.loads(experiments["metrics"].iloc[0])["lift"] == metrics["lift"]


def test_benchmark_top_n_applies_liquidity_floor_selection_only(con, monkeypatch, caplog):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    # Every runner is below the floor: the top-N selection must be unable to
    # pick them (precision collapses to the flats' zero hit rate) while the
    # scoring and label populations still contain them (AUC and base_rate
    # unchanged) — the benchmark now selects exactly like the shipped product.
    con.execute("UPDATE prices SET volume = 50_000 WHERE symbol LIKE 'RUN%'")
    with caplog.at_level("WARNING", logger="twopercent.backtest"):
        metrics = backtest.run_benchmark(con, "logreg_v1", months=2, top_n=5)

    assert metrics["auc"] > 0.9  # all-names scoring population still sees runners
    assert 0.3 < metrics["base_rate"] < 0.7  # all-names label population too
    assert metrics["precision_at_n"] == 0.0  # selection could only pick flats
    assert "liquidity floor" in caplog.text

    params = json.loads(store.list_experiments(con)["params"].iloc[0])
    assert params["selection"] == "liquidity_floor_100k"


def test_benchmark_persists_per_rank_daily_rows(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)  # 60 symbols, all above the liquidity floor
    metrics = backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5)

    experiments = store.list_experiments(con)
    seq = int(experiments["id"].iloc[0])
    daily = con.execute(
        "SELECT * FROM experiment_daily WHERE seq = ? ORDER BY target_date, rank", [seq]
    ).df()
    # 20 ranks per scored test day (plenty of eligible names), inside the window.
    assert len(daily) == metrics["test_days"] * backtest.RECORD_RANKS
    per_day = daily.groupby("target_date")["rank"].agg(["count", "min", "max"])
    assert (per_day["count"] == backtest.RECORD_RANKS).all()
    assert (per_day["min"] == 1).all() and (per_day["max"] == backtest.RECORD_RANKS).all()
    dates = sorted({pd.Timestamp(d).date() for d in daily["target_date"]})
    assert len(dates) == metrics["test_days"]
    assert dates[0] >= pd.Timestamp(experiments["test_start"].iloc[0]).date()
    assert dates[-1] <= pd.Timestamp(experiments["test_end"].iloc[0]).date()

    # The persisted rank rows recompound to exactly the recorded aggregates:
    # rank 1 → sim_top1_growth, mean of ranks 1-5 → sim_top5_growth.
    from twopercent import track

    top1 = daily[daily["rank"] == 1]
    growth1 = float((1 + top1["ret"] - track.COST_ROUND_TRIP).prod())
    assert abs(round(growth1, 4) - metrics["sim_top1_growth"]) < 1e-9
    day5 = daily[daily["rank"] <= 5].groupby("target_date")["ret"].mean()
    growth5 = float((1 + day5 - track.COST_ROUND_TRIP).prod())
    assert abs(round(growth5, 4) - metrics["sim_top5_growth"]) < 1e-9

    result = store.latest_experiment_daily(con, "baseline_gbm_v1")
    assert result is not None
    meta, latest_daily = result
    assert meta["seq"] == seq
    assert len(latest_daily) == len(daily)
    # Metrics JSON stays aggregate-only — per-day rows live in their own table.
    recorded = json.loads(experiments["metrics"].iloc[0])
    assert "daily" not in recorded and "daily_picks" not in recorded


def test_benchmark_records_fewer_ranks_when_fewer_eligible(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 200)
    seed_planted(con, n_each=4)  # 8 symbols total — fewer than RECORD_RANKS
    metrics = backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5)
    daily = con.execute("SELECT * FROM experiment_daily").df()
    per_day = daily.groupby("target_date")["rank"].count()
    assert len(per_day) == metrics["test_days"]
    assert (per_day == 8).all()  # every eligible name recorded, no phantom ranks


def test_benchmark_unrecorded_run_persists_nothing(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5, record=False)
    assert store.list_experiments(con).empty
    assert con.execute("SELECT count(*) FROM experiment_daily").fetchone()[0] == 0


def test_benchmark_records_first_run_fold_as_test_start(con, monkeypatch, caplog):
    # MIN_TRAIN_ROWS sits between the train-row counts of the first and second
    # requested folds: the first is skipped, and the recorded test_start must
    # be the first fold that RAN — not the skipped one.
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 2000)
    seed_planted(con)
    with caplog.at_level("WARNING", logger="twopercent.backtest"):
        metrics = backtest.run_benchmark(con, "baseline_gbm_v1", months=3, top_n=5)
    assert "skipped" in caplog.text  # the first fold really was skipped
    assert metrics["folds"] < 3  # premise: at least one requested fold did not run
    test_start = pd.Timestamp(store.list_experiments(con)["test_start"].iloc[0]).date()
    first_recorded = con.execute("SELECT min(target_date) FROM experiment_daily").fetchone()[0]
    first_recorded = pd.Timestamp(first_recorded).date()
    # test_start is the month start of the first fold that RAN, which is the
    # month of the first recorded sim day — not the skipped first fold's month.
    assert test_start == first_recorded.replace(day=1)


def test_month_folds_shape():
    dates = pd.Series(pd.bdate_range("2026-01-05", periods=90).date)
    folds = backtest.month_folds(dates, months=2)
    assert len(folds) == 2
    for start, end in folds:
        assert start < end
    assert folds[0][1] < folds[1][0]


def test_record_is_atomic_when_daily_rows_fail(con, monkeypatch):
    """A crash between the experiments row and its daily rows must leave
    NOTHING: a half-recorded experiment would be counted "done" by the
    research queue forever while its sim-panel rows are missing."""
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)

    def boom(con_, seq, rows):
        raise ValueError("synthetic daily-row corruption")

    monkeypatch.setattr(backtest.store, "record_experiment_daily", boom)
    import pytest

    with pytest.raises(ValueError, match="synthetic daily-row corruption"):
        backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5)
    assert con.execute("SELECT count(*) FROM experiments").fetchone()[0] == 0
    assert con.execute("SELECT count(*) FROM experiment_daily").fetchone()[0] == 0
