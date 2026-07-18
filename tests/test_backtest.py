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


def test_benchmark_persists_daily_sim_rows(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    metrics = backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5)

    experiments = store.list_experiments(con)
    seq = int(experiments["id"].iloc[0])
    daily = con.execute(
        "SELECT * FROM experiment_daily WHERE seq = ? ORDER BY target_date", [seq]
    ).df()
    # One row per scored test day, all inside the recorded test window.
    assert len(daily) == metrics["test_days"]
    dates = [pd.Timestamp(d).date() for d in daily["target_date"]]
    assert dates == sorted(set(dates))  # unique, ordered — one row per day
    assert dates[0] >= pd.Timestamp(experiments["test_start"].iloc[0]).date()
    assert dates[-1] <= pd.Timestamp(experiments["test_end"].iloc[0]).date()
    # The persisted rows recompound to exactly the recorded aggregate.
    from twopercent import track

    growth = float((1 + daily["top1_ret"] - track.COST_ROUND_TRIP).prod())
    assert abs(round(growth, 4) - metrics["sim_top1_growth"]) < 1e-9

    result = store.latest_experiment_daily(con, "baseline_gbm_v1")
    assert result is not None
    meta, latest_daily = result
    assert meta["seq"] == seq
    assert len(latest_daily) == len(daily)
    # Metrics JSON stays aggregate-only — per-day rows live in their own table.
    recorded = json.loads(experiments["metrics"].iloc[0])
    assert "daily" not in recorded and "daily_picks" not in recorded


def test_benchmark_unrecorded_run_persists_nothing(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5, record=False)
    assert store.list_experiments(con).empty
    assert con.execute("SELECT count(*) FROM experiment_daily").fetchone()[0] == 0


def test_month_folds_shape():
    dates = pd.Series(pd.bdate_range("2026-01-05", periods=90).date)
    folds = backtest.month_folds(dates, months=2)
    assert len(folds) == 2
    for start, end in folds:
        assert start < end
    assert folds[0][1] < folds[1][0]
