import json

import pandas as pd

from tests.conftest import seed_history
from twopercent import backtest, store
from twopercent.strategies.base import _REGISTRY, register

# Slight deterministic variation keeps every feature column multi-valued
# (sklearn's binner rejects constant columns).
RUNNER_OC = [0.03 + 0.001 * (i % 5) for i in range(100)]  # +3.0–3.4% every day
FLAT_OC = [0.002 + 0.001 * (i % 3) for i in range(100)]  # +0.2–0.4%, never 2%


def _seed_planted(con, n_each: int = 30):
    data = {}
    for i in range(n_each):
        data[f"RUN{i:02d}"] = RUNNER_OC
        data[f"FLT{i:02d}"] = FLAT_OC
    seed_history(con, data, vary_volume=True)
    store.upsert_universe(
        con,
        pd.DataFrame(
            {
                "symbol": list(data),
                "name": list(data),
                "market_cap": [1e9 * (i + 1) for i in range(len(data))],
            }
        ),
        as_of=pd.Timestamp("2026-06-01").date(),
    )


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
    _seed_planted(con)
    _REGISTRY["split_spy"].observed.clear()
    backtest.run_benchmark(con, "split_spy", months=2, top_n=5, record=False)
    assert _REGISTRY["split_spy"].observed  # folds actually ran
    for train_max, test_min in _REGISTRY["split_spy"].observed:
        assert train_max < test_min


def test_planted_signal_is_detected_and_recorded(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    _seed_planted(con)
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


def test_month_folds_shape():
    dates = pd.Series(pd.bdate_range("2026-01-05", periods=90).date)
    folds = backtest.month_folds(dates, months=2)
    assert len(folds) == 2
    for start, end in folds:
        assert start < end
    assert folds[0][1] < folds[1][0]
