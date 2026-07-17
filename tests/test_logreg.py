import json

import pandas as pd

from tests.conftest import seed_history
from twopercent import backtest, store, strategies

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


def test_logreg_registered():
    assert "logreg_v1" in strategies.names()
    strat = strategies.get("logreg_v1")
    assert strat.name == "logreg_v1"
    assert isinstance(strat, strategies.Strategy)


def test_logreg_detects_planted_signal_end_to_end(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    _seed_planted(con)
    metrics = backtest.run_benchmark(con, "logreg_v1", months=2, top_n=5)

    # Runners are perfectly identified by oc_return_today: near-perfect ranking.
    assert metrics["auc"] > 0.9
    assert metrics["lift"] > 1.5

    # The run landed in the experiments table with parseable metrics.
    experiments = store.list_experiments(con)
    assert len(experiments) == 1
    assert experiments["strategy"].iloc[0] == "logreg_v1"
    assert json.loads(experiments["metrics"].iloc[0])["lift"] == metrics["lift"]
