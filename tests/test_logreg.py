import json
import math

import numpy as np
import pandas as pd

from tests.conftest import seed_planted
from twopercent import backtest, store, strategies
from twopercent.cli import _compare_verdict
from twopercent.features import FEATURE_COLUMNS


def test_logreg_registered():
    assert "logreg_v1" in strategies.names()
    strat = strategies.get("logreg_v1")
    assert strat.name == "logreg_v1"
    assert isinstance(strat, strategies.Strategy)


def test_logreg_detects_planted_signal_end_to_end(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    metrics = backtest.run_benchmark(con, "logreg_v1", months=2, top_n=5)

    # Runners are perfectly identified by oc_return_today: near-perfect ranking.
    assert metrics["auc"] > 0.9
    assert metrics["lift"] > 1.5
    assert metrics["base_rate"] < 0.6

    # The run landed in the experiments table with parseable metrics.
    experiments = store.list_experiments(con)
    assert len(experiments) == 1
    assert experiments["strategy"].iloc[0] == "logreg_v1"
    assert json.loads(experiments["metrics"].iloc[0])["lift"] == metrics["lift"]


def test_logreg_imputes_symbols_missing_from_universe(con, monkeypatch):
    """Symbols absent from the universe flow NULL log_mcap through the LEFT
    JOIN; the imputer must absorb the NaNs, not crash or emit NaN metrics."""
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    covered = [f"RUN{i:02d}" for i in range(30)] + [f"FLT{i:02d}" for i in range(26)]
    all_symbols = seed_planted(con, universe_symbols=covered)
    assert len(all_symbols) == len(covered) + 4  # four symbols really lack universe rows

    metrics = backtest.run_benchmark(con, "logreg_v1", months=2, top_n=5, record=False)
    for key in ("precision_at_n", "base_rate", "lift", "auc", "brier"):
        assert metrics[key] is not None
        assert math.isfinite(metrics[key])


def _feature_frame_rows(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    frame["did_2pct_next"] = (frame["oc_return_today"] > 0).astype(int)
    return frame


def test_logreg_warns_loudly_on_all_nan_feature_column(caplog):
    frame = _feature_frame_rows()
    frame["log_mcap"] = np.nan
    strat = strategies.get("logreg_v1")
    with caplog.at_level("WARNING", logger="twopercent.strategies.logreg"):
        strat.fit(frame)
    assert any("log_mcap" in record.message for record in caplog.records)
    probs = strat.predict_proba(frame)
    assert probs.index.equals(frame.index)
    assert np.isfinite(probs).all()


def test_logreg_silent_when_all_features_observed(caplog):
    strat = strategies.get("logreg_v1")
    with caplog.at_level("WARNING", logger="twopercent.strategies.logreg"):
        strat.fit(_feature_frame_rows())
    assert not caplog.records


def test_compare_verdict_clear_winner():
    verdict = _compare_verdict("gbm", 2.0, "logreg", 1.5)
    assert "gbm" in verdict
    assert "2.0 vs 1.5" in verdict
    reverse = _compare_verdict("gbm", 1.5, "logreg", 2.0)
    assert "logreg" in reverse
    assert "2.0 vs 1.5" in reverse


def test_compare_verdict_tie():
    assert _compare_verdict("gbm", 2.0, "logreg", 2.0) == "Winner on lift: tie at 2.0"


def test_compare_verdict_within_noise_band():
    verdict = _compare_verdict("gbm", 2.0, "logreg", 1.95)
    assert "within noise" in verdict
    assert "gbm" not in verdict
    assert "logreg" not in verdict
    # Adversarial boundary: a difference at/above the band still crowns a winner.
    assert "gbm" in _compare_verdict("gbm", 2.05, "logreg", 1.95)


def test_compare_verdict_lift_unavailable():
    assert "undecided" in _compare_verdict("gbm", None, "logreg", 1.5)
    assert "undecided" in _compare_verdict("gbm", 1.5, "logreg", None)
