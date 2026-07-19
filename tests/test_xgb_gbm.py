"""xgb_gbm_v1: CPU-fallback probe (the CI path), dropped-columns semantics,
planted-signal walk-forward, imbalance weighting. Offline and GPU-free: the
CUDA probe is monkeypatched both ways, so these tests never touch a device."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from tests.conftest import seed_planted
from twopercent import backtest, store, strategies
from twopercent.features import FEATURE_COLUMNS
from twopercent.strategies import xgb_gbm

XGB_LOGGER = "twopercent.strategies.xgb_gbm"
FAST = {"n_estimators": 30}  # small forests keep the walk-forward tests quick


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """CI has no GPU and the dev box must not be touched by tests: force the
    probe onto the CPU-fallback path and reset its once-per-process cache."""
    xgb_gbm._cuda_available.cache_clear()
    monkeypatch.setattr(xgb_gbm, "_probe_cuda", lambda: False)
    yield
    xgb_gbm._cuda_available.cache_clear()


def _feature_frame_rows(n: int = 200, positives: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    if positives is None:
        frame["did_2pct_next"] = (frame["oc_return_today"] > 0).astype(int)
    else:
        frame["did_2pct_next"] = np.array([1] * positives + [0] * (n - positives))
    return frame


def test_xgb_registered():
    assert "xgb_gbm_v1" in strategies.names()
    strat = strategies.get("xgb_gbm_v1")
    assert strat.name == "xgb_gbm_v1"
    assert isinstance(strat, strategies.Strategy)


def test_cpu_fallback_is_loud_but_warns_only_once(caplog):
    strat = strategies.get("xgb_gbm_v1", **FAST)
    with caplog.at_level("WARNING", logger=XGB_LOGGER):
        strat.fit(_feature_frame_rows())
        strat.fit(_feature_frame_rows())  # second fit: cached probe, no re-warn

    warnings = [r.message for r in caplog.records if "CUDA" in r.message]
    assert len(warnings) == 1
    assert "falling back to CPU" in warnings[0]
    assert strat._model.get_params()["device"] == "cpu"
    probs = strat.predict_proba(_feature_frame_rows())
    assert np.isfinite(probs).all()
    assert xgb_gbm.device_in_use() == "cpu"


def test_cuda_kept_when_probe_succeeds(monkeypatch):
    xgb_gbm._cuda_available.cache_clear()
    monkeypatch.setattr(xgb_gbm, "_probe_cuda", lambda: True)
    assert xgb_gbm._resolve_device("cuda") == "cuda"
    assert xgb_gbm.device_in_use() == "cuda"


def test_explicit_cpu_never_probes(monkeypatch, caplog):
    monkeypatch.setattr(
        xgb_gbm, "_probe_cuda", lambda: pytest.fail("probe ran for an explicit-CPU strategy")
    )
    strat = strategies.get("xgb_gbm_v1", device="cpu", **FAST)
    with caplog.at_level("WARNING", logger=XGB_LOGGER):
        strat.fit(_feature_frame_rows())
    assert not caplog.records
    assert strat._model.get_params()["device"] == "cpu"
    assert xgb_gbm.device_in_use() is None  # probe cache untouched


def test_constructor_params_reach_the_model():
    strat = strategies.get("xgb_gbm_v1", n_estimators=10, learning_rate=0.2, max_depth=2)
    strat.fit(_feature_frame_rows())
    params = strat._model.get_params()
    assert params["n_estimators"] == 10
    assert params["learning_rate"] == 0.2
    assert params["max_depth"] == 2
    # Untouched knobs keep the documented defaults.
    assert params["min_child_weight"] == 5
    assert params["subsample"] == 0.8
    assert params["colsample_bytree"] == 0.8


def test_scale_pos_weight_computed_from_training_labels():
    strat = strategies.get("xgb_gbm_v1", **FAST)
    strat.fit(_feature_frame_rows(n=200, positives=40))
    assert strat._model.get_params()["scale_pos_weight"] == 160 / 40


def test_xgb_detects_planted_signal_end_to_end(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    metrics = backtest.run_benchmark(con, "xgb_gbm_v1", months=2, top_n=5, strategy_params=FAST)

    assert metrics["auc"] > 0.9
    assert metrics["lift"] > 1.5
    experiments = store.list_experiments(con)
    assert experiments["strategy"].iloc[0] == "xgb_gbm_v1"
    params = json.loads(experiments["params"].iloc[0])
    assert params["strategy_params"] == FAST


def test_xgb_survives_all_nan_sector_features(con, monkeypatch, caplog):
    """Same store state that crashed the baseline pre-#26: sector features all
    NaN. XGBoost tolerates NaN, but zero-observed columns must still be dropped
    LOUDLY so dropped_columns semantics match the baseline's exactly."""
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    con.execute("UPDATE universe SET sector = NULL")

    with caplog.at_level("WARNING", logger=XGB_LOGGER):
        metrics = backtest.run_benchmark(con, "xgb_gbm_v1", months=2, top_n=5, strategy_params=FAST)

    drops = [r.message for r in caplog.records if "zero observed values" in r.message]
    assert len(drops) == metrics["folds"]  # every fold refits and must re-warn
    assert all("sector_breadth" in msg and "sector_excess" in msg for msg in drops)
    for key in ("precision_at_n", "base_rate", "lift", "auc", "brier"):
        assert metrics[key] is not None
        assert math.isfinite(metrics[key])
    params = json.loads(store.list_experiments(con)["params"].iloc[0])
    assert params["dropped_columns"] == ["sector_breadth", "sector_excess"]


def test_xgb_all_nan_features_raise_clear_error():
    n = 60
    frame = pd.DataFrame({col: np.full(n, np.nan) for col in FEATURE_COLUMNS})
    frame["did_2pct_next"] = np.tile([0, 1], n // 2)
    strat = strategies.get("xgb_gbm_v1", **FAST)
    with pytest.raises(ValueError, match="every feature column has zero observed values"):
        strat.fit(frame)


def test_xgb_keeps_partially_observed_column(caplog):
    """One observed value = kept (NaN-native), matching the baseline boundary."""
    frame = _feature_frame_rows(n=120)
    frame["log_mcap"] = np.nan
    frame.loc[0, "log_mcap"] = 3.7
    strat = strategies.get("xgb_gbm_v1", **FAST)
    with caplog.at_level("WARNING", logger=XGB_LOGGER):
        strat.fit(frame)
    assert strat.dropped_columns == []
    assert not [r for r in caplog.records if "zero observed values" in r.message]
    assert np.isfinite(strat.predict_proba(frame)).all()


def test_unknown_param_rejected_at_construction():
    """XGBoost forwards unknown kwargs with only a printed notice — the
    whitelist keeps the registry's promise that a typo can never silently
    run the defaults."""
    with pytest.raises(ValueError, match="unknown param"):
        strategies.get("xgb_gbm_v1", n_estimatorz=100)
    # Whitelisted regularizers pass through.
    strat = strategies.get("xgb_gbm_v1", gamma=0.1, reg_alpha=0.5)
    assert strat._params["gamma"] == 0.1


def test_resolved_device_exposed_for_the_referee():
    strat = strategies.get("xgb_gbm_v1", **FAST)
    assert strat.resolved_device is None  # unknown until fit
    strat.fit(_feature_frame_rows())
    assert strat.resolved_device == "cpu"  # probe forced to the fallback path
