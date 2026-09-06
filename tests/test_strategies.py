import json
import math

import numpy as np
import pandas as pd
import pytest

from tests.conftest import seed_planted
from twopercent import backtest, features, store, strategies
from twopercent.features import FEATURE_COLUMNS
from twopercent.strategies.base import _REGISTRY, register

GBM_LOGGER = "twopercent.strategies.baseline_gbm"


def test_builtin_strategy_registered():
    assert "baseline_gbm_v1" in strategies.names()
    strat = strategies.get("baseline_gbm_v1")
    assert strat.name == "baseline_gbm_v1"


def test_unknown_strategy_error_names_available():
    with pytest.raises(ValueError, match="baseline_gbm_v1"):
        strategies.get("nope")


def test_duplicate_registration_rejected():
    with pytest.raises(ValueError, match="already registered"):

        @register("baseline_gbm_v1")
        class Clash:
            pass


def test_gbm_survives_all_nan_sector_features(con, monkeypatch, caplog):
    """Migrated store before a universe refresh: sector features come back
    all-NaN; the benchmark must warn and complete, not crash the binner."""
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    con.execute("UPDATE universe SET sector = NULL")

    with caplog.at_level("WARNING", logger=GBM_LOGGER):
        metrics = backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5)

    warnings = [r.message for r in caplog.records if r.name == GBM_LOGGER]
    assert len(warnings) == metrics["folds"]  # every fold refits and must re-warn
    assert all("sector_breadth" in msg and "sector_excess" in msg for msg in warnings)
    for key in ("precision_at_n", "base_rate", "lift", "auc", "brier"):
        assert metrics[key] is not None
        assert math.isfinite(metrics[key])
    # The planted signal lives in oc_return_today, so dropping the sector
    # columns must not cost the ranking.
    assert metrics["auc"] > 0.9
    assert metrics["lift"] > 1.5
    # The ledger records which columns the model actually fit without, so a
    # 9-feature run is distinguishable from an 11-feature run by params alone.
    params = json.loads(store.list_experiments(con)["params"].iloc[0])
    assert params["dropped_columns"] == ["sector_breadth", "sector_excess"]


def test_gbm_silent_when_all_features_observed(con, monkeypatch, caplog):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)

    with caplog.at_level("WARNING", logger=GBM_LOGGER):
        metrics = backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5)

    assert not [r for r in caplog.records if r.name == GBM_LOGGER]
    assert metrics["auc"] > 0.9
    assert metrics["lift"] > 1.5
    params = json.loads(store.list_experiments(con)["params"].iloc[0])
    assert params["dropped_columns"] == []


def test_gbm_all_nan_features_raise_clear_error():
    n = 60
    frame = pd.DataFrame({col: np.full(n, np.nan) for col in FEATURE_COLUMNS})
    frame["did_2pct_next"] = np.tile([0, 1], n // 2)
    strat = strategies.get("baseline_gbm_v1")
    with pytest.raises(ValueError, match="every feature column has zero observed values"):
        strat.fit(frame)


def test_gbm_keeps_column_with_single_observed_value(caplog):
    """Pins the sklearn crash boundary at zero observed values: one observed
    value must be kept and fit cleanly (an sklearn upgrade moving the
    boundary shows up here, not in production)."""
    n = 120
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({col: rng.normal(size=n) for col in FEATURE_COLUMNS})
    frame["did_2pct_next"] = (frame["oc_return_today"] > 0).astype(int)
    frame["log_mcap"] = np.nan
    frame.loc[0, "log_mcap"] = 3.7

    strat = strategies.get("baseline_gbm_v1")
    with caplog.at_level("WARNING", logger=GBM_LOGGER):
        strat.fit(frame)

    assert strat.dropped_columns == []
    assert not caplog.records
    probs = strat.predict_proba(frame)
    assert np.isfinite(probs).all()


def test_benchmark_warns_when_folds_drop_different_columns(con, monkeypatch, caplog):
    """A benchmark that mixes 9-feature and 11-feature fits under one strategy
    name must say so, not average them silently."""
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)

    real_get = strategies.get
    fold_index = iter(range(10))

    def flaky_get(name):
        strat = real_get(name)
        real_fit = strat.fit

        def fit(train):
            real_fit(train)
            strat.dropped_columns = ["log_mcap"] if next(fold_index) == 0 else []

        strat.fit = fit
        return strat

    monkeypatch.setattr(backtest.strategies, "get", flaky_get)
    with caplog.at_level("WARNING", logger="twopercent.backtest"):
        backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5)

    mixed = [
        r.message
        for r in caplog.records
        if r.name == "twopercent.backtest" and "structurally different" in r.message
    ]
    assert len(mixed) == 1
    assert "log_mcap" in mixed[0]
    params = json.loads(store.list_experiments(con)["params"].iloc[0])
    assert params["dropped_columns"] == ["log_mcap"]  # union across folds


# --- parameterized registry (research loop) -----------------------------------


def test_get_passes_constructor_params_and_defaults():
    @register("param_probe_v1")
    class ParamProbe:
        def __init__(self, alpha=1, beta="x"):
            self.alpha = alpha
            self.beta = beta

    try:
        got = strategies.get("param_probe_v1", alpha=7)
        assert got.alpha == 7 and got.beta == "x"
        default = strategies.get("param_probe_v1")
        assert default.alpha == 1 and default.beta == "x"  # no params -> historical behavior
    finally:
        _REGISTRY.pop("param_probe_v1")


def test_get_unknown_param_is_loud():
    with pytest.raises(TypeError):
        strategies.get("baseline_gbm_v1", bogus_knob=1)


def test_baseline_gbm_constructor_kwargs_reach_sklearn():
    tuned = strategies.get("baseline_gbm_v1", max_iter=60, learning_rate=0.05, max_depth=3)
    params = tuned._model.get_params()
    assert params["max_iter"] == 60
    assert params["learning_rate"] == 0.05
    assert params["max_depth"] == 3
    # No-params regression: the historical baseline config, exactly.
    default = strategies.get("baseline_gbm_v1")._model.get_params()
    assert default["max_iter"] == 150
    assert default["learning_rate"] == 0.1
    assert default["random_state"] == 42
    assert default["max_depth"] is None


def test_benchmark_records_strategy_params(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)

    backtest.run_benchmark(
        con, "baseline_gbm_v1", months=2, top_n=5, strategy_params={"max_iter": 60}
    )
    params = json.loads(store.list_experiments(con)["params"].iloc[0])
    assert params["strategy_params"] == {"max_iter": 60}


def test_benchmark_default_records_empty_strategy_params(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)

    metrics = backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5)
    assert metrics["lift"] > 1.5  # unchanged no-params behavior
    params = json.loads(store.list_experiments(con)["params"].iloc[0])
    assert params["strategy_params"] == {}


def test_feature_columns_override_reaches_every_strategy():
    """A strategy is a (features-used + model + params) unit — the features half
    must be settable, or an A/B has to edit the shipped list to run."""
    subset = ["oc_return_today", "ret_5d", "vol_20d"]
    for name in ("baseline_gbm_v1", "logreg_v1", "xgb_gbm_v1"):
        assert strategies.get(name, feature_columns=subset).configured_columns == subset
        assert strategies.get(name).configured_columns == list(features.FEATURE_COLUMNS)


def test_feature_columns_may_name_the_held_intraday_columns():
    columns = [*features.FEATURE_COLUMNS, *features.INTRADAY_FEATURE_COLUMNS]
    assert strategies.get("baseline_gbm_v1", feature_columns=columns).configured_columns == columns


def test_feature_columns_is_a_whitelist_not_a_passthrough():
    """The override is the one seam a LABEL could reach fit() through."""
    for leak in ("did_2pct_next", "next_oc_return", "target_date", "median_vol_20"):
        with pytest.raises(ValueError, match="unknown feature_columns"):
            strategies.get("baseline_gbm_v1", feature_columns=[*features.FEATURE_COLUMNS, leak])


def test_feature_columns_rejects_empty_and_duplicates():
    with pytest.raises(ValueError, match="feature_columns is empty"):
        strategies.get("baseline_gbm_v1", feature_columns=[])
    with pytest.raises(ValueError, match="duplicate feature_columns"):
        strategies.get("baseline_gbm_v1", feature_columns=["ret_5d", "ret_5d"])


def test_overridden_strategy_trains_on_only_those_columns(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    train = features.feature_frame(con)
    train = train[train["did_2pct_next"].notna()]
    strategy = strategies.get("baseline_gbm_v1", feature_columns=["oc_return_today", "ret_5d"])
    strategy.fit(train)
    assert strategy._columns == ["oc_return_today", "ret_5d"]
    assert len(strategy.predict_proba(train)) == len(train)


def test_benchmark_records_the_feature_set_the_model_actually_saw(con, monkeypatch):
    """A ledger row naming the shipped feature set while the model trained on a
    different one is exactly the lie the fingerprint exists to prevent (#110)."""
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    columns = [*features.FEATURE_COLUMNS, *features.INTRADAY_FEATURE_COLUMNS]
    backtest.run_benchmark(
        con,
        "baseline_gbm_v1",
        months=2,
        top_n=5,
        strategy_params={"feature_columns": columns},
    )
    params = json.loads(store.list_experiments(con)["params"].iloc[0])
    assert params["feature_set"] == features.feature_set_version(columns)
    assert params["feature_set"] != features.feature_set_version()
