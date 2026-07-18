import math

import numpy as np
import pandas as pd
import pytest

from tests.conftest import seed_planted
from twopercent import backtest, strategies
from twopercent.features import FEATURE_COLUMNS
from twopercent.strategies.base import register

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
        metrics = backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5, record=False)

    warnings = [r.message for r in caplog.records if r.name == GBM_LOGGER]
    assert warnings
    assert all("sector_breadth" in msg and "sector_excess" in msg for msg in warnings)
    for key in ("precision_at_n", "base_rate", "lift", "auc", "brier"):
        assert metrics[key] is not None
        assert math.isfinite(metrics[key])
    # The planted signal lives in oc_return_today, so dropping the sector
    # columns must not cost the ranking.
    assert metrics["auc"] > 0.9
    assert metrics["lift"] > 1.5


def test_gbm_silent_when_all_features_observed(con, monkeypatch, caplog):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)

    with caplog.at_level("WARNING", logger=GBM_LOGGER):
        metrics = backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5, record=False)

    assert not [r for r in caplog.records if r.name == GBM_LOGGER]
    assert metrics["auc"] > 0.9
    assert metrics["lift"] > 1.5


def test_gbm_all_nan_features_raise_clear_error():
    n = 60
    frame = pd.DataFrame({col: np.full(n, np.nan) for col in FEATURE_COLUMNS})
    frame["did_2pct_next"] = np.tile([0, 1], n // 2)
    strat = strategies.get("baseline_gbm_v1")
    with pytest.raises(ValueError, match="every feature column has zero observed values"):
        strat.fit(frame)
