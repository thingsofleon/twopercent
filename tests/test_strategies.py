import pytest

from twopercent import strategies
from twopercent.strategies.base import register


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
