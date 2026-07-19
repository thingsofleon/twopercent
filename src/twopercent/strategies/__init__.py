"""Strategy plugins. Importing this package registers all built-in strategies."""

from twopercent.strategies import (
    baseline_gbm,  # noqa: F401  (registers on import)
    logreg,  # noqa: F401  (registers on import)
    xgb_gbm,  # noqa: F401  (registers on import)
)
from twopercent.strategies.base import Strategy, get, names, register

__all__ = ["Strategy", "get", "names", "register"]
