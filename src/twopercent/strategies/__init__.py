"""Strategy plugins. Importing this package registers all built-in strategies."""

from twopercent.strategies import (
    baseline_gbm,  # noqa: F401  (registers on import)
    logreg,  # noqa: F401  (registers on import)
    xgb_gbm,  # noqa: F401  (registers on import)
)
from twopercent.strategies.base import (
    SELECTABLE_FEATURE_COLUMNS,
    Strategy,
    get,
    names,
    register,
    resolve_feature_columns,
)

__all__ = [
    "SELECTABLE_FEATURE_COLUMNS",
    "Strategy",
    "get",
    "names",
    "register",
    "resolve_feature_columns",
]
