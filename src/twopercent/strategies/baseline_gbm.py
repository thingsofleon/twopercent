"""Baseline strategy: gradient-boosted trees on the canonical features."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from twopercent.strategies.base import register, resolve_feature_columns

logger = logging.getLogger(__name__)


DEFAULT_PARAMS = {"max_iter": 150, "learning_rate": 0.1, "random_state": 42}


@register("baseline_gbm_v1")
class BaselineGBM:
    """HistGradientBoosting: fast on millions of rows, NaN-tolerant, no tuning.

    Constructor kwargs pass straight through to HistGradientBoostingClassifier
    (research configs use max_iter/learning_rate/max_depth); defaults are
    DEFAULT_PARAMS, so no-arg construction is the historical baseline. An
    unknown kwarg raises TypeError at construction — never a silent default.
    `feature_columns` overrides the model inputs (validated whitelist, see
    resolve_feature_columns); omitted, it is FEATURE_COLUMNS as always.
    """

    def __init__(self, feature_columns: list[str] | None = None, **params) -> None:
        self._model = HistGradientBoostingClassifier(**{**DEFAULT_PARAMS, **params})
        self.configured_columns: list[str] = resolve_feature_columns(
            "baseline_gbm_v1", feature_columns
        )
        self._columns: list[str] = list(self.configured_columns)
        self.dropped_columns: list[str] = []

    def fit(self, train: pd.DataFrame) -> None:
        empty = [col for col in self.configured_columns if train[col].notna().sum() == 0]
        if len(empty) == len(self.configured_columns):
            raise ValueError(
                "baseline_gbm_v1: every feature column has zero observed values in "
                "training data — nothing to train on (migrated store before a universe "
                f"refresh? columns: {', '.join(empty)})"
            )
        if empty:
            logger.warning(
                "baseline_gbm_v1: %d feature column(s) have zero observed values in training "
                "data and carry no signal (dropped; all-NaN columns crash HistGBM's binner): %s",
                len(empty),
                ", ".join(empty),
            )
        self.dropped_columns = empty
        self._columns = [col for col in self.configured_columns if col not in empty]
        self._model.fit(train[self._columns], train["did_2pct_next"])

    def predict_proba(self, rows: pd.DataFrame) -> pd.Series:
        probs = self._model.predict_proba(rows[self._columns])[:, 1]
        return pd.Series(probs, index=rows.index)
