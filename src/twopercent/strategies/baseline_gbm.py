"""Baseline strategy: gradient-boosted trees on the canonical features."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from twopercent.features import FEATURE_COLUMNS
from twopercent.strategies.base import register

logger = logging.getLogger(__name__)


@register("baseline_gbm_v1")
class BaselineGBM:
    """HistGradientBoosting: fast on millions of rows, NaN-tolerant, no tuning."""

    def __init__(self) -> None:
        self._model = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, random_state=42
        )
        self._columns: list[str] = list(FEATURE_COLUMNS)

    def fit(self, train: pd.DataFrame) -> None:
        empty = [col for col in FEATURE_COLUMNS if train[col].notna().sum() == 0]
        if len(empty) == len(FEATURE_COLUMNS):
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
        self._columns = [col for col in FEATURE_COLUMNS if col not in empty]
        self._model.fit(train[self._columns], train["did_2pct_next"])

    def predict_proba(self, rows: pd.DataFrame) -> pd.Series:
        probs = self._model.predict_proba(rows[self._columns])[:, 1]
        return pd.Series(probs, index=rows.index)
