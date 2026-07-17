"""Baseline strategy: gradient-boosted trees on the canonical features."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from twopercent.features import FEATURE_COLUMNS
from twopercent.strategies.base import register


@register("baseline_gbm_v1")
class BaselineGBM:
    """HistGradientBoosting: fast on millions of rows, NaN-tolerant, no tuning."""

    def __init__(self) -> None:
        self._model = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.1, random_state=42
        )

    def fit(self, train: pd.DataFrame) -> None:
        self._model.fit(train[FEATURE_COLUMNS], train["did_2pct_next"])

    def predict_proba(self, rows: pd.DataFrame) -> pd.Series:
        probs = self._model.predict_proba(rows[FEATURE_COLUMNS])[:, 1]
        return pd.Series(probs, index=rows.index)
