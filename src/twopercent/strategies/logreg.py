"""Logistic-regression strategy: linear baseline on the canonical features."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from twopercent.features import FEATURE_COLUMNS
from twopercent.strategies.base import register


@register("logreg_v1")
class LogReg:
    """Impute → scale → logistic regression: a calibrated linear yardstick."""

    def __init__(self) -> None:
        self._model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

    def fit(self, train: pd.DataFrame) -> None:
        self._model.fit(train[FEATURE_COLUMNS], train["did_2pct_next"])

    def predict_proba(self, rows: pd.DataFrame) -> pd.Series:
        probs = self._model.predict_proba(rows[FEATURE_COLUMNS])[:, 1]
        return pd.Series(probs, index=rows.index)
