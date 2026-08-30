"""Logistic-regression strategy: linear baseline on the canonical features."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from twopercent.strategies.base import register, resolve_feature_columns

logger = logging.getLogger(__name__)


@register("logreg_v1")
class LogReg:
    """Impute → scale → logistic regression: a linear yardstick
    (class-balanced; ranking metrics comparable, brier not).

    `feature_columns` overrides the model inputs (validated whitelist, see
    resolve_feature_columns); omitted, it is FEATURE_COLUMNS as always.
    """

    def __init__(self, feature_columns: list[str] | None = None) -> None:
        self.configured_columns: list[str] = resolve_feature_columns("logreg_v1", feature_columns)
        # Always empty: unobserved columns are imputed as constants, never dropped.
        self.dropped_columns: list[str] = []
        self._model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

    def fit(self, train: pd.DataFrame) -> None:
        empty = [col for col in self.configured_columns if train[col].notna().sum() == 0]
        if empty:
            logger.warning(
                "logreg_v1: %d feature column(s) have zero observed values in training "
                "data and carry no signal (imputed as a constant): %s",
                len(empty),
                ", ".join(empty),
            )
        self._model.fit(train[self.configured_columns], train["did_2pct_next"])

    def predict_proba(self, rows: pd.DataFrame) -> pd.Series:
        probs = self._model.predict_proba(rows[self.configured_columns])[:, 1]
        return pd.Series(probs, index=rows.index)
