"""XGBoost challenger: GPU-first gradient-boosted trees on the canonical features.

Defaults (overridable per experiment via constructor kwargs):

- n_estimators=300, learning_rate=0.05 — more, slower trees than the baseline's
  150 @ 0.1; the usual trade for tabular noise.
- max_depth=6 — moderate interaction depth.
- min_child_weight=5 — leaves must summarize several rows; single-row leaves
  memorize noise on financial data.
- subsample=0.8, colsample_bytree=0.8 — stochastic rows/columns per tree, the
  standard variance dampener.
- device="cuda" — trains on the GPU when one is visible. The probe runs once
  per process; when CUDA is unavailable (CI, driver trouble) fit falls back to
  CPU with a single LOUD warning. Results are equivalent, just slower.

Class imbalance mirrors the baseline's intent: scale_pos_weight is computed
from each fold's training labels (negatives/positives), never hard-coded.

XGBoost tolerates NaN feature values natively, but columns with ZERO observed
values are still dropped with the same loud warning as baseline_gbm_v1 so the
benchmark's `dropped_columns` semantics (fold-drop warnings, ledger params)
stay identical across strategies. NOTE: XGBoost only warns (does not raise) on
unknown booster params, so keep queue configs to the documented knobs.
"""

from __future__ import annotations

import functools
import logging

import numpy as np
import pandas as pd
import xgboost

from twopercent.features import FEATURE_COLUMNS
from twopercent.strategies.base import register

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def _probe_cuda() -> bool:
    """Can XGBoost actually train on CUDA here? A 2-row boost round answers."""
    try:
        xgboost.train(
            {"device": "cuda", "tree_method": "hist", "verbosity": 0},
            xgboost.DMatrix(np.array([[0.0], [1.0]]), label=[0, 1]),
            num_boost_round=1,
        )
        return True
    except xgboost.core.XGBoostError:
        return False


@functools.cache
def _cuda_available() -> bool:
    """One probe, one warning, per process."""
    ok = _probe_cuda()
    if not ok:
        logger.warning(
            "xgb_gbm_v1: CUDA is unavailable — falling back to CPU training "
            "(expected on CI; on the GPU box this means a driver problem and "
            "a much slower research night)"
        )
    return ok


def _resolve_device(requested: str) -> str:
    if requested != "cuda":
        return requested
    return "cuda" if _cuda_available() else "cpu"


def device_in_use() -> str | None:
    """'cuda' or 'cpu' once a fit has run the probe; None before that."""
    if _cuda_available.cache_info().currsize == 0:
        return None
    return "cuda" if _cuda_available() else "cpu"


@register("xgb_gbm_v1")
class XGBoostGBM:
    """XGBoost hist trees; see module docstring for defaults and GPU fallback."""

    def __init__(self, device: str = "cuda", **params) -> None:
        self._device = device
        self._params = {**DEFAULT_PARAMS, **params}
        self._model: xgboost.XGBClassifier | None = None
        self._columns: list[str] = list(FEATURE_COLUMNS)
        self.dropped_columns: list[str] = []

    def fit(self, train: pd.DataFrame) -> None:
        empty = [col for col in FEATURE_COLUMNS if train[col].notna().sum() == 0]
        if len(empty) == len(FEATURE_COLUMNS):
            raise ValueError(
                "xgb_gbm_v1: every feature column has zero observed values in "
                "training data — nothing to train on (migrated store before a universe "
                f"refresh? columns: {', '.join(empty)})"
            )
        if empty:
            logger.warning(
                "xgb_gbm_v1: %d feature column(s) have zero observed values in training "
                "data and carry no signal (dropped; keeps dropped_columns semantics "
                "aligned with baseline_gbm_v1 even though XGBoost tolerates NaN): %s",
                len(empty),
                ", ".join(empty),
            )
        self.dropped_columns = empty
        self._columns = [col for col in FEATURE_COLUMNS if col not in empty]
        labels = train["did_2pct_next"].astype(int)
        positives = int(labels.sum())
        negatives = len(labels) - positives
        # Imbalance weight from THIS fold's training labels (like the baseline's
        # intent); degenerate one-class training keeps a neutral 1.0.
        scale_pos_weight = negatives / positives if positives and negatives else 1.0
        self._model = xgboost.XGBClassifier(
            **self._params,
            device=_resolve_device(self._device),
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )
        self._model.fit(train[self._columns], labels)

    def predict_proba(self, rows: pd.DataFrame) -> pd.Series:
        probs = self._model.predict_proba(rows[self._columns])[:, 1]
        return pd.Series(probs, index=rows.index)
