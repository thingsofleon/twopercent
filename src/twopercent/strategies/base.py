"""Strategy protocol and registry.

A strategy is a self-contained (features-used + model + params) unit behind a
two-method interface. Research agents add new strategies as new modules with
an @register decorator; they never modify the pipeline or the benchmark
referee. See ROADMAP.md "Architecture constraint".
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

_REGISTRY: dict[str, type] = {}


@runtime_checkable
class Strategy(Protocol):
    name: str

    def fit(self, train: pd.DataFrame) -> None:
        """Train on labeled feature rows (columns per features.feature_frame)."""

    def predict_proba(self, rows: pd.DataFrame) -> pd.Series:
        """Probability of did_2pct_next=1 for each row, aligned to rows.index."""


def register(name: str):
    def decorator(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(f"strategy {name!r} already registered")
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get(name: str) -> Strategy:
    try:
        return _REGISTRY[name]()
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; available: {sorted(_REGISTRY)}") from None


def names() -> list[str]:
    return sorted(_REGISTRY)
