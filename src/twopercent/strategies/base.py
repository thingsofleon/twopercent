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


def get(name: str, **params) -> Strategy:
    """Instantiate a registered strategy, passing `params` to its constructor.

    No params → identical to the historical no-arg behavior. A strategy whose
    constructor rejects a param raises TypeError here — loud, so a typo in an
    experiment config can never silently run the defaults instead.
    """
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; available: {sorted(_REGISTRY)}") from None
    return cls(**params)


def names() -> list[str]:
    return sorted(_REGISTRY)
