"""Strategy protocol and registry.

A strategy is a self-contained (features-used + model + params) unit behind a
two-method interface. Research agents add new strategies as new modules with
an @register decorator; they never modify the pipeline or the benchmark
referee. See ROADMAP.md "Architecture constraint".
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import pandas as pd

from twopercent.features import FEATURE_COLUMNS, INTRADAY_FEATURE_COLUMNS

_REGISTRY: dict[str, type] = {}

# Every column a strategy may be pointed at. The intraday four are computed and
# canary-watched but held out of FEATURE_COLUMNS (#115); they are selectable so
# the A/B that decides them (#116) can build the with-arm without editing the
# shipped list. Nothing else in the feature frame is a legal model input.
SELECTABLE_FEATURE_COLUMNS = frozenset(FEATURE_COLUMNS) | frozenset(INTRADAY_FEATURE_COLUMNS)


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


def resolve_feature_columns(strategy_name: str, feature_columns: Sequence[str] | None) -> list[str]:
    """The model inputs a strategy will use — `FEATURE_COLUMNS` unless overridden.

    An override exists so a feature set can be A/B'd against another on
    identical rows (ab.py) without editing the shipped list. It is a WHITELIST,
    not a passthrough: the only legal names are the canonical feature columns
    plus the computed-but-held intraday ones. That is the leakage guard — the
    feature frame also carries `did_2pct_next`, `next_oc_return` and
    `target_date`, and an override is exactly the seam through which a label
    column could otherwise reach `fit`.

    Raises on an unknown name, a duplicate, or an empty list — never silently
    falls back to the default, which would make an A/B compare an arm against
    itself and report the answer as "no difference".
    """
    if feature_columns is None:
        return list(FEATURE_COLUMNS)
    columns = list(feature_columns)
    if not columns:
        raise ValueError(f"{strategy_name}: feature_columns is empty — nothing to train on")
    duplicates = sorted({col for col in columns if columns.count(col) > 1})
    if duplicates:
        raise ValueError(f"{strategy_name}: duplicate feature_columns: {', '.join(duplicates)}")
    unknown = [col for col in columns if col not in SELECTABLE_FEATURE_COLUMNS]
    if unknown:
        raise ValueError(
            f"{strategy_name}: unknown feature_columns: {', '.join(unknown)}; "
            f"selectable: {', '.join(sorted(SELECTABLE_FEATURE_COLUMNS))}"
        )
    return columns
