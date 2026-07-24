"""Tier-1 config generator: bounded, deterministic hyperparameter search.

First tier of the autonomous research engine (see ROADMAP "Autonomous research
engine"). When the hand-curated `research/queue.json` is exhausted, the nightly
runner tops its batch up from a FIXED, checked-in search grid so the idle GPU
keeps working instead of no-op'ing. This module only proposes hyperparameter
variants of already-registered strategies — it never writes code, touches the
champion, or promotes anything. State stays the experiments ledger: the runner
skips grid configs already recorded, so generation is idempotent and writes
nothing to the repo.

The grid is bounded ON PURPOSE. Every config is another trial against the same
walk-forward folds, so an unbounded generator would be a multiple-comparisons
engine. `family_size()` is asserted <= `MAX_AUTO_FAMILY`; widen the grid only
with the promotion-band accounting in mind (that widening is quant-skeptic-gated
per the ROADMAP). The grid deliberately probes REGULARISATION — the lever the
original 24-config sweep left at its default while lift plateaued (~2.1-2.19) —
rather than re-treading learning-rate/depth grids that already flat-lined. When
the grid is also exhausted the runner falls back to the `research-queue-empty`
signal, which now means "even auto-search is tapped out; add new *kinds* of work
(features, algorithms)."
"""

from __future__ import annotations

import itertools
import math

# Per-strategy discrete search grids. Only params valid for that strategy's
# constructor (baseline: HistGradientBoosting kwargs; xgb: within ALLOWED_PARAMS
# — l2_regularization/reg_lambda are the previously-unexplored regularisation
# axis). Values are chosen to bound the Cartesian product; see MAX_AUTO_FAMILY.
SEARCH_SPACE: dict[str, dict[str, list]] = {
    "baseline_gbm_v1": {
        "learning_rate": [0.05, 0.1],
        "max_depth": [4, 6],
        "max_iter": [150, 300],
        "l2_regularization": [1.0, 5.0],
    },
    "xgb_gbm_v1": {
        "learning_rate": [0.05, 0.1],
        "max_depth": [6, 8],
        "n_estimators": [200, 400],
        "reg_lambda": [5.0, 10.0],
    },
}

# Hard backstop on total trials the generator can ever emit across all
# strategies. The grid must stay under this; raising it is a deliberate
# multiple-comparisons decision, not an accident.
MAX_AUTO_FAMILY = 64


def family_size() -> int:
    """Total number of distinct configs the grid can ever produce."""
    return sum(
        math.prod(len(values) for values in space.values()) for space in SEARCH_SPACE.values()
    )


assert family_size() <= MAX_AUTO_FAMILY, (
    f"auto-search grid has {family_size()} configs > MAX_AUTO_FAMILY "
    f"({MAX_AUTO_FAMILY}); widening the grid inflates multiple comparisons — "
    "size the promotion band for it first (quant-skeptic-gated, see ROADMAP)"
)


def grid_configs() -> list[dict]:
    """The full search grid as ordered {strategy, params, note} dicts.

    Deterministic: strategies sorted, params sorted by key, values in listed
    order. The runner filters out configs already recorded in the ledger or
    present in the curated queue, then runs the rest up to its budget — so the
    order here only sets which un-run configs get tried first, never whether a
    config is tried twice.
    """
    configs: list[dict] = []
    for strategy in sorted(SEARCH_SPACE):
        space = SEARCH_SPACE[strategy]
        keys = sorted(space)
        for combo in itertools.product(*(space[k] for k in keys)):
            params = dict(zip(keys, combo, strict=True))
            note = "auto-search: " + " ".join(f"{k}={params[k]}" for k in keys)
            configs.append({"strategy": strategy, "params": params, "note": note})
    return configs
