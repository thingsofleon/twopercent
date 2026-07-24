"""Canonical config identity — the ONE way a (strategy, params) is keyed.

Extracted from research.py so both the research runner and the shadow-trading
engine can share it without an import cycle (research imports routine, routine
imports shadow, shadow needs this). Numeric-normalizes so 200 and 200.0 are the
same config, and order-normalizes so key order never changes identity.
"""

from __future__ import annotations

import json


def _canonical(value):
    """Integral floats become ints so 200 and 200.0 are the SAME config."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_canonical(v) for v in value]
    return value


def canonical_params(params: dict) -> str:
    """Order- and numeric-normalized config identity for done-matching."""
    return json.dumps(_canonical(params), sort_keys=True)
