"""The active ("champion") strategy: a one-line versioned config.

Promotion = editing champion.json via PR after the benchmark shows a
challenger beating the champion on the standard folds. Rollback = revert.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CHAMPION = "baseline_gbm_v1"
CHAMPION_FILE = Path("champion.json")


def get_champion(path: Path = CHAMPION_FILE) -> str:
    if path.exists():
        return json.loads(path.read_text())["champion"]
    return DEFAULT_CHAMPION
