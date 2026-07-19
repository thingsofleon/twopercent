"""Noise-band verdict on benchmark lift, shared by the compare CLI and the
research runner. Referee-adjacent: like backtest.py, "better" is defined here
and only here, changed only by human-reviewed PR (kept stdlib-only so the CLI
can import it without paying for sklearn)."""

from __future__ import annotations

LIFT_NOISE_BAND = 0.1
_NOISE_BAND_EPSILON = 1e-9  # FP: 2.05 - 1.95 == 0.0999...987, yet is a true 0.1 gap


def lift_winner(
    strat_a: str,
    lift_a: float | None,
    strat_b: str,
    lift_b: float | None,
    band: float = LIFT_NOISE_BAND,
) -> str | None:
    """The strategy winning on lift OUTSIDE `band`, else None.

    None means undecided: lift unavailable, an exact tie, or a difference
    inside the band. The default band is calibrated for a SINGLE comparison
    (the compare CLI); multiple-comparison callers (the research sweep) must
    pass a wider band. Callers comparing a strategy against itself must pass
    distinct role names (e.g. "challenger"/"champion") to tell the sides apart.
    """
    if lift_a is None or lift_b is None or lift_a == lift_b:
        return None
    if abs(lift_a - lift_b) < band - _NOISE_BAND_EPSILON:
        return None
    return strat_a if lift_a > lift_b else strat_b


def compare_verdict(strat_a: str, lift_a: float | None, strat_b: str, lift_b: float | None) -> str:
    """One-line verdict on lift; refuses to crown a winner inside the noise band."""
    if lift_a is None or lift_b is None:
        return "Winner on lift: undecided (lift unavailable for at least one strategy)"
    if lift_a == lift_b:
        return f"Winner on lift: tie at {lift_a}"
    winner = lift_winner(strat_a, lift_a, strat_b, lift_b)
    if winner is None:
        return (
            "Winner on lift: within noise — no meaningful difference "
            "(same-window comparison; repeated trials against the same months "
            "inflate the best result)"
        )
    return f"Winner on lift: {winner} ({max(lift_a, lift_b)} vs {min(lift_a, lift_b)})"
