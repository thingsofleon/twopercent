"""Walk-forward probability calibration: reliability + Brier skill score.

The app outputs P(reach +2% intraday) and the user ACTS on the number, so the
probability must be honest — "a 78% pick should reach 2% ~78% of the time."
This module MEASURES that honesty on the benchmark's held-out test-fold
predictions (out-of-sample), never in-sample. It does not recalibrate the model.

Two honesty choices, both from the quant-skeptic Stage-C notes:

- **Brier SKILL score, not raw Brier.** Touching +2% is far more common than
  closing +2%, and the daily base rate swings ~7%→98%. Raw Brier drops "for
  free" at a higher base rate, so it is misleading across the close→touch
  cutover. `BSS = 1 − Brier_model / Brier_ref` measures skill over the
  no-skill forecast — and the reference is each test day's OWN base rate (a
  perfect climatology forecast), not the pooled global rate, so the score
  credits only skill the ranking adds beyond simply knowing the day's rate.
- **Regime-conditioned, not only pooled.** A single pooled reliability table
  can look calibrated on average while being miscalibrated inside every
  regime. Calibration is also computed within low / medium / high
  base-rate-day strata so a regime-specific miscalibration cannot hide.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_BINS = 10
# Bin-edge FP guard: (0.3 * 10) is 2.9999999999999996 in binary, which would
# drop a predicted 0.30 into the [0.2, 0.3) bucket. A round-number bin test is
# a false all-clear (CLAUDE.md: test numeric boundaries at adversarial values) —
# nudge before floor so an exact edge lands in the bucket it names.
_EDGE_EPSILON = 1e-9


def _bin_indices(probs: np.ndarray, n_bins: int) -> np.ndarray:
    """Fixed-width bucket index in [0, n_bins-1] for each predicted prob.

    Bucket b spans [b/n_bins, (b+1)/n_bins); the top edge 1.0 falls in the last
    bucket. Epsilon-guarded so a value exactly on an edge lands in the bucket it
    names, not one below it through float representation error.
    """
    idx = np.floor(probs * n_bins + _EDGE_EPSILON).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def reliability_table(probs, labels, n_bins: int = DEFAULT_BINS) -> list[dict]:
    """Per-bucket (mean predicted prob, realized reach-rate, count).

    The core "does 70% mean 70%" artifact: for each fixed 0.1-wide predicted
    bucket, how often the event actually happened. Empty buckets carry None
    means (never a silent 0 that a `>= 0` check would swallow) with count 0.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"probs/labels length mismatch: {p.shape} vs {y.shape}")
    bins = _bin_indices(p, n_bins)
    table = []
    for b in range(n_bins):
        mask = bins == b
        count = int(mask.sum())
        table.append(
            {
                "lo": round(b / n_bins, 4),
                "hi": round((b + 1) / n_bins, 4),
                "mean_pred": round(float(p[mask].mean()), 4) if count else None,
                "reach_rate": round(float(y[mask].mean()), 4) if count else None,
                "count": count,
            }
        )
    return table


def calibration_slope(table: list[dict]) -> float | None:
    """Count-weighted least-squares slope of realized reach-rate on predicted
    prob across occupied buckets.

    Slope ≈ 1 with the points near the diagonal means calibrated; slope < 1 is
    overconfident (predictions too extreme), slope > 1 underconfident. None when
    fewer than two occupied buckets or no spread in predicted prob (slope
    undefined), so a degenerate stratum never fabricates a number.
    """
    pts = [(r["mean_pred"], r["reach_rate"], r["count"]) for r in table if r["count"] > 0]
    if len(pts) < 2:
        return None
    x = np.array([p for p, _, _ in pts], dtype=float)
    y = np.array([r for _, r, _ in pts], dtype=float)
    w = np.array([c for _, _, c in pts], dtype=float)
    xbar = np.average(x, weights=w)
    denom = float(np.sum(w * (x - xbar) ** 2))
    if denom <= 0:
        return None
    ybar = np.average(y, weights=w)
    slope = float(np.sum(w * (x - xbar) * (y - ybar)) / denom)
    return round(slope, 4)


def brier_skill_score(probs, labels, dates) -> dict:
    """Brier skill score vs the DAILY base-rate reference forecast.

    Brier_model = mean((p − y)²). The reference forecast for every row is that
    row's own day base rate (mean realized reach on that test day) — the honest
    no-skill "just predict the day's base rate" baseline. BSS = 1 − model/ref;
    positive means the ranking adds skill beyond knowing the daily base rate.

    Returns brier (model), brier_ref, and bss (None when the reference is a
    perfect 0 — every day all-0 or all-1 — so BSS is undefined rather than
    dividing by zero).
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    frame = pd.DataFrame({"p": p, "y": y, "d": np.asarray(dates)})
    base_d = frame.groupby("d")["y"].transform("mean").to_numpy()
    brier_model = float(np.mean((p - y) ** 2))
    brier_ref = float(np.mean((base_d - y) ** 2))
    bss = 1.0 - brier_model / brier_ref if brier_ref > 0 else None
    return {
        "brier": round(brier_model, 5),
        "brier_ref": round(brier_ref, 5),
        "bss": round(float(bss), 4) if bss is not None else None,
    }


# Base-rate-day strata: the daily base rate swings ~7%→98%, so calibration is
# also measured within tertiles of the per-day base rate. Tertile cuts adapt to
# the data (fixed cuts would misclassify a whole regime); ordered low→high.
_REGIME_NAMES = ("low", "medium", "high")


def _regime_of(base: float, lo_cut: float, hi_cut: float) -> str:
    if base <= lo_cut:
        return "low"
    if base > hi_cut:
        return "high"
    return "medium"


def regime_reports(probs, labels, dates, n_bins: int = DEFAULT_BINS) -> list[dict]:
    """Per-regime calibration: BSS + slope within low/medium/high base-rate days.

    Days are stratified by tertiles of their own base rate (the reference the
    BSS already uses), so a pooled table that looks calibrated on average cannot
    hide a regime that is systematically over/under-confident. Each report
    carries the stratum's base-rate span, day/row counts, BSS, and slope. Empty
    strata (possible under ties in the daily base rate) are omitted.
    """
    frame = pd.DataFrame(
        {"p": np.asarray(probs, float), "y": np.asarray(labels, float), "d": np.asarray(dates)}
    )
    day_base = frame.groupby("d")["y"].mean()
    lo_cut = float(day_base.quantile(1 / 3))
    hi_cut = float(day_base.quantile(2 / 3))
    regime = {d: _regime_of(b, lo_cut, hi_cut) for d, b in day_base.items()}
    frame["regime"] = frame["d"].map(regime)
    frame["base_d"] = frame["d"].map(day_base)

    reports = []
    for name in _REGIME_NAMES:
        sub = frame[frame["regime"] == name]
        if sub.empty:
            continue
        bss = brier_skill_score(sub["p"], sub["y"], sub["d"])
        table = reliability_table(sub["p"], sub["y"], n_bins)
        reports.append(
            {
                "name": name,
                "base_lo": round(float(sub["base_d"].min()), 4),
                "base_hi": round(float(sub["base_d"].max()), 4),
                "n_days": int(sub["d"].nunique()),
                "n": int(len(sub)),
                "bss": bss["bss"],
                "slope": calibration_slope(table),
            }
        )
    return reports


def calibration_report(probs, labels, dates, n_bins: int = DEFAULT_BINS) -> dict:
    """The full walk-forward calibration block recorded into experiment metrics.

    Population: the all-names test-fold predictions — the SAME rows that feed the
    benchmark's AUC and Brier, out-of-sample by construction (each fold predicts
    only test months it never trained on). Pooled reliability + BSS, plus the
    per-regime reports so a regime-specific miscalibration is visible.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"probs/labels length mismatch: {p.shape} vs {y.shape}")
    table = reliability_table(p, y, n_bins)
    skill = brier_skill_score(p, y, dates)
    return {
        "population": "all_names_test",
        "n": int(len(p)),
        "n_days": int(pd.Series(np.asarray(dates)).nunique()),
        "n_bins": n_bins,
        "brier": skill["brier"],
        "brier_ref": skill["brier_ref"],
        "bss": skill["bss"],
        "slope": calibration_slope(table),
        "reliability": table,
        "regimes": regime_reports(p, y, dates, n_bins),
    }
