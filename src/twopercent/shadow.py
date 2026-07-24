"""Shadow-trading engine: challenger strategies run in parallel with the
champion, logged and scored, but NEVER emailed and NEVER touching the
champion's live record.

Tier-2-foundation of the autonomous research engine (ROADMAP "Autonomous
research engine"). This tier is the ISOLATION FOUNDATION only — it LOGS and
SCORES shadow picks; it promotes nothing (that is #59, separately gated).

**The invariant this module exists to guarantee:** nothing that reads the
`predictions` table — `track.score_predictions`, the money tiles, the
dashboard, the signal email — may ever be affected by shadow picks. Shadow
storage is a SEPARATE table (`shadow_predictions`); a challenger is scored with
the same walk-forward machinery (`predict_for(..., save=False)` then
`store.save_shadow_predictions`) and the same 09:30-ET live/late rule as the
champion, but its rows physically cannot reach the champion's path.

Each predict morning `run_shadow` produces + logs the rostered challengers'
picks (per-challenger crash isolation — one bad challenger never stops the
others or the routine). Next day `score_shadow` scores them against actuals,
accumulating a genuine FORWARD live track record (live-only, backfills
excluded exactly like the champion).

Roster: `research/shadow.json`, a JSON list of {strategy, params, note} (same
shape as research/queue.json), edited via PR. It ships EMPTY — nothing is
shadowed until a human/PR adds a challenger.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np

from twopercent import store, track
from twopercent.canonical import canonical_params
from twopercent.predict import predict_for

logger = logging.getLogger(__name__)

SHADOW_PATH = Path("research/shadow.json")
# Concurrency cap: at most this many challengers shadowed at once. It bounds
# the compute added to every predict morning AND is the K that the shadow-gate
# forward-margin accounting (#60) scales against — a max-over-K selection.
MAX_SHADOW = 4
# The forward record uses the shipped top-20 basket, same as the champion's
# degradation detector, so champion and challenger records are comparable.
SHADOW_TOP_N = 20


@dataclass(frozen=True)
class ShadowEntry:
    strategy: str
    params: dict
    note: str = ""

    def challenger(self) -> str:
        """Canonical identity: 'strategy {canonical-json-params}'. Two entries
        that differ only by params are distinct challengers; the champion's
        strategy name with default params is still a distinct identity."""
        return f"{self.strategy} {canonical_params(self.params)}"


def load_roster(path: Path | str | None = None) -> tuple[list[ShadowEntry], int, int]:
    """(entries, malformed_count, dropped_over_cap).

    Mirrors research.load_queue's hygiene: a malformed ENTRY is skipped LOUDLY
    and counted (one bad hand-edit must not cancel the roster). A missing or
    unreadable file → EMPTY roster with a WARN (not fatal — no roster just means
    nothing is shadowed). If the roster exceeds MAX_SHADOW, the first MAX_SHADOW
    are kept and the rest DROPPED with a loud warning naming the count — never a
    silent truncation. `path` defaults (None) to SHADOW_PATH resolved at call
    time, so the shipped roster location can be overridden per call.
    """
    path = Path(path) if path is not None else SHADOW_PATH
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        logger.warning("shadow roster %s does not exist — no challengers shadowed", path)
        return [], 0, 0
    except (OSError, ValueError) as exc:
        # ValueError covers json.JSONDecodeError (and a non-UTF-8 read).
        logger.warning("shadow roster %s is unreadable (%s) — no challengers shadowed", path, exc)
        return [], 0, 0
    if not isinstance(raw, list):
        logger.warning("shadow roster %s is not a JSON list — no challengers shadowed", path)
        return [], 0, 0
    entries: list[ShadowEntry] = []
    malformed = 0
    for i, item in enumerate(raw):
        strategy = item.get("strategy") if isinstance(item, dict) else None
        params = item.get("params", {}) if isinstance(item, dict) else None
        if not isinstance(strategy, str) or not strategy or not isinstance(params, dict):
            malformed += 1
            logger.warning("shadow roster entry %d is malformed — SKIPPED: %r", i, item)
            continue
        entries.append(ShadowEntry(strategy, params, str(item.get("note", ""))))
    dropped = 0
    if len(entries) > MAX_SHADOW:
        dropped = len(entries) - MAX_SHADOW
        logger.warning(
            "shadow roster has %d valid challenger(s), over the MAX_SHADOW cap of %d — "
            "DROPPING %d (kept the first %d): %s",
            len(entries),
            MAX_SHADOW,
            dropped,
            MAX_SHADOW,
            ", ".join(e.challenger() for e in entries[MAX_SHADOW:]),
        )
        entries = entries[:MAX_SHADOW]
    return entries, malformed, dropped


@dataclass
class ShadowReport:
    """Outcome of a shadow predict run — pure counts; it gates nothing."""

    rostered: int = 0  # valid parsed challengers (kept + dropped-over-cap)
    ran: int = 0
    failed: int = 0
    dropped: int = 0  # over the MAX_SHADOW cap
    malformed: int = 0
    challengers: list[str] = field(default_factory=list)  # identities that ran

    @property
    def had_trouble(self) -> bool:
        return bool(self.failed or self.malformed or self.dropped)

    def summary(self) -> str:
        return (
            f"{self.ran} challenger(s) shadowed, {self.failed} failed, "
            f"{self.dropped} over-cap dropped, {self.malformed} malformed "
            f"({self.rostered} rostered)"
        )


def run_shadow(
    con: duckdb.DuckDBPyConnection, signal_date, roster_path: Path | str | None = None
) -> ShadowReport:
    """Produce and log each rostered challenger's picks for the day after
    `signal_date`, into shadow_predictions ONLY.

    Per-challenger crash isolation: a challenger that raises WARNs and the
    others still run; a total failure never raises to the caller (the routine's
    real signal email must never be delayed or failed by shadow compute). Writes
    nothing but shadow_predictions — never predictions, champion.json, or the
    dashboard."""
    entries, malformed, dropped = load_roster(roster_path)
    report = ShadowReport(rostered=len(entries) + dropped, dropped=dropped, malformed=malformed)
    for entry in entries:
        challenger = entry.challenger()
        try:
            result = predict_for(
                con,
                entry.strategy,
                signal_date,
                save=False,
                strategy_params=entry.params,
            )
            store.save_shadow_predictions(
                con,
                challenger,
                entry.strategy,
                canonical_params(entry.params),
                result.signal_date,
                result.scored,
            )
        except Exception as exc:
            report.failed += 1
            logger.warning(
                "shadow challenger %s crashed (%s) — SKIPPED; other challengers and the "
                "champion routine continue",
                challenger,
                exc,
            )
            continue
        report.ran += 1
        report.challengers.append(challenger)
        logger.info(
            "shadowed %s for day after %s: %d picks", challenger, signal_date, len(result.scored)
        )
    return report


@dataclass
class ShadowChallengerScore:
    """A challenger's accumulating FORWARD (live-only) record."""

    challenger: str
    strategy: str
    params: str
    live_days: int
    late_days: int
    pending: int
    mean_precision: float | None  # over live days
    mean_base_rate: float | None  # over live days
    mean_lift: float | None  # over live days with a defined (finite) lift
    live_lift_days: int  # live days that had a finite lift (contributed to mean_lift)


@dataclass
class ShadowScoreReport:
    """Outcome of scoring every challenger present in shadow_predictions."""

    scores: list[ShadowChallengerScore] = field(default_factory=list)
    failed: int = 0

    @property
    def had_trouble(self) -> bool:
        return bool(self.failed)

    def summary(self) -> str:
        return f"{len(self.scores)} challenger(s) scored, {self.failed} failed"


def _mean_or_none(values) -> float | None:
    """Mean of the finite values only, or None when none are finite. Guards the
    DuckDB/NumPy NaN traps: a NULL-lift (zero base rate) day must not poison the
    mean, and isfinite() keeps a stray NaN from sorting/aggregating wrong."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else None


def score_shadow(con: duckdb.DuckDBPyConnection) -> ShadowScoreReport:
    """Score every challenger in shadow_predictions, computing its accumulating
    FORWARD (LIVE-ONLY) record: mean per-day precision, base rate, and lift over
    days whose picks were created before the target day's 09:30 ET open.

    This tier LOGS and RETURNS the record; it makes NO promotion decision.
    Per-challenger isolation: one challenger's scoring crash WARNs and the rest
    still score. Reads only shadow_predictions."""
    report = ShadowScoreReport()
    for challenger, strategy, params in store.shadow_challengers(con):
        try:
            record = track.score_shadow_predictions(con, challenger, top_n=SHADOW_TOP_N)
        except Exception as exc:
            report.failed += 1
            logger.warning("shadow scoring for %s crashed (%s) — SKIPPED", challenger, exc)
            continue
        scored = record.scored
        if scored.empty:
            live = scored
            late_days = 0
        else:
            late_mask = scored["late"].astype(bool)
            live = scored[~late_mask]
            late_days = int(late_mask.sum())
        live_lift = live["lift"] if len(live) else []
        finite_lift = int(np.isfinite(np.asarray(live_lift, dtype=float)).sum()) if len(live) else 0
        score = ShadowChallengerScore(
            challenger=challenger,
            strategy=strategy,
            params=params,
            live_days=len(live),
            late_days=late_days,
            pending=len(record.pending),
            mean_precision=_mean_or_none(live["precision"]) if len(live) else None,
            mean_base_rate=_mean_or_none(live["base_rate"]) if len(live) else None,
            mean_lift=_mean_or_none(live_lift) if len(live) else None,
            live_lift_days=finite_lift,
        )
        report.scores.append(score)
        logger.info(
            "shadow record %s: %d live day(s) (%d late, %d pending), "
            "mean live lift %s over %d day(s) with a defined lift",
            challenger,
            score.live_days,
            score.late_days,
            score.pending,
            f"{score.mean_lift:.3g}" if score.mean_lift is not None else "n/a",
            score.live_lift_days,
        )
    return report
