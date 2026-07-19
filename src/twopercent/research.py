"""Overnight research loop: bounded experiment queue → referee → candidates.

Level-4 shape: the GPU works through `research/queue.json` overnight. Every
config runs through the SAME walk-forward referee as any human benchmark
(12-month standard, top-20, recorded to the experiments ledger), and anything
that beats the champion on lift beyond the PROMOTION band AND survives the
disjoint-halves confirmation is surfaced as a `promotion-candidate` GitHub
issue. Champion promotion stays human-only, by PR, after the standard referee
run, quant-skeptic review, and a wall-clock holdout (see the issue template).

Guardrails (recorded in ROADMAP Level 4):

- **Clock gate.** Runs only between 16:30 and 05:00 America/Denver, any day
  (weekends fine — pure offline compute). This one rule stays clear of US
  market hours AND the 06:00 predict / 14:45 score routine windows: DuckDB is
  single-writer, so research I/O must never collide with them. Defense in
  depth: the routine's market-hours helper is consulted too. The window is
  also rechecked before EACH experiment with a one-hour must-finish margin
  (no new config starts after 04:00), so a slow night can never hold the
  store into the 06:00 predict run.
- **Budget cap** (--budget, default 8, must be >= 1) bounds a night's compute.
- **Queue state lives in the experiments ledger, not a state file.** A config
  whose (strategy, strategy_params) already has a recorded standard benchmark
  is skipped (loudly counted), so the queue is idempotent and re-runnable and
  a scheduled run never dirties the git-tracked queue file.
- **One crash must not kill the night.** A crashed config WARNs, is NOT
  recorded, and therefore retries next night (noted in the digest).
- **The runner writes NOTHING except experiments-ledger rows** and possibly
  the promotion-candidate issue — never champion.json, predictions, or prices.
- **Multiple comparisons.** An overnight sweep's best result is the best of
  many trials against the same test months. Three defenses: the promotion
  band is 0.25 lift (vs compare's 0.1 — see RESEARCH_PROMOTION_BAND), a
  candidate must also beat the champion on BOTH disjoint halves of their
  shared test days, and every candidate issue demands a wall-clock holdout
  (>= 2 post-candidate months) before any promotion PR. Still a hypothesis,
  never a promotion.

Exit codes: 0 clean (including empty queue), 1 some experiments failed or
queue entries were malformed, 2 the runner itself failed.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb

from twopercent import backtest, champion, issues, store
from twopercent.compare import compare_verdict, lift_winner
from twopercent.routine import _RANK, FAIL, OK, WARN, Step, _market_is_open
from twopercent.strategies import xgb_gbm

logger = logging.getLogger(__name__)

DEFAULT_BUDGET = 8
STANDARD_MONTHS = 12  # the referee standard: promotion evidence is 12-month walk-forward
STANDARD_TOP_N = 20
QUEUE_PATH = Path("research/queue.json")
PROMOTION_LABEL = "promotion-candidate"

# Promotion band, deliberately wider than compare.LIFT_NOISE_BAND (0.1): the
# compare CLI makes ONE comparison; a seeded sweep makes ~24 against the same
# test months. With SE(delta-lift) ~ 0.065-0.08 on ~250 shared test days, 0.1
# is only a ~1.5-sigma test — a null 24-config sweep would produce a spurious
# "candidate" with ~45% probability. 0.25 corresponds to z ~ 3.1-3.8, holding
# the family-wise false-candidate rate near a few percent.
RESEARCH_PROMOTION_BAND = 0.25

_DENVER = ZoneInfo("America/Denver")
WINDOW_OPEN = dt.time(16, 30)  # research may start (post score-run, post-close)
WINDOW_CLOSE = dt.time(5, 0)  # research must be done (pre predict-run, pre-open)
# Must-finish margin: no NEW experiment starts after this, leaving a full hour
# of the window for the one already running to finish before 05:00.
WINDOW_LAST_START = dt.time(4, 0)


def _now_denver() -> dt.datetime:
    return dt.datetime.now(tz=_DENVER)


def _in_research_window(now: dt.datetime) -> bool:
    """Only between 16:30 and 05:00 America/Denver (any day) — clear of market
    hours and both routine runs. The market-hours helper is consulted as
    defense in depth (it can only fire if the window rule is ever broken)."""
    t = now.time()
    in_window = t >= WINDOW_OPEN or t < WINDOW_CLOSE
    return in_window and not _market_is_open(now.astimezone(ZoneInfo("America/New_York")))


def _may_start_next(now: dt.datetime) -> bool:
    """May another experiment START now? Entry gating alone is not enough: a
    late start or a slow (CPU-fallback) night would otherwise run past 05:00
    and hold the single-writer store into the 06:00 predict run."""
    t = now.time()
    return t >= WINDOW_OPEN or t < WINDOW_LAST_START


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


@dataclass(frozen=True)
class QueueEntry:
    strategy: str
    params: dict
    note: str = ""

    def key(self) -> tuple[str, str]:
        return self.strategy, canonical_params(self.params)

    def label(self) -> str:
        return f"{self.strategy} {canonical_params(self.params)}"


@dataclass
class ResearchReport:
    budget: int = DEFAULT_BUDGET
    steps: list[Step] = field(default_factory=list)
    fatal: bool = False
    n_ran: int = 0
    n_failed: int = 0
    n_skipped_done: int = 0
    n_malformed: int = 0
    best: dict | None = None  # strategy, params, metrics, experiment_id, verdict
    champion_lift: float | None = None
    champion_auc: float | None = None

    def add(self, name: str, status: str, detail: str) -> None:
        self.steps.append(Step(name, status, detail))
        logger.info("research step %-10s %-4s %s", name, status.upper(), detail)

    @property
    def status(self) -> str:
        return max((s.status for s in self.steps), key=_RANK.get, default=FAIL)

    @property
    def exit_code(self) -> int:
        """0 clean/empty-queue, 1 experiments failed or queue entries malformed,
        2 runner failed — a scheduled job's headline signal."""
        if self.fatal:
            return 2
        if self.n_failed or self.n_malformed:
            return 1
        return 0

    def summary_lines(self) -> list[str]:
        lines = [f"research: {self.status.upper()}"]
        lines += [f"  [{s.status:^4}] {s.name:<10} {s.detail}" for s in self.steps]
        lines.append(
            f"  night: {self.n_ran} run, {self.n_skipped_done} skipped (already recorded), "
            f"{self.n_failed} failed, {self.n_malformed} malformed, budget {self.budget}"
        )
        if self.n_failed:
            lines.append("  crashed configs were NOT recorded — they retry next night")
        if self.best is not None:
            champ = (
                f"vs champion lift {self.champion_lift} auc {self.champion_auc}"
                if self.champion_lift is not None
                else "champion comparison unavailable"
            )
            lines.append(
                f"  best: {self.best['strategy']} {json.dumps(self.best['params'], sort_keys=True)}"
                f" lift {self.best['metrics'].get('lift')} auc {self.best['metrics'].get('auc')}"
                f" (exp #{self.best['experiment_id']}) {champ}"
            )
        elif self.n_ran:
            lines.append("  best: none (no successful experiment produced a lift)")
        device = xgb_gbm.device_in_use()
        if device is not None:
            lines.append(
                f"  xgb device: {'cuda' if device == 'cuda' else 'CPU FALLBACK (no CUDA)'}"
            )
        return lines


def load_queue(path: Path | str = QUEUE_PATH) -> tuple[list[QueueEntry] | None, int]:
    """(entries, malformed_count); entries is None when the file itself is unusable.

    A malformed ENTRY is skipped loudly and counted (exit 1, not a crash): one
    bad hand-edit must not cancel the rest of the night.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        logger.warning("research queue %s does not exist", path)
        return None, 0
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("research queue %s is unreadable: %s", path, exc)
        return None, 0
    if not isinstance(raw, list):
        logger.warning("research queue %s is not a JSON list", path)
        return None, 0
    entries: list[QueueEntry] = []
    malformed = 0
    for i, item in enumerate(raw):
        strategy = item.get("strategy") if isinstance(item, dict) else None
        params = item.get("params", {}) if isinstance(item, dict) else None
        if not isinstance(strategy, str) or not strategy or not isinstance(params, dict):
            malformed += 1
            logger.warning("research queue entry %d is malformed — SKIPPED: %r", i, item)
            continue
        entries.append(QueueEntry(strategy, params, str(item.get("note", ""))))
    return entries, malformed


def recorded_configs(con: duckdb.DuckDBPyConnection) -> set[tuple[str, str]]:
    """(strategy, canonical strategy_params JSON) for every recorded STANDARD
    benchmark — the queue's done-ledger. Non-standard runs (other months/top_n)
    never satisfy a queue config."""
    keys: set[tuple[str, str]] = set()
    for strategy, params_json in con.execute("SELECT strategy, params FROM experiments").fetchall():
        try:
            params = json.loads(params_json) if params_json else {}
        except json.JSONDecodeError:
            logger.warning("experiments row for %s has unparseable params — ignored", strategy)
            continue
        if params.get("months") != STANDARD_MONTHS or params.get("top_n") != STANDARD_TOP_N:
            continue
        keys.add((strategy, canonical_params(params.get("strategy_params") or {})))
    return keys


def _latest_standard_experiment(
    con: duckdb.DuckDBPyConnection, strategy: str
) -> tuple[int, dict, dt.date | None] | None:
    """The champion's newest recorded standard (12-month, top-20) DEFAULT-CONFIG
    benchmark, as (id, metrics, test_end).

    Rows with non-empty strategy_params are research variants recorded under
    the champion's strategy name — they must never be quoted as the champion
    (pre-research rows lack the key entirely; that counts as default-config).
    """
    rows = con.execute(
        "SELECT id, params, metrics, test_end FROM experiments WHERE strategy = ? "
        "ORDER BY run_ts DESC, id DESC",
        [strategy],
    ).fetchall()
    for exp_id, params_json, metrics_json, test_end in rows:
        try:
            params = json.loads(params_json) if params_json else {}
            metrics = json.loads(metrics_json)
        except json.JSONDecodeError:
            logger.warning("experiments row #%s has unparseable JSON — ignored", exp_id)
            continue
        if params.get("strategy_params"):
            continue  # parameterized variant, not the champion's own config
        if params.get("months") == STANDARD_MONTHS and params.get("top_n") == STANDARD_TOP_N:
            return exp_id, metrics, test_end
    return None


def _halves_hold(
    con: duckdb.DuckDBPyConnection, challenger_id: int, champion_id: int
) -> tuple[bool, str]:
    """Disjoint-halves confirmation: the challenger's daily top-N precision
    must beat the champion's on BOTH halves of their SHARED test days (split
    at the midpoint date) — a one-hot-month win fails this. Both sides' per-day
    outcomes are already in experiment_daily; on shared days the base rates are
    identical, so the precision-margin sign IS the lift-margin sign."""
    daily = con.execute(
        """
        SELECT target_date,
               avg(CASE WHEN seq = ? THEN hit END) AS challenger,
               avg(CASE WHEN seq = ? THEN hit END) AS champion
        FROM experiment_daily
        WHERE seq IN (?, ?)
        GROUP BY target_date
        HAVING challenger IS NOT NULL AND champion IS NOT NULL
        ORDER BY target_date
        """,
        [challenger_id, champion_id, challenger_id, champion_id],
    ).df()
    if len(daily) < 4:
        return False, (
            f"only {len(daily)} shared test day(s) between challenger #{challenger_id} "
            f"and champion #{champion_id} daily rows — cannot confirm"
        )
    mid = len(daily) // 2
    first, second = daily.iloc[:mid], daily.iloc[mid:]
    margin_1 = float(first["challenger"].mean() - first["champion"].mean())
    margin_2 = float(second["challenger"].mean() - second["champion"].mean())
    detail = (
        f"precision margin {margin_1:+.4f} (first half) / {margin_2:+.4f} (second half) "
        f"over {len(daily)} shared days"
    )
    return margin_1 > 0 and margin_2 > 0, detail


def _promotion_body(
    best: dict,
    n_swept: int,
    champ: str,
    champ_id: int,
    champ_metrics: dict,
    halves_detail: str,
) -> str:
    metric_keys = sorted(set(best["metrics"]) | set(champ_metrics))
    device = best.get("device")
    lines = [
        "Auto-filed by `twopercent research`: an overnight experiment beat the champion "
        f"on lift beyond the promotion band ({RESEARCH_PROMOTION_BAND}) and held the "
        "margin on both disjoint halves of the shared test window. "
        "**This is a hypothesis, not a promotion.**",
        "",
        f"- Challenger: `{best['strategy']}` — experiments ledger id **#{best['experiment_id']}**",
        f"- Challenger params: `{json.dumps(best['params'], sort_keys=True)}`"
        + (f" ({best['note']})" if best.get("note") else ""),
        f"- Trained on device: `{device or 'unrecorded'}`",
        f"- Champion: `{champ}` — experiments ledger id **#{champ_id}**",
        f"- Verdict: {best['verdict']}",
        f"- Disjoint-halves confirmation: {halves_detail}",
        "",
        "## Metrics (challenger vs champion, standard 12-month walk-forward)",
        "",
        "| metric | challenger | champion |",
        "|---|---|---|",
    ]
    for key in metric_keys:
        lines.append(f"| {key} | {best['metrics'].get(key)} | {champ_metrics.get(key)} |")
    lines += [
        "",
        "## Out-of-sample confirmation (REQUIRED before any promotion PR)",
        "",
        f"The lift margin must hold on **>= 2 months of data arriving AFTER this "
        f"candidate's test_end** (see ledger row #{best['experiment_id']}), computed by "
        "the same referee restricted to that window. The sweep months are burned by "
        "selection — re-running the deterministic referee on them proves nothing; only "
        "wall-clock-new data counts (#45).",
        "",
        "## Promotion rules",
        "",
        "- Promotion is HUMAN-ONLY, by PR editing `champion.json`, after the wall-clock "
        "holdout above plus `quant-skeptic` review of the strategy/config.",
        "- Decide on lift/AUC — NEVER on sim growth (tail-dominated, survivorship-"
        "flattered; see backtest.py).",
        "",
        "## Multiple-comparisons caveat",
        "",
        f"This candidate is the best of an overnight sweep ({n_swept} config(s) run "
        "tonight; more across nights) evaluated against the SAME 12 test months. The "
        "best of many trials is inflated by selection — treat it as a hypothesis and "
        "expect the edge to shrink out of sample.",
    ]
    return "\n".join(lines) + "\n"


def _file_candidate_issue(
    report: ResearchReport,
    best: dict,
    n_swept: int,
    champ: str,
    champ_id: int,
    champ_metrics: dict,
    halves_detail: str,
) -> None:
    title = (
        f"Research: {best['strategy']} beats champion {champ} on lift "
        f"({best['metrics'].get('lift')} vs {champ_metrics.get('lift')})"
    )
    result = issues.file_issue(
        label=PROMOTION_LABEL,
        title=title,
        body=_promotion_body(best, n_swept, champ, champ_id, champ_metrics, halves_detail),
        color="1D76DB",
        description="Auto-filed by twopercent research: challenger beat champion on lift",
    )
    if result.outcome == issues.FILED:
        report.add("issue", OK, f"filed {result.url} (conversation locked)")
    elif result.outcome == issues.DUPLICATE:
        report.add(
            "issue",
            WARN,
            f"open {PROMOTION_LABEL} issue already filed ({result.existing}) — not filing "
            "a duplicate; tonight's winner is in the digest and the experiments ledger",
        )
    elif result.outcome == issues.LOCK_FAILED:
        report.add(
            "issue",
            WARN,
            f"filed {result.url} but could not lock the conversation ({result.error}) — "
            "treat comments on it as untrusted",
        )
    else:
        report.add(
            "issue",
            WARN,
            f"gh failed ({result.error}) — a champion-beating candidate was found but NO "
            "issue was filed; see the digest and the experiments ledger",
        )


def run(
    db_path: Path | str = store.DEFAULT_DB_PATH,
    budget: int = DEFAULT_BUDGET,
    queue_path: Path | str = QUEUE_PATH,
) -> ResearchReport:
    report = ResearchReport(budget=budget)
    if budget < 1:
        report.add("budget", FAIL, f"--budget must be >= 1 (got {budget})")
        report.fatal = True
        return report
    now = _now_denver()
    if not _in_research_window(now):
        report.add(
            "clock",
            FAIL,
            f"{now:%a %H:%M} Denver is outside the research window — runs only between "
            f"{WINDOW_OPEN:%H:%M} and {WINDOW_CLOSE:%H:%M} Denver (any day), clear of "
            "market hours and the 06:00/14:45 routine runs (single-writer DuckDB)",
        )
        report.fatal = True
        return report
    report.add("clock", OK, f"{now:%a %H:%M} Denver, research window")

    try:
        con = store.connect(db_path)
    except duckdb.IOException as exc:
        report.add("connect", FAIL, f"database locked or unreadable: {exc}")
        report.fatal = True
        return report

    entries, report.n_malformed = load_queue(queue_path)
    if entries is None:
        report.add("queue", FAIL, f"queue {queue_path} missing or unreadable")
        report.fatal = True
        return report
    if not entries:
        detail = "queue is empty — nothing to research"
        if report.n_malformed:
            detail = (
                f"queue is empty after skipping {report.n_malformed} malformed "
                "entries — nothing to research"
            )
        report.add("queue", WARN if report.n_malformed else OK, detail)
        return report

    done = recorded_configs(con)
    pending = [e for e in entries if e.key() not in done]
    report.n_skipped_done = len(entries) - len(pending)
    batch = pending[:budget]
    report.add(
        "queue",
        WARN if report.n_malformed else OK,
        f"{len(entries)} valid config(s): {len(pending)} pending, "
        f"{report.n_skipped_done} already recorded, {report.n_malformed} malformed; "
        f"running {len(batch)} (budget {budget})",
    )
    if not batch:
        return report

    champ = None
    champ_id = None
    champ_metrics = None
    champ_test_end = None
    try:
        champ = champion.get_champion()
        latest = _latest_standard_experiment(con, champ)
        if latest is None:
            report.add(
                "champion",
                WARN,
                f"champion {champ} has no recorded standard ({STANDARD_MONTHS}-month, "
                f"top-{STANDARD_TOP_N}) benchmark — comparisons and promotion detection "
                "are OFF tonight; run `twopercent benchmark` first",
            )
        else:
            champ_id, champ_metrics, champ_test_end = latest
            report.champion_lift = champ_metrics.get("lift")
            report.champion_auc = champ_metrics.get("auc")
            report.add(
                "champion",
                OK,
                f"{champ} exp #{champ_id}: lift {report.champion_lift} auc {report.champion_auc}",
            )
    except Exception as exc:
        report.add(
            "champion",
            WARN,
            f"cannot resolve champion ({exc}) — comparisons and promotion detection "
            "are OFF tonight",
        )

    cpu_fallback_recorded = 0
    stale_warned = False
    for i, entry in enumerate(batch):
        now = _now_denver()
        if not _may_start_next(now):
            report.add(
                "window",
                WARN,
                f"stopped after {i} of {len(batch)}: research window closing "
                f"({now:%H:%M} Denver is past the {WINDOW_LAST_START:%H:%M} last-start "
                "margin) — remaining configs run next night",
            )
            break
        try:
            metrics = backtest.run_benchmark(
                con,
                entry.strategy,
                months=STANDARD_MONTHS,
                top_n=STANDARD_TOP_N,
                record=True,
                strategy_params=entry.params,
            )
        except Exception as exc:
            report.n_failed += 1
            report.add(
                "experiment",
                WARN,
                f"{entry.label()} crashed ({exc}) — NOT recorded, retries next night",
            )
            continue
        report.n_ran += 1
        # Single-process, single-writer store: the newest experiments row is
        # the one run_benchmark just recorded (atomically, with daily rows).
        exp_id, rec_params_json, rec_test_end = con.execute(
            "SELECT id, params, test_end FROM experiments ORDER BY id DESC LIMIT 1"
        ).fetchone()
        rec_params = json.loads(rec_params_json) if rec_params_json else {}
        device = rec_params.get("device")
        if device == "cpu" and entry.params.get("device") != "cpu":
            cpu_fallback_recorded += 1
        if (
            not stale_warned
            and champ_test_end is not None
            and rec_test_end is not None
            and (rec_test_end - champ_test_end).days > 31
        ):
            stale_warned = True
            report.add(
                "champion",
                WARN,
                f"champion reference benchmark is stale (test_end {champ_test_end} vs "
                f"challenger {rec_test_end}) — comparisons span different windows; "
                "re-run `twopercent benchmark` for a fresh reference",
            )
        verdict = ""
        if champ_metrics is not None:
            # Role names, not strategy names: a parameterized variant of the
            # champion strategy would otherwise be indistinguishable from it.
            verdict = compare_verdict(
                "challenger", metrics.get("lift"), f"champion {champ}", champ_metrics.get("lift")
            )
        detail = f"{entry.label()}: lift {metrics.get('lift')} auc {metrics.get('auc')}"
        detail += f" (exp #{exp_id})" + (f"; {verdict}" if verdict else "")
        report.add("experiment", OK, detail)
        lift = metrics.get("lift")
        if lift is not None and (report.best is None or lift > report.best["metrics"]["lift"]):
            report.best = {
                "strategy": entry.strategy,
                "params": entry.params,
                "note": entry.note,
                "metrics": metrics,
                "experiment_id": exp_id,
                "verdict": verdict,
                "device": device,
            }

    if cpu_fallback_recorded:
        report.add(
            "device",
            WARN,
            f"{cpu_fallback_recorded} config(s) recorded under CPU FALLBACK — they are "
            "done-keyed with CPU numbers; delete their ledger rows to re-run on GPU",
        )

    if report.best is not None and champ_metrics is not None:
        best_lift = report.best["metrics"].get("lift")
        champ_lift = champ_metrics.get("lift")
        winner = lift_winner(
            "challenger", best_lift, "champion", champ_lift, band=RESEARCH_PROMOTION_BAND
        )
        if winner == "challenger":
            halves_ok, halves_detail = _halves_hold(con, report.best["experiment_id"], champ_id)
            if halves_ok:
                _file_candidate_issue(
                    report, report.best, report.n_ran, champ, champ_id, champ_metrics, halves_detail
                )
            else:
                report.add(
                    "candidate",
                    WARN,
                    f"best lift {best_lift} beats champion {champ_lift} beyond the "
                    f"promotion band but FAILED the disjoint-halves confirmation "
                    f"({halves_detail}) — no issue filed (likely a one-hot-window win)",
                )
        elif best_lift is not None and champ_lift is not None and best_lift > champ_lift:
            report.add(
                "candidate",
                OK,
                f"no promotion candidate: best lift {best_lift} vs champion {champ_lift} "
                f"is within the promotion band ({RESEARCH_PROMOTION_BAND})",
            )
    return report
