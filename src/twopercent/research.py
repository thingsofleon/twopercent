"""Overnight research loop: bounded experiment queue → referee → candidates.

Level-4 shape: the GPU works through `research/queue.json` overnight. Every
config runs through the SAME walk-forward referee as any human benchmark
(12-month standard, top-20, recorded to the experiments ledger), and anything
that beats the champion on lift OUTSIDE the noise band is surfaced as a
`promotion-candidate` GitHub issue. Champion promotion stays human-only, by
PR, after the standard referee run plus quant-skeptic review.

Guardrails (recorded in ROADMAP Level 4):

- **Clock gate.** Runs only between 16:30 and 05:00 America/Denver, any day
  (weekends fine — pure offline compute). This one rule stays clear of US
  market hours AND the 06:00 predict / 14:45 score routine windows: DuckDB is
  single-writer, so research I/O must never collide with them. Defense in
  depth: the routine's market-hours helper is consulted too.
- **Budget cap** (--budget, default 8) bounds a night's compute.
- **Queue state lives in the experiments ledger, not a state file.** A config
  whose (strategy, strategy_params) already has a recorded standard benchmark
  is skipped (loudly counted), so the queue is idempotent and re-runnable and
  a scheduled run never dirties the git-tracked queue file.
- **One crash must not kill the night.** A crashed config WARNs, is NOT
  recorded, and therefore retries next night (noted in the digest).
- **The runner writes NOTHING except experiments-ledger rows** and possibly
  the promotion-candidate issue — never champion.json, predictions, or prices.
- **Multiple comparisons.** An overnight sweep's best result is the best of
  many trials against the same test months; the candidate issue says so and
  must be treated as a hypothesis, not a promotion.

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

_DENVER = ZoneInfo("America/Denver")
WINDOW_OPEN = dt.time(16, 30)  # research may start (post score-run, post-close)
WINDOW_CLOSE = dt.time(5, 0)  # research must be done (pre predict-run, pre-open)


def _now_denver() -> dt.datetime:
    return dt.datetime.now(tz=_DENVER)


def _in_research_window(now: dt.datetime) -> bool:
    """Only between 16:30 and 05:00 America/Denver (any day) — clear of market
    hours and both routine runs. The market-hours helper is consulted as
    defense in depth (it can only fire if the window rule is ever broken)."""
    t = now.time()
    in_window = t >= WINDOW_OPEN or t < WINDOW_CLOSE
    return in_window and not _market_is_open(now.astimezone(ZoneInfo("America/New_York")))


@dataclass(frozen=True)
class QueueEntry:
    strategy: str
    params: dict
    note: str = ""

    def key(self) -> tuple[str, str]:
        return self.strategy, json.dumps(self.params, sort_keys=True)

    def label(self) -> str:
        return f"{self.strategy} {json.dumps(self.params, sort_keys=True)}"


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
        keys.add((strategy, json.dumps(params.get("strategy_params") or {}, sort_keys=True)))
    return keys


def _latest_standard_experiment(
    con: duckdb.DuckDBPyConnection, strategy: str
) -> tuple[int, dict] | None:
    """The champion's newest recorded standard (12-month, top-20) benchmark."""
    rows = con.execute(
        "SELECT id, params, metrics FROM experiments WHERE strategy = ? "
        "ORDER BY run_ts DESC, id DESC",
        [strategy],
    ).fetchall()
    for exp_id, params_json, metrics_json in rows:
        try:
            params = json.loads(params_json) if params_json else {}
            metrics = json.loads(metrics_json)
        except json.JSONDecodeError:
            logger.warning("experiments row #%s has unparseable JSON — ignored", exp_id)
            continue
        if params.get("months") == STANDARD_MONTHS and params.get("top_n") == STANDARD_TOP_N:
            return exp_id, metrics
    return None


def _promotion_body(
    best: dict,
    n_swept: int,
    champ: str,
    champ_id: int,
    champ_metrics: dict,
) -> str:
    metric_keys = sorted(set(best["metrics"]) | set(champ_metrics))
    lines = [
        "Auto-filed by `twopercent research`: an overnight experiment beat the champion "
        "on lift outside the noise band. **This is a hypothesis, not a promotion.**",
        "",
        f"- Challenger: `{best['strategy']}` — experiments ledger id **#{best['experiment_id']}**",
        f"- Challenger params: `{json.dumps(best['params'], sort_keys=True)}`"
        + (f" ({best['note']})" if best.get("note") else ""),
        f"- Champion: `{champ}` — experiments ledger id **#{champ_id}**",
        f"- Verdict: {best['verdict']}",
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
        "## Promotion rules",
        "",
        "- Promotion is HUMAN-ONLY, by PR editing `champion.json`, after a standard "
        "referee benchmark plus `quant-skeptic` review of the strategy/config.",
        "- Decide on lift/AUC — NEVER on sim growth (tail-dominated, survivorship-"
        "flattered; see backtest.py).",
        "",
        "## Multiple-comparisons caveat",
        "",
        f"This candidate is the best of an overnight sweep ({n_swept} config(s) run "
        "tonight; more across nights) evaluated against the SAME 12 test months. The "
        "best of many trials is inflated by selection — treat it as a hypothesis and "
        "expect the edge to shrink out of sample (holdout-months discipline is tracked "
        "as a follow-up issue).",
    ]
    return "\n".join(lines) + "\n"


def _file_candidate_issue(
    report: ResearchReport, best: dict, n_swept: int, champ: str, champ_id: int, champ_metrics: dict
) -> None:
    title = (
        f"Research: {best['strategy']} beats champion {champ} on lift "
        f"({best['metrics'].get('lift')} vs {champ_metrics.get('lift')})"
    )
    result = issues.file_issue(
        label=PROMOTION_LABEL,
        title=title,
        body=_promotion_body(best, n_swept, champ, champ_id, champ_metrics),
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
            champ_id, champ_metrics = latest
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

    for entry in batch:
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
        # the one run_benchmark just recorded.
        exp_id = con.execute("SELECT max(id) FROM experiments").fetchone()[0]
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
            }

    if report.best is not None and champ_metrics is not None:
        winner = lift_winner(
            "challenger",
            report.best["metrics"].get("lift"),
            "champion",
            champ_metrics.get("lift"),
        )
        if winner == "challenger":
            _file_candidate_issue(report, report.best, report.n_ran, champ, champ_id, champ_metrics)
    return report
