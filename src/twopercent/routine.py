"""The morning routine: the daily cycle as one doctor-gated command.

Level-3 shape: a deterministic pipeline that runs unattended and reports by
exception. Hard data problems (invalid bars, coverage holes) abort BEFORE the
model runs; soft problems (gaps, staleness, suspicious bars) degrade the run
to a warning but let it finish. The summary is the exception report.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from twopercent import champion, dashboard, doctor, ingest, store, track, universe
from twopercent.predict import predict_for

logger = logging.getLogger(__name__)

UNIVERSE_MAX_AGE_DAYS = 7
INGEST_FAIL_FRACTION = 0.05  # more than this share of symbols failing = step failure

OK, WARN, FAIL = "ok", "warn", "fail"
_RANK = {OK: 0, WARN: 1, FAIL: 2}


@dataclass
class Step:
    name: str
    status: str
    detail: str


@dataclass
class RoutineReport:
    steps: list[Step] = field(default_factory=list)
    top_candidates: list[tuple[str, float]] = field(default_factory=list)
    last_scored: str = ""

    def add(self, name: str, status: str, detail: str) -> None:
        self.steps.append(Step(name, status, detail))
        logger.info("routine step %-10s %-4s %s", name, status.upper(), detail)

    @property
    def status(self) -> str:
        return max((s.status for s in self.steps), key=_RANK.get, default=FAIL)

    @property
    def exit_code(self) -> int:
        return _RANK[self.status]

    def summary_lines(self) -> list[str]:
        lines = [f"routine: {self.status.upper()}"]
        lines += [f"  [{s.status:^4}] {s.name:<10} {s.detail}" for s in self.steps]
        if self.top_candidates:
            ranked = ", ".join(f"{sym} p={p:.3f}" for sym, p in self.top_candidates)
            lines.append(f"  top candidates: {ranked}")
        if self.last_scored:
            lines.append(f"  last scored day: {self.last_scored}")
        return lines


def _doctor_gate(report: doctor.DoctorReport) -> tuple[str, str]:
    """Invalid bars and coverage holes are hard failures; the rest degrade."""
    hard = len(report.invalid) + len(report.universe_missing_prices)
    soft = (
        len(report.gaps)
        + len(report.stale)
        + len(report.extreme)
        + len(report.zero_runs)
        + len(report.prices_missing_meta)
    )
    detail = f"{hard} hard / {soft} soft problems"
    if hard:
        return FAIL, detail + " (invalid bars or coverage holes — aborting before model)"
    return (WARN if soft else OK), detail


def run(
    db_path: Path | str = store.DEFAULT_DB_PATH,
    out_path: str = "dashboard.html",
    top: int = 20,
    universe_max_age_days: int = UNIVERSE_MAX_AGE_DAYS,
) -> RoutineReport:
    report = RoutineReport()
    try:
        con = store.connect(db_path)
    except duckdb.IOException as exc:
        report.add("connect", FAIL, f"database locked or unreadable: {exc}")
        return report

    pre = doctor.run(con)
    status, detail = _doctor_gate(pre)
    report.add("doctor", status, detail)
    if status == FAIL:
        return report

    try:
        uni = store.latest_universe(con)
        age = (dt.date.today() - uni["as_of"].iloc[0].date()).days if not uni.empty else 10**6
        if age > universe_max_age_days:
            fresh = universe.refresh_universe()
            n = store.upsert_universe(con, fresh, as_of=dt.date.today())
            report.add("universe", OK, f"refreshed: {n} symbols (was {age}d old)")
        else:
            report.add("universe", OK, f"current ({age}d old, {len(uni)} symbols)")
    except Exception as exc:
        # A stale-but-present universe is usable; refresh failure degrades only.
        report.add("universe", WARN, f"refresh failed, using existing: {exc}")

    try:
        result = ingest.ingest(con, store.all_universe_symbols(con))
        n_failed, n_total = (
            len(result.symbols_failed),
            max(
                1, len(result.symbols_ok) + len(result.symbols_skipped) + len(result.symbols_failed)
            ),
        )
        detail = (
            f"{result.rows_written} rows, {len(result.symbols_ok)} fetched, "
            f"{len(result.symbols_skipped)} current, {n_failed} failed"
        )
        if n_failed / n_total > INGEST_FAIL_FRACTION:
            report.add("ingest", FAIL, detail + " (failure rate over threshold)")
            return report
        report.add("ingest", WARN if n_failed else OK, detail)
    except Exception as exc:
        report.add("ingest", FAIL, f"crashed: {exc}")
        return report

    post = doctor.run(con)
    new_invalid = len(post.invalid) - len(pre.invalid)
    if new_invalid > 0:
        report.add("recheck", FAIL, f"today's ingest introduced {new_invalid} invalid-bar rows")
        return report
    report.add("recheck", OK, "no new invalid bars from today's ingest")

    try:
        name = champion.get_champion()
        prediction = predict_for(con, name, save=True)
        head = prediction.scored.head(5)
        report.top_candidates = list(zip(head["symbol"], head["prob"], strict=True))
        report.add(
            "predict",
            OK,
            f"{name}: {len(prediction.scored)} symbols scored for day after "
            f"{prediction.signal_date}, trained on {prediction.trained_rows:,} rows",
        )
    except Exception as exc:
        report.add("predict", FAIL, f"crashed: {exc}")
        return report

    try:
        dashboard.render(con, name, out_path, top=top)
        report.add("dashboard", OK, f"written to {out_path}")
    except Exception as exc:
        report.add("dashboard", WARN, f"render failed (predictions are logged): {exc}")

    scored = track.score_predictions(con, name, top_n=top).scored
    if not scored.empty:
        last = scored.iloc[-1]
        report.last_scored = (
            f"{last['target_date']}: {int(last['hits'])}/{int(last['n_scored'])} hit "
            f"({last['precision']:.0%} vs base {last['base_rate']:.0%})"
        )
    return report
