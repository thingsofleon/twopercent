"""The morning routine: the daily cycle as one gated command.

Level-3 shape: a deterministic pipeline that runs unattended and reports by
exception. Gate design (rebuilt after quant-skeptic review of the first
draft):

- **Market-hours guard first.** A mid-session run would ingest today's
  partial bar, fabricating features AND the prior day's label. The routine
  refuses to run during US market hours. (Defense in depth: ingest always
  refetches each symbol's last stored bar — never skips it — so a partial
  bar written by any bypassed run is overwritten by the next run.)
- **Wall-clock staleness gate.** A store whose newest bar is old produces
  "predictions" for days that already happened; those must never enter the
  track record.
- **The real corruption gate runs AFTER ingest**, on what ingest could
  actually introduce: newly extreme bars, new zero-volume runs, new invalid
  rows (symbol-set difference vs the pre-ingest baseline). Pre-existing
  problems warn; new ones abort before the model trains.
- Coverage holes (universe symbols without prices) only WARN — the ingest
  step each morning is exactly what heals them, so hard-failing on them
  before ingest would deadlock the routine permanently.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from twopercent import champion, dashboard, doctor, ingest, store, track, universe
from twopercent.predict import predict_for

logger = logging.getLogger(__name__)

UNIVERSE_MAX_AGE_DAYS = 7
INGEST_FAIL_FRACTION = 0.05  # more than this share of symbols failing = step failure
MAX_STORE_AGE_DAYS = 5  # newest bar older than this = predictions would be stale

_EASTERN = ZoneInfo("America/New_York")

OK, WARN, FAIL = "ok", "warn", "fail"
_RANK = {OK: 0, WARN: 1, FAIL: 2}


def _now_eastern() -> dt.datetime:
    return dt.datetime.now(tz=_EASTERN)


def _market_is_open(now: dt.datetime) -> bool:
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dt.time(9, 25) <= t < dt.time(16, 15)


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


def _symbols(frame: pd.DataFrame) -> set[str]:
    return set(frame["symbol"]) if not frame.empty and "symbol" in frame.columns else set()


def _bar_keys(frame: pd.DataFrame) -> set[tuple[str, str]]:
    """(symbol, date) pairs — same-symbol new corruption must not hide behind
    a baseline extreme on that symbol."""
    if frame.empty or "symbol" not in frame.columns or "date" not in frame.columns:
        return set()
    return {(s, str(d)) for s, d in zip(frame["symbol"], frame["date"], strict=True)}


def run(
    db_path: Path | str = store.DEFAULT_DB_PATH,
    out_path: str = "dashboard.html",
    top: int = 20,
    universe_max_age_days: int = UNIVERSE_MAX_AGE_DAYS,
) -> RoutineReport:
    report = RoutineReport()

    now = _now_eastern()
    if _market_is_open(now):
        report.add(
            "clock",
            FAIL,
            f"US market is open ({now:%H:%M} ET) — a run now would ingest a partial "
            "bar and fabricate labels; run pre-open or post-close",
        )
        return report
    report.add("clock", OK, f"{now:%a %H:%M} ET, market closed")
    today = now.date()  # one clock: ET everywhere (a UTC host is 'tomorrow' after 20:00 ET)

    try:
        con = store.connect(db_path)
    except duckdb.IOException as exc:
        report.add("connect", FAIL, f"database locked or unreadable: {exc}")
        return report

    pre = doctor.run(con)
    pre_problems = pre.problem_count
    report.add(
        "doctor", WARN if pre_problems else OK, f"{pre_problems} pre-existing problems (baseline)"
    )

    uni = store.latest_universe(con)
    if uni.empty:
        try:
            fresh = universe.refresh_universe()
            n = store.upsert_universe(con, fresh, as_of=today)
            report.add("universe", OK, f"bootstrapped: {n} symbols")
        except Exception as exc:
            report.add("universe", FAIL, f"no universe stored and refresh failed: {exc}")
            return report
    else:
        age = (today - uni["as_of"].iloc[0].date()).days
        if age > universe_max_age_days:
            try:
                fresh = universe.refresh_universe()
                n = store.upsert_universe(con, fresh, as_of=today)
                report.add("universe", OK, f"refreshed: {n} symbols (was {age}d old)")
            except Exception as exc:
                report.add("universe", WARN, f"refresh failed, using {age}d-old snapshot: {exc}")
        else:
            report.add("universe", OK, f"current ({age}d old, {len(uni)} symbols)")

    symbols = store.all_universe_symbols(con)
    if not symbols:
        report.add("ingest", FAIL, "no symbols to ingest — empty universe")
        return report
    try:
        result = ingest.ingest(con, symbols)
        accounted = (
            len(result.symbols_ok) + len(result.symbols_skipped) + len(result.symbols_failed)
        )
        n_failed = len(result.symbols_failed)
        detail = (
            f"{result.rows_written} rows, {len(result.symbols_ok)} fetched, "
            f"{len(result.symbols_skipped)} current, {n_failed} failed"
        )
        if accounted != len(symbols):
            report.add(
                "ingest",
                FAIL,
                detail + f" — {len(symbols) - accounted} symbols unaccounted for (silent loss)",
            )
            return report
        if n_failed / len(symbols) > INGEST_FAIL_FRACTION:
            report.add("ingest", FAIL, detail + " (failure rate over threshold)")
            return report
        report.add("ingest", WARN if n_failed else OK, detail)
    except Exception as exc:
        report.add("ingest", FAIL, f"crashed: {exc}")
        return report

    post = doctor.run(con)
    max_date = post.max_date
    if max_date is None or (today - max_date).days > MAX_STORE_AGE_DAYS:
        report.add(
            "freshness",
            FAIL,
            f"newest bar is {max_date} — too old to predict from; "
            "predictions for elapsed days must not enter the track record",
        )
        return report
    report.add("freshness", OK, f"newest bar {max_date}")

    # New corruption = (symbol, date) pairs absent from the pre-ingest
    # baseline. Only RECENT pairs hard-fail: a weekly-refresh backfill of a
    # volatile symbol legitimately adds years-old extreme bars, and a gate
    # that cries wolf weekly trains its operator to ignore exit 2.
    recent_cutoff = (pre.max_date or dt.date.min) - dt.timedelta(days=3)
    new_extreme = _bar_keys(post.extreme) - _bar_keys(pre.extreme)
    new_invalid = _symbols(post.invalid) - _symbols(pre.invalid)
    new_zero = _symbols(post.zero_runs) - _symbols(pre.zero_runs)
    recent_extreme = {k for k in new_extreme if k[1] >= str(recent_cutoff)}
    backfill_extreme = new_extreme - recent_extreme
    if recent_extreme or new_invalid:
        examples = ", ".join(sorted({s for s, _ in recent_extreme} | new_invalid)[:5])
        report.add(
            "recheck",
            FAIL,
            f"today's ingest introduced corruption ({len(recent_extreme)} recent extreme "
            f"bars, {len(new_invalid)} invalid symbols: {examples}) — aborting before the model",
        )
        return report
    coverage_holes = len(post.universe_missing_prices)
    recheck_detail = "no new corruption from today's ingest"
    if backfill_extreme:
        recheck_detail += (
            f"; {len(backfill_extreme)} historical extreme bars from backfill (review)"
        )
    if new_zero:
        recheck_detail += f"; {len(new_zero)} new zero-volume symbols"
    if coverage_holes:
        recheck_detail += f"; {coverage_holes} universe symbols still without prices"
    soft = backfill_extreme or new_zero or coverage_holes
    report.add("recheck", WARN if soft else OK, recheck_detail)

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
        dashboard.render(con, name, out_path, top=top, result=prediction)
        report.add("dashboard", OK, f"written to {out_path}")
    except Exception as exc:
        report.add("dashboard", WARN, f"render failed (predictions are logged): {exc}")

    try:
        scored = track.score_predictions(con, name, top_n=top).scored
        if not scored.empty:
            last = scored.iloc[-1]
            report.last_scored = (
                f"{last['target_date']}: {int(last['hits'])}/{int(last['n_scored'])} hit "
                f"({last['precision']:.0%} vs base {last['base_rate']:.0%})"
            )
        report.add("scoring", OK, f"{len(scored)} days in track record")
    except Exception as exc:
        report.add("scoring", WARN, f"track-record scoring failed: {exc}")
    return report
