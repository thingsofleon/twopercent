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

Two modes (level 4):

- **predict** (default) — the pre-open cycle above: universe refresh, ingest,
  gates, champion predict, dashboard, track-record scoring.
- **score** — the post-close run: ingest today's final bars, score pending
  predictions, run the degradation detector, and on DEGRADED auto-file an
  `auto-degradation` GitHub issue for the investigator agent. No predict
  step (nothing is logged to the predictions table), no universe refresh.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import subprocess  # noqa: F401  (tests patch the gh transport via routine.subprocess.run)
from dataclasses import dataclass, field
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from twopercent import champion, dashboard, doctor, ingest, issues, store, track, universe
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


SCORE_EARLIEST = dt.time(16, 15)
# The detector's lift series is defined on the top-20 daily basket (ROADMAP
# Level 4 locked decision). Pinned here so a --top override, which may still
# restyle the dashboard, can never change what the detector measures.
DETECTOR_TOP_N = 20


def _score_too_early(now: dt.datetime) -> bool:
    """Score mode is post-close only: on weekdays it refuses during market
    hours (same guard as predict) AND any earlier time — before 16:15 ET
    today's bar is partial or absent. Weekends are allowed: scoring time
    never affects the live/late flag (that depends only on when a prediction
    was CREATED relative to its target day's 09:30 ET open), so a weekend
    run just resolves pending days."""
    return now.weekday() < 5 and (_market_is_open(now) or now.time() < SCORE_EARLIEST)


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
    mode: str = "predict",
) -> RoutineReport:
    if mode not in ("predict", "score"):
        raise ValueError(f"unknown routine mode {mode!r} (expected 'predict' or 'score')")
    report = RoutineReport()
    now = _now_eastern()
    if mode == "score":
        return _run_score(report, now, db_path, out_path, top)
    return _run_predict(report, now, db_path, out_path, top, universe_max_age_days)


def _connect_step(report: RoutineReport, db_path: Path | str):
    try:
        return store.connect(db_path)
    except duckdb.IOException as exc:
        report.add("connect", FAIL, f"database locked or unreadable: {exc}")
        return None


def _doctor_baseline_step(report: RoutineReport, con) -> doctor.DoctorReport:
    pre = doctor.run(con)
    pre_problems = pre.problem_count
    report.add(
        "doctor", WARN if pre_problems else OK, f"{pre_problems} pre-existing problems (baseline)"
    )
    return pre


def _ingest_step(report: RoutineReport, con, symbols: list[str]) -> bool:
    try:
        result = ingest.ingest(con, symbols)
        accounted = (
            len(result.symbols_ok)
            + len(result.symbols_skipped)
            + len(result.symbols_failed)
            + len(result.symbols_dormant)
        )
        n_failed = len(result.symbols_failed)
        # Dormant (delisted/halted) names are excluded from the live pool —
        # counting them as failures would trip this gate every day forever.
        live_pool = max(1, len(symbols) - len(result.symbols_dormant))
        detail = (
            f"{result.rows_written} rows, {len(result.symbols_ok)} fetched, "
            f"{len(result.symbols_skipped)} current, {n_failed} failed, "
            f"{len(result.symbols_dormant)} dormant"
        )
        if accounted != len(symbols):
            report.add(
                "ingest",
                FAIL,
                detail + f" — {len(symbols) - accounted} symbols unaccounted for (silent loss)",
            )
            return False
        if n_failed / live_pool > INGEST_FAIL_FRACTION:
            report.add("ingest", FAIL, detail + " (failure rate over threshold)")
            return False
        report.add("ingest", WARN if n_failed else OK, detail)
        return True
    except Exception as exc:
        report.add("ingest", FAIL, f"crashed: {exc}")
        return False


def _freshness_step(
    report: RoutineReport, post: doctor.DoctorReport, today: dt.date, consequence: str
) -> bool:
    max_date = post.max_date
    if max_date is None or (today - max_date).days > MAX_STORE_AGE_DAYS:
        report.add("freshness", FAIL, f"newest bar is {max_date} — {consequence}")
        return False
    report.add("freshness", OK, f"newest bar {max_date}")
    return True


def _recheck_step(
    report: RoutineReport, pre: doctor.DoctorReport, post: doctor.DoctorReport
) -> bool:
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
        return False
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
    return True


def _run_predict(
    report: RoutineReport,
    now: dt.datetime,
    db_path: Path | str,
    out_path: str,
    top: int,
    universe_max_age_days: int,
) -> RoutineReport:
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

    con = _connect_step(report, db_path)
    if con is None:
        return report

    pre = _doctor_baseline_step(report, con)

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
    if not _ingest_step(report, con, symbols):
        return report

    post = doctor.run(con)
    if not _freshness_step(
        report,
        post,
        today,
        "too old to predict from; predictions for elapsed days must not enter the track record",
    ):
        return report

    if not _recheck_step(report, pre, post):
        return report

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


def _scored_target_days(con, strategy: str, top_n: int) -> set[dt.date] | None:
    """Target days already resolved BEFORE today's ingest, so the score step
    can report exactly how many days this run added. None means the pre-ingest
    scoring itself failed — the caller must report the new-day count as
    unknown, never as zero."""
    try:
        frame = track.score_predictions(con, strategy, top_n=top_n).scored
    except Exception as exc:
        logger.warning("pre-ingest scoring failed (%s) — new-day count will be unknown", exc)
        return None
    if frame.empty:
        return set()
    return set(pd.to_datetime(frame["target_date"]).dt.date)


def _run_score(
    report: RoutineReport,
    now: dt.datetime,
    db_path: Path | str,
    out_path: str,
    top: int,
) -> RoutineReport:
    if _score_too_early(now):
        report.add(
            "clock",
            FAIL,
            f"score mode is a post-close run ({now:%a %H:%M} ET is before "
            f"{SCORE_EARLIEST:%H:%M} ET) — today's bar is partial or absent; "
            "run after close (weekends may score pending days)",
        )
        return report
    report.add("clock", OK, f"{now:%a %H:%M} ET, post-close scoring window")
    today = now.date()  # one clock: ET everywhere (a UTC host is 'tomorrow' after 20:00 ET)

    con = _connect_step(report, db_path)
    if con is None:
        return report

    pre = _doctor_baseline_step(report, con)
    try:
        # Same protection predict mode gets from its predict-step try: a
        # malformed champion.json (partial write) must FAIL with a report,
        # not escape as a traceback with exit 1.
        name = champion.get_champion()
    except Exception as exc:
        report.add("score", FAIL, f"cannot resolve champion strategy: {exc}")
        return report
    prior_days = _scored_target_days(con, name, DETECTOR_TOP_N)

    symbols = store.all_universe_symbols(con)
    if not symbols:
        report.add("ingest", FAIL, "no symbols to ingest — empty universe")
        return report
    if not _ingest_step(report, con, symbols):
        return report

    post = doctor.run(con)
    if not _freshness_step(
        report,
        post,
        today,
        "too old to score against; today's close never arrived, so pending days cannot resolve",
    ):
        return report

    if not _recheck_step(report, pre, post):
        return report

    try:
        # Always the top-20 basket, whatever --top says: the detector's lift
        # series is locked to the shipped basket definition.
        record = track.score_predictions(con, name, top_n=DETECTOR_TOP_N)
        perf = track.daily_pick_performance(con, name)
    except Exception as exc:
        report.add("score", FAIL, f"crashed: {exc}")
        return report

    scored = record.scored
    post_days = set() if scored.empty else set(pd.to_datetime(scored["target_date"]).dt.date)
    if prior_days is None:
        n_new = None
        new_detail = "unknown new days (pre-ingest scoring failed)"
    else:
        n_new = len(post_days - prior_days)
        new_detail = f"{n_new} new day(s) scored"
    detail = (
        f"{name}: {new_detail}; {len(scored)} total ({record.late_days} late), "
        f"{len(record.pending)} pending"
    )
    live_p1 = perf.precision_at_1()
    if live_p1 is not None:
        detail += f"; live top-1 precision {live_p1:.0%} over {len(perf.live)} day(s)"
    # 0 new days after a trading day means today was never scored — loud, not ok.
    report.add("score", WARN if (scored.empty or not n_new) else OK, detail)
    if not scored.empty:
        last = scored.sort_values("target_date").iloc[-1]
        lift_txt = f"{last['lift']:.2f}x" if pd.notna(last["lift"]) else "n/a (zero base rate)"
        report.last_scored = (
            f"{pd.Timestamp(last['target_date']).date()}: "
            f"{int(last['hits'])}/{int(last['n_scored'])} hit "
            f"({last['precision']:.0%} vs base {last['base_rate']:.0%}, lift {lift_txt})"
        )

    verdict = track.degradation_verdict(scored)
    if verdict.degraded:
        report.add("detector", FAIL, verdict.detail)
        _file_issue_step(report, con, name, verdict, scored, pre, today)
    else:
        report.add("detector", OK if verdict.armed else WARN, verdict.detail)

    try:
        # Display-only rescore for the dashboard: save=False, so score mode
        # NEVER writes to the predictions log (an evening-of save would be a
        # late prediction; the morning predict run is the only logger).
        result = predict_for(con, name, save=False)
        dashboard.render(con, name, out_path, top=top, result=result)
        report.add("dashboard", OK, f"refreshed {out_path} (predictions log untouched)")
    except Exception as exc:
        report.add("dashboard", WARN, f"render failed (scoring is complete): {exc}")
    return report


AUTO_DEGRADATION_LABEL = "auto-degradation"


def _issue_body(
    con,
    strategy: str,
    verdict: track.DegradationVerdict,
    scored: pd.DataFrame,
    pre: doctor.DoctorReport,
    today: dt.date,
) -> str:
    lines = [
        "Auto-filed by `twopercent routine --mode score`: the champion strategy is "
        "underperforming the all-names base rate on live-scored days.",
        "",
        f"- Champion: `{strategy}`",
        f"- Trailing-{verdict.window} live mean lift: **{verdict.trailing_mean_lift:.4f}** "
        f"(DEGRADED when < 1.0; {verdict.days_below_1} of {verdict.window} window "
        "day(s) individually below 1.0)",
        f"- Live days scored: {verdict.live_days}"
        + (
            f" ({verdict.excluded_null_lift} zero-base-rate day(s) excluded)"
            if verdict.excluded_null_lift
            else ""
        ),
    ]
    uni = store.latest_universe(con)
    if uni.empty:
        lines.append("- Universe snapshot: NONE STORED")
    else:
        as_of = pd.Timestamp(uni["as_of"].iloc[0]).date()
        lines.append(f"- Universe snapshot: {as_of} ({(today - as_of).days}d old)")
    lines.append(
        f"- Doctor baseline this run: {pre.problem_count} problems "
        f"({len(pre.gaps)} gap symbols, {len(pre.stale)} stale, {len(pre.extreme)} extreme "
        f"bars, {len(pre.zero_runs)} zero-volume, {len(pre.invalid)} invalid, "
        f"{len(pre.universe_missing_prices)} universe symbols without prices, "
        f"{len(pre.prices_missing_meta)} price symbols without meta)"
    )
    # `*` marks the rows forming the trailing-window the detector averaged,
    # so the headline mean is recomputable from the table by eye.
    ordered = scored.sort_values("target_date")
    live = ordered[~ordered["late"].astype(bool) & ordered["lift"].notna()]
    window_days = set(pd.to_datetime(live["target_date"]).dt.date.tail(verdict.window))
    lines += [
        "",
        "## Last 10 scored days",
        "",
        f"(`*` = in the trailing-{verdict.window} detector window)",
        "",
        "| target date | hits/N | precision | base rate | lift | late | window |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in ordered.tail(10).itertuples():
        lift = f"{row.lift:.2f}" if pd.notna(row.lift) else "n/a"
        day = pd.Timestamp(row.target_date).date()
        lines.append(
            f"| {day} | {int(row.hits)}/{int(row.n_scored)} "
            f"| {row.precision:.0%} | {row.base_rate:.1%} | {lift} "
            f"| {'yes' if row.late else ''} | {'*' if day in window_days else ''} |"
        )
    lines += ["", "## Latest champion benchmark (experiments table)", ""]
    # Select by strategy in SQL: filtering a recent-N page could falsely say
    # "no experiments" once other strategies crowd the table. Rows with
    # non-empty strategy_params are research variants recorded under the
    # champion's name — they must never be quoted as the champion's benchmark.
    bench = con.execute(
        "SELECT run_ts, test_start, test_end, params, metrics FROM experiments "
        "WHERE strategy = ? ORDER BY run_ts DESC, id DESC",
        [strategy],
    ).df()
    row = None
    for cand in bench.itertuples():
        try:
            parsed = json.loads(cand.params) if cand.params else {}
        except (TypeError, json.JSONDecodeError):
            continue
        if not parsed.get("strategy_params"):
            row = cand
            break
    if row is None:
        lines.append(f"No experiments recorded for `{strategy}` — run `twopercent benchmark`.")
    else:
        lines.append(
            f"Run {pd.Timestamp(row.run_ts)}, test window {row.test_start} → {row.test_end}:"
        )
        metrics = row.metrics
        lines += ["```json", metrics if isinstance(metrics, str) else json.dumps(metrics), "```"]
    lines += [
        "",
        "Investigate per the charter in `.claude/agents/investigator.md`: classify the "
        "cause (data problem / feature drift / regime change / genuine model decay), "
        "post findings as a comment on this issue, and file scoped issues for fixes.",
    ]
    return "\n".join(lines) + "\n"


def _file_issue_step(
    report: RoutineReport,
    con,
    strategy: str,
    verdict: track.DegradationVerdict,
    scored: pd.DataFrame,
    pre: doctor.DoctorReport,
    today: dt.date,
) -> None:
    """File the investigation issue via the shared hardened helper (issues.py:
    arg lists, stdin body, dedup, lock — never shell=True). Any failure WARNs
    loudly; the detector step is already FAIL, so the run exits 2 either way
    and the scoring stands."""
    title = (
        f"Auto: champion underperforming baseline "
        f"(trailing-{verdict.window} live lift {verdict.trailing_mean_lift:.2f})"
    )
    body = _issue_body(con, strategy, verdict, scored, pre, today)
    result = issues.file_issue(
        label=AUTO_DEGRADATION_LABEL,
        title=title,
        body=body,
        color="B60205",
        description="Auto-filed by routine score mode: champion below baseline",
    )
    if result.outcome == issues.FILED:
        report.add("issue", OK, f"filed {result.url} (conversation locked)")
    elif result.outcome == issues.DUPLICATE:
        report.add(
            "issue",
            WARN,
            f"open {AUTO_DEGRADATION_LABEL} issue already filed ({result.existing}) — "
            "not filing a duplicate; degradation persists",
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
            f"gh failed ({result.error}) — degradation detected but NO issue was filed; "
            "investigate manually",
        )
