# twopercent

Stock scanner + ML predictor for US tickers likely to **reach +2% intraday**
(`(high − open) / open ≥ 2%`, the open-to-high touch). DuckDB store,
walk-forward-validated gradient
boosting, static HTML dashboard. See ROADMAP.md for scope and locked decisions;
CLAUDE.md for project standards.

## Setup

```sh
uv sync
uv run pytest                      # offline suite (live smoke: -m live)
uv run ruff check --no-cache .
```

## Commands

```sh
uv run twopercent universe --refresh   # top-3000 universe snapshot
uv run twopercent ingest               # daily OHLCV into data/twopercent.duckdb
uv run twopercent scan --date 2026-07-17
uv run twopercent doctor               # data-quality checks (read-only)
uv run twopercent predict              # ranked +2% candidates (logged)
uv run twopercent dashboard            # static dashboard.html
uv run twopercent benchmark            # walk-forward benchmark -> experiments row
uv run twopercent routine              # pre-open daily cycle (predict mode)
uv run twopercent routine --mode score # post-close scoring + degradation check
uv run twopercent research             # overnight experiment queue (budget 8/night)
```

## The +2% event: intraday reach (open-to-high)

The event this system predicts is **reaching +2% intraday** —
`high >= open × (1 + 0.02)` — not closing +2%. A pre-placed +2% limit order
would have filled on the day's high (deterministic on that day's own bar, no
lookahead). Open-to-close return stays available as a momentum FEATURE, never
the label. One predicate (`scan.touch_event_predicate`, a column on the
`daily_returns` view) defines the event for the training label, the scanner, the
base rate, precision/lift, and the rolling `cnt_2pct_20d` feature, so they can
never disagree. A narrow **high-spike guard** (`high_glitch_suspect`) excludes
only implausible high prints — an isolated high the close did not confirm and
volume did not corroborate (~95 bars in 5y on the live store, 0.009% of touch
bars); real high-volume squeezes are kept. `twopercent doctor` surfaces the
flagged bars.

Touching +2% is ~2× as common as closing +2% (live store: 14.7% → 30.8% base
rate), so **lift and calibration are the honesty metrics**, not headline reach
rate. Metric definitions are versioned: every experiments/predictions/shadow row
carries `event = 'open_to_high'`; pre-pivot rows are NULL (close-era) and are
walled off — a close-era benchmark is never compared to a touch benchmark, and
the live track record starts fresh at the first touch prediction (earlier days
are archived, never silently re-scored).

> **Required post-merge step:** run `uv run twopercent benchmark` to record a
> touch-era champion benchmark. Until one exists, champion-benchmark reads
> (dashboard SIM panel, research comparison, signal email) degrade gracefully to
> "no touch-era benchmark yet" rather than quoting a close-era number. Old
> experiments and predictions remain as a close-era archive — there is no
> migration.

## The daily cycle (two runs)

`twopercent routine` is the whole day as two gated commands, each reporting by
exception (exit 0 clean / 1 ran-with-warnings / 2 failed-or-degraded):

- **predict mode** (default, pre-open): market-hours guard → doctor baseline →
  universe refresh (if stale) → tail ingest → freshness + corruption gates →
  completeness note → champion predict (logged) → dashboard → track-record
  scoring → signal email
  (skipped loudly when unconfigured — see "Daily signal email").
- **score mode** (post-close): refuses before 16:15 ET on weekdays; weekends
  may run — scoring time never affects the live/late flag, which depends only
  on when a prediction was *created* relative to its target day's 09:30 ET
  open (a Friday-created prediction scored on Saturday is still live; an
  after-open backfill is late no matter when it gets scored). Then: doctor
  baseline → tail ingest of today's final bars → freshness + corruption
  gates → completeness note → score pending predictions → **degradation
  detector** → dashboard refresh. Score mode never writes to the predictions
  log and never
  refreshes the universe.

**Complete trading days (data invariant):** scoring and prediction never touch
an INCOMPLETE (provisional, in-progress) day. Every predict/score run ingests a
partial pre-market bar for today covering only a fraction of the universe (e.g.
2026-07-24 held 2,438 of ~3,035 symbols); resolving a prediction's target onto
that day, or forecasting from it, would count/forecast a day that has not
finished trading. A date is COMPLETE iff its valid-bar coverage is at least 90%
of the median coverage over the 20 trading days *strictly before* it
(trailing-only window → no lookahead; a later day never changes an earlier day's
verdict). The single definition lives in the `complete_trading_days` view
(store.py) and gates target resolution, base rates, and predict's default signal
day. A held-back edge day is expected pre-close (not a failure): the routine's
`completeness` step and the doctor's `incomplete` check WARN loudly, the
prediction is forecast from the latest complete day, and the day shows "Awaiting
outcomes" until it completes.

**Degradation detector:** over scored days with `late == false` (ordered by
target date), once ≥5 such live days exist, the model is DEGRADED when the
mean lift of the most recent 5 falls below 1.0 (epsilon-guarded). On DEGRADED
the run exits 2 and auto-files a GitHub issue labeled `auto-degradation`
(deduped — one open at a time) carrying the evidence bundle; the
`investigator` agent (.claude/agents/investigator.md) picks it up, classifies
the cause (data problem / feature drift / regime change / genuine model
decay), and posts findings back to the issue. With fewer than 5 live days the
detector loudly reports it is not yet armed.

## Daily signal email

The predict run's last step emails the day's signal. The email body IS the
rendered dashboard: `dashboard.html` is rendered headlessly at send time
(Playwright chromium, dark theme, full page) to a PNG embedded inline via
CID — no attachment, plus a one-line plain-text alternative for text-only
clients. One-time setup on the box: `uv run playwright install chromium`
(CI never needs browsers — tests mock the renderer). When rendering is
unavailable (missing browsers, a render crash, an implausibly large PNG),
the step WARNs and falls back to the previous composed email: a trade
suggestion (the top-5 equal-weight open-to-close basket), the top-10
candidate table with model scores, a system summary whose
precision/base-rate/lift figures are pulled from the champion's latest
standard benchmark in the experiments ledger (never hard-coded), the
live-record status, and a disclaimer — with `dashboard.html` attached.
Score-mode runs (`--mode score`) never email.

Configuration is environment-only (see `.env.example`; copy it to `.env`,
which is gitignored — never commit a real `.env`). Two variables are common
to both transports:

- `TWOPERCENT_EMAIL_TO` — recipient address(es), comma-separated for several
- `TWOPERCENT_EMAIL_FROM` — the From address

The transport is selected by which credentials are present:

**Resend HTTP API (recommended).** Set `TWOPERCENT_RESEND_API_KEY`. Setup is
about five minutes: create a free account at [resend.com](https://resend.com)
→ API Keys → create a key with **sending access only** — it is scoped and
revocable, and no personal account credential ever lands on this box. On the
free tier without a verified domain, use `onboarding@resend.dev` as
`TWOPERCENT_EMAIL_FROM` and note that **recipients are restricted to the
account owner's own address**. Verifying a domain you own (e.g.
upanddownai.com) lifts that restriction and lets the mail come from
`signals@your-domain` — that is the path if the signal is ever distributed
to anyone else.

**Generic SMTP (fallback).** Set `TWOPERCENT_SMTP_HOST`,
`TWOPERCENT_SMTP_PORT` (default 587), `TWOPERCENT_SMTP_USER`, and
`TWOPERCENT_SMTP_PASSWORD` for a STARTTLS SMTP relay. **Dedicated sending
account only — never a personal credential.** When both transports are
configured, Resend wins and the routine WARNs that the SMTP settings are
ignored; a partially set SMTP trio WARNs as misconfiguration rather than
silently skipping.

For the scheduled predict run, hand the variables to the systemd service by
adding one line to the `[Service]` section of
`~/.config/systemd/user/twopercent-routine.service`, then
`systemctl --user daemon-reload`:

```ini
EnvironmentFile=%h/projects/twopercent/.env
```

Behavior when things go wrong: recipient/sender or every transport unset →
the step reports a loud "email not configured — skipping" line without
degrading the exit code; a send failure (rejected API key, SMTP trouble) →
WARN (exit 1 class), never exit 2 — the prediction is already logged either
way, and no credential ever appears in logs or summaries. Dashboard render
trouble WARNs and falls back to the composed text email; on that fallback a
missing `dashboard.html` warns again and sends without the attachment.

## The research loop (overnight)

`twopercent research` works through `research/queue.json` — a checked-in list
of `{strategy, params, note}` configs (edited only via PR, so every sweep is
auditable). Each config runs the standard referee benchmark (12-month
walk-forward, top-20, recorded to the experiments ledger) with the params
passed to the strategy constructor. The `xgb_gbm_v1` challenger trains on the
GPU (`device="cuda"`) and falls back to CPU with a loud warning when CUDA is
unavailable.

Guardrails:

- **Clock gate:** runs only between 16:30 and 05:00 America/Denver (any day —
  offline compute, weekends fine), keeping clear of market hours and the
  06:00/14:45 routine runs (DuckDB is single-writer). The window is rechecked
  before each experiment with a one-hour must-finish margin (no new config
  starts after 04:00), so a slow night can never hold the store into the
  06:00 predict run.
- **Budget:** at most `--budget` (default 8, minimum 1) experiments per night.
- **Idempotent, no state file:** a config whose (strategy, params) already has
  a recorded standard benchmark in the experiments ledger is skipped (loudly
  counted; numeric identity is canonicalized, so 200 == 200.0), so a scheduled
  run never dirties the repo and reruns are safe. The experiments row and its
  daily rows land atomically; a crashed config records NOTHING and retries the
  next night. The training device (cuda/cpu) is recorded per run, and a night
  that recorded configs under CPU fallback warns loudly.
- **Write-nothing:** the runner writes only experiments-ledger rows — never
  champion.json, predictions, or prices.
- **Promotion stays human:** a config beating the champion on lift beyond the
  PROMOTION band (0.25 — family-wise, wider than compare's single-comparison
  0.1 because a sweep is ~24 comparisons against the same months) AND holding
  the margin on both disjoint halves of the shared test days files ONE
  deduped, locked `promotion-candidate` GitHub issue. It is a hypothesis, not
  a promotion: the issue demands the wall-clock holdout (margin holds on >= 2
  months of data arriving after the candidate's test_end, #45), quant-skeptic
  review, and a human PR — never sim growth. The champion's reference
  benchmark is always its own default-config run (parameterized sweep
  variants recorded under the same strategy name are excluded).

Exit codes: 0 clean or empty queue, 1 some experiments failed or queue entries
were malformed, 2 the runner itself failed.

## Shadow-trading engine (challenger isolation)

Each predict morning, zero or more *challenger* strategies run IN PARALLEL with
the champion — their picks are generated, logged, and (next day) scored against
actuals to accumulate a genuine FORWARD live track record — but they are NEVER
emailed and NEVER touch the champion's live record, dashboard, or signal email.
This is the isolation foundation the forward shadow-gate promotion (#59) will
sit on; this tier only LOGS and SCORES — it promotes nothing.

- **Roster: `research/shadow.json`** — a checked-in JSON list of
  `{strategy, params, note}` (same shape as `research/queue.json`), edited only
  via PR. It ships **empty (`[]`)**: nothing is shadowed until a human/PR adds a
  challenger. Add an entry (e.g.
  `[{"strategy": "logreg_v1", "params": {}, "note": "linear yardstick"}]`) and
  the next `twopercent routine` predict run starts shadowing it; the following
  score run begins reporting its forward record.
- **Concurrency cap `MAX_SHADOW` (4):** if the roster exceeds it, the first 4
  are kept and the rest are DROPPED with a loud warning naming the count (never
  a silent truncation). The cap bounds the compute added to each morning and is
  the K the shadow-gate forward-margin accounting (#60) will scale against.
- **Isolation invariant:** shadow picks live in a SEPARATE table
  (`shadow_predictions`), keyed by challenger identity
  (`strategy {canonical-json-params}`, so a challenger can share a strategy name
  and differ only by params). Nothing that reads `predictions` — the track
  record, money tiles, dashboard, or email — can see them.
- **Crash-isolated and non-gating:** the shadow steps run LAST (after the
  champion predict/dashboard/notify, so shadow compute never delays the real
  signal email); one failing challenger warns and the others still run, and a
  shadow failure WARNs but never fails the routine. Same 09:30-ET live/late
  rule as the champion, so backfilled shadow picks are excluded from the
  forward record exactly like the champion's money tiles.

## Scheduling (systemd user timers, local)

The DuckDB store is on-box, so v1 schedules locally (cloud runs are deferred
until data lives off-box). Two timers, both `Persistent=true`, output appended
to `logs/routine.log`:

| Run | OnCalendar | ET equivalent | Command |
|---|---|---|---|
| predict | `Mon..Fri 06:00 America/Denver` | 08:00 ET, pre-open | `uv run twopercent routine` |
| score | `Mon..Fri 14:45 America/Denver` | 16:45 ET, post-close | `uv run twopercent routine --mode score` |
| research | `daily 22:00 America/Denver` | 00:00 ET, overnight | `uv run twopercent research` |

(Research runs every day, weekends included — it is offline compute against
the local store, and its own clock gate refuses anything outside 16:30–05:00
Denver.)

The predict timer is already installed at
`~/.config/systemd/user/twopercent-routine.{service,timer}`. The score timer
is a post-merge step — install it as follows: copy the routine units to
`twopercent-score.{service,timer}`, change `ExecStart` to
`.../uv run twopercent routine --mode score` and `OnCalendar` to
`Mon..Fri 14:45 America/Denver`, then
`systemctl --user daemon-reload && systemctl --user enable --now
twopercent-score.timer`. On WSL the machine must be running with lingering
enabled (`loginctl enable-linger`) for unattended fire.

The research timer installs the same way: copy the routine units to
`twopercent-research.{service,timer}`, change `ExecStart` to
`.../uv run twopercent research` and `OnCalendar` to
`*-*-* 22:00 America/Denver` (every day), then
`systemctl --user daemon-reload && systemctl --user enable --now
twopercent-research.timer`.
