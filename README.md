# twopercent

Stock scanner + ML predictor for US tickers likely to move **+2% open-to-close**
(`(close − open) / open ≥ 2%`). DuckDB store, walk-forward-validated gradient
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
```

## The daily cycle (two runs)

`twopercent routine` is the whole day as two gated commands, each reporting by
exception (exit 0 clean / 1 ran-with-warnings / 2 failed-or-degraded):

- **predict mode** (default, pre-open): market-hours guard → doctor baseline →
  universe refresh (if stale) → tail ingest → freshness + corruption gates →
  champion predict (logged) → dashboard → track-record scoring.
- **score mode** (post-close): refuses before 16:15 ET on weekdays; weekends
  may run — scoring time never affects the live/late flag, which depends only
  on when a prediction was *created* relative to its target day's 09:30 ET
  open (a Friday-created prediction scored on Saturday is still live; an
  after-open backfill is late no matter when it gets scored). Then: doctor
  baseline → tail ingest of today's final bars → freshness + corruption
  gates → score pending predictions → **degradation detector** → dashboard
  refresh. Score mode never writes to the predictions log and never
  refreshes the universe.

**Degradation detector:** over scored days with `late == false` (ordered by
target date), once ≥5 such live days exist, the model is DEGRADED when the
mean lift of the most recent 5 falls below 1.0 (epsilon-guarded). On DEGRADED
the run exits 2 and auto-files a GitHub issue labeled `auto-degradation`
(deduped — one open at a time) carrying the evidence bundle; the
`investigator` agent (.claude/agents/investigator.md) picks it up, classifies
the cause (data problem / feature drift / regime change / genuine model
decay), and posts findings back to the issue. With fewer than 5 live days the
detector loudly reports it is not yet armed.

## Scheduling (systemd user timers, local)

The DuckDB store is on-box, so v1 schedules locally (cloud runs are deferred
until data lives off-box). Two timers, both `Persistent=true`, output appended
to `logs/routine.log`:

| Run | OnCalendar | ET equivalent | Command |
|---|---|---|---|
| predict | `Mon..Fri 06:00 America/Denver` | 08:00 ET, pre-open | `uv run twopercent routine` |
| score | `Mon..Fri 14:45 America/Denver` | 16:45 ET, post-close | `uv run twopercent routine --mode score` |

The predict timer is already installed at
`~/.config/systemd/user/twopercent-routine.{service,timer}`. The score timer
is a post-merge step — install it as follows: copy the routine units to
`twopercent-score.{service,timer}`, change `ExecStart` to
`.../uv run twopercent routine --mode score` and `OnCalendar` to
`Mon..Fri 14:45 America/Denver`, then
`systemctl --user daemon-reload && systemctl --user enable --now
twopercent-score.timer`. On WSL the machine must be running with lingering
enabled (`loginctl enable-linger`) for unattended fire.
