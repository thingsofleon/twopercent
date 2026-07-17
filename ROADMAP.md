# twopercent — Roadmap

A stock-scanning and prediction system, built deliberately through the levels of
Boris Cherny's "Steps of AI Adoption" (Jul 2026). Each level produces concrete
artifacts in this repo, and each level's exit criteria are the entry
requirements for the next.

## The product

Every trading day, many tickers move 2%+. The system:

1. **Historical processing** — per day, which tickers did +2% (open-to-close)?
2. **Signal analysis** — features/signals explaining why a ticker moved.
3. **Prediction dashboard** — before/during market hours, a ranked list of
   tickers likely to do +2% today; after close, tomorrow's candidates.

## Locked-in decisions

| Decision | Choice | Notes |
|---|---|---|
| Universe | Russell 3000 (broad US market) | Biggest signal surface; heaviest ingestion load |
| Data source | yfinance (free) | Daily OHLCV, batched + cached locally; no true real-time/pre-market. Revisit paid API (e.g. Polygon) if used daily |
| "Did 2%" definition | Open-to-close: `(close − open) / open ≥ 2%` | Regular-hours move only; gaps excluded; +2% direction only |
| Signals/prediction | ML model from the start | Gradient boosting on engineered features; **walk-forward validation only** — no lookahead |
| Storage | DuckDB / parquet | Columnar, local, zero server; fits 3,000 tickers × years of daily bars |
| Language/tooling | Python, `uv`, `pytest`, `ruff` | |
| Dashboard v1 | Locally-served web page / static HTML from data | Form finalized when we get there |

Known tension: Russell 3000 on free yfinance makes ingestion the slowest,
flakiest part of the system. Mitigate with batching and an aggressive local
cache; it also gives levels 3–4 something real to automate and monitor.

## Level 1 — Assisted (one agent, review everything)

Plan mode on, every diff reviewed. Goal: working core **and** the
verification loop (tests + lint) that level 2 depends on.

- **Session 1 — skeleton + ingestion.** Python project, first commit.
  Russell 3000 constituent list, batched yfinance download of daily OHLCV,
  cached to DuckDB/parquet. Tests from the first file.
- **Session 2 — 2% scanner.** Open-to-close returns per ticker/day, flag ≥2%
  events, store them. CLI: `twopercent scan --date YYYY-MM-DD`.
- **Session 3 — features + first model.** Features (prior-day return, volume
  ratio, volatility, gap, sector move, days-to-earnings), gradient-boosting
  classifier, walk-forward validation, ranked daily probability list.
- **Session 4 — dashboard v1.** Today's/tomorrow's top candidates with
  probabilities and top signals; history of predictions vs. actuals.

**Exit criteria:** tests pass, lint clean, pipeline runs end-to-end on a real
historical day.

## Level 2 — Parallel (orchestrate ~5–10 agents)

Trust infrastructure first, then fan out.

- Pre-approve safe commands (`pytest`, `ruff`, `uv`, read-only git) in
  `.claude/settings.json` so agents don't block on prompts.
- Agent team in `.claude/agents/`: `builder`, `reviewer`, and a
  `quant-skeptic` whose only job is attacking methodology (lookahead bias,
  survivorship bias, data leakage).
- First `CLAUDE.md`: how to run tests, data layout, "all model evaluation must
  be walk-forward."
- Then: batches of independent features built by parallel agents in separate
  worktrees, each self-verifying before review. Candidate batch: backtesting
  harness, sector/regime features, data-quality checks (halts, splits,
  delistings), dashboard filters. Review finished diffs with `/code-review`.

**Exit criteria:** parallel agent work merged that you didn't watch being
written, and the tests — not your eyeballs — caught at least one problem.

## Level 3 — Supervised autonomy (loops, routines, ~100 agent-runs)

- Daily routine: refresh data → retrain/score → regenerate dashboard → flag
  anomalies (data gaps, prediction drift, degradation) as an exception report.
- `/loop` maintenance: score yesterday's predictions vs. actuals, log
  calibration drift, hunt corrupted tickers.
- Workflows for heavy jobs: multi-year walk-forward backtests as fan-out
  pipelines (one agent per year-slice, verifier per result, synthesizer).
- Every agent mistake → encode the missing context into CLAUDE.md or a Skill
  ("how we validate a new feature," "how we handle corporate actions") instead
  of reviewing harder.

**Exit criteria:** a morning where data refreshed, model retrained, and
quality checks ran — and you only read the exception report.

## Level 4 — AI-native (steer by intent)

At solo scale: scheduled autonomy, not thousand-agent fleets.

- Scheduled cloud agents (`/schedule`): pre-market run (~8:00 ET) produces the
  day's dashboard; post-close run scores the day and appends to the track
  record — no laptop open.
- Closed loop: when the post-close scorer detects the model underperforming
  its baseline for N days, it kicks off an investigation agent (feature
  drift? data issue? regime change?) and files findings for review.
- Your role: "keep prediction quality above X, keep data clean, surface
  anything unusual" — monitor by exception.

## Status

- [ ] Level 1, Session 1 — skeleton + ingestion
- [ ] Level 1, Session 2 — 2% scanner
- [ ] Level 1, Session 3 — features + model
- [ ] Level 1, Session 4 — dashboard v1
- [ ] Level 2 — trust infra + first parallel batch
- [ ] Level 3 — routines, loops, workflows
- [ ] Level 4 — scheduled autonomy, closed loop
