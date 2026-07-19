# twopercent

Stock scanner + ML predictor for tickers likely to move +2% open-to-close.
Built deliberately through the levels of AI adoption (assisted → parallel →
supervised autonomy → AI-native).

## Rules

- **ROADMAP.md is the source of truth** for scope, locked-in decisions, and
  the level-by-level plan. Read it before starting substantive work.
- **Keep ROADMAP.md updated**: when a decision changes, new information
  invalidates part of the plan, or a session/level completes, update the
  relevant section and the status checklist in the same piece of work.
- All model evaluation must be walk-forward — no lookahead. The 2% target is
  open-to-close: `(close − open) / open ≥ 2%`.
- Push to GitHub (`origin main`) after committing — the remote is the source
  of truth, and local-only commits defeat cloud/scheduled agents at levels 3–4.

## Project standards (each earned by a real failure — see git history)

- **Silent success is the enemy.** This is a data pipeline: the worst bugs
  report success while losing data (skip logic that wrongly says "current",
  filters that silently drop symbols, parses that swallow rows). Any code that
  skips, caches, resumes, or filters must warn loudly about what it excluded,
  and must have tests for the backfill and partial-coverage paths — not just
  the happy path. (Session 1's review found four of these on "all-green" code.)
- **Lint cold before pushing:** `uv run ruff check --no-cache .` — a stale
  ruff cache passed locally and failed in CI after a package-structure change.
- **DuckDB uses total ordering in comparisons: `NaN > 0` is TRUE and NaN sorts
  above every number.** Any SQL filtering or ranking float columns must guard
  with `isfinite()` — a NaN row otherwise tops every ORDER BY DESC. (Session 2:
  a NaN open would have ranked first in every scan.)
- **Test numeric boundaries at adversarial values, not round ones.** The
  exactly-2% boundary passed at open=100.0 but failed at open=5.00 (FP
  rounding); a round-number boundary test can be a false all-clear. Threshold
  comparisons on derived floats need an epsilon.
- **sklearn's HistGradientBoosting binner crashes on single-valued feature
  columns** ("window shape cannot be larger than input array shape"). Synthetic
  test/experiment data must vary every feature column (see tests/conftest.py
  seed_history vary_volume). Real market data never triggers this. An all-NaN
  feature column crashes the same way — seeded test universes need non-empty
  sectors or the sector features are all NaN (Batch 1b).
- **Diagnostics must read raw tables, never filtered views.** The doctor's
  first draft read daily_returns — whose WHERE clause hides exactly the
  corrupt bars a doctor exists to find, while making broken symbols look
  fresh. A view's filter is invisible data loss to anything built on it.
- **Green CI proves the code, not the system.** The sector-features batch
  passed every test and review, then crashed live on a migrated-but-
  unrefreshed store (all-NaN columns). Failures can live in operational
  state no diff contains — run the real pipeline after merging anything
  that changes schema or features (level 3 automates this as a routine).
- **Generated HTML/visual output must be looked at, not just grepped.** The
  dashboard's missing charset (mojibake in every dash) passed all
  string-assertion tests and was caught only by rendering a screenshot.
  Verify visual output by rendering it (Playwright) before shipping; always
  declare `<meta charset="utf-8">` in generated HTML.
- **Network code test pattern:** offline unit tests against canned payloads
  (fixtures in tests/conftest.py) plus `@pytest.mark.live` smoke tests; CI
  runs offline only. Follow it; don't invent a new pattern per module.

## GitHub workflow

- Work is tracked as **GitHub issues** grouped into **milestones** (one per
  adoption level). Start work by picking up an issue; file new issues for new
  work instead of keeping private todo lists.
- **Everything lands via pull request** from a feature branch, referencing the
  issue (`Closes #N`). `main` is branch-protected (required check: `test`,
  strict up-to-date, admins included) — direct pushes are rejected, including
  docs. Reflection/roadmap updates ride in the PR that finished the work.
- A PR merges only when CI is green and review has run. Same bar for human-
  and agent-written code. The user merges; agents never merge.
- Parallel agents each work in their **own git worktree** on their own branch
  — one agent, one branch, one PR.

## Agent team (.claude/agents/)

- **builder** — implements one scoped feature end-to-end on its own branch
  and opens the PR. Spawn one per independent work item; worktree isolation
  when parallel.
- **reviewer** — reviews a PR diff against the earned standards above;
  read/test only. Run on every agent-built PR before merge.
- **quant-skeptic** — adversarial methodology review (lookahead, leakage,
  contamination, survivorship, overfitting, regime). Mandatory for any PR
  touching features, labels, training, evaluation, or reported metrics.
- **investigator** — diagnoses auto-degradation issues (data problem / feature
  drift / regime change / model decay); read-only, findings as issue comments.
- **strategy-researcher** — surveys published research and files referee-ready
  strategy proposals as `strategy-proposal` design-doc issues, never code;
  every proposal follows the `validate-new-strategy` skill checklist.

## Working loop

Every substantive piece of work follows design → code → test → review →
document → reflect:

1. **Design** — state the approach before writing code (plan mode for sessions,
   a short written plan for agents). Anything that changes scope or a locked-in
   decision goes to ROADMAP.md first.
2. **Code** — small vertical slices that run end-to-end; match existing style.
3. **Test** — tests land in the same change as the code, never after. `pytest`
   and `ruff` must pass before work is presented for review. Model-related
   changes additionally prove no lookahead (walk-forward).
4. **Review** — run `/code-review` on the diff before merge. Run
   `/security-review` for changes touching ingestion, network calls,
   credentials, subprocess use, or the dashboard. Hold agent-written and
   human-written code to the same bar.
5. **Document** — update README/CLAUDE.md only when commands, layout, or
   invariants change. No narration comments; code should carry itself.
6. **Reflect** — after each session or merged batch: tick the ROADMAP.md status
   checklist, and if anything went wrong or required un-encoded knowledge, add
   the missing context here or as a skill — that is the fix, not "review
   harder next time."

Keep this rule set minimal: add a rule only when a real failure shows the need,
and delete rules that stop paying for themselves.
