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

## GitHub workflow

- Work is tracked as **GitHub issues** grouped into **milestones** (one per
  adoption level). Start work by picking up an issue; file new issues for new
  work instead of keeping private todo lists.
- Code changes land via **pull request** from a feature branch, referencing
  the issue (`Closes #N`). Direct pushes to `main` are for docs/config only,
  and stop entirely once branch protection is on (#6).
- A PR merges only when CI is green and `/code-review` has run. Same bar for
  human- and agent-written code.
- At level 2+, parallel agents each work in their **own git worktree** on
  their own branch — one agent, one branch, one PR.

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
