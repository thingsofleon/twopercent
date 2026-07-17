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
