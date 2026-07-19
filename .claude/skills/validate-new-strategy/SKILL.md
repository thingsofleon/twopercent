---
name: validate-new-strategy
description: The mandatory checklist for proposing, implementing, or reviewing a new prediction strategy — timing model, leakage checks, referee integration, and the promotion gauntlet.
---

# Validating a new strategy

Every strategy proposal, implementation, and review in this repo follows this
checklist. It exists so that an idea from a paper becomes a plugin the referee
can score without anyone re-deriving the rules — and so no approach sneaks
past the standards the earned failures paid for.

## 1. The timing model (non-negotiable)

A row is keyed by `(symbol, signal_date)`. Every feature must be computable
from data through the END of signal_date S — known after S's close. The label
is the NEXT trading day's +2% open-to-close outcome (`(close − open)/open ≥
0.02 − 1e-9`, the epsilon is deliberate). Before implementation, write a
feature-timing table: each proposed feature, its data source, and the moment
it becomes known. For the current daily-bar pipeline, anything not strictly
≤ S's close is lookahead — including "tomorrow's open" (unknown at the
pre-open prediction moment).

**Post-close data carve-out.** Sources that arrive after S's close but
strictly before the next open (pre-market quotes/volume, overnight news) are
legitimate to PROPOSE (data-gated), but they change the prediction-time
contract: the proposal must state the exact cutoff moment (e.g. "known by
05:55 Denver"), the routine change it implies, and how the feature-timing
table and the canary (section 2) will be extended to cover the new source.
The base law is never loosened silently — a post-close feature that skips
this carve-out is lookahead.

## 2. Leakage checklist

- No `LEAD`/future joins except the label columns (`did_2pct_next`,
  `next_oc_return`, `target_date`) — which must never appear in
  `FEATURE_COLUMNS` or `METADATA_COLUMNS`.
- The lookahead canary test must pass for any new feature SQL — and know its
  ACTUAL scope before trusting it: today it mutates only `close`, `high`, and
  `volume` in the `prices` table after the cutoff (a leak of future `open` or
  `low`, or of any other table, passes vacuously — `LEAD(open)` would be
  green). Any new feature's inputs — every column and every table it reads —
  must be ADDED to the canary's mutation in the same PR, so the
  byte-identical assertion actually covers the new surface. Do not route
  around it; do not cite it for coverage it doesn't have.
- Universe-snapshot fields (sector, market cap) are TODAY'S values applied to
  history — survivorship in feature values. Allowed (documented ROADMAP
  caveat) but any new snapshot-derived feature must restate it (#24).
- Median-volume liquidity floor applies at SELECTION (prediction + benchmark
  top-N) only — never to training or labels.
- Scalers/imputers/calibrators fit inside the fold on train rows only.
- No fitting to the test window's identity: no early stopping on test, no
  eval_set, no per-fold hyperparameter tuning.
- No unrecorded test-window mining: iterating a config against the standard
  12-month window with `benchmark --no-record` and then submitting only the
  polished winner is global tuning on the test window — invisible to the
  band's family-wise calibration, which counts recorded runs only. Any
  development-loop referee runs against the standard window must be disclosed
  in the PR, and a config developed that way gets no benefit of the doubt at
  the band.

## 3. Integration steps

- One class in `src/twopercent/strategies/`, registered via `@register`;
  constructor kwargs for anything sweepable (validated against a whitelist —
  unknown params must raise, never silently run defaults).
- Implement `fit`/`predict_proba` over `FEATURE_COLUMNS` only; mirror the
  loud `dropped_columns` semantics (all-unobserved columns dropped with a
  warning; all-unusable raises).
- GPU strategies need a capability probe with loud CPU fallback — CI has no
  GPU; the resolved device is recorded in the experiments ledger.
- Tests land in the same PR: fit/predict on the seeded conftest universe
  (vary every feature column; non-empty sectors), dropped-column behavior,
  param-whitelist rejection, and the canary if feature SQL changed.
- Sweepable configs are `research/queue.json` entries in the same PR.

## 4. Evaluation and the referee

- The referee (`backtest.py`) is the only scorer: expanding-window monthly
  folds, standard = 12 months, top-20. Strategies never influence it.
- Lift (vs the all-names base rate) and AUC are the decision numbers. Brier
  matters for display: poorly calibrated probabilities (e.g. from class
  reweighting) must be flagged — rankings may ship, raw probabilities must
  not be shown as chances without a calibration layer.
- Sim growth ($1 compounding) is NEVER used for champion selection — it is
  tail-dominated and regime-dependent by construction.

## 5. The promotion gauntlet

1. Nightly research runs record standard benchmarks to the experiments ledger.
2. A candidate needs: lift margin over the params-free champion row above
   `RESEARCH_PROMOTION_BAND` (0.25 — multiple-comparisons widened; the
   compare CLI's 0.1 band is for single comparisons only. The band's
   guarantee is family-size-dependent: it was calibrated for ~24 configs, and
   every queue-growing PR erodes it), AND the daily top-N precision margin
   strictly positive on each disjoint half of the shared test window (that is
   what `_halves_hold` checks — a weaker per-half condition than the band;
   don't credit it with more).
3. Wall-clock holdout (#45): the margin must hold on ≥2 months of data dated
   AFTER the candidate's recorded `test_end`, scored by the same referee.
   Holdout months are a consumable: confirming several candidates on the SAME
   trailing months is not several independent confirmations — #45 requires
   tracking and warning on reuse.
4. Promotion is a one-line `champion.json` PR — quant-skeptic review
   mandatory, the human merges. Agents never promote.

## 6. Known traps (paid for by real failures — see CLAUDE.md)

sklearn's HistGBM crashes on all-NaN/constant columns; DuckDB NaN sorts above
every number (`isfinite()` guards); FP boundaries need epsilons tested at
adversarial values (open=5.00, not 100.0); diagnostics read raw tables, never
filtered views; silent skips/drops are the enemy — warn loudly and test the
unhappy path in the same PR.
