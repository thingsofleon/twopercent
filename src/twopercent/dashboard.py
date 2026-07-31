"""Static HTML dashboard: next-day candidates + honest track record.

Self-contained output — inline CSS, inline SVG, no external requests — so the
file opens anywhere and satisfies a strict CSP (claude.ai Artifact
compatible). Terminal-style dark-first design with a full light theme;
viewer's data-theme toggle overrides the OS preference in both directions.
Semantic color only: green = hit/up, red = down/miss, amber = base rate.
"""

from __future__ import annotations

import html
import json
import math
from decimal import ROUND_HALF_UP, Decimal

import duckdb
import pandas as pd

from twopercent import backtest, store, strategy, track
from twopercent.predict import PredictResult, predict_for

_CSS = """
<style>
:root {
  --bg: #f6f8f7; --card: #ffffff; --card-border: rgba(10, 40, 30, 0.12);
  --ink-1: #10201a; --ink-2: #46605a; --ink-muted: #7d938d;
  --grid: #dde5e2; --baseline: #b8c6c1;
  --up: #067647; --down: #b42334; --amber: #9a6700; --info: #0e7f7d;
  --up-dim: rgba(6, 118, 71, 0.12); --hero-glow: none;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    --bg: #0a0f16; --card: #101823; --card-border: rgba(112, 255, 190, 0.10);
    --ink-1: #e6efec; --ink-2: #9fb6ae; --ink-muted: #62796f;
    --grid: #1c2733; --baseline: #2d3b46;
    --up: #2fd980; --down: #ff5d6c; --amber: #ffc247; --info: #38d1cf;
    --up-dim: rgba(47, 217, 128, 0.13);
    --hero-glow: 0 0 18px rgba(47, 217, 128, 0.35);
  }
}
:root[data-theme="dark"] {
  --bg: #0a0f16; --card: #101823; --card-border: rgba(112, 255, 190, 0.10);
  --ink-1: #e6efec; --ink-2: #9fb6ae; --ink-muted: #62796f;
  --grid: #1c2733; --baseline: #2d3b46;
  --up: #2fd980; --down: #ff5d6c; --amber: #ffc247; --info: #38d1cf;
  --up-dim: rgba(47, 217, 128, 0.13);
  --hero-glow: 0 0 18px rgba(47, 217, 128, 0.35);
}
html, body { background: var(--bg); margin: 0; }
.tp-root {
  color-scheme: light dark;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink-1); background:
    linear-gradient(transparent, var(--bg) 420px),
    repeating-linear-gradient(0deg, transparent 0 31px, var(--grid) 31px 32px),
    repeating-linear-gradient(90deg, transparent 0 31px, var(--grid) 31px 32px),
    var(--bg);
  min-height: 100vh; padding: 28px 16px 48px;
}
.tp-wrap { max-width: 920px; margin: 0 auto; }
.tp-root h1 { font-size: 1.3rem; margin: 0 0 2px; letter-spacing: 0.01em; }
.tp-root h1 .tick { display: inline-block; width: 0; height: 0; margin-right: 6px;
  border-left: 7px solid transparent; border-right: 7px solid transparent;
  border-bottom: 12px solid var(--up); vertical-align: 1px; }
.tp-root h2 { font-size: 0.8rem; margin: 30px 0 8px; color: var(--ink-muted);
  text-transform: uppercase; letter-spacing: 0.12em; }
.tp-root .sub { color: var(--ink-2); font-size: 0.86rem; margin: 0; }
.tp-root .mono, .tp-root td, .tp-root .tile b {
  font-family: ui-monospace, "Cascadia Code", "SF Mono", Consolas, monospace;
  font-variant-numeric: tabular-nums;
}
.tp-root .card {
  background: var(--card); border: 1px solid var(--card-border);
  border-radius: 8px; padding: 14px 16px; margin-top: 10px; overflow: visible;
}
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; margin-top: 16px; }
.tile { background: var(--card); border: 1px solid var(--card-border);
  border-radius: 8px; padding: 12px 14px; }
.tile .label { font-size: 0.68rem; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--ink-muted); }
.tile b { display: block; font-size: 1.55rem; font-weight: 600; margin-top: 2px;
  color: var(--ink-1); }
.tile b.up { color: var(--up); text-shadow: var(--hero-glow); }
.tile .cmp { font-size: 0.72rem; color: var(--ink-2); }
.tp-root table { border-collapse: collapse; width: 100%; font-size: 0.86rem; }
.tp-root th { text-align: left; color: var(--ink-muted); font-weight: 600;
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.08em;
  border-bottom: 1px solid var(--baseline); padding: 6px 10px 6px 0; }
.tp-root td { border-bottom: 1px solid var(--grid); padding: 6px 10px 6px 0;
  color: var(--ink-1); }
.tp-root td.name { color: var(--ink-2); font-family: system-ui, sans-serif; }
.tp-root td .sym { color: var(--ink-1); font-weight: 700; }
.tp-root .pos { color: var(--up); }
.tp-root .neg { color: var(--down); }
.probbar { display: inline-block; width: 52px; height: 5px; border-radius: 3px;
  background: var(--grid); vertical-align: 2px; margin-left: 7px; overflow: hidden; }
.probbar i { display: block; height: 100%; background: var(--info); }
.tp-root .legend { display: flex; gap: 16px; font-size: 0.78rem;
  color: var(--ink-2); margin: 2px 0 6px; }
.tp-root .chip { display: inline-block; width: 10px; height: 10px;
  border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
.tp-root .empty { color: var(--ink-muted); font-style: italic; }
.badge-sim, .badge-live { display: inline-block; color: var(--bg);
  font-size: 0.62rem; font-weight: 700; padding: 2px 7px; border-radius: 4px;
  letter-spacing: 0.12em; vertical-align: 1px; margin-right: 6px; }
.badge-sim { background: var(--amber); }
.badge-live { background: var(--up); }
.stamp-ub { display: inline-block; background: var(--amber); color: var(--bg);
  font-size: 0.6rem; font-weight: 800; padding: 2px 6px; border-radius: 4px;
  letter-spacing: 0.08em; margin-left: 6px; vertical-align: 1px;
  font-family: system-ui, sans-serif; white-space: nowrap; }
.tp-root select { background: var(--card); color: var(--ink-1);
  border: 1px solid var(--card-border); border-radius: 6px; padding: 3px 6px;
  font: inherit; font-size: 0.84rem; }
.tp-root .note { color: var(--ink-muted); font-size: 0.74rem; margin-top: 26px; }
.tp-i {
  display: inline-flex; align-items: center; justify-content: center;
  width: 14px; height: 14px; margin-left: 5px; border-radius: 50%;
  border: 1px solid var(--ink-muted); color: var(--ink-muted);
  font: 600 9px/1 system-ui, sans-serif; font-style: normal;
  text-transform: none; letter-spacing: 0; cursor: help;
  position: relative; vertical-align: middle; user-select: none; flex: none;
}
.tp-i:hover, .tp-i:focus-visible { color: var(--info); border-color: var(--info); }
.tp-i:focus-visible { outline: 2px solid var(--info); outline-offset: 1px; }
.tp-tip {
  display: none; position: absolute; top: calc(100% + 9px); left: 50%;
  transform: translateX(-50%); width: max-content; max-width: min(260px, 82vw);
  background: var(--card); border: 1px solid var(--card-border); border-radius: 8px;
  padding: 8px 11px; box-shadow: 0 8px 24px rgba(0, 0, 0, 0.30); z-index: 50;
  color: var(--ink-2); font: 400 0.72rem/1.45 system-ui, sans-serif;
  font-style: normal; text-transform: none; letter-spacing: normal;
  text-align: left; white-space: normal; pointer-events: none;
}
.tp-i:hover .tp-tip, .tp-i:focus-visible .tp-tip { display: block; }
.tp-tip--start { left: 0; transform: none; }
.tp-tip--end { left: auto; right: 0; transform: none; }
.tp-tip::after, .tp-tip::before {
  content: ""; position: absolute; bottom: 100%; left: 50%; border: 6px solid transparent;
  border-bottom-color: var(--card); margin-left: -6px;
}
.tp-tip::before { border-width: 7px; border-bottom-color: var(--card-border); margin-left: -7px; }
.tp-tip--start::after, .tp-tip--start::before { left: 15px; }
.tp-tip--end::after, .tp-tip--end::before { left: auto; right: 15px; margin-left: 0; }
@media (max-width: 520px) {
  .tp-root .col-co { display: none; }
  .tp-root table { font-size: 0.8rem; }
  .tp-root .probbar { width: 34px; }
}
</style>
"""


_INFO_TEXT = {
    # Header tiles — walk-forward benchmark (leads with base-rate-adjusted,
    # base-rate-invariant evidence) then the accumulating live reach record.
    "t_auc": "Walk-forward AUC of the ranking: the chance a random reacher is "
    "ranked above a random non-reacher. Base-rate-invariant — it does not move "
    "when the +2% base rate does, so it is the honest headline. Backtest number: "
    "the model never trains on the days it scores, but the system was designed "
    "with this history visible, so it is not yet live performance.",
    "t_bench_lift": "Walk-forward lift: how much more often the top-N picks reach "
    "+2% than a random symbol would. Above 1× beats chance. Lift compresses at "
    "high base rates (it is capped at 1/base-rate), so read AUC and excess too. "
    "Backtest number, designed with this history visible — not yet live "
    "performance; see the live tiles for the clean forward record.",
    "t_reach": "Walk-forward reach-rate: the share of the top-N picks that reached "
    "+2% intraday, over the benchmark test window, versus the all-names base rate. "
    "Backtest number, designed with this history visible — not yet live "
    "performance; the live tiles carry the clean forward record.",
    "t_cands": "How many symbols the model ranked for the next trading day.",
    "t_live_reach": "Live reach-rate: of the top-N picks logged before the market "
    "opened, the share that then reached +2% intraday. Backfilled days excluded.",
    "t_live_lift": "Live reach-rate ÷ base rate, pooled across live scored days. "
    "Above 1× beats picking at random. Read alongside AUC and excess.",
    "t_excess": "Live reach-rate minus the base rate (in percentage points), pooled "
    "across live days. Positive means the ranking beat the market — the honesty "
    "metric when lift compresses at a high base rate.",
    # Calibration panel — does the probability mean what it says.
    "t_bss": "Brier skill score, walk-forward, ACROSS ALL NAMES (not the picks): "
    "1 − Brier_model / Brier_reference, reference = each day's own base rate (which "
    "swings ~7%→74% in the touch-era test window). Positive means the ranking's "
    "probabilities add skill BEYOND simply knowing the daily base rate — a measure "
    "of ranking DISCRIMINATION over all names, NOT a statement about the top-N picks "
    "you act on (the reliability curve answers that). Raw Brier drops for free at a "
    "higher base rate, so this skill form is the one comparable across the "
    "close→touch cutover.",
    "cal_curve": "Reliability, walk-forward, on the population named in the legend: "
    "each point is a predicted-probability bucket — its mean predicted chance (x) "
    "against how often those actually reached +2% (y), sized by how many fell in it. "
    "On the dashed diagonal means the probability is honest (a 74% call → ~74% "
    "realized); below the line means overconfident, above means underconfident.",
    # Candidates table columns
    "c_rank": "Rank by model probability — 1 is the most likely to reach +2%.",
    "c_sym": "Ticker symbol.",
    "c_prob": "Model's estimated chance the symbol reaches +2% intraday. The "
    "bar is relative to today's top candidate — a ranking, not calibrated odds.",
    "c_prev": "The symbol's open-to-close return on the most recent completed day.",
    "c_vol": "Latest volume ÷ its trailing-20-day average volume (today included). "
    "Above 1× means unusually active.",
    "c_cnt": "How many of the last 20 trading days this symbol reached +2% intraday.",
    "c_co": "Company name.",
    # Track-record chart + table
    "chart": "One bar per scored day: the top-N reach-rate. Green beat the market "
    "base rate, red missed it. The amber dash marks that day's base rate — the "
    "share of all symbols that reached +2% intraday.",
    "r_day": "The trading day these picks were made for.",
    "r_pick": "The #1 ranked pick and whether it reached +2% intraday (✓/✗). A † "
    "means the day was backfilled, not a live forecast.",
    "r_hits": "How many of that day's scored top-N picks reached +2%.",
    "r_hit": "Reachers ÷ picks scored that day. Green beat the market base rate.",
    "r_base": "Share of all symbols that reached +2% that day — the bar to beat.",
    "r_lift": "Reach-rate ÷ base rate. Above 1× is better than picking at random.",
    # Explorer controls + table
    "e_basket": "Basket size — the top-N ranked picks each day drive the numbers below.",
    "e_window": "How many trailing trading days to summarize.",
    "e_record": "SIM is walk-forward simulation: the model never trains on the days "
    "it predicts, but the system was built with this history visible. LIVE is the "
    "picks actually logged before each open — the clean test.",
    "e_hit": "Share of the basket that reached +2% intraday, averaged over the window.",
    "e_base": "Average share of all symbols that reached +2% over the window.",
    "e_lift": "Reach-rate ÷ base rate over the window. Above 1× beats random.",
    "e_days": "Trading days included in this window.",
    "e_strategy": "What-if exit rules replayed on the SAME picks — an opt-in "
    "backtest view, never the headline. Reach rate keeps the default "
    "prediction-quality view. Every P&L number is gross — before trading costs "
    "(not estimated) — and the SIM row is an optimistic upper bound "
    "(survivorship). The trailing stop needs minute-by-minute data we don't "
    "have, so it is disabled.",
    "s_growth": "What $1 became following this exit rule: equal-weight basket, "
    "compounded daily, gross — before trading costs (not estimated). Fills are "
    "assumed at exactly the trigger price. For the limit+stop rule, daily data "
    "can't tell which order filled first when both triggered, so the result is "
    "a worst–best band.",
    "s_win": "Share of individual picks whose strategy return was positive. "
    "Never read it without the growth number beside it: a high win rate can "
    "coexist with losing most of your capital when the losers are bigger than "
    "the winners.",
}


def _info(key: str, variant: str = "") -> str:
    """A hoverable/focusable 'i' icon with a CSS tooltip. Text is authored here
    (never user data), but escape anyway — belt and suspenders, and the aria
    label must not break the attribute. variant is '' | 'start' | 'end' to
    anchor edge-column tooltips inside the viewport."""
    text = _INFO_TEXT[key]
    tip_cls = "tp-tip" + (f" tp-tip--{variant}" if variant else "")
    return (
        f'<span class="tp-i" tabindex="0" role="note" '
        f'aria-label="{html.escape(text, quote=True)}">i'
        f'<span class="{tip_cls}">{html.escape(text, quote=False)}</span></span>'
    )


def _chart_svg(scored: pd.DataFrame) -> str:
    """Per-day hit-rate bars with a base-rate dash marker (shared 0..max axis)."""
    width, height, pad_l, pad_b, pad_t = 780, 200, 46, 26, 12
    plot_w, plot_h = width - pad_l - 10, height - pad_t - pad_b
    top = max(0.05, float(scored["precision"].max()), float(scored["base_rate"].max())) * 1.15

    n = len(scored)
    slot = plot_w / n
    bar_w = min(40.0, slot * 0.5)

    def y(v: float) -> float:
        return pad_t + plot_h * (1 - v / top)

    parts = [
        '<defs><filter id="glow" x="-40%" y="-40%" width="180%" height="180%">'
        '<feGaussianBlur stdDeviation="2.2" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        "</filter></defs>"
    ]
    for frac in (0.25, 0.5, 0.75, 1.0):
        gy = y(top * frac / 1.15)
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - 10}" y2="{gy:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{pad_l - 7}" y="{gy + 3:.1f}" text-anchor="end" font-size="10" '
            f'font-family="ui-monospace, Consolas, monospace" '
            f'fill="var(--ink-muted)">{top * frac / 1.15:.0%}</text>'
        )
    for i, row in enumerate(scored.itertuples()):
        cx = pad_l + slot * (i + 0.5)
        bx = cx - bar_w / 2
        by = y(row.precision)
        beat = row.precision >= row.base_rate
        color = "var(--up)" if beat else "var(--down)"
        date = pd.Timestamp(row.target_date).strftime("%b %d")
        parts.append(
            f"<g><title>{date}: top-{int(row.n_scored)} hit rate "
            f"{row.precision:.0%}, base rate {row.base_rate:.0%}</title>"
            f'<path d="M{bx:.1f} {y(0):.1f} V{by + 4:.1f} Q{bx:.1f} {by:.1f} '
            f"{bx + 4:.1f} {by:.1f} H{bx + bar_w - 4:.1f} Q{bx + bar_w:.1f} {by:.1f} "
            f'{bx + bar_w:.1f} {by + 4:.1f} V{y(0):.1f} Z" fill="{color}" '
            f'fill-opacity="0.85" filter="url(#glow)"/>'
            f'<line x1="{bx - 4:.1f}" y1="{y(row.base_rate):.1f}" x2="{bx + bar_w + 4:.1f}" '
            f'y2="{y(row.base_rate):.1f}" stroke="var(--amber)" stroke-width="2.5"/>'
            f'<text x="{cx:.1f}" y="{by - 6:.1f}" text-anchor="middle" font-size="11" '
            f'font-family="ui-monospace, Consolas, monospace" '
            f'fill="var(--ink-1)">{row.precision:.0%}</text>'
            f'<text x="{cx:.1f}" y="{height - 8}" text-anchor="middle" font-size="10" '
            f'fill="var(--ink-muted)">{date}</text></g>'
        )
    parts.append(
        f'<line x1="{pad_l}" y1="{y(0):.1f}" x2="{width - 10}" y2="{y(0):.1f}" '
        f'stroke="var(--baseline)" stroke-width="1"/>'
    )
    return (
        '<div class="legend">'
        '<span><span class="chip" style="background:var(--up)"></span>hit rate (beat market)</span>'
        '<span><span class="chip" style="background:var(--down)"></span>hit rate (missed)</span>'
        '<span><span class="chip" style="background:var(--amber)"></span>market base rate'
        f"{_info('chart')}</span>"
        "</div>"
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Daily top-N hit rate vs market base rate">{"".join(parts)}</svg>'
    )


def _finite(value) -> bool:
    """A real, plottable number — not None and not a NaN (NaN survives a JSON
    round-trip out of the ledger and passes every `>= 0` comparison silently)."""
    return isinstance(value, int | float) and math.isfinite(value)


def _benchmark_tiles(benchmark: tuple[int, dict, object, object] | None, top: int) -> str:
    """Lead tiles: the walk-forward touch-era benchmark (AUC, lift, reach-rate).

    AUC leads because it is base-rate-invariant; lift compresses at the high
    touch base rate, so its tooltip and the caveats say to read AUC/excess too.
    Degrades gracefully to one honest tile when no touch benchmark is recorded
    (the required post-Stage-A `twopercent benchmark` op has not run yet)."""
    if benchmark is None:
        return (
            '<div class="tile"><span class="label">Walk-forward benchmark</span>'
            '<b>—</b><span class="cmp">no touch-era benchmark yet — '
            "run twopercent benchmark</span></div>"
        )
    _exp_id, metrics, _ts, _te = benchmark
    auc = metrics.get("auc")
    lift = metrics.get("lift")
    reach = metrics.get("precision_at_n")
    base = metrics.get("base_rate")
    bn = metrics.get("top_n", top)
    auc_val = f"{auc:.3f}" if _finite(auc) else "—"
    lift_val = f"{lift:.2f}×" if _finite(lift) else "—"
    lift_up = "up" if _finite(lift) and lift >= 1 else ""
    reach_val = f"{reach:.0%}" if _finite(reach) else "—"
    base_cmp = f"base rate {base:.0%}" if _finite(base) else "base rate —"
    return (
        f'<div class="tile"><span class="label">Walk-forward AUC{_info("t_auc")}</span>'
        f'<b>{auc_val}</b><span class="cmp">base-rate-invariant ranking</span></div>'
        f'<div class="tile"><span class="label">Walk-forward lift{_info("t_bench_lift")}</span>'
        f'<b class="{lift_up}">{lift_val}</b>'
        f'<span class="cmp">top-{bn} reach vs base</span></div>'
        f'<div class="tile"><span class="label">Walk-forward reach-rate (top {bn})'
        f"{_info('t_reach')}</span>"
        f'<b>{reach_val}</b><span class="cmp">{base_cmp}</span></div>'
    )


def _reliability_svg(reliability: list[dict], pop_label: str) -> str:
    """Reliability curve: predicted-bucket mean (x) vs realized reach (y), with
    the perfect-calibration diagonal. Self-contained inline SVG, theme-aware —
    same var(--*) palette as _chart_svg. Points sized by bucket count; empty
    buckets are simply absent (never plotted as a silent 0). pop_label names the
    population the curve is computed on (picks vs all names) in the legend."""
    width, height, pad_l, pad_b, pad_t, pad_r = 460, 300, 42, 30, 12, 14
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b

    def x(v: float) -> float:
        return pad_l + plot_w * v

    def y(v: float) -> float:
        return pad_t + plot_h * (1 - v)

    parts = [
        '<defs><filter id="cglow" x="-40%" y="-40%" width="180%" height="180%">'
        '<feGaussianBlur stdDeviation="2" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        "</filter></defs>"
    ]
    for frac in (0.25, 0.5, 0.75, 1.0):
        gy, gx = y(frac), x(frac)
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{pad_l - 6}" y="{gy + 3:.1f}" text-anchor="end" font-size="10" '
            f'font-family="ui-monospace, Consolas, monospace" '
            f'fill="var(--ink-muted)">{frac:.0%}</text>'
            f'<text x="{gx:.1f}" y="{height - 10}" text-anchor="middle" font-size="10" '
            f'fill="var(--ink-muted)">{frac:.0%}</text>'
        )
    # Perfect-calibration diagonal (predicted == realized).
    parts.append(
        f'<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(1):.1f}" y2="{y(1):.1f}" '
        f'stroke="var(--baseline)" stroke-width="1.5" stroke-dasharray="5 4"/>'
    )
    populated = [r for r in reliability if r["count"] and _finite(r["mean_pred"])]
    max_count = max((r["count"] for r in populated), default=1)
    pts = [(x(r["mean_pred"]), y(r["reach_rate"])) for r in populated]
    if len(pts) >= 2:
        path = "M" + " L".join(f"{px:.1f} {py:.1f}" for px, py in pts)
        parts.append(
            f'<path d="{path}" fill="none" stroke="var(--info)" stroke-width="2" '
            f'stroke-opacity="0.75"/>'
        )
    for r, (px, py) in zip(populated, pts, strict=True):
        radius = 3 + 6 * math.sqrt(r["count"] / max_count)
        parts.append(
            f"<g><title>predicted {r['mean_pred']:.0%}, realized {r['reach_rate']:.0%} "
            f"(n={r['count']})</title>"
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius:.1f}" fill="var(--info)" '
            f'fill-opacity="0.85" filter="url(#cglow)"/></g>'
        )
    parts.append(
        f'<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(0):.1f}" y2="{y(1):.1f}" '
        f'stroke="var(--baseline)" stroke-width="1"/>'
        f'<line x1="{x(0):.1f}" y1="{y(0):.1f}" x2="{x(1):.1f}" y2="{y(0):.1f}" '
        f'stroke="var(--baseline)" stroke-width="1"/>'
    )
    # Axis names live in the legend, not as inline SVG labels — an inline label
    # at the 100% corner collides with the 100% tick text ("predict100%").
    return (
        '<div class="legend">'
        '<span><span class="chip" style="background:var(--info)"></span>'
        f"realized reach (y) vs predicted prob (x) · {html.escape(pop_label)}"
        f"{_info('cal_curve')}</span>"
        "<span>· · · perfect calibration</span>"
        "</div>"
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Reliability curve: predicted probability (x) vs realized reach rate (y)">'
        f"{''.join(parts)}</svg>"
    )


# A confident bucket (predicted >= this) whose realized reach misses its
# prediction by more than the tolerance must be NAMED in the read regardless of
# its count — the count-weighted verdict alone drowns the small confident tail
# where the user actually acts (FIX A / both reviewers).
_CONFIDENT_FLOOR = 0.70
_TAIL_TOLERANCE = 0.075


def _calibration_read(reliability: list[dict], slope, noun_cap: str, noun_low: str) -> str:
    """One honest plain-language sentence computed FROM the buckets — never a
    canned claim. Reports a representative bucket (predicted vs realized), a
    count-weighted verdict, AND — separately — any confident-tail miss, which the
    weighted verdict would otherwise average away. noun_cap/noun_low name the
    population (Picks / predictions) so the language matches what was measured."""
    populated = [
        r
        for r in reliability
        if r["count"] and _finite(r["mean_pred"]) and _finite(r["reach_rate"])
    ]
    if not populated:
        return f"Not enough test {noun_low} to read calibration."
    total = sum(r["count"] for r in populated)
    # Weighted mean |predicted − realized| across occupied buckets (expected
    # calibration error): the honest average gap between promise and reality.
    ece = sum(r["count"] * abs(r["mean_pred"] - r["reach_rate"]) for r in populated) / total
    # Representative bucket: the most-populated confident (>=50%) bucket, else the
    # highest-predicted occupied bucket — the number the user most acts on.
    confident = [r for r in populated if r["mean_pred"] >= 0.5]
    rep = max(confident or populated, key=lambda r: (r["count"], r["mean_pred"]))
    # The confident-tail miss: an occupied confident bucket (predicted >= floor)
    # that OVERPROMISES — realized reach falls short by more than tolerance. This
    # is the trust hazard (the user over-trusts the number); an underconfident
    # bucket, or the tiny top noise bucket that happens to reach 100%, is not.
    # Detected regardless of count (it must NOT be averaged away), but among
    # candidates the most material (highest count) is named, not the noisiest.
    overpromise = [
        r
        for r in populated
        if r["mean_pred"] >= _CONFIDENT_FLOOR
        and (r["mean_pred"] - r["reach_rate"]) >= _TAIL_TOLERANCE
    ]
    tail = max(overpromise, key=lambda r: (r["count"], r["mean_pred"])) if overpromise else None
    verdict = (
        "well calibrated"
        if ece < 0.05
        else "roughly calibrated"
        if ece < 0.10
        else "materially miscalibrated"
    )
    if tail is not None and verdict in ("well calibrated", "roughly calibrated"):
        verdict += " in the bulk"
    # Global slope tilt only when NO specific tail is named. Over the compressed
    # pick range the slope is dragged by the tail bucket, so a blanket
    # "overconfident" would contradict "well calibrated in the bulk" — the tail
    # sentence carries the overconfidence signal more honestly there.
    tilt = ""
    if tail is None and _finite(slope):
        if slope < 0.85:
            tilt = f" — the {noun_low} are overconfident (too extreme)"
        elif slope > 1.15:
            tilt = f" — the {noun_low} are underconfident (too timid)"
    tail_txt = ""
    if tail is not None:
        # Direction (overconfident) is robust; the tail is concentrated in a few
        # regime-days, so hedge the magnitude (quant-skeptic Note C).
        tail_txt = (
            f" But {noun_low} it calls ~{tail['mean_pred']:.0%} reach only "
            f"~{tail['reach_rate']:.0%} — overconfident in the tail (direction robust; the "
            f"tail is regime-clustered, so read it as a direction, not a hard magnitude)."
        )
    return (
        f"{noun_cap} the model calls ~{rep['mean_pred']:.0%} reached +2% about "
        f"{rep['reach_rate']:.0%} of the time. Averaged over all buckets the "
        f"probability is off by {ece:.0%} — {verdict}{tilt}.{tail_txt}"
    )


def _regime_note(cal: dict) -> str:
    """Surface calibration that differs across base-rate regimes — a pooled curve
    can look honest while every regime is off. Renders each stratum's base-rate
    span, BSS, and slope; flags when the BSS spread across regimes is wide."""
    regimes = cal.get("regimes") or []
    if not regimes:
        return ""
    cells = []
    for r in regimes:
        bss = f"{r['bss']:+.2f}" if _finite(r["bss"]) else "—"
        slope = f"{r['slope']:.2f}" if _finite(r["slope"]) else "—"
        cells.append(
            f"<b>{html.escape(r['name'])}</b> base {r['base_lo']:.0%}–{r['base_hi']:.0%} "
            f"({r['n_days']}d): BSS {bss}, slope {slope}"
        )
    bsss = [r["bss"] for r in regimes if _finite(r["bss"])]
    flag = ""
    if len(bsss) >= 2 and (max(bsss) - min(bsss)) >= 0.10:
        flag = (
            " Calibration DIFFERS by regime — read per-regime, the pooled curve "
            "can average away a regime-specific miss."
        )
    return (
        '<p class="sub" style="margin-top:8px">By base-rate regime (walk-forward, '
        "all names, ranking skill): " + " · ".join(cells) + f".{flag}</p>"
    )


def _calibration_panel(benchmark: tuple[int, dict, object, object] | None) -> str:
    """ "Calibration (walk-forward)" section. The PRIMARY reliability curve + read
    are on the TOP-N LIQUID PICKS — the population the user acts on ("a pick
    called X% reaches ~X%?"). The all-names Brier skill score rides as an
    explicitly-labelled SECONDARY tile (ranking discrimination over the daily
    base rate — never implying it describes the picks). Degrades gracefully: no
    touch-era benchmark, or a pre-Stage-C row with no calibration block, or an
    older touch row that carries all-names calibration but no pick block (the
    curve then falls back to all names, loudly labelled). Backtest; no dollars."""
    header = "<h2>Calibration (walk-forward)</h2>"
    empty = (
        header + '<div class="card"><p class="empty">No touch-era benchmark yet — run '
        "twopercent benchmark to measure whether a pick the model calls ~70% "
        "reaches +2% about 70% of the time.</p></div>"
    )
    if benchmark is None:
        return empty
    cal = benchmark[1].get("calibration")
    if not cal or not cal.get("reliability"):
        return empty

    # Secondary: all-names Brier skill score (ranking discrimination). Labelled so
    # it is never read as "the picks add no skill" — it is a different question.
    bss = cal.get("bss")
    bss_val = f"{bss:+.2f}" if _finite(bss) else "—"
    bss_up = "up" if _finite(bss) and bss > 0 else ""
    tile = (
        '<div class="tiles">'
        f'<div class="tile"><span class="label">Ranking skill · BSS (all names)'
        f'{_info("t_bss")}</span><b class="{bss_up}">{bss_val}</b>'
        f'<span class="cmp">vs the daily base rate · {cal["n"]:,} all-names test '
        "predictions</span></div></div>"
    )

    # Primary: the pick population the user acts on, falling back to all names
    # (loudly) only when an older row carries no pick block.
    picks = cal.get("picks")
    if picks and any(r["count"] for r in picks.get("reliability", [])):
        reliability = picks["reliability"]
        slope = picks.get("slope")
        pop_label = f"top-{picks['top_n']} picks"
        read = _calibration_read(reliability, slope, "Picks", "picks")
        floor_note = (
            f'<p class="sub" style="margin-top:10px">Reliability of the top-'
            f"{picks['top_n']} liquidity-floored PICKS — the population you act on — "
            f"over {picks['n']:,} pick-predictions on {picks['n_days']} test days, "
            "out-of-sample per fold.</p>"
        )
    else:
        reliability = cal["reliability"]
        slope = cal.get("slope")
        pop_label = "all names"
        read = _calibration_read(reliability, slope, "Predictions", "predictions")
        floor_note = (
            '<p class="sub" style="margin-top:10px"><b>No pick-population block '
            "recorded yet</b> — showing all-names reliability (a different question "
            "from the picks you act on). Re-run twopercent benchmark to record it.</p>"
        )

    return (
        header
        + tile
        + "<div class='card'>"
        + _reliability_svg(reliability, pop_label)
        + floor_note
        + f'<p class="sub" style="margin-top:6px">{html.escape(read)}</p>'
        + _regime_note(cal)
        + '<p class="sub" style="margin-top:8px">Reliability curve: the top-N '
        "liquidity-floored picks, out-of-sample per fold — the population you act "
        "on. BSS tile + regime line: all names (ranking skill), same population as "
        "the benchmark AUC/Brier. Walk-forward — the model never trains on the days "
        "it scores. Backtest, not the live record; measured, not recalibrated.</p>" + "</div>"
    )


_LIVE_EMPTY_TILE = (
    '<div class="tile"><span class="label">Live reach-rate{info}</span>'
    '<b>—</b><span class="cmp">no live days yet — the live record starts with '
    "the first touch prediction, kept separate from the backtest tiles</span></div>"
)


def _live_tiles(record: track.TrackRecord, top: int) -> str:
    """Accumulating LIVE reach record: reach-rate, lift, and excess (reach −
    base). LIVE days ONLY (late == False): a backfilled day's outcome was known
    when the pick was saved, so pooling it in would inflate a "live" number —
    exactly the silent success this project guards against. Pooled (base rate
    pick-weighted like precision). Never dollars.

    Until the first live day the tile does NOT vanish (that would leave a skimmer
    reading the walk-forward benchmark tiles as system performance): it renders an
    explicit "no live days yet" placeholder so there is visible tile-level
    contrast between the backtest and the empty forward record."""
    live = (
        record.scored[~record.scored["late"].astype(bool)]
        if not record.scored.empty
        else record.scored
    )
    if live.empty:
        return _LIVE_EMPTY_TILE.format(info=_info("t_live_reach"))
    total_n = live["n_scored"].sum()
    reach = live["hits"].sum() / total_n
    # Weight base rates like precision is pooled, or the headline lift is
    # inconsistent when day sizes / base rates differ.
    base = (live["base_rate"] * live["n_scored"]).sum() / total_n
    lift = reach / base if base > 0 else float("nan")
    excess = reach - base
    lift_txt = f"{lift:.2f}×" if math.isfinite(lift) else "—"
    return (
        f'<div class="tile"><span class="label">Live reach-rate (top {top})'
        f"{_info('t_live_reach')}</span><b>{reach:.0%}</b>"
        f'<span class="cmp">market base {base:.0%}</span></div>'
        f'<div class="tile"><span class="label">Live lift{_info("t_live_lift")}</span>'
        f'<b class="{"up" if math.isfinite(lift) and lift >= 1 else ""}">{lift_txt}</b>'
        f'<span class="cmp">vs picking at random</span></div>'
        f'<div class="tile"><span class="label">Live excess{_info("t_excess")}</span>'
        f'<b class="{"up" if excess >= 0 else ""}">{excess:+.1%}</b>'
        f'<span class="cmp">reach-rate − base rate</span></div>'
    )


def _tiles(
    record: track.TrackRecord,
    benchmark: tuple[int, dict, object, object] | None,
    n_candidates: int,
    top: int,
) -> str:
    """Summary tiles: lead with the walk-forward benchmark, then candidates
    today, then the accumulating live reach record. No dollars anywhere."""
    return (
        '<div class="tiles">'
        + _benchmark_tiles(benchmark, top)
        + f'<div class="tile"><span class="label">Candidates today{_info("t_cands")}</span>'
        f"<b>{n_candidates}</b>"
        f'<span class="cmp">ranked by probability</span></div>'
        + _live_tiles(record, top)
        + "</div>"
    )


_SIM_CAVEAT = (
    "Walk-forward simulation: the model never trains on the days it predicts, "
    "but the system was designed with this history visible. Picks require a "
    "next-day bar and today's universe is applied to history, so delisted names "
    "can never contribute their final catastrophic day — the reach-rate and any "
    "strategy P&L are biased up (survivorship). Browsing strategy, basket, and "
    "window combinations is itself a form of selection — flipping "
    "strategies/baskets/windows to find a good-looking cell is itself luck — and "
    "short windows are small samples. Lift compresses at a high base "
    "rate (it is capped at 1 / base-rate), so read AUC and excess (reach − base) "
    "too. The live record above is the clean test."
)

BASKET_CHOICES = [1, 5, 10, 15, 20]
BASKET_DEFAULT = 5
WINDOW_DEFAULT = 126  # "6 months" in track.SIM_WINDOW_SPECS


def _embed_json(payload: dict) -> str:
    """Serialize the payload into its inline script tag, breakout-proof.

    json.dumps leaves "<" unescaped; today no field carries free-form text,
    but the obvious future addition (symbols per pick) would let a literal
    "</script>" end the tag early. \\u003c/\\u003e are valid JSON escapes that
    JSON.parse restores, so escaping costs nothing. NaN must fail loudly."""
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    text = text.replace("<", "\\u003c").replace(">", "\\u003e")
    return f'<script type="application/json" id="tp-data">{text}</script>'


def _payload_num(value) -> float | None:
    """Payload outcome return: rounded to 6 places; None for a NULL/NaN (a
    pre-upgrade sim row without oh/ol) → JSON null, which BOTH summarize
    mirrors treat as 'no strategy data recorded', never as a silent zero.
    json.dumps(allow_nan=False) could not carry a NaN anyway."""
    if value is None:
        return None
    value = float(value)
    return round(value, 6) if math.isfinite(value) else None


def _payload_days(frame: pd.DataFrame, bases: dict, late: bool = False) -> list[dict]:
    """Per-rank frame -> ordered payload day dicts {d, base, [late,] picks}.

    Each pick is [rank, oh, ol, oc, hit]: the reach view reads only rank+hit;
    the strategy explorer replays exit rules from the oh/ol/oc outcome returns
    (null on rows recorded before the strategy upgrade — degradation, not 0).
    No dollar figure rides in the payload; growth is computed client/server in
    lockstep from these raw returns."""
    days: list[dict] = []
    for d, grp in frame.groupby("target_date"):
        day = pd.Timestamp(d).date()
        base = bases.get(day)
        entry = {
            "d": str(day),
            "base": round(float(base), 6) if base is not None else None,
            "picks": [
                [int(rank), _payload_num(oh), _payload_num(ol), _payload_num(oc), int(hit)]
                for rank, oh, ol, oc, hit in zip(
                    grp["rank"], grp["oh"], grp["ol"], grp["oc"], grp["hit"], strict=True
                )
            ],
        }
        if late:
            entry["late"] = bool(grp["late"].iloc[0])
        days.append(entry)
    return days


def _summarize_days(days: list[dict], n: int) -> dict:
    """Basket reach stats over payload days — EXACT server-side mirror of the
    inline JS tpMath.summarize(): basket = first n picks of each day (rank
    order; taking the next available name IS the substitution rule), day reach
    = mean of the basket's hit flags (pick index 4 — [rank, oh, ol, oc, hit]).
    A non-finite basket reach is counted as corrupt, never silently averaged
    around."""
    hit_sum, base_sum = 0.0, 0.0
    base_known = short = subst = corrupt = 0
    for day in days:
        picks = day["picks"][:n]
        if len(picks) < n:
            short += 1
        if any(p[0] != i + 1 for i, p in enumerate(picks)):
            # The basket traded lower-ranked names in place of missing picks —
            # the same event daily_pick_performance warns about (a missing
            # rank may hide exactly the catastrophic day). Count and disclose.
            subst += 1
        hit = sum(p[4] for p in picks) / len(picks)
        if not math.isfinite(hit):
            corrupt += 1
            continue
        hit_sum += hit
        base = day.get("base")
        if base is not None and math.isfinite(base):
            base_sum += base
            base_known += 1
    clean = len(days) - corrupt
    return {
        "hit": hit_sum / clean if clean else float("nan"),
        "base": base_sum / base_known if base_known else None,
        "days": len(days),
        "short": short,
        "subst": subst,
        "base_days": base_known,
        "clean": clean,
        "corrupt": corrupt,
    }


def _row_notes(prefix: str, s: dict, n: int) -> list[str]:
    """Per-row disclosures — mirrors JS rowNotes(). Generic for SIM and LIVE:
    sim ranks are contiguous by construction so substitution can only fire for
    live frames today, but the check is not special-cased."""
    notes: list[str] = []
    if s["short"]:
        notes.append(f"{prefix}: {s['short']} day(s) had fewer than {n} picks")
    if s["subst"]:
        notes.append(
            f"{prefix}: {s['subst']} day(s) substituted lower-ranked names for missing picks"
        )
    if s["base_days"] and s["base_days"] < s["clean"]:
        notes.append(f"{prefix}: base rate from {s['base_days']} of {s['clean']} day(s)")
    return notes


def _explorer_state(
    sim_days: list[dict], live_days: list[dict], n: int, w: int
) -> tuple[dict | None, dict | None, list[str]]:
    """Selection outcome for both rows + disclosure notes — mirrors JS update().

    Honesty rules enforced identically on both sides: a SIM window renders
    only when that many sim days exist; LIVE uses live-only days and says
    "all M live day(s)" when short of the window instead of pretending;
    short-picks days, within-basket substitutions, partial base-rate coverage,
    and corrupt days are disclosed, never silent."""
    notes: list[str] = []
    sim_s = None
    if not sim_days:
        notes.append("No touch-era benchmark yet — run twopercent benchmark.")
    elif len(sim_days) < w:
        notes.append(f"SIM: needs {w} trading days — {len(sim_days)} available")
    else:
        s = _summarize_days(sim_days[-w:], n)
        if s["corrupt"]:
            notes.append(f"SIM: {s['corrupt']} corrupt day(s) in window — data error")
        else:
            sim_s = s
            notes.extend(_row_notes("SIM", s, n))
    live = [d for d in live_days if not d["late"]]
    live_s = None
    if not live:
        notes.append("LIVE: no live days yet")
    else:
        s = _summarize_days(live[-min(w, len(live)) :], n)
        if s["corrupt"]:
            notes.append(f"LIVE: {s['corrupt']} corrupt day(s) — data error")
        else:
            live_s = s
            if len(live) < w:
                notes.append(f"LIVE: all {len(live)} live day(s) — fewer than the {w}-day window")
            notes.extend(_row_notes("LIVE", s, n))
    return sim_s, live_s, notes


def _pct_half_up(x: float) -> str:
    """Percent rounded half-UP, matching JS Math.round — Python's :.0% rounds
    half-even (0.125 → "12%") and would visibly flicker to the JS "13%" on the
    first select change."""
    return f"{math.floor(100 * x + 0.5)}%"


def _fixed_half_up(x: float, digits: int) -> str:
    """Non-negative x to fixed decimals, exactly matching JS toFixed.

    toFixed rounds the double's EXACT decimal value, ties away from zero —
    Decimal(x) carries that exact value. Multiply-then-floor would inject its
    own FP error and disagree on values like 1.0005 (×1000 rounds UP to
    1000.5 in binary while the double itself sits below the half)."""
    return str(Decimal(x).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP))


def _explorer_cells(prefix: str, s: dict | None) -> str:
    if s is None or not math.isfinite(s["hit"]):
        return "".join(
            f'<td id="tp-{prefix}-{col}">—</td>' for col in ("hit", "base", "lift", "days")
        )
    base = s["base"]
    base_txt = _pct_half_up(base) if base is not None else "—"
    lift_txt = f"{_fixed_half_up(s['hit'] / base, 2)}×" if base else "—"
    return (
        f'<td id="tp-{prefix}-hit">{_pct_half_up(s["hit"])}</td>'
        f'<td id="tp-{prefix}-base">{base_txt}</td>'
        f'<td id="tp-{prefix}-lift">{lift_txt}</td>'
        f'<td id="tp-{prefix}-days">{s["days"]}</td>'
    )


def _strategy_state(
    sim_days: list[dict], live_days: list[dict], n: int, w: int, strat: str
) -> tuple[dict | None, dict | None, list[str]]:
    """Strategy-view outcome for both rows + disclosure notes — mirrors the JS
    updateStrat(). Same honesty rules as _explorer_state, plus: a window
    containing ANY day without recorded oh/ol/oc (a pre-upgrade benchmark)
    yields NO number, only 're-run twopercent benchmark' — a growth compounded
    over a silently different set of days would be the classic silent-drop bug.
    LIVE uses live-only days (late/backfilled excluded): the survivorship-free
    framing is only honest if the row truly is genuinely-live picks."""
    notes: list[str] = []
    sim_s = None
    if not sim_days:
        notes.append("No touch-era benchmark yet — run twopercent benchmark.")
    elif len(sim_days) < w:
        notes.append(f"SIM: needs {w} trading days — {len(sim_days)} available")
    else:
        s = strategy.summarize_strategy_days(sim_days[-w:], n, strat)
        if s["missing"]:
            notes.append(
                f"SIM: no strategy data recorded on {s['missing']} of {s['days']} "
                "day(s) — re-run twopercent benchmark"
            )
        elif s["corrupt"]:
            notes.append(f"SIM: {s['corrupt']} corrupt day(s) in window — data error")
        else:
            sim_s = s
            notes.extend(_row_notes("SIM", s, n))
    live = [d for d in live_days if not d["late"]]
    live_s = None
    if not live:
        notes.append("LIVE: no live days yet")
    else:
        s = strategy.summarize_strategy_days(live[-min(w, len(live)) :], n, strat)
        if s["missing"]:
            notes.append(f"LIVE: {s['missing']} day(s) missing strategy data — data error")
        elif s["corrupt"]:
            notes.append(f"LIVE: {s['corrupt']} corrupt day(s) — data error")
        else:
            live_s = s
            if len(live) < w:
                notes.append(f"LIVE: all {len(live)} live day(s) — fewer than the {w}-day window")
            notes.extend(_row_notes("LIVE", s, n))
    return sim_s, live_s, notes


def _growth_text(s: dict) -> str:
    """ "$1 → $X", or "$1 → $worst–$best" when the stop/limit band is open —
    mirrors JS tpMath.growthText(). Band collapse compares the FORMATTED ends
    (like the JS) so a sub-cent band never renders as a fake spread."""
    lo, hi = _fixed_half_up(s["gw"], 2), _fixed_half_up(s["gb"], 2)
    return f"$1 → ${lo}" if lo == hi else f"$1 → ${lo}–${hi}"


def _win_text(s: dict) -> str:
    """Pick win rate, "X%" or band "X%–Y%" — mirrors JS tpMath.winText()."""
    if not math.isfinite(s["ww"]):
        return "—"
    lo, hi = _pct_half_up(s["ww"]), _pct_half_up(s["wb"])
    return lo if lo == hi else f"{lo}–{hi}"


def _became_text(s: dict) -> str:
    """The read-sentence growth phrase — mirrors JS tpMath.becameText()."""
    lo, hi = _fixed_half_up(s["gw"], 2), _fixed_half_up(s["gb"], 2)
    return f"${lo}" if lo == hi else f"between ${lo} (worst case) and ${hi} (best case)"


def _strategy_read(sim_s: dict | None, live_s: dict | None) -> str:
    """One plain-language sentence translating the selected strategy cell for a
    non-financial reader — mirrors JS tpMath.readText(). SIM leads (it is the
    cell with the most days) but is always framed as the gross upper bound;
    with no SIM the LIVE row is translated with its small-sample framing."""
    if sim_s is not None and math.isfinite(sim_s["gw"]):
        return (
            f"$1 following this rule on the SIM picks became {_became_text(sim_s)} over "
            f"these {sim_s['days']} trading days — gross (before trading costs, not "
            "estimated), and an optimistic upper bound: the delisted losers are missing "
            "from the history."
        )
    if live_s is not None and math.isfinite(live_s["gw"]):
        return (
            f"$1 following this rule on the LIVE picks became {_became_text(live_s)} over "
            f"{live_s['days']} live day(s) — gross (before trading costs, not estimated); "
            "real logged picks, but a small sample."
        )
    return "No strategy result to translate yet — see the notes below."


def _strategy_cells(prefix: str, s: dict | None, stamp: str = "") -> str:
    """Growth + win rate + days cells for one strategy-table row. Growth and win
    rate are ALWAYS emitted together — a win rate without its growth number is
    the exact half-truth quant-skeptic barred (79% wins can lose 60% of $1).
    The growth number lives in an inner span so the JS can update it while the
    stamp markup (SIM's OPTIMISTIC UPPER BOUND) stays put beside it."""
    if s is None or not math.isfinite(s["gw"]):
        growth = win = days = "—"
    else:
        growth, win, days = _growth_text(s), _win_text(s), str(s["days"])
    return (
        f'<td><span id="tp-{prefix}-growth">{growth}</span>{stamp}</td>'
        f'<td id="tp-{prefix}-win">{win}</td>'
        f'<td id="tp-{prefix}-days">{days}</td>'
    )


# The strategy table is server-rendered for the FIRST enabled exit rule (the
# page defaults to the reach view; this hidden default keeps the server the
# source of truth for the numbers the JS re-derives — lockstep, not trust).
STRATEGY_SERVER_DEFAULT = "hold_close"

_STAMP_UB = ' <span class="stamp-ub">OPTIMISTIC UPPER BOUND</span>'


def _strategy_card(sim_days: list[dict], live_days: list[dict], n: int, w: int) -> str:
    """The P&L what-if card (hidden while Strategy = Reach rate): SIM/LIVE rows
    of $1-growth + win rate, the survivorship stamp ON the SIM number, and the
    plain-language framing lines. Every number gross; no cost constant exists."""
    sim_s, live_s, notes = _strategy_state(sim_days, live_days, n, w, STRATEGY_SERVER_DEFAULT)
    return (
        '<div class="card" id="tp-card-strat" hidden><table><tr>'
        f"<th>Record{_info('e_record', 'start')}</th>"
        f"<th>$1 → growth{_info('s_growth')}</th>"
        f"<th>Win rate{_info('s_win')}</th>"
        f"<th>Days{_info('e_days', 'end')}</th></tr>"
        '<tr><td><span class="badge-sim">SIM</span> walk-forward backtest</td>'
        + _strategy_cells("strat-sim", sim_s, _STAMP_UB)
        + "</tr>"
        '<tr><td><span class="badge-live">LIVE</span> real logged picks — '
        "survivorship-free, small sample</td>"
        + _strategy_cells("strat-live", live_s)
        + "</tr></table>"
        f'<p class="sub" id="tp-strat-read">{html.escape(_strategy_read(sim_s, live_s))}</p>'
        '<p class="sub"><b>SIM is an optimistic upper bound:</b> the backtest history '
        "excludes every company that delisted; if even a few of their blow-up days "
        "were real, this could be a loss. The LIVE row is the honest number — real "
        "logged picks, real outcomes, survivorship-free — but a small sample.</p>"
        '<p class="sub">All returns are gross — before trading costs (not estimated). '
        "Limit and stop fills are assumed at exactly the trigger price; a "
        "glitch-suspect high never counts as a fill. For the limit+stop rule, when "
        "both could fill on the same day the result is shown as best case / worst "
        "case — daily data can't tell which came first.</p>"
        f'<p class="sub" id="tp-strat-note">{html.escape(" · ".join(notes))}</p></div>'
    )


def _record_explorer(meta: dict | None, sim_days: list[dict], live_days: list[dict]) -> str:
    """Interactive basket/window/strategy explorer: amber SIM row vs green LIVE
    row, reach-rate by default with opt-in exit-rule P&L views.

    Server-side renders the default selection (top-5, 6 months, reach rate) so
    the page is meaningful without JS — the reach card is EXACTLY the pre-
    strategy explorer — plus the hidden strategy card at its server default;
    the inline script recomputes on select change from the embedded JSON
    payload."""
    n, w = BASKET_DEFAULT, WINDOW_DEFAULT
    sim_s, live_s, notes = _explorer_state(sim_days, live_days, n, w)
    basket_opts = "".join(
        f'<option value="{b}"{" selected" if b == n else ""}>Top {b}</option>'
        for b in BASKET_CHOICES
    )
    window_opts = "".join(
        f'<option value="{d}"{" selected" if d == w else ""}>{label} ({d} trading days)</option>'
        for label, d in track.SIM_WINDOW_SPECS
    )
    strategy_opts = "".join(
        f'<option value="{key}"{" selected" if key == "reach" else ""}'
        f"{'' if enabled else ' disabled'}>{html.escape(label)}</option>"
        for key, label, enabled in strategy.STRATEGY_CHOICES
    )
    controls = (
        '<p class="sub controls"><label>Basket '
        f'<select id="tp-basket">{basket_opts}</select></label>{_info("e_basket")} '
        f'<label>Window <select id="tp-window">{window_opts}</select></label>'
        f"{_info('e_window')} "
        f'<label>Strategy <select id="tp-strategy">{strategy_opts}</select></label>'
        f"{_info('e_strategy')}</p>"
    )
    span = ""
    if meta is not None:
        months = meta["params"].get("months")
        run_day = pd.Timestamp(meta["run_ts"]).date()
        span = (
            f'<p class="sub">SIM test span <b class="mono">{meta["test_start"]}</b> → '
            f'<b class="mono">{meta["test_end"]}</b> · months={months} · '
            f'simulated <b class="mono">{run_day}</b> · '
            f'<span class="mono">{len(sim_days)}</span> sim days available</p>'
        )
    table = (
        "<div class='card' id='tp-card-reach'><table><tr>"
        f"<th>Record{_info('e_record', 'start')}</th>"
        f"<th>Reach rate{_info('e_hit')}</th><th>Base rate{_info('e_base')}</th>"
        f"<th>Lift{_info('e_lift')}</th><th>Days{_info('e_days', 'end')}</th></tr>"
        '<tr><td><span class="badge-sim">SIM</span> walk-forward, monthly retrain</td>'
        + _explorer_cells("sim", sim_s)
        + "</tr>"
        '<tr><td><span class="badge-live">LIVE</span> logged picks</td>'
        + _explorer_cells("live", live_s)
        + "</tr></table>"
        f'<p class="sub" id="tp-note">{html.escape(" · ".join(notes))}</p></div>'
    )
    return (
        "<h2>Record over trailing windows</h2>"
        + controls
        + table
        + _strategy_card(sim_days, live_days, n, w)
        + span
        + f'<p class="sub" style="margin-top:8px">{_SIM_CAVEAT}</p>'
    )


# Pure explorer math, no DOM: the browser-side HALF of the Python/JS lockstep
# pair (dashboard._summarize_days / strategy.summarize_strategy_days and the
# _growth_text/_win_text/_strategy_read formatters are the server mirrors).
# Kept DOM-free and exposed as `tpMath` so the test suite can execute EXACTLY
# this script under node and diff its numbers against Python's — the P&L math
# is the highest-stakes surface on the page, so the mirror is proven, not
# trusted. The __LIMIT__/__STOP__/__EPS__ constants are injected from
# strategy.py below so the two sides cannot drift.
_JS_MATH = """
<script>
var tpMath = (function () {
  var LIMIT = __LIMIT__, STOP = __STOP__, EPS = __EPS__;
  function pickBand(strat, ol, oc, filled) {
    if (strat === "hold_close") return [oc, oc];
    if (strat === "limit_2pct") { var r = filled ? LIMIT : oc; return [r, r]; }
    if (strat === "limit_stop") {
      var stopped = ol <= STOP + EPS;
      if (filled && stopped) return [STOP, LIMIT];
      if (stopped) return [STOP, STOP];
      if (filled) return [LIMIT, LIMIT];
      return [oc, oc];
    }
    throw new Error("unknown strategy: " + strat);
  }
  function summarize(days, n) {
    var hitSum = 0, baseSum = 0;
    var baseKnown = 0, shortDays = 0, substDays = 0, corrupt = 0;
    for (var i = 0; i < days.length; i++) {
      var picks = days[i].picks.slice(0, n);
      if (picks.length < n) shortDays += 1;
      var hit = 0, sub = false;
      for (var j = 0; j < picks.length; j++) {
        hit += picks[j][4];
        if (picks[j][0] !== j + 1) sub = true;
      }
      if (sub) substDays += 1;
      hit /= picks.length;
      if (!isFinite(hit)) { corrupt += 1; continue; }
      hitSum += hit;
      var b = days[i].base;
      if (b != null && isFinite(b)) { baseSum += b; baseKnown += 1; }
    }
    var clean = days.length - corrupt;
    return { hit: clean ? hitSum / clean : NaN,
             base: baseKnown ? baseSum / baseKnown : null,
             days: days.length, shortDays: shortDays, substDays: substDays,
             baseDays: baseKnown, clean: clean, corrupt: corrupt };
  }
  function stratSummarize(days, n, strat) {
    var gw = 1, gb = 1, winsW = 0, winsB = 0, picksN = 0;
    var shortDays = 0, substDays = 0, missing = 0, corrupt = 0;
    for (var i = 0; i < days.length; i++) {
      var picks = days[i].picks.slice(0, n);
      if (picks.length < n) shortDays += 1;
      var sub = false, miss = false;
      for (var j = 0; j < picks.length; j++) {
        if (picks[j][0] !== j + 1) sub = true;
        if (picks[j][1] == null || picks[j][2] == null || picks[j][3] == null) miss = true;
      }
      if (sub) substDays += 1;
      if (miss) { missing += 1; continue; }
      if (!picks.length) { corrupt += 1; continue; }
      var sumW = 0, sumB = 0, ok = true;
      for (j = 0; j < picks.length; j++) {
        var band = pickBand(strat, picks[j][2], picks[j][3], !!picks[j][4]);
        if (!isFinite(band[0]) || !isFinite(band[1])) { ok = false; break; }
        sumW += band[0]; sumB += band[1];
        if (band[0] > 0) winsW += 1;
        if (band[1] > 0) winsB += 1;
      }
      if (!ok) { corrupt += 1; continue; }
      picksN += picks.length;
      gw *= 1 + sumW / picks.length;
      gb *= 1 + sumB / picks.length;
    }
    var clean = days.length - missing - corrupt;
    return { gw: clean ? gw : NaN, gb: clean ? gb : NaN,
             ww: picksN ? winsW / picksN : NaN, wb: picksN ? winsB / picksN : NaN,
             picks: picksN, days: days.length, clean: clean,
             shortDays: shortDays, substDays: substDays, baseDays: 0,
             missing: missing, corrupt: corrupt };
  }
  function rowNotes(prefix, s, n) {
    var notes = [];
    if (s.shortDays) notes.push(prefix + ": " + s.shortDays +
                                " day(s) had fewer than " + n + " picks");
    if (s.substDays) notes.push(prefix + ": " + s.substDays +
                                " day(s) substituted lower-ranked names for missing picks");
    if (s.baseDays && s.baseDays < s.clean) notes.push(prefix + ": base rate from " +
                                                       s.baseDays + " of " + s.clean + " day(s)");
    return notes;
  }
  function growthText(s) {
    var lo = s.gw.toFixed(2), hi = s.gb.toFixed(2);
    return lo === hi ? "$1 \\u2192 $" + lo : "$1 \\u2192 $" + lo + "\\u2013$" + hi;
  }
  function winText(s) {
    if (!isFinite(s.ww)) return "\\u2014";
    var lo = Math.round(100 * s.ww) + "%", hi = Math.round(100 * s.wb) + "%";
    return lo === hi ? lo : lo + "\\u2013" + hi;
  }
  function becameText(s) {
    var lo = s.gw.toFixed(2), hi = s.gb.toFixed(2);
    return lo === hi ? "$" + lo
                     : "between $" + lo + " (worst case) and $" + hi + " (best case)";
  }
  function readText(simS, liveS) {
    if (simS != null && isFinite(simS.gw))
      return "$1 following this rule on the SIM picks became " + becameText(simS) +
             " over these " + simS.days + " trading days \\u2014 gross (before trading " +
             "costs, not estimated), and an optimistic upper bound: the delisted losers " +
             "are missing from the history.";
    if (liveS != null && isFinite(liveS.gw))
      return "$1 following this rule on the LIVE picks became " + becameText(liveS) +
             " over " + liveS.days + " live day(s) \\u2014 gross (before trading costs, " +
             "not estimated); real logged picks, but a small sample.";
    return "No strategy result to translate yet \\u2014 see the notes below.";
  }
  return { pickBand: pickBand, summarize: summarize, stratSummarize: stratSummarize,
           rowNotes: rowNotes, growthText: growthText, winText: winText,
           becameText: becameText, readText: readText };
})();
</script>
"""
_JS_MATH = (
    _JS_MATH.replace("__LIMIT__", repr(strategy.LIMIT_PROFIT))
    .replace("__STOP__", repr(strategy.STOP_LEVEL))
    .replace("__EPS__", repr(strategy._STOP_EPSILON))
)

_JS = """
<script>
(function () {
  var el = function (id) { return document.getElementById(id); };
  var dataEl = el("tp-data");
  var basket = el("tp-basket");
  var win = el("tp-window");
  var strat = el("tp-strategy");
  if (!dataEl || !basket || !win || !strat) return;
  var data = JSON.parse(dataEl.textContent);
  var DASH = "\\u2014";
  var cardReach = el("tp-card-reach");
  var cardStrat = el("tp-card-strat");
  function setRow(prefix, s) {
    var h = el("tp-" + prefix + "-hit");
    if (s == null || !isFinite(s.hit)) {
      h.textContent = DASH;
      el("tp-" + prefix + "-base").textContent = DASH;
      el("tp-" + prefix + "-lift").textContent = DASH;
      el("tp-" + prefix + "-days").textContent = DASH;
      return;
    }
    h.textContent = Math.round(100 * s.hit) + "%";
    el("tp-" + prefix + "-base").textContent =
      s.base == null ? DASH : Math.round(100 * s.base) + "%";
    el("tp-" + prefix + "-lift").textContent =
      s.base != null && s.base > 0 ? (s.hit / s.base).toFixed(2) + "\\u00d7" : DASH;
    el("tp-" + prefix + "-days").textContent = String(s.days);
  }
  function setStratRow(prefix, s) {
    var g = el("tp-" + prefix + "-growth");
    if (s == null || !isFinite(s.gw)) {
      g.textContent = DASH;
      el("tp-" + prefix + "-win").textContent = DASH;
      el("tp-" + prefix + "-days").textContent = DASH;
      return;
    }
    g.textContent = tpMath.growthText(s);
    el("tp-" + prefix + "-win").textContent = tpMath.winText(s);
    el("tp-" + prefix + "-days").textContent = String(s.days);
  }
  function updateReach(n, w) {
    var notes = [];
    if (!data.sim.length) {
      setRow("sim", null);
      notes.push("No touch-era benchmark yet " + DASH +
                 " run twopercent benchmark.");
    } else if (data.sim.length < w) {
      setRow("sim", null);
      notes.push("SIM: needs " + w + " trading days " + DASH + " " +
                 data.sim.length + " available");
    } else {
      var s = tpMath.summarize(data.sim.slice(-w), n);
      if (s.corrupt) {
        setRow("sim", null);
        notes.push("SIM: " + s.corrupt + " corrupt day(s) in window " + DASH + " data error");
      } else {
        setRow("sim", s);
        notes = notes.concat(tpMath.rowNotes("SIM", s, n));
      }
    }
    var live = [];
    for (var i = 0; i < data.live.length; i++) if (!data.live[i].late) live.push(data.live[i]);
    if (!live.length) { setRow("live", null); notes.push("LIVE: no live days yet"); }
    else {
      var s2 = tpMath.summarize(live.slice(-Math.min(w, live.length)), n);
      if (s2.corrupt) {
        setRow("live", null);
        notes.push("LIVE: " + s2.corrupt + " corrupt day(s) " + DASH + " data error");
      } else {
        setRow("live", s2);
        if (live.length < w) notes.push("LIVE: all " + live.length + " live day(s) " + DASH +
                                        " fewer than the " + w + "-day window");
        notes = notes.concat(tpMath.rowNotes("LIVE", s2, n));
      }
    }
    el("tp-note").textContent = notes.join(" \\u00b7 ");
  }
  function updateStrat(n, w, key) {
    var notes = [];
    var simS = null;
    if (!data.sim.length) {
      notes.push("No touch-era benchmark yet " + DASH +
                 " run twopercent benchmark.");
    } else if (data.sim.length < w) {
      notes.push("SIM: needs " + w + " trading days " + DASH + " " +
                 data.sim.length + " available");
    } else {
      var s = tpMath.stratSummarize(data.sim.slice(-w), n, key);
      if (s.missing) {
        notes.push("SIM: no strategy data recorded on " + s.missing + " of " + s.days +
                   " day(s) " + DASH + " re-run twopercent benchmark");
      } else if (s.corrupt) {
        notes.push("SIM: " + s.corrupt + " corrupt day(s) in window " + DASH + " data error");
      } else {
        simS = s;
        notes = notes.concat(tpMath.rowNotes("SIM", s, n));
      }
    }
    var live = [];
    for (var i = 0; i < data.live.length; i++) if (!data.live[i].late) live.push(data.live[i]);
    var liveS = null;
    if (!live.length) { notes.push("LIVE: no live days yet"); }
    else {
      var s2 = tpMath.stratSummarize(live.slice(-Math.min(w, live.length)), n, key);
      if (s2.missing) {
        notes.push("LIVE: " + s2.missing + " day(s) missing strategy data " + DASH +
                   " data error");
      } else if (s2.corrupt) {
        notes.push("LIVE: " + s2.corrupt + " corrupt day(s) " + DASH + " data error");
      } else {
        liveS = s2;
        if (live.length < w) notes.push("LIVE: all " + live.length + " live day(s) " + DASH +
                                        " fewer than the " + w + "-day window");
        notes = notes.concat(tpMath.rowNotes("LIVE", s2, n));
      }
    }
    setStratRow("strat-sim", simS);
    setStratRow("strat-live", liveS);
    el("tp-strat-read").textContent = tpMath.readText(simS, liveS);
    el("tp-strat-note").textContent = notes.join(" \\u00b7 ");
  }
  function update() {
    var n = parseInt(basket.value, 10);
    var w = parseInt(win.value, 10);
    if (strat.value === "reach") {
      cardReach.hidden = false;
      cardStrat.hidden = true;
      updateReach(n, w);
      return;
    }
    cardReach.hidden = true;
    cardStrat.hidden = false;
    updateStrat(n, w, strat.value);
  }
  basket.addEventListener("change", update);
  win.addEventListener("change", update);
  strat.addEventListener("change", update);
})();
</script>
"""


def build_html(
    con: duckdb.DuckDBPyConnection,
    result: PredictResult,
    top: int = 20,
) -> str:
    """Render the dashboard for an already-computed prediction result."""
    record = track.score_predictions(con, result.strategy, top_n=top)
    picks = track.daily_pick_performance(con, result.strategy)
    benchmark = backtest.latest_standard_experiment(con, result.strategy)
    pick_by_day = (
        {pd.Timestamp(r.target_date).date(): r for r in picks.daily.itertuples()}
        if picks.days
        else {}
    )
    uni = store.latest_universe(con).set_index("symbol")
    n_prices = store.price_row_count(con)
    max_prob = float(result.scored["prob"].max()) or 1.0

    head = (
        '<meta charset="utf-8">'
        "<title>twopercent — reach +2% intraday</title>"
        + _CSS
        + '<div class="tp-root"><div class="tp-wrap">'
        + '<h1><span class="tick"></span>twopercent — tickers likely to REACH +2% intraday</h1>'
        + '<p class="sub">Ranked P(reach +2% intraday, open-to-high) for the trading day after '
        + f'<b class="mono">{result.signal_date}</b> · strategy '
        + f"<b>{html.escape(result.strategy)}</b> · trained on "
        + f'<span class="mono">{result.trained_rows:,}</span> rows · '
        + f'<span class="mono">{n_prices:,}</span> price rows in store</p>'
        + _tiles(record, benchmark, len(result.scored), top)
    )

    rows = []
    for row in result.scored.head(top).itertuples():
        name = html.escape(str(uni["name"].get(row.symbol, "?")))[:48]
        move_cls = "pos" if row.oc_return_today >= 0 else "neg"
        bar = int(round(row.prob / max_prob * 100))
        rows.append(
            f'<tr><td>{row.rank}</td><td><span class="sym">{html.escape(row.symbol)}</span></td>'
            f'<td>{row.prob:.3f}<span class="probbar"><i style="width:{bar}%"></i></span></td>'
            f'<td class="{move_cls}">{row.oc_return_today:+.1%}</td>'
            f"<td>{row.volume_ratio:.2f}×</td><td>{int(row.cnt_2pct_20d)}</td>"
            f'<td class="name col-co">{name}</td></tr>'
        )
    candidates = (
        f"<h2>Top {top} candidates</h2><div class='card'><table>"
        f"<tr><th>#{_info('c_rank', 'start')}</th><th>Symbol{_info('c_sym')}</th>"
        f"<th>Probability{_info('c_prob')}</th><th>Prev day{_info('c_prev')}</th>"
        f"<th>Vol ratio{_info('c_vol')}</th><th>2% days /20d{_info('c_cnt')}</th>"
        f'<th class="col-co">Company{_info("c_co", "end")}</th></tr>'
        + "".join(rows)
        + "</table></div>"
    )

    if record.scored.empty:
        body = (
            '<p class="empty">No scored days yet — predictions are logged; '
            "outcomes appear here once the next trading day's data is ingested.</p>"
        )
    else:

        def _pick_cell(day) -> str:
            # Prediction quality, not P&L: show whether the #1 pick REACHED +2%
            # intraday (✓/✗), never a dollar return.
            p = pick_by_day.get(day)
            if p is None:
                return "<td>—</td>"
            reached = bool(p.top1_hit)
            cls = "pos" if reached else "neg"
            mark = "✓" if reached else "✗"
            marker = " †" if p.late else ""
            return (
                f'<td><span class="sym">{html.escape(p.top1_symbol)}</span> '
                f'<span class="{cls}">{mark}</span>{marker}</td>'
            )

        trs = "".join(
            f"<tr><td>{pd.Timestamp(r.target_date).date()}</td>"
            + _pick_cell(pd.Timestamp(r.target_date).date())
            + f"<td>{int(r.hits)}/{int(r.n_scored)}</td>"
            f'<td class="{"pos" if r.precision >= r.base_rate else "neg"}">{r.precision:.0%}</td>'
            f"<td>{r.base_rate:.0%}</td>"
            f'<td class="{"pos" if r.lift >= 1 else "neg"}">{r.lift:.2f}×</td></tr>'
            for r in record.scored.itertuples()
        )
        late_note = ""
        if record.late_days:
            late_note = (
                f'<p class="sub"><b>{record.late_days} of {len(record.scored)} days '
                "were backfilled after the fact</b> (marked †) — not live forecasting "
                "skill, and excluded from the live reach tiles above.</p>"
            )
        body = late_note + (
            f"<div class='card'>{_chart_svg(record.scored)}</div>"
            + "<div class='card'><table><tr>"
            + f"<th>Day{_info('r_day', 'start')}</th><th>Top pick{_info('r_pick')}</th>"
            + f"<th>Reachers{_info('r_hits')}</th><th>Reach rate{_info('r_hit')}</th>"
            + f"<th>Base rate{_info('r_base')}</th>"
            + f"<th>Lift{_info('r_lift', 'end')}</th></tr>{trs}</table></div>"
        )
    pending = ""
    if record.pending:
        dates = ", ".join(f'<span class="mono">{d}</span>' for d in record.pending)
        pending = f'<p class="sub" style="margin-top:8px">Awaiting outcomes for: {dates}</p>'

    # Clean-reset disclosure (M1): the live touch record counts only touch-era
    # (event='open_to_high') predictions; pre-cutover days are archived, never
    # silently re-scored under the new definition. Rendered whenever a real cutover
    # left archived close-era days behind — INCLUDING the exact cutover moment when
    # archived days exist but no touch prediction has been made yet (first_touch is
    # None); suppressing it then would silently hide the archive while the record
    # shows "No scored days yet". A fresh touch-only store (no archive) shows nothing.
    first_touch, archived_days = store.touch_record_bounds(con, result.strategy)
    reset_note = ""
    if archived_days:
        if first_touch is not None:
            reset_note = (
                f'<p class="sub" style="margin-top:8px">Live reach-record began '
                f'<b class="mono">{first_touch}</b> (reached +2% intraday, open-to-high); '
                f"{archived_days} earlier day(s) targeted open-to-close and are archived.</p>"
            )
        else:
            reset_note = (
                f'<p class="sub" style="margin-top:8px">{archived_days} earlier day(s) '
                "targeted open-to-close and are archived — the reach-record starts with "
                "the first touch prediction.</p>"
            )

    sim = store.latest_experiment_daily(con, result.strategy)
    outcomes = track.daily_rank_outcomes(con, result.strategy)
    base_dates = [
        pd.Timestamp(d).date()
        for frame in ((sim[1] if sim is not None else None), outcomes)
        if frame is not None and not frame.empty
        for d in frame["target_date"]
    ]
    bases = track.daily_base_rates(con, base_dates)
    # One payload shape for both rows ([rank, oh, ol, oc, hit] per pick): the
    # sim frame's ret IS the pick's open-to-close return; the live frame comes
    # straight off daily_returns with the view's column names.
    sim_days = _payload_days(sim[1].rename(columns={"ret": "oc"}), bases) if sim is not None else []
    live_days = (
        _payload_days(
            outcomes.rename(columns={"high_return": "oh", "low_return": "ol", "oc_return": "oc"}),
            bases,
            late=True,
        )
        if len(outcomes)
        else []
    )
    explorer = _record_explorer(sim[0] if sim is not None else None, sim_days, live_days)
    data_tag = _embed_json({"sim": sim_days, "live": live_days})

    return (
        head
        + candidates
        + _calibration_panel(benchmark)
        + "<h2>Track record</h2>"
        + reset_note
        + body
        + pending
        + explorer
        + '<p class="note">Generated by twopercent. Model output, not investment advice.</p>'
        + "</div></div>"
        + data_tag
        + _JS_MATH
        + _JS
    )


def render(
    con: duckdb.DuckDBPyConnection,
    strategy_name: str,
    out_path: str,
    top: int = 20,
    result: PredictResult | None = None,
) -> str:
    """Render the dashboard; pass a precomputed PredictResult to avoid retraining."""
    if result is None:
        result = predict_for(con, strategy_name, save=True)
    content = build_html(con, result, top=top)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return out_path
