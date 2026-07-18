"""Static HTML dashboard: next-day candidates + honest track record.

Self-contained output — inline CSS, inline SVG, no external requests — so the
file opens anywhere and satisfies a strict CSP (claude.ai Artifact
compatible). Terminal-style dark-first design with a full light theme;
viewer's data-theme toggle overrides the OS preference in both directions.
Semantic color only: green = hit/up, red = down/miss, amber = base rate.
"""

from __future__ import annotations

import html

import duckdb
import pandas as pd

from twopercent import store, track
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
  border-radius: 8px; padding: 14px 16px; margin-top: 10px; overflow-x: auto;
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
.tp-root .note { color: var(--ink-muted); font-size: 0.74rem; margin-top: 26px; }
</style>
"""


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
        '<span><span class="chip" style="background:var(--amber)"></span>market base rate</span>'
        "</div>"
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Daily top-N hit rate vs market base rate">{"".join(parts)}</svg>'
    )


def _tiles(
    record: track.TrackRecord, picks: track.PickPerformance, n_candidates: int, top: int
) -> str:
    if record.scored.empty:
        return ""
    total_n = record.scored["n_scored"].sum()
    overall_p = record.scored["hits"].sum() / total_n
    # Weight base rates like precision is pooled, or the headline lift is
    # inconsistent when day sizes / base rates differ.
    overall_b = (record.scored["base_rate"] * record.scored["n_scored"]).sum() / total_n
    lift = overall_p / overall_b if overall_b > 0 else float("nan")
    tiles = (
        '<div class="tiles">'
        f'<div class="tile"><span class="label">Cumulative lift</span>'
        f'<b class="{"up" if lift >= 1 else ""}">{lift:.2f}×</b>'
        f'<span class="cmp">vs picking at random</span></div>'
        f'<div class="tile"><span class="label">Hit rate (top {top})</span>'
        f"<b>{overall_p:.0%}</b>"
        f'<span class="cmp">market base {overall_b:.0%}</span></div>'
    )
    if picks.days:
        # Live days only: backfilled days must never inflate the money tiles.
        p1 = picks.precision_at_1()
        g1 = picks.growth("top1_return")
        g5 = picks.growth("topn_return")
        late_note = f" · excludes {picks.late_days} backfilled" if picks.late_days else ""
        if g1 is not None:
            live = picks.live
            tiles += (
                f'<div class="tile"><span class="label">Top pick hit rate (live)</span>'
                f"<b>{p1:.0%}</b>"
                f'<span class="cmp">{int(live["top1_hit"].sum())}/{len(live)} days'
                f" did +2%{late_note}</span></div>"
                f'<div class="tile"><span class="label">$1 → top pick daily (live)</span>'
                f'<b class="{"up" if g1 >= 1 else ""}">${g1:.3f}</b>'
                f'<span class="cmp">top-5: ${g5:.3f} · net of '
                f"{track.COST_ROUND_TRIP:.1%}/day assumed costs{late_note}</span></div>"
            )
        else:
            tiles += (
                f'<div class="tile"><span class="label">$1 → top pick daily (live)</span>'
                f"<b>—</b>"
                f'<span class="cmp">all {picks.late_days} scored days were backfilled — '
                f"live record starts with the next scheduled run</span></div>"
            )
    tiles += (
        f'<div class="tile"><span class="label">Candidates today</span>'
        f"<b>{n_candidates}</b>"
        f'<span class="cmp">ranked by probability</span></div>'
        "</div>"
    )
    return tiles


def build_html(
    con: duckdb.DuckDBPyConnection,
    result: PredictResult,
    top: int = 20,
) -> str:
    """Render the dashboard for an already-computed prediction result."""
    record = track.score_predictions(con, result.strategy, top_n=top)
    picks = track.daily_pick_performance(con, result.strategy)
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
        "<title>twopercent dashboard</title>"
        + _CSS
        + '<div class="tp-root"><div class="tp-wrap">'
        + '<h1><span class="tick"></span>twopercent — +2% open-to-close candidates</h1>'
        + '<p class="sub">Trading day after '
        + f'<b class="mono">{result.signal_date}</b> · strategy '
        + f"<b>{html.escape(result.strategy)}</b> · trained on "
        + f'<span class="mono">{result.trained_rows:,}</span> rows · '
        + f'<span class="mono">{n_prices:,}</span> price rows in store</p>'
        + _tiles(record, picks, len(result.scored), top)
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
            f'<td class="name">{name}</td></tr>'
        )
    candidates = (
        f"<h2>Top {top} candidates</h2><div class='card'><table>"
        "<tr><th>#</th><th>Symbol</th><th>Probability</th><th>Prev day</th>"
        "<th>Vol ratio</th><th>2% days /20d</th><th>Company</th></tr>"
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
            p = pick_by_day.get(day)
            if p is None:
                return "<td>—</td>"
            cls = "pos" if p.top1_return >= 0 else "neg"
            marker = " †" if p.late else ""
            return (
                f'<td><span class="sym">{html.escape(p.top1_symbol)}</span> '
                f'<span class="{cls}">{p.top1_return:+.1%}</span>{marker}</td>'
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
                "skill, and excluded from the money tiles above.</p>"
            )
        body = late_note + (
            f"<div class='card'>{_chart_svg(record.scored)}</div>"
            + "<div class='card'><table><tr><th>Day</th><th>Top pick</th><th>Hits</th>"
            + f"<th>Hit rate</th><th>Base rate</th><th>Lift</th></tr>{trs}</table></div>"
        )
    pending = ""
    if record.pending:
        dates = ", ".join(f'<span class="mono">{d}</span>' for d in record.pending)
        pending = f'<p class="sub" style="margin-top:8px">Awaiting outcomes for: {dates}</p>'

    return (
        head
        + candidates
        + "<h2>Track record</h2>"
        + body
        + pending
        + '<p class="note">Generated by twopercent. Model output, not investment advice.</p>'
        + "</div></div>"
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
