"""Static HTML dashboard: next-day candidates + honest track record.

Self-contained output — inline CSS, inline SVG, no external requests — so the
file opens anywhere and satisfies a strict CSP (claude.ai Artifact
compatible). Light/dark theming follows prefers-color-scheme with a
data-theme override. Colors are the validated reference dataviz palette
(slots 1–2 + ink/surface tokens).
"""

from __future__ import annotations

import html

import duckdb
import pandas as pd

from twopercent import store, track
from twopercent.predict import PredictResult, predict_for

_CSS = """
<style>
.tp-root {
  color-scheme: light;
  --surface-1: #fcfcfb; --page: #f9f9f7;
  --ink-1: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
  --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
  --series-1: #2a78d6; --series-2: #008300;
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink-1); background: var(--page);
  max-width: 880px; margin: 0 auto; padding: 24px 16px 48px;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .tp-root {
    color-scheme: dark;
    --surface-1: #1a1a19; --page: #0d0d0d;
    --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
    --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
    --series-1: #3987e5; --series-2: #008300;
  }
}
:root[data-theme="dark"] .tp-root {
  color-scheme: dark;
  --surface-1: #1a1a19; --page: #0d0d0d;
  --ink-1: #ffffff; --ink-2: #c3c2b7; --ink-muted: #898781;
  --grid: #2c2c2a; --baseline: #383835; --border: rgba(255,255,255,0.10);
  --series-1: #3987e5; --series-2: #008300;
}
.tp-root h1 { font-size: 1.4rem; margin: 0 0 4px; }
.tp-root h2 { font-size: 1.05rem; margin: 28px 0 8px; }
.tp-root .sub { color: var(--ink-2); font-size: 0.9rem; margin: 0 0 4px; }
.tp-root .card {
  background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 10px; padding: 14px 16px; margin-top: 10px; overflow-x: auto;
}
.tp-root table { border-collapse: collapse; width: 100%; font-size: 0.88rem; }
.tp-root th { text-align: left; color: var(--ink-muted); font-weight: 600;
  border-bottom: 1px solid var(--baseline); padding: 5px 10px 5px 0; }
.tp-root td { border-bottom: 1px solid var(--grid); padding: 5px 10px 5px 0;
  font-variant-numeric: tabular-nums; }
.tp-root td.name { color: var(--ink-2); }
.tp-root .legend { display: flex; gap: 16px; font-size: 0.82rem;
  color: var(--ink-2); margin: 2px 0 6px; }
.tp-root .chip { display: inline-block; width: 10px; height: 10px;
  border-radius: 3px; margin-right: 5px; vertical-align: -1px; }
.tp-root .empty { color: var(--ink-muted); font-style: italic; }
.tp-root .note { color: var(--ink-muted); font-size: 0.78rem; margin-top: 24px; }
</style>
"""


def _chart_svg(scored: pd.DataFrame) -> str:
    """Per-day precision bars with a base-rate dash marker (shared 0..max axis)."""
    width, height, pad_l, pad_b, pad_t = 760, 190, 44, 24, 8
    plot_w, plot_h = width - pad_l - 8, height - pad_t - pad_b
    top = max(0.05, float(scored["precision"].max()), float(scored["base_rate"].max())) * 1.15

    n = len(scored)
    slot = plot_w / n
    bar_w = min(38.0, slot * 0.55)

    def y(v: float) -> float:
        return pad_t + plot_h * (1 - v / top)

    parts = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        gy = y(top * frac / 1.15)
        parts.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - 8}" y2="{gy:.1f}" '
            f'stroke="var(--grid)" stroke-width="1"/>'
            f'<text x="{pad_l - 6}" y="{gy + 3:.1f}" text-anchor="end" font-size="10" '
            f'fill="var(--ink-muted)">{top * frac / 1.15:.0%}</text>'
        )
    for i, row in enumerate(scored.itertuples()):
        cx = pad_l + slot * (i + 0.5)
        bx = cx - bar_w / 2
        by = y(row.precision)
        date = pd.Timestamp(row.target_date).strftime("%b %d")
        parts.append(
            f"<g><title>{date}: top-{int(row.n_scored)} precision "
            f"{row.precision:.0%}, base rate {row.base_rate:.0%}</title>"
            f'<path d="M{bx:.1f} {y(0):.1f} V{by + 4:.1f} Q{bx:.1f} {by:.1f} '
            f"{bx + 4:.1f} {by:.1f} H{bx + bar_w - 4:.1f} Q{bx + bar_w:.1f} {by:.1f} "
            f'{bx + bar_w:.1f} {by + 4:.1f} V{y(0):.1f} Z" fill="var(--series-1)"/>'
            f'<line x1="{bx - 3:.1f}" y1="{y(row.base_rate):.1f}" x2="{bx + bar_w + 3:.1f}" '
            f'y2="{y(row.base_rate):.1f}" stroke="var(--series-2)" stroke-width="2.5"/>'
            f'<text x="{cx:.1f}" y="{by - 5:.1f}" text-anchor="middle" font-size="10" '
            f'fill="var(--ink-2)">{row.precision:.0%}</text>'
            f'<text x="{cx:.1f}" y="{height - 8}" text-anchor="middle" font-size="10" '
            f'fill="var(--ink-muted)">{date}</text></g>'
        )
    parts.append(
        f'<line x1="{pad_l}" y1="{y(0):.1f}" x2="{width - 8}" y2="{y(0):.1f}" '
        f'stroke="var(--baseline)" stroke-width="1"/>'
    )
    return (
        '<div class="legend">'
        '<span><span class="chip" style="background:var(--series-1)"></span>top-20 hit rate</span>'
        '<span><span class="chip" style="background:var(--series-2)"></span>market base rate</span>'
        "</div>"
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
        f'aria-label="Daily top-20 hit rate vs market base rate">{"".join(parts)}</svg>'
    )


def build_html(
    con: duckdb.DuckDBPyConnection,
    result: PredictResult,
    top: int = 20,
) -> str:
    """Render the dashboard for an already-computed prediction result."""
    record = track.score_predictions(con, result.strategy, top_n=top)
    uni = store.latest_universe(con).set_index("symbol")
    n_prices = store.price_row_count(con)

    head = (
        '<meta charset="utf-8">'
        "<title>twopercent dashboard</title>"
        + _CSS
        + '<div class="tp-root">'
        + "<h1>twopercent — +2% open-to-close candidates</h1>"
        + '<p class="sub">Candidates for the trading day after '
        + f"<b>{result.signal_date}</b> · strategy <b>{html.escape(result.strategy)}</b> · "
        + f"trained on {result.trained_rows:,} rows · {n_prices:,} price rows in store</p>"
    )

    rows = []
    for row in result.scored.head(top).itertuples():
        name = html.escape(str(uni["name"].get(row.symbol, "?")))[:48]
        rows.append(
            f"<tr><td>{row.rank}</td><td><b>{html.escape(row.symbol)}</b></td>"
            f"<td>{row.prob:.3f}</td><td>{row.oc_return_today:+.1%}</td>"
            f"<td>{row.volume_ratio:.2f}×</td><td>{int(row.cnt_2pct_20d)}</td>"
            f'<td class="name">{name}</td></tr>'
        )
    candidates = (
        f"<h2>Top {top} candidates</h2><div class='card'><table>"
        "<tr><th>#</th><th>Symbol</th><th>Prob</th><th>Prev day</th>"
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
        overall_p = record.scored["hits"].sum() / record.scored["n_scored"].sum()
        overall_b = record.scored["base_rate"].mean()
        trs = "".join(
            f"<tr><td>{pd.Timestamp(r.target_date).date()}</td>"
            f"<td>{int(r.hits)}/{int(r.n_scored)}</td><td>{r.precision:.0%}</td>"
            f"<td>{r.base_rate:.0%}</td><td>{r.lift:.2f}×</td></tr>"
            for r in record.scored.itertuples()
        )
        body = (
            f'<p class="sub">Cumulative: <b>{overall_p:.0%}</b> of top-{top} picks hit, '
            f"vs {overall_b:.0%} base rate — "
            f"<b>{overall_p / overall_b:.2f}× lift</b> over {len(record.scored)} scored days.</p>"
            + f"<div class='card'>{_chart_svg(record.scored)}</div>"
            + "<div class='card'><table><tr><th>Day</th><th>Hits</th><th>Hit rate</th>"
            + f"<th>Base rate</th><th>Lift</th></tr>{trs}</table></div>"
        )
    pending = ""
    if record.pending:
        dates = ", ".join(str(d) for d in record.pending)
        pending = f'<p class="sub">Awaiting outcomes for: {dates}</p>'

    return (
        head
        + candidates
        + "<h2>Track record</h2>"
        + body
        + pending
        + '<p class="note">Generated by twopercent. Model output, not investment advice.</p>'
        + "</div>"
    )


def render(
    con: duckdb.DuckDBPyConnection,
    strategy_name: str,
    out_path: str,
    top: int = 20,
) -> str:
    result = predict_for(con, strategy_name, save=True)
    content = build_html(con, result, top=top)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return out_path
