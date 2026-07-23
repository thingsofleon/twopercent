import pandas as pd
import pytest

from tests.conftest import seed_history
from twopercent import dashboard, store, track
from twopercent.predict import predict_for

RUNNER_OC = [0.03 + 0.001 * (i % 5) for i in range(60)]
FLAT_OC = [0.002 + 0.001 * (i % 3) for i in range(60)]


@pytest.fixture
def modeled(con):
    data = {f"RUN{i}": RUNNER_OC for i in range(4)} | {f"FLT{i}": FLAT_OC for i in range(4)}
    seed_history(con, data, vary_volume=True)
    store.upsert_universe(
        con,
        pd.DataFrame(
            {
                "symbol": list(data),
                "name": [f"{s} <Corp> & Sons" for s in data],  # needs HTML escaping
                "market_cap": [1e9 * (i + 1) for i in range(len(data))],
                # Runners and flats share a sector so sector_excess varies per row;
                # an all-NaN sector column would crash the binner (see CLAUDE.md).
                "sector": ["Tech"] * len(data),
            }
        ),
        as_of=pd.Timestamp("2026-03-01").date(),
    )
    return con


def test_predict_for_backfill_trains_walk_forward(modeled):
    dates = sorted(pd.bdate_range("2026-01-05", periods=60).date)
    past = dates[-5]
    result = predict_for(modeled, "baseline_gbm_v1", signal_date=past, save=True)
    assert result.signal_date == past
    # Training could only use outcomes known by the past signal date's close.
    frame_rows = modeled.execute(
        "SELECT count(*) FROM predictions WHERE signal_date = ?", [past]
    ).fetchone()[0]
    assert frame_rows == len(result.scored)
    assert result.scored["rank"].tolist() == list(range(1, len(result.scored) + 1))


def test_dashboard_renders_scored_track_record(modeled, tmp_path):
    dates = sorted(pd.bdate_range("2026-01-05", periods=60).date)
    predict_for(modeled, "baseline_gbm_v1", signal_date=dates[-3], save=True)
    predict_for(modeled, "baseline_gbm_v1", signal_date=dates[-2], save=True)

    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    assert "Top 5 candidates" in content
    assert "Track record" in content
    assert "RUN0" in content
    assert "&lt;Corp&gt; &amp; Sons" in content  # names HTML-escaped
    assert "<Corp>" not in content
    assert 'src="http' not in content and 'href="http' not in content  # self-contained
    assert '<meta charset="utf-8">' in content  # no mojibake when served header-less
    assert "<svg" in content  # chart rendered for the scored days
    assert "Awaiting outcomes" in content  # the render-day prediction is pending


def test_dashboard_info_tooltips_present(modeled, tmp_path):
    dates = sorted(pd.bdate_range("2026-01-05", periods=60).date)
    predict_for(modeled, "baseline_gbm_v1", signal_date=dates[-3], save=True)
    predict_for(modeled, "baseline_gbm_v1", signal_date=dates[-2], save=True)

    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    # An "i" icon with its tooltip appears for a tile, a candidate column, the
    # chart, a track-record column, and an explorer control — one per surface.
    assert 'class="tp-i"' in content
    for key in ("t_lift", "c_prob", "chart", "r_lift", "e_basket"):
        assert dashboard._INFO_TEXT[key] in content, key
    # Edge columns anchor their tooltip inward so it can't clip the viewport.
    assert "tp-tip--start" in content and "tp-tip--end" in content
    # Company column carries the class the mobile media query hides — on both
    # the header and the body cell, or the columns misalign when it's dropped.
    assert content.count("col-co") >= 2
    assert "@media (max-width: 520px)" in content


def test_info_helper_escapes_and_labels():
    # Authored text only, but the icon must stay breakout-proof and carry an
    # accessible label for keyboard/screen-reader users (tooltip is hover-only).
    dashboard._INFO_TEXT["_probe"] = 'a <b> "quote" & amp'
    try:
        html_out = dashboard._info("_probe")
    finally:
        del dashboard._INFO_TEXT["_probe"]
    assert 'aria-label="a &lt;b&gt; &quot;quote&quot; &amp; amp"' in html_out
    assert "<b>" not in html_out  # raw tag never reaches the markup
    assert 'tabindex="0"' in html_out


def test_dashboard_empty_track_record_state(modeled, tmp_path):
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()
    assert "No scored days yet" in content
    assert "<svg" not in content
    # No benchmark recorded daily rows yet — the SIM row must say so loudly.
    assert "No walk-forward simulation recorded yet" in content
    assert "twopercent benchmark" in content
    assert "LIVE: no live days yet" in content


def _record_sim(con, n_days, ranks_per_day=6, strategy="baseline_gbm_v1"):
    """Per-rank sim rows: mostly-up rank-1 returns, adversarial values."""
    seq = store.record_experiment(
        con,
        strategy=strategy,
        params={"months": 12, "top_n": 5},
        train_start=pd.Timestamp("2025-06-02").date(),
        test_start=pd.Timestamp("2026-01-05").date(),
        test_end=pd.Timestamp("2026-12-31").date(),
        metrics={"lift": 2.1},
    )
    dates = sorted(pd.bdate_range("2026-01-05", periods=n_days).date)
    rows = []
    for i, d in enumerate(dates):
        for rank in range(1, ranks_per_day + 1):
            ret = -0.0117 if (i + rank) % 5 == 0 else 0.0203
            rows.append({"target_date": d, "rank": rank, "ret": ret, "hit": int(ret > 0.02)})
    store.record_experiment_daily(con, seq, pd.DataFrame(rows))
    return seq


def test_dashboard_explorer_defaults_match_python_math(modeled, tmp_path):
    # Server-side default view (top-5 basket, 6-month window) must carry the
    # exact numbers track.sim_windows computes — the JS payload is generated
    # from the same frames, so this pins Python and JS to one source of truth.
    _record_sim(modeled, n_days=130)
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    _, daily = store.latest_experiment_daily(modeled, "baseline_gbm_v1")
    summary = track.sim_windows(daily, n=5)
    w126 = next(w for w in summary.windows if w["days"] == 126)
    growth_txt = dashboard._fixed_half_up(w126["growth"], 3)
    assert f'id="tp-sim-growth" class="pos">${growth_txt}</td>' in content
    assert f'<td id="tp-sim-hit">{dashboard._pct_half_up(w126["hit_rate"])}</td>' in content
    assert '<td id="tp-sim-days">126</td>' in content  # selection day count visible

    assert 'class="badge-sim">SIM</span>' in content
    assert 'class="badge-live">LIVE</span>' in content
    assert "walk-forward, monthly retrain" in content
    # Both selects, with per-window trading-day counts, defaults marked.
    assert '<option value="5" selected>Top 5</option>' in content
    assert '<option value="126" selected>6 months (126 trading days)</option>' in content
    assert "1 year (252 trading days)" in content
    # Span line: test span, months, run staleness, sim day count.
    assert "SIM test span" in content and "months=12" in content
    assert "simulated <b" in content
    assert '<span class="mono">130</span> sim days available' in content
    # Strengthened survivorship + multiple-comparisons caveats, verbatim tails.
    assert "delisted names can never contribute their final catastrophic day" in content
    assert "itself a form of selection" in content
    assert "dominated by a handful of days" in content
    assert "The live record above is the clean test." in content
    assert '<meta charset="utf-8">' in content


def test_dashboard_explorer_payload_json(modeled, tmp_path):
    import json
    import re

    _record_sim(modeled, n_days=6)
    dates = sorted(pd.bdate_range("2026-01-05", periods=60).date)
    predict_for(modeled, "baseline_gbm_v1", signal_date=dates[-2], save=True)
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    match = re.search(r'<script type="application/json" id="tp-data">(.*?)</script>', content)
    assert match, "payload script tag missing"
    payload = json.loads(match.group(1))
    assert payload["cost"] == track.COST_ROUND_TRIP
    assert len(payload["sim"]) == 6
    day0 = payload["sim"][0]
    assert day0["d"] == "2026-01-05"
    # Hand-check day 0: (i + rank) % 5 == 0 at rank 5 → one down day.
    assert day0["picks"] == [
        [1, 0.0203, 1],
        [2, 0.0203, 1],
        [3, 0.0203, 1],
        [4, 0.0203, 1],
        [5, -0.0117, 0],
        [6, 0.0203, 1],
    ]
    # Base rate on 2026-01-05: 4 runners of 8 names did ≥2%.
    assert abs(day0["base"] - 0.5) < 1e-9
    # Live day present with late flag and rank-ordered picks.
    assert len(payload["live"]) == 1
    live0 = payload["live"][0]
    assert live0["late"] is True  # backfilled save, created after the target open
    assert [p[0] for p in live0["picks"]] == sorted(p[0] for p in live0["picks"])


def test_dashboard_explorer_too_few_sim_days_says_so(modeled, tmp_path):
    _record_sim(modeled, n_days=4)
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    assert 'class="badge-sim">SIM</span>' in content
    assert "SIM: needs 126 trading days — 4 available" in content
    assert '<td id="tp-sim-growth">—</td>' in content  # no number pretended
    assert "The live record above is the clean test." in content


def test_summarize_days_first_available_substitution_and_short_days():
    days = [
        {"d": "a", "base": 0.25, "picks": [[2, 0.0203, 1], [3, -0.0117, 0]]},  # rank 1 missing
        {"d": "b", "base": 0.15, "picks": [[1, 0.04, 1]]},
    ]
    s1 = dashboard._summarize_days(days, 1)
    # First-available rule: day a's top pick is rank 2 (the trader takes it).
    expected = (1 + 0.0203 - track.COST_ROUND_TRIP) * (1 + 0.04 - track.COST_ROUND_TRIP)
    assert abs(s1["growth"] - expected) < 1e-12
    assert s1["hit"] == 1.0
    assert s1["short"] == 0
    assert s1["subst"] == 1  # ...but the substitution is counted, never silent
    assert abs(s1["base"] - 0.2) < 1e-12

    s2 = dashboard._summarize_days(days, 2)
    assert s2["short"] == 1  # day b has only one pick for a 2-basket
    assert s2["subst"] == 1  # day a's 2-basket is ranks (2, 3), not (1, 2)
    assert abs(s2["hit"] - (0.5 + 1.0) / 2) < 1e-12


def test_explorer_substitution_and_base_coverage_notes():
    days = [
        {"d": "a", "late": False, "base": 0.25, "picks": [[2, 0.0203, 1]]},  # rank 1 missing
        {"d": "b", "late": False, "base": None, "picks": [[1, 0.011, 0]]},  # base unknown
    ]
    _, live_s, notes = dashboard._explorer_state([], days, 1, 5)
    assert live_s is not None
    assert "LIVE: 1 day(s) substituted lower-ranked names for missing picks" in notes
    assert "LIVE: base rate from 1 of 2 day(s)" in notes
    # A contiguous full-coverage frame emits neither disclosure.
    clean_days = [
        {"d": "a", "late": False, "base": 0.25, "picks": [[1, 0.0203, 1]]},
        {"d": "b", "late": False, "base": 0.25, "picks": [[1, 0.011, 0]]},
    ]
    _, _, clean_notes = dashboard._explorer_state([], clean_days, 1, 5)
    assert not any("substituted" in n or "base rate from" in n for n in clean_notes)


def test_dashboard_renders_live_substitution_note(modeled, tmp_path):
    import datetime as dt

    dates = sorted(pd.bdate_range("2026-01-05", periods=60).date)
    predict_for(modeled, "baseline_gbm_v1", signal_date=dates[-2], save=True)
    # Stamp the save live (created before the target day's open)...
    modeled.execute(
        "UPDATE predictions SET created_ts = ?",
        [dt.datetime.combine(dates[-1], dt.time(6, 0))],
    )
    # ...then remove the rank-2 pick's target-day bar: the top-5 basket
    # becomes ranks 1,3,4,5,6 and the default view must disclose it.
    rank2 = modeled.execute("SELECT symbol FROM predictions WHERE rank = 2").fetchone()[0]
    modeled.execute("DELETE FROM prices WHERE symbol = ? AND date = ?", [rank2, dates[-1]])

    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()
    assert "LIVE: 1 day(s) substituted lower-ranked names for missing picks" in content


def test_half_up_formatting_matches_js():
    # 0.125 is an exact half: Python's :.0% gives "12%" (half-even) while JS
    # Math.round gives 13 — the server must render the JS answer or the cell
    # visibly flickers on the first select change.
    assert f"{0.125:.0%}" == "12%"  # the trap this guards against
    assert dashboard._pct_half_up(0.125) == "13%"
    assert dashboard._fixed_half_up(0.125, 2) == "0.13"
    assert f"{0.125:.2f}" == "0.12"  # ditto for lift-style fixed decimals
    # Non-representable "halves" follow the stored double, exactly like toFixed
    # (both verified against node): naive multiply-then-floor gets BOTH wrong.
    assert dashboard._fixed_half_up(1.2345, 3) == "1.234"  # double sits below the half
    assert dashboard._fixed_half_up(1.0005, 3) == "1.000"  # ditto, ×1000 FP rounds up
    cells = dashboard._explorer_cells(
        "sim",
        {"growth": 1.5, "hit": 0.125, "base": 0.125, "days": 8, "short": 0, "corrupt": 0},
    )
    assert '<td id="tp-sim-hit">13%</td>' in cells
    assert '<td id="tp-sim-base">13%</td>' in cells
    assert '<td id="tp-sim-lift">1.00×</td>' in cells


def test_embed_json_is_breakout_proof():
    import json

    prefix = '<script type="application/json" id="tp-data">'
    tag = dashboard._embed_json({"cost": 0.003, "note": "</script><b>x</b>"})
    assert tag.startswith(prefix) and tag.endswith("</script>")
    inner = tag[len(prefix) : -len("</script>")]
    assert "<" not in inner and ">" not in inner  # nothing can end the tag early
    assert tag.count("</script>") == 1  # only the real closer
    assert json.loads(inner) == {"cost": 0.003, "note": "</script><b>x</b>"}


def test_summarize_days_counts_corrupt_never_averages_around():
    days = [
        {"d": "a", "late": False, "base": None, "picks": [[1, float("nan"), 1]]},
        {"d": "b", "late": False, "base": None, "picks": [[1, 0.01, 0]]},
    ]
    s = dashboard._summarize_days(days, 1)
    assert s["corrupt"] == 1
    # The corrupt window renders as an error state, never as a shorter product.
    _, live_s, notes = dashboard._explorer_state([], days, 1, 5)
    assert live_s is None
    assert any("corrupt day(s)" in n for n in notes)


def test_explorer_state_live_short_window_disclosures():
    live = [{"d": str(i), "late": False, "base": None, "picks": [[1, 0.011, 0]]} for i in range(3)]
    sim_s, live_s, notes = dashboard._explorer_state([], live, 5, 126)
    assert sim_s is None
    assert any("No walk-forward simulation recorded yet" in n for n in notes)
    assert live_s is not None and live_s["days"] == 3
    assert any("all 3 live day(s)" in n for n in notes)  # window shortfall disclosed
    assert any("3 day(s) had fewer than 5 picks" in n for n in notes)
    # Late days are excluded from LIVE, loudly absent when that empties it.
    _, live_s2, notes2 = dashboard._explorer_state([], [dict(d, late=True) for d in live], 5, 126)
    assert live_s2 is None
    assert any("no live days yet" in n for n in notes2)
