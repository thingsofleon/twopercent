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
    assert f'id="tp-sim-growth" class="pos">${w126["growth"]:.3f}</td>' in content
    assert f'<td id="tp-sim-hit">{w126["hit_rate"]:.0%}</td>' in content
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
    # Strengthened survivorship caveat, verbatim tail.
    assert "delisted names can never contribute their final catastrophic day" in content
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
    assert abs(s1["base"] - 0.2) < 1e-12

    s2 = dashboard._summarize_days(days, 2)
    assert s2["short"] == 1  # day b has only one pick for a 2-basket
    assert abs(s2["hit"] - (0.5 + 1.0) / 2) < 1e-12


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
