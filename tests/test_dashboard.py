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
    for key in ("t_cands", "c_prob", "chart", "r_lift", "e_basket"):
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
    assert "No touch-era benchmark yet" in content
    assert "twopercent benchmark" in content
    assert "LIVE: no live days yet" in content
    # The empty live record must not VANISH (that reads the backtest tiles as
    # system performance) — an explicit placeholder tile provides the contrast.
    assert "no live days yet — the live record starts" in content


def _scored(rows: list[dict]) -> track.TrackRecord:
    """A TrackRecord.scored-shaped frame from per-day dicts (hits, n_scored,
    base_rate, late) — precision/lift derived so the frame is self-consistent."""
    frame = pd.DataFrame(rows)
    frame["precision"] = frame["hits"] / frame["n_scored"]
    frame["lift"] = frame["precision"] / frame["base_rate"]
    frame["signal_date"] = pd.bdate_range("2026-06-01", periods=len(frame)).date
    frame["target_date"] = pd.bdate_range("2026-06-02", periods=len(frame)).date
    return track.TrackRecord(scored=frame, pending=[])


def test_live_tiles_pool_live_days_only_never_the_backfilled_mix():
    # THE backfill-path guard (CLAUDE.md signature failure mode): a live day and
    # a backfilled/late day. Wrong-pooling both gives reach 9/20 = 45%; the live
    # tiles must count the LIVE day ONLY → reach 80%, excess +60.0%, lift 4.00×.
    record = _scored(
        [
            {"hits": 8, "n_scored": 10, "base_rate": 0.20, "late": False},  # live
            {"hits": 1, "n_scored": 10, "base_rate": 0.20, "late": True},  # backfilled
        ]
    )
    tiles = dashboard._live_tiles(record, top=10)
    assert "Live reach-rate" in tiles
    assert ">80%<" in tiles  # 8/10, the live day alone
    assert "45%" not in tiles  # never the (8+1)/(10+10) wrong-pooled mix
    assert "+60.0%" in tiles  # excess = 0.80 − 0.20 (base from the live day only)
    assert "4.00×" in tiles  # lift = 0.80 / 0.20
    # Deleting the `~late` filter (the ship-green regression) flips all three.


def test_live_tiles_empty_when_all_days_backfilled():
    # Only backfilled days scored: no live number may be shown — the placeholder
    # tile appears instead of a wrongly-populated "live" reach-rate.
    record = _scored([{"hits": 9, "n_scored": 10, "base_rate": 0.20, "late": True}])
    tiles = dashboard._live_tiles(record, top=10)
    assert "no live days yet" in tiles
    assert "90%" not in tiles  # the backfilled day's reach never surfaces as "live"


def test_benchmark_tiles_populated_render_walk_forward_values():
    # FIX 4: the populated benchmark path renders the three Walk-forward tiles
    # with the ledger values, all carrying the "Walk-forward" qualifier so none
    # reads as the live/default number once the live record fills.
    benchmark = (
        7,
        {"auc": 0.72, "lift": 2.15, "precision_at_n": 0.74, "base_rate": 0.34, "top_n": 20},
        None,
        None,
    )
    tiles = dashboard._benchmark_tiles(benchmark, top=20)
    assert "Walk-forward AUC" in tiles and ">0.720<" in tiles
    assert "Walk-forward lift" in tiles and ">2.15×<" in tiles
    assert "Walk-forward reach-rate (top 20)" in tiles and ">74%<" in tiles
    assert "base rate 34%" in tiles
    # Every benchmark tile is qualified "Walk-forward" — no bare "Reach-rate".
    assert "Reach-rate (top 20)" not in tiles.replace("Walk-forward reach-rate", "")


def test_title_and_h1_say_reach_not_open_to_close(modeled, tmp_path):
    # FIX 4: guard a title/h1 regression back to the misleading close wording.
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()
    assert "<title>twopercent — reach +2% intraday</title>" in content
    assert "tickers likely to REACH +2% intraday" in content  # h1
    assert "open-to-close candidates" not in content  # the stale, wrong label


def test_dashboard_clean_reset_note(modeled, tmp_path):
    # When a real cutover left archived close-era predictions behind, the M1
    # clean-reset note discloses when the touch record began; a fresh touch-only
    # store shows nothing.
    dates = sorted(pd.bdate_range("2026-01-05", periods=60).date)
    predict_for(modeled, "baseline_gbm_v1", signal_date=dates[-3], save=True)  # touch era
    picks = pd.DataFrame({"symbol": ["RUN0"], "prob": [0.9], "rank": [1]})
    store.save_predictions(modeled, "baseline_gbm_v1", dates[-6], picks, event=None)  # archived

    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5, result=None)
    content = out.read_text()
    assert "Live reach-record began" in content
    assert str(dates[-3]) in content  # first touch-era signal date
    assert "archived" in content


def test_dashboard_reset_note_at_cutover_without_touch_prediction(modeled):
    # The exact cutover: archived close-era predictions exist but NO touch
    # prediction has been logged yet (first_touch is None). The archive must
    # still be disclosed — not silently suppressed (F3).
    dates = sorted(pd.bdate_range("2026-01-05", periods=60).date)
    picks = pd.DataFrame({"symbol": ["RUN0"], "prob": [0.9], "rank": [1]})
    store.save_predictions(modeled, "baseline_gbm_v1", dates[-6], picks, event=None)  # archived
    result = predict_for(modeled, "baseline_gbm_v1", save=False)  # nothing logged
    content = dashboard.build_html(modeled, result, top=5)
    assert "targeted open-to-close and are archived" in content
    assert "the reach-record starts with the first touch prediction" in content
    assert "Live reach-record began" not in content  # no touch prediction yet


def _calibration_metrics(bss=0.18, reliability=None, regimes=None, picks=True) -> dict:
    """A recorded touch-era benchmark metrics dict carrying a calibration block."""
    if reliability is None:
        reliability = [
            {
                "lo": round(b / 10, 4),
                "hi": round((b + 1) / 10, 4),
                "mean_pred": round(b / 10 + 0.05, 4),
                "reach_rate": round(b / 10 + 0.05, 4),
                "count": 100,
            }
            for b in range(10)
        ]
    if regimes is None:
        regimes = [
            {
                "name": "low",
                "base_lo": 0.05,
                "base_hi": 0.20,
                "n_days": 40,
                "n": 4000,
                "bss": 0.22,
                "slope": 0.98,
            },
            {
                "name": "medium",
                "base_lo": 0.20,
                "base_hi": 0.45,
                "n_days": 40,
                "n": 4000,
                "bss": 0.15,
                "slope": 0.90,
            },
            {
                "name": "high",
                "base_lo": 0.45,
                "base_hi": 0.95,
                "n_days": 40,
                "n": 4000,
                "bss": 0.05,
                "slope": 0.70,
            },
        ]
    calibration = {
        "population": "all_names_test",
        "n": 12000,
        "n_days": 120,
        "n_bins": 10,
        "brier": 0.19,
        "brier_ref": 0.22,
        "bss": bss,
        "slope": 0.9,
        "reliability": reliability,
        "regimes": regimes,
    }
    if picks:
        # Picks concentrate at the modal ~0.74 (well calibrated) with a small
        # overconfident 0.8-0.9 tail — the real champion shape.
        calibration["picks"] = {
            "population": "top_n_liquid_picks",
            "top_n": 20,
            "n": 9000,
            "n_days": 120,
            "n_bins": 10,
            "slope": 0.88,
            "reliability": [
                {"lo": 0.6, "hi": 0.7, "mean_pred": 0.65, "reach_rate": 0.64, "count": 1200},
                {"lo": 0.7, "hi": 0.8, "mean_pred": 0.74, "reach_rate": 0.744, "count": 7000},
                {"lo": 0.8, "hi": 0.9, "mean_pred": 0.83, "reach_rate": 0.72, "count": 800},
            ],
        }
    return {
        "auc": 0.72,
        "lift": 2.15,
        "precision_at_n": 0.74,
        "base_rate": 0.34,
        "top_n": 20,
        "brier": 0.19,
        "calibration": calibration,
    }


def test_calibration_panel_leads_with_picks_and_labels_bss_all_names():
    # FIX B: the PRIMARY curve + read are the top-N picks (what the user acts on);
    # the all-names BSS is a SECONDARY, explicitly-labelled tile — never implying
    # it describes the picks.
    panel = dashboard._calibration_panel((7, _calibration_metrics(bss=0.18), None, None))
    assert "Calibration (walk-forward)" in panel
    assert "Ranking skill · BSS (all names)" in panel and ">+0.18<" in panel  # labelled BSS
    assert "all-names test predictions" in panel  # the tile names its population
    assert "top-20 picks" in panel  # the curve legend names the pick population
    assert "top-20 liquidity-floored PICKS" in panel  # the read caption
    assert "Picks the model calls" in panel  # read uses pick language, not "70% pick"
    assert "<svg" in panel and "perfect calibration" in panel
    assert "By base-rate regime" in panel and "all names, ranking skill" in panel
    assert "$" not in panel


def test_calibration_panel_curve_is_picks_reliability_with_tail_named():
    # The picks curve concentrates at the modal ~74% (calibrated) but carries a
    # small 0.8-0.9 overconfident tail — the read must show BOTH, honestly.
    panel = dashboard._calibration_panel((7, _calibration_metrics(), None, None))
    assert "~74% reached +2% about 74%" in panel  # modal pick is calibrated
    assert "in the bulk" in panel  # not an unqualified "well calibrated"
    assert "~83% reach only ~72%" in panel  # the overconfident tail is named


def test_calibration_panel_falls_back_to_all_names_when_no_pick_block():
    # An older touch row recorded before pick-calibration existed: the curve falls
    # back to all-names, LOUDLY labelled (not silently mislabelled as picks).
    panel = dashboard._calibration_panel((7, _calibration_metrics(picks=False), None, None))
    assert "No pick-population block recorded yet" in panel
    assert "all names" in panel
    assert "Re-run twopercent benchmark" in panel
    assert "Predictions the model calls" in panel  # all-names noun, not "Picks"


def test_calibration_panel_flags_regime_disagreement():
    # BSS spread across regimes is wide (0.22 vs 0.05) — the panel must say so, not
    # let the pooled curve average a regime-specific miss away.
    panel = dashboard._calibration_panel((7, _calibration_metrics(), None, None))
    assert "Calibration DIFFERS by regime" in panel


def test_calibration_read_is_honest_about_miscalibration():
    # Overconfident model: the 0.9 bucket only reaches 0.4. The read must NOT claim
    # calibration — it reports the gap and calls it materially miscalibrated.
    reliability = [
        {"lo": 0.0, "hi": 0.1, "mean_pred": 0.05, "reach_rate": 0.05, "count": 100},
        {"lo": 0.9, "hi": 1.0, "mean_pred": 0.9, "reach_rate": 0.4, "count": 100},
    ]
    read = dashboard._calibration_read(reliability, 0.6, "Picks", "picks")
    assert "materially miscalibrated" in read
    assert "overconfident" in read
    assert "~90%" in read and "~40%" in read  # honest to the actual bucket


def test_calibration_read_names_a_small_overconfident_tail_not_averaged_away():
    # THE adversarial case both reviewers asked for: a large well-calibrated bulk
    # plus a SMALL overconfident high-prob bucket. The count-weighted ECE is tiny
    # (the bulk dominates), so a weighted verdict alone would print "well
    # calibrated" — but the read must NAME the tail miss regardless of its count.
    reliability = [
        {"lo": 0.1, "hi": 0.2, "mean_pred": 0.15, "reach_rate": 0.15, "count": 5000},
        {"lo": 0.2, "hi": 0.3, "mean_pred": 0.25, "reach_rate": 0.25, "count": 5000},
        {"lo": 0.3, "hi": 0.4, "mean_pred": 0.35, "reach_rate": 0.35, "count": 5000},
        {"lo": 0.8, "hi": 0.9, "mean_pred": 0.85, "reach_rate": 0.60, "count": 50},  # tail
    ]
    read = dashboard._calibration_read(reliability, 0.95, "Picks", "picks")
    assert "well calibrated" not in read or "in the bulk" in read  # never unqualified
    assert "in the bulk" in read
    assert "~85%" in read and "reach only ~60%" in read  # the tail is named
    assert "regime-clustered" in read  # magnitude hedged (quant-skeptic Note C)


def test_calibration_read_uses_population_noun():
    reliability = [{"lo": 0.7, "hi": 0.8, "mean_pred": 0.74, "reach_rate": 0.74, "count": 900}]
    picks_read = dashboard._calibration_read(reliability, 1.0, "Picks", "picks")
    names_read = dashboard._calibration_read(reliability, 1.0, "Predictions", "predictions")
    assert picks_read.startswith("Picks the model calls")
    assert names_read.startswith("Predictions the model calls")


def test_calibration_panel_degrades_without_benchmark_or_block():
    # No touch-era benchmark, and a pre-Stage-C benchmark with no calibration block:
    # both degrade gracefully to the same honest placeholder — never a blank/crash.
    for benchmark in (None, (7, {"auc": 0.7}, None, None)):
        panel = dashboard._calibration_panel(benchmark)
        assert "Calibration (walk-forward)" in panel
        assert "No touch-era benchmark yet" in panel
        assert "<svg" not in panel


def test_dashboard_render_includes_calibration_panel(modeled, tmp_path):
    # End-to-end: a recorded touch-era benchmark with a calibration block surfaces
    # the panel in the rendered page, walk-forward-labelled and dollar-free.
    store.record_experiment(
        modeled,
        strategy="baseline_gbm_v1",
        params={"months": 12, "top_n": 20},
        train_start=pd.Timestamp("2025-06-02").date(),
        test_start=pd.Timestamp("2026-01-05").date(),
        test_end=pd.Timestamp("2026-06-30").date(),
        metrics=_calibration_metrics(bss=0.12),
    )
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()
    section = content.split("Calibration (walk-forward)", 1)[1].split("Track record", 1)[0]
    assert "Ranking skill · BSS (all names)" in section and ">+0.12<" in section
    assert "<svg" in section
    assert "top-20 liquidity-floored PICKS" in section  # picks-primary curve
    assert "same population as the benchmark AUC/Brier" in section  # WF labelling
    assert "$" not in section


def _record_sim(con, n_days, ranks_per_day=6, strategy="baseline_gbm_v1", strategy_cols=True):
    """Per-rank sim rows: mostly-up rank-1 returns, adversarial values.

    strategy_cols=False seeds LEGACY rows (recorded before the oh/ol upgrade)
    for the degradation path. When present: winners touch +2% intraday (a limit
    fill) with a shallow low; losers never touch and their low trips the −1%
    stop — so every exit rule separates from hold-to-close."""
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
            row = {"target_date": d, "rank": rank, "ret": ret, "hit": int(ret > 0.02)}
            if strategy_cols:
                row["oh"] = 0.0231 if ret > 0.02 else 0.0042
                row["ol"] = -0.0009 if ret > 0.02 else -0.0117
            rows.append(row)
    store.record_experiment_daily(con, seq, pd.DataFrame(rows))
    return seq


def test_dashboard_explorer_defaults_match_python_math(modeled, tmp_path):
    # Server-side default view (top-5 basket, 6-month window) must carry the
    # exact reach-rate _summarize_days computes — the JS payload is generated
    # from the same frames, so this pins Python and JS to one source of truth.
    # No dollars: the explorer reports reach-rate / base rate / lift only.
    _record_sim(modeled, n_days=130)
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    _, daily = store.latest_experiment_daily(modeled, "baseline_gbm_v1")
    sim_days = dashboard._payload_days(daily.rename(columns={"ret": "oc"}), bases={})
    s126 = dashboard._summarize_days(sim_days[-126:], 5)
    assert f'<td id="tp-sim-hit">{dashboard._pct_half_up(s126["hit"])}</td>' in content
    assert '<td id="tp-sim-days">126</td>' in content  # selection day count visible

    assert 'class="badge-sim">SIM</span>' in content
    assert 'class="badge-live">LIVE</span>' in content
    assert "walk-forward, monthly retrain" in content
    # The DEFAULT (reach) card reports reach only — dollars live exclusively in
    # the opt-in strategy card, which stays hidden until selected.
    reach_card = content.split("id='tp-card-reach'")[1].split("</div>")[0]
    assert "<th>Reach rate" in content and "$" not in reach_card
    # Both selects, with per-window trading-day counts, defaults marked.
    assert '<option value="5" selected>Top 5</option>' in content
    assert '<option value="126" selected>6 months (126 trading days)</option>' in content
    assert "1 year (252 trading days)" in content
    # Span line: test span, months, run staleness, sim day count.
    assert "SIM test span" in content and "months=12" in content
    assert "simulated <b" in content
    assert '<span class="mono">130</span> sim days available' in content
    # Survivorship + selection + base-rate caveats, verbatim tails.
    assert "delisted names can never contribute their final catastrophic day" in content
    assert "itself a form of selection" in content
    assert "Lift compresses at a high base rate" in content
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
    assert "cost" not in payload  # GROSS returns: no trading-cost constant anywhere
    assert len(payload["sim"]) == 6
    day0 = payload["sim"][0]
    assert day0["d"] == "2026-01-05"
    # Each pick is [rank, oh, ol, oc, hit] — raw outcome returns + the guarded
    # touch/fill flag; growth is DERIVED in lockstep, never shipped as dollars.
    # Hand-check day 0: (i + rank) % 5 == 0 at rank 5 → that one is not a reacher.
    assert day0["picks"] == [
        [1, 0.0231, -0.0009, 0.0203, 1],
        [2, 0.0231, -0.0009, 0.0203, 1],
        [3, 0.0231, -0.0009, 0.0203, 1],
        [4, 0.0231, -0.0009, 0.0203, 1],
        [5, 0.0042, -0.0117, -0.0117, 0],
        [6, 0.0231, -0.0009, 0.0203, 1],
    ]
    # Base rate on 2026-01-05: 4 runners of 8 names reached ≥2%.
    assert abs(day0["base"] - 0.5) < 1e-9
    # Live day present with late flag and rank-ordered picks; oh/ol/oc are the
    # daily_returns outcome magnitudes (never null on the live path).
    assert len(payload["live"]) == 1
    live0 = payload["live"][0]
    assert live0["late"] is True  # backfilled save, created after the target open
    assert [p[0] for p in live0["picks"]] == sorted(p[0] for p in live0["picks"])
    assert all(len(p) == 5 for p in live0["picks"])
    assert all(p[1] is not None and p[2] is not None and p[3] is not None for p in live0["picks"])


def test_dashboard_explorer_too_few_sim_days_says_so(modeled, tmp_path):
    _record_sim(modeled, n_days=4)
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    assert 'class="badge-sim">SIM</span>' in content
    assert "SIM: needs 126 trading days — 4 available" in content
    assert '<td id="tp-sim-hit">—</td>' in content  # no number pretended
    assert "The live record above is the clean test." in content


def _pick(rank, hit, oh=0.021, ol=-0.002, oc=0.005):
    """A payload pick [rank, oh, ol, oc, hit] with harmless default returns."""
    return [rank, oh, ol, oc, hit]


def test_summarize_days_first_available_substitution_and_short_days():
    days = [
        {"d": "a", "base": 0.25, "picks": [_pick(2, 1), _pick(3, 0)]},  # rank 1 missing
        {"d": "b", "base": 0.15, "picks": [_pick(1, 1)]},
    ]
    s1 = dashboard._summarize_days(days, 1)
    # First-available rule: day a's top pick is rank 2 (the trader takes it) —
    # a reacher; day b's rank 1 reached too, so reach-rate is 1.0.
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
        {"d": "a", "late": False, "base": 0.25, "picks": [_pick(2, 1)]},  # rank 1 missing
        {"d": "b", "late": False, "base": None, "picks": [_pick(1, 0)]},  # base unknown
    ]
    _, live_s, notes = dashboard._explorer_state([], days, 1, 5)
    assert live_s is not None
    assert "LIVE: 1 day(s) substituted lower-ranked names for missing picks" in notes
    assert "LIVE: base rate from 1 of 2 day(s)" in notes
    # A contiguous full-coverage frame emits neither disclosure.
    clean_days = [
        {"d": "a", "late": False, "base": 0.25, "picks": [_pick(1, 1)]},
        {"d": "b", "late": False, "base": 0.25, "picks": [_pick(1, 0)]},
    ]
    _, _, clean_notes = dashboard._explorer_state([], clean_days, 1, 5)
    assert not any("substituted" in n or "base rate from" in n for n in clean_notes)


def test_dashboard_renders_live_substitution_note(modeled, tmp_path):
    import datetime as dt

    dates = sorted(pd.bdate_range("2026-01-05", periods=60).date)
    # Pad the universe so removing ONE pick's target-day bar keeps the day
    # COMPLETE (coverage vs trailing median, #65) — in an 8-name fixture a single
    # deletion is a 12% cliff that the gate would hold; in production it is ~0.03%.
    seed_history(modeled, {f"PAD{i:02d}": FLAT_OC for i in range(12)}, vary_volume=True)
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
        {"hit": 0.125, "base": 0.125, "days": 8, "short": 0, "corrupt": 0},
    )
    assert "tp-sim-growth" not in cells  # no $-growth cell
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
    # A non-finite reach flag (a data error) must be counted as corrupt, never
    # silently averaged around into a shorter window.
    days = [
        {"d": "a", "late": False, "base": None, "picks": [_pick(1, float("nan"))]},
        {"d": "b", "late": False, "base": None, "picks": [_pick(1, 0)]},
    ]
    s = dashboard._summarize_days(days, 1)
    assert s["corrupt"] == 1
    # The corrupt window renders as an error state, never as a shorter product.
    _, live_s, notes = dashboard._explorer_state([], days, 1, 5)
    assert live_s is None
    assert any("corrupt day(s)" in n for n in notes)


def test_explorer_state_live_short_window_disclosures():
    live = [{"d": str(i), "late": False, "base": None, "picks": [_pick(1, 0)]} for i in range(3)]
    sim_s, live_s, notes = dashboard._explorer_state([], live, 5, 126)
    assert sim_s is None
    assert any("No touch-era benchmark yet" in n for n in notes)
    assert live_s is not None and live_s["days"] == 3
    assert any("all 3 live day(s)" in n for n in notes)  # window shortfall disclosed
    assert any("3 day(s) had fewer than 5 picks" in n for n in notes)
    # Late days are excluded from LIVE, loudly absent when that empties it.
    _, live_s2, notes2 = dashboard._explorer_state([], [dict(d, late=True) for d in live], 5, 126)
    assert live_s2 is None
    assert any("no live days yet" in n for n in notes2)


# --- strategy explorer (exit-rule what-if) ------------------------------------


def test_strategy_dropdown_defaults_to_reach_and_disables_trailing(modeled, tmp_path):
    # Reach rate (prediction quality) is the DEFAULT view; the P&L card exists
    # but is hidden until an exit rule is chosen; the trailing stop is listed
    # DISABLED with its reason (daily bars cannot replay it) — never simulated.
    _record_sim(modeled, n_days=130)
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    assert '<option value="reach" selected>Reach rate (prediction quality)</option>' in content
    assert '<option value="trailing" disabled>Trailing stop (needs intraday data)' in content
    assert 'id="tp-card-strat" hidden' in content  # opt-in, never the default
    assert "id='tp-card-reach'" in content  # the default card is the reach table
    # Selection caveat names strategies explicitly (browsing IS selection).
    assert "flipping strategies/baskets/windows to find a good-looking cell" in content


def test_strategy_card_framing_stamps_and_gross_labels(modeled, tmp_path):
    _record_sim(modeled, n_days=130)
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()
    card = content.split('id="tp-card-strat"')[1].split("</table>")[0]

    # SIM growth cell: the stamp sits NEXT TO the number, not in a footnote.
    sim_cell = card.split('id="tp-strat-sim-growth"')[1].split("</td>")[0]
    assert "OPTIMISTIC UPPER BOUND" in sim_cell
    # LIVE row is framed as the honest, survivorship-free, small sample.
    assert "real logged picks — survivorship-free, small sample" in card
    # Plain-language survivorship line + gross labels ride in the same card.
    strat_block = content.split('id="tp-card-strat"')[1].split("</div>")[0]
    assert "excludes every company that delisted" in strat_block
    assert "if even a few of their blow-up days were real, this could be a loss" in strat_block
    assert "before trading costs (not estimated)" in strat_block
    assert "daily data can't tell which came first" in strat_block
    assert "glitch-suspect high never counts as a fill" in strat_block


def test_strategy_row_never_shows_win_rate_without_growth(modeled, tmp_path):
    # The inviolable pairing (quant-skeptic must-fix #2): every populated
    # strategy row carries $1→growth AND win rate together; a dashed row dashes
    # BOTH. Checked on the rendered server default (hold_close).
    _record_sim(modeled, n_days=130)
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    sim_growth = content.split('id="tp-strat-sim-growth">')[1].split("<")[0]
    sim_win = content.split('id="tp-strat-sim-win">')[1].split("<")[0]
    assert sim_growth.startswith("$1 → $")
    assert sim_win.endswith("%")
    # Structural guard on the cell builder itself: populated → both; empty → both dashed.
    s = {"gw": 1.1, "gb": 1.1, "ww": 0.6, "wb": 0.6, "days": 5}
    cells = dashboard._strategy_cells("strat-sim", s)
    assert "$1 → $1.10" in cells and ">60%<" in cells
    dashed = dashboard._strategy_cells("strat-sim", None)
    assert dashed.count("—") == 3  # growth, win rate, days — all dashed together


def test_strategy_server_default_matches_python_math(modeled, tmp_path):
    # The hidden strategy table is server-rendered for hold_close from the SAME
    # payload the JS consumes — Python is the source of truth the JS re-derives.
    _record_sim(modeled, n_days=130)
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    _, daily = store.latest_experiment_daily(modeled, "baseline_gbm_v1")
    sim_days = dashboard._payload_days(daily.rename(columns={"ret": "oc"}), bases={})
    sim_s, live_s, _ = dashboard._strategy_state(sim_days, [], 5, 126, "hold_close")
    assert sim_s is not None
    expected = dashboard._growth_text(sim_s)
    assert f'id="tp-strat-sim-growth">{expected}</span>' in content
    assert f'id="tp-strat-sim-win">{dashboard._win_text(sim_s)}</td>' in content
    read = dashboard._strategy_read(sim_s, live_s)
    assert "$1 following this rule on the SIM picks became" in read
    assert read in content


def test_strategy_state_missing_ohol_degrades_to_rebenchmark_message(modeled, tmp_path):
    # Pre-upgrade benchmark rows (no oh/ol): the SIM strategy row must show NO
    # number — only the re-benchmark message — and the page must not crash.
    # Reach view stays fully functional (it never needed oh/ol).
    _record_sim(modeled, n_days=130, strategy_cols=False)
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    assert "no strategy data recorded on 126 of 126 day(s) — re-run twopercent benchmark" in content
    assert 'id="tp-strat-sim-growth">—</span>' in content  # no number pretended
    # Reach default is untouched by the missing strategy data.
    assert '<td id="tp-sim-days">126</td>' in content


def test_strategy_state_live_excludes_late_days(caplog):
    # The survivorship-free LIVE framing is only honest on genuinely-live picks:
    # a backfilled day's outcome was known at save time. Late days must vanish
    # from the strategy LIVE row exactly as they do from the reach row.
    live_day = {"d": "a", "late": False, "base": None, "picks": [_pick(1, 1, oc=0.03)]}
    late_day = {"d": "b", "late": True, "base": None, "picks": [_pick(1, 1, oc=0.50)]}
    sim_s, live_s, notes = dashboard._strategy_state([], [live_day, late_day], 1, 5, "hold_close")
    assert sim_s is None
    assert live_s is not None and live_s["days"] == 1  # the late +50% day never counts
    assert abs(live_s["gw"] - 1.03) < 1e-12
    assert any("all 1 live day(s)" in n for n in notes)
    # All days late → no LIVE number at all, said loudly.
    _, live_s2, notes2 = dashboard._strategy_state([], [late_day], 1, 5, "hold_close")
    assert live_s2 is None
    assert any("no live days yet" in n for n in notes2)


def test_strategy_growth_and_win_math_hand_checked():
    # Two days, basket of 2, hold_close: day returns are means of oc, growth
    # compounds them, win rate counts PICKS (3 of 4 positive).
    days = [
        {"d": "a", "base": None, "picks": [_pick(1, 1, oc=0.02), _pick(2, 0, oc=-0.01)]},
        {"d": "b", "base": None, "picks": [_pick(1, 1, oc=0.04), _pick(2, 1, oc=0.02)]},
    ]
    s = dashboard.strategy.summarize_strategy_days(days, 2, "hold_close")
    assert abs(s["gw"] - (1 + 0.005) * (1 + 0.03)) < 1e-12
    assert s["gw"] == s["gb"]  # hold_close is exact — no band
    assert s["ww"] == s["wb"] == 0.75
    assert s["picks"] == 4 and s["days"] == 2
    assert dashboard._growth_text(s) == "$1 → $1.04"
    assert dashboard._win_text(s) == "75%"


def test_strategy_band_renders_worst_and_best():
    # A both-triggered limit+stop day yields a BAND, never a single number.
    days = [{"d": "a", "base": None, "picks": [_pick(1, 1, ol=-0.015, oc=0.001)]}]
    s = dashboard.strategy.summarize_strategy_days(days, 1, "limit_stop")
    assert abs(s["gw"] - 0.99) < 1e-12 and abs(s["gb"] - 1.02) < 1e-12
    assert s["ww"] == 0.0 and s["wb"] == 1.0
    assert dashboard._growth_text(s) == "$1 → $0.99–$1.02"
    assert dashboard._win_text(s) == "0%–100%"
    assert "between $0.99 (worst case) and $1.02 (best case)" in dashboard._strategy_read(s, None)


def test_strategy_lockstep_python_vs_node():
    # THE lockstep proof for the highest-stakes math on the page: execute the
    # page's own tpMath script under node on an adversarial payload and demand
    # bit-identical numbers and identical rendered strings from the Python
    # mirrors. Covers: fills, stops, both-triggered bands, exact-boundary stop
    # lows, substitution, short days, and a missing-data (pre-upgrade) day.
    import json
    import shutil
    import subprocess

    import pytest

    if shutil.which("node") is None:
        pytest.skip("node not available for the JS lockstep")

    days = [
        {"d": "a", "base": 0.5, "picks": [_pick(1, 1, ol=-0.015, oc=0.001), _pick(2, 0)]},
        {"d": "b", "base": None, "picks": [_pick(2, 0, ol=-0.01, oc=-0.0117)]},  # subst + short
        {"d": "c", "base": 0.25, "picks": [_pick(1, 0, ol=-0.009999999999999964, oc=0.0)]},
        {"d": "d", "base": 0.25, "picks": [_pick(1, 1, oh=0.0203, ol=-0.0009, oc=0.0203)]},
    ]
    missing_days = days + [{"d": "e", "base": None, "picks": [[1, None, None, None, 0]]}]

    js = dashboard._JS_MATH.replace("<script>", "").replace("</script>", "")
    driver = f"""
{js}
const days = {json.dumps(days)};
const missingDays = {json.dumps(missing_days)};
const out = {{}};
for (const strat of ["hold_close", "limit_2pct", "limit_stop"]) {{
  for (const n of [1, 2, 5]) {{
    const s = tpMath.stratSummarize(days, n, strat);
    out[strat + ":" + n] = [s.gw, s.gb, s.ww, s.wb, s.picks, s.days, s.clean,
                            s.shortDays, s.substDays, s.missing, s.corrupt,
                            tpMath.growthText(s), tpMath.winText(s),
                            tpMath.readText(s, null)];
  }}
  out[strat + ":missing"] = tpMath.stratSummarize(missingDays, 2, strat).missing;
}}
const r = tpMath.summarize(days, 2);
out.reach = [r.hit, r.base, r.days, r.shortDays, r.substDays, r.baseDays, r.corrupt];
console.log(JSON.stringify(out));
"""
    proc = subprocess.run(
        ["node", "-e", driver], capture_output=True, text=True, timeout=60, check=True
    )
    node_out = json.loads(proc.stdout)

    def _nn(x):  # JSON.stringify(NaN) -> null; mirror for comparison
        import math

        return None if isinstance(x, float) and math.isnan(x) else x

    for strat in ("hold_close", "limit_2pct", "limit_stop"):
        for n in (1, 2, 5):
            s = dashboard.strategy.summarize_strategy_days(days, n, strat)
            expected = [
                _nn(s["gw"]),
                _nn(s["gb"]),
                _nn(s["ww"]),
                _nn(s["wb"]),
                s["picks"],
                s["days"],
                s["clean"],
                s["short"],
                s["subst"],
                s["missing"],
                s["corrupt"],
                dashboard._growth_text(s),
                dashboard._win_text(s),
                dashboard._strategy_read(s, None),
            ]
            assert node_out[f"{strat}:{n}"] == expected, f"{strat} n={n} diverged"
        s_missing = dashboard.strategy.summarize_strategy_days(missing_days, 2, strat)
        assert node_out[f"{strat}:missing"] == s_missing["missing"] == 1
    r = dashboard._summarize_days(days, 2)
    assert node_out["reach"] == [
        _nn(r["hit"]),
        r["base"],
        r["days"],
        r["short"],
        r["subst"],
        r["base_days"],
        r["corrupt"],
    ]
