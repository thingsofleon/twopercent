import pandas as pd
import pytest

from tests.conftest import seed_history
from twopercent import dashboard, store
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
    # No benchmark recorded daily rows yet — the SIM panel must say so loudly.
    assert "No walk-forward simulation recorded yet" in content
    assert "twopercent benchmark" in content


def _record_sim(con, n_days, strategy="baseline_gbm_v1"):
    seq = store.record_experiment(
        con,
        strategy=strategy,
        params={"months": 2, "top_n": 5},
        train_start=pd.Timestamp("2025-06-02").date(),
        test_start=pd.Timestamp("2026-01-05").date(),
        test_end=pd.Timestamp("2026-02-27").date(),
        metrics={"lift": 2.1},
    )
    daily = pd.DataFrame(
        {
            "target_date": sorted(pd.bdate_range("2026-01-05", periods=n_days).date),
            "top1_ret": [0.0203 if i % 2 == 0 else -0.0117 for i in range(n_days)],
            "top1_hit": [1 if i % 2 == 0 else 0 for i in range(n_days)],
            "top5_ret": [0.0041] * n_days,
            "top5_hits": [0.4] * n_days,
        }
    )
    store.record_experiment_daily(con, seq, daily)
    return seq


def test_dashboard_sim_panel_renders_windows(modeled, tmp_path):
    _record_sim(modeled, n_days=6)
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    assert "Simulated record — walk-forward, monthly retrain" in content
    assert 'class="badge-sim">SIM</span>' in content
    assert "1 week" in content  # 6 days: only the 1-week window qualifies
    assert "1 month" not in content
    assert '<span class="mono">6</span> sim days available' in content
    assert "2026-01-05" in content and "2026-02-27" in content  # test span
    assert "The live record above is the clean test." in content
    assert "No walk-forward simulation recorded yet" not in content


def test_dashboard_sim_panel_too_few_days_says_so(modeled, tmp_path):
    _record_sim(modeled, n_days=4)
    out = tmp_path / "dash.html"
    dashboard.render(modeled, "baseline_gbm_v1", str(out), top=5)
    content = out.read_text()

    assert 'class="badge-sim">SIM</span>' in content
    assert "Only 4 sim day(s) recorded" in content
    assert "needs 5 trading days" in content  # names the omitted shortest window
    assert "The live record above is the clean test." in content
