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
