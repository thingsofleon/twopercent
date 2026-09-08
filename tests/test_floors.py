"""Liquidity-floor study: selection-only sweep on shared fits (#121)."""

from __future__ import annotations

import logging

import pytest

from tests.conftest import seed_planted
from twopercent import backtest, floors, store, strategy


@pytest.fixture
def planted(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    return con


def test_all_arms_reported_and_baseline_anchors(planted):
    result = floors.run_study(planted, "baseline_gbm_v1", months=2, top_n=5, seeds=[42])
    assert set(result["arms"]) == set(floors._arms())
    assert result["baseline_arm"] in result["arms"]
    assert "precision_vs_baseline" not in result["arms"][result["baseline_arm"]]
    # Every non-baseline arm with days carries the paired blocks.
    for name, row in result["arms"].items():
        if name != result["baseline_arm"] and row["days"]:
            assert "precision_vs_baseline" in row and "net_vs_baseline" in row
    # Nothing recorded: a study is not a benchmark.
    assert len(store.list_experiments(planted)) == 0


def test_identical_selection_means_exactly_zero_delta(planted):
    """The whole design: arms share ONE fitted model, so two floors that admit
    the same rows must produce bit-identical days — any nonzero delta would
    mean the sweep is refitting per arm and the pairing is fiction."""
    # Seeded volumes are ~1.0-1.016M shares, so 100k/250k/500k floors admit
    # every symbol identically.
    result = floors.run_study(planted, "baseline_gbm_v1", months=2, top_n=5, seeds=[42])
    for arm in ("shares>=250k", "shares>=500k"):
        row = result["arms"][arm]
        assert row["days"] == result["arms"][result["baseline_arm"]]["days"]
        for key in ("precision_vs_baseline", "net_vs_baseline"):
            test = row[key]
            assert test["mean"] == pytest.approx(0.0, abs=1e-12)
            assert test["positive"] == test["negative"] == 0


def test_a_floor_above_the_universe_yields_zero_days_not_a_crash(planted):
    # Seeded volumes top out ~1.016M: the 2M arm must report zero days and the
    # report must render it without fabricating numbers.
    result = floors.run_study(planted, "baseline_gbm_v1", months=2, top_n=5, seeds=[42])
    row = result["arms"]["shares>=2000k"]
    assert row["days"] == 0
    assert row["precision"] is None and row["net_daily"] is None
    report = floors.format_report(result)
    assert "no day had any eligible name" in report
    assert "OPTIMISTIC" in report


def test_tick_cost_is_one_tick_over_signal_close(planted):
    """A $5.00 name costs 20bps round trip under the stated model — the exact
    convention ROADMAP's paper study used, pinned at an adversarial price."""
    # Rescale each bar so close is EXACTLY 5.00 with OHLC ratios preserved —
    # setting close alone would violate the ordering gate and silently drop
    # every bar from daily_returns (the frame would be empty, not cheap).
    planted.execute(
        "UPDATE prices SET open = open / close * 5, high = high / close * 5, "
        "low = low / close * 5, close = 5.0"
    )
    result = floors.run_study(planted, "baseline_gbm_v1", months=2, top_n=5, seeds=[42])
    row = result["arms"][result["baseline_arm"]]
    assert row["tick_cost_bps"] == pytest.approx(20.0, abs=1e-6)
    assert row["median_price"] == pytest.approx(5.00)
    assert row["net_daily"] == pytest.approx(row["gross_daily"] - 0.002, abs=1e-9)


def test_gross_limit_matches_the_single_exit_rule_definition(planted):
    """floors inlines `0.02 if hit else oc` for speed; strategy.pick_return_band
    is the single definition of exit rules. Pin them together so the inline
    copy cannot drift (the repo's one-predicate discipline)."""
    from twopercent import features

    frame = features.feature_frame(planted)
    labeled = frame[frame["did_2pct_next"].notna()].head(200)
    for row in labeled.itertuples():
        expected = strategy.pick_return_band(
            "limit_2pct",
            float(row.next_low_return),
            float(row.next_oc_return),
            bool(row.did_2pct_next),
        )[0]
        inline = 0.02 if bool(row.did_2pct_next) else float(row.next_oc_return)
        assert inline == pytest.approx(expected, abs=1e-12)


def test_missing_signal_close_is_counted_and_excluded_from_cost(planted, caplog):
    """A pick without a price stays in PRECISION for share arms (matching the
    shipped selection exactly) but must never enter a cost mean silently."""
    # A NEGATIVE close passes the daily_returns validity gate (open>0, finite,
    # ordering — it never checks close>0), so such a bar reaches the feature
    # frame while the study's price query rightly refuses it as a divisor.
    sym = planted.execute("SELECT symbol FROM prices LIMIT 1").fetchone()[0]
    planted.execute("UPDATE prices SET close = -1, low = least(low, -2) WHERE symbol = ?", [sym])
    with caplog.at_level(logging.WARNING, logger="twopercent.floors"):
        result = floors.run_study(planted, "baseline_gbm_v1", months=2, top_n=5, seeds=[42])
    assert result["rows_without_close"] > 0
    assert "no usable signal close" in caplog.text


def test_duplicate_seeds_rejected(planted):
    with pytest.raises(ValueError, match="duplicate seeds"):
        floors.run_study(planted, "baseline_gbm_v1", seeds=[42, 42])
