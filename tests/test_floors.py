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


def test_baseline_arm_reproduces_the_referee_on_the_floored_path(planted, caplog):
    """The baseline arm IS the shipped selection — pinned against run_benchmark
    with sub-floor symbols planted, because with every symbol eligible this
    test would stay green if the floor filter were deleted (the exact trap
    ab.py's lockstep test fell into first; reviewer, PR #122)."""
    planted.execute("UPDATE prices SET volume = 50_000 WHERE symbol LIKE 'RUN0%'")
    with caplog.at_level(logging.WARNING):
        metrics = backtest.run_benchmark(
            planted, "baseline_gbm_v1", months=2, top_n=5, record=False
        )
    result = floors.run_study(planted, "baseline_gbm_v1", months=2, top_n=5, seeds=[42])
    base = result["arms"][result["baseline_arm"]]
    assert base["precision"] == pytest.approx(metrics["precision_at_n"], abs=5e-5)
    assert base["days"] == metrics["test_days"]
    assert result["folds"] == metrics["folds"]
    # The floored path actually ran, or the pin proves nothing.
    assert "liquidity floor" in caplog.text


def test_everything_below_the_baseline_floor_is_a_hard_error(planted):
    """A table of nan-deltas against a nonexistent baseline reads like a
    measurement; ab.py earned the hard error and this module inherits it."""
    planted.execute("UPDATE prices SET volume = 50_000")
    with pytest.raises(RuntimeError, match="liquidity floor"):
        floors.run_study(planted, "baseline_gbm_v1", months=2, top_n=5, seeds=[42])


def test_paired_blocks_carry_the_studys_own_family_alpha(planted):
    """mde_80 against the wrong family is how #115 misread its evidence: the
    persisted blocks must carry THIS study's 18-test alpha, not ab's 2-test
    default (reviewer, PR #122)."""
    result = floors.run_study(planted, "baseline_gbm_v1", months=2, top_n=5, seeds=[42])
    expected = 0.05 / result["multiplicity"]["non_baseline_arms"] / 2
    assert result["multiplicity"]["bonferroni_alpha"] == pytest.approx(expected)
    for name, row in result["arms"].items():
        if name != result["baseline_arm"] and row["days"]:
            assert row["net_vs_baseline"]["alpha"] == pytest.approx(expected)
            assert row["precision_vs_baseline"]["alpha"] == pytest.approx(expected)


def test_test_end_is_the_last_day_with_data_not_the_nominal_fold_end(planted):
    """The first live run claimed test_end 2026-09-30 while its series ended
    2026-09-04 — a 26-day phantom in a checked-in artifact (quant-skeptic,
    PR #122)."""
    result = floors.run_study(planted, "baseline_gbm_v1", months=2, top_n=5, seeds=[42])
    base_days = result["arms"][result["baseline_arm"]]["by_day"]["precision"]
    assert result["test_end"] == max(base_days)
    assert result["final_fold_end_nominal"] >= result["test_end"]
