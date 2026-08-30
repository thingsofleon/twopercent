import logging

import pandas as pd
import pytest

from tests.conftest import seed_planted
from twopercent import ab, backtest, features, store
from twopercent.strategies.base import _REGISTRY, register, resolve_feature_columns

WITHOUT = list(features.FEATURE_COLUMNS)
WITH = list(features.FEATURE_COLUMNS) + list(features.INTRADAY_FEATURE_COLUMNS)


if "ab_seed_blind" not in _REGISTRY:

    @register("ab_seed_blind")
    class SeedBlind:
        """Deterministic ranker that ACCEPTS a seed and ignores it."""

        def __init__(self, feature_columns=None, random_state=0):
            self.configured_columns = resolve_feature_columns("ab_seed_blind", feature_columns)
            self.dropped_columns: list[str] = []

        def fit(self, train):
            pass

        def predict_proba(self, rows):
            return pd.Series(rows["oc_return_today"].rank(pct=True).to_numpy(), index=rows.index)


def _arms(**kwargs):
    return {"without": WITHOUT, "with": WITH, **kwargs}


def test_reference_arm_reproduces_the_referee(con, monkeypatch):
    """The A/B's ruler IS the benchmark's ruler.

    ab.py runs its own fold loop, so nothing but this test stops the two from
    drifting into measuring subtly different things — at which point the
    decision rule ("must reach the SHIPPED metric") would be enforced with a
    ruler the shipped metric is not measured on.
    """
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    metrics = backtest.run_benchmark(con, "baseline_gbm_v1", months=2, top_n=5, record=False)
    result = ab.run_ab(
        con,
        arms=_arms(),
        strategy_name="baseline_gbm_v1",
        months=2,
        top_n=5,
        seeds=[42],
    )
    reference = result["arms"]["without"]
    assert result["base_rate"] == pytest.approx(metrics["base_rate"], abs=5e-5)
    assert reference["precision_at_n"] == pytest.approx(metrics["precision_at_n"], abs=5e-5)
    assert reference["auc"] == pytest.approx(metrics["auc"], abs=5e-5)
    assert reference["lift"] == pytest.approx(metrics["lift"], abs=5e-4)
    assert result["test_days"] == metrics["test_days"]
    assert result["folds"] == metrics["folds"]


def test_arms_are_the_only_difference_and_are_recorded_by_fingerprint(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    result = ab.run_ab(
        con, arms=_arms(), strategy_name="baseline_gbm_v1", months=2, top_n=5, seeds=[42]
    )
    assert result["columns_under_test"] == sorted(features.INTRADAY_FEATURE_COLUMNS)
    assert result["arms"]["without"]["feature_set"] == features.feature_set_version()
    assert result["arms"]["with"]["feature_set"] != features.feature_set_version()
    assert result["comparisons"]["with"]["vs"] == "without"
    # Nothing recorded: an A/B is not a benchmark.
    assert len(store.list_experiments(con)) == 0


def test_all_nan_column_under_test_raises_instead_of_reporting_no_difference(con, monkeypatch):
    """The failure this guard exists for reports SUCCESS without it.

    Strategies drop an all-NaN column (HistGBM's binner crashes on one), so an
    arm distinguished only by unobserved columns trains byte-identically to the
    reference and the A/B answers "no difference" — a false all-clear that looks
    exactly like a real measurement.
    """
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con, with_intraday=False)
    with pytest.raises(RuntimeError, match="ZERO observed values"):
        ab.run_ab(con, arms=_arms(), strategy_name="baseline_gbm_v1", months=2, top_n=5, seeds=[42])


def test_identical_arms_rejected(con):
    with pytest.raises(ValueError, match="identical feature columns"):
        ab.run_ab(
            con,
            arms={"a": WITHOUT, "b": list(WITHOUT)},
            strategy_name="baseline_gbm_v1",
        )


def test_single_arm_rejected(con):
    with pytest.raises(ValueError, match="at least two arms"):
        ab.run_ab(con, arms={"only": WITHOUT}, strategy_name="baseline_gbm_v1")


def test_train_start_warns_that_the_run_is_not_a_benchmark(con, monkeypatch, caplog):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    days = [
        row[0] for row in con.execute("SELECT DISTINCT date FROM prices ORDER BY date").fetchall()
    ]
    with caplog.at_level(logging.WARNING, logger="twopercent.ab"):
        result = ab.run_ab(
            con,
            arms=_arms(),
            strategy_name="baseline_gbm_v1",
            months=2,
            top_n=5,
            seeds=[42],
            train_start=days[5],
        )
    assert result["train_start"] == days[5].isoformat()
    assert "NOT comparable to a standard-window benchmark" in caplog.text
    assert result["labeled_rows"] < len(features.feature_frame(con))


def test_a_seed_ignoring_strategy_is_called_out(con, monkeypatch, caplog):
    """Averaging over a seed the strategy ignores removes no noise.

    Silently, the paired vectors would look like a 3-seed average and be a
    1-seed run, which understates the noise the pairing is meant to shrink.
    """
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    with caplog.at_level(logging.WARNING, logger="twopercent.ab"):
        result = ab.run_ab(
            con,
            arms=_arms(),
            strategy_name="ab_seed_blind",
            months=2,
            top_n=5,
            seeds=[1, 2, 3],
        )
    assert set(result["seed_ignored_by"]) == {"without", "with"}
    assert "ignored 'random_state'" in caplog.text


def test_coverage_is_reported_for_every_column_under_test(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    result = ab.run_ab(
        con, arms=_arms(), strategy_name="baseline_gbm_v1", months=2, top_n=5, seeds=[42]
    )
    for col in features.INTRADAY_FEATURE_COLUMNS:
        assert 0.0 < result["coverage_all_rows"][col] <= 1.0
        assert 0.0 < result["coverage_min_fold_train"][col] <= 1.0
    report = ab.format_report(result)
    assert "Bonferroni" in report
    assert "nothing here is recorded" in report


def test_paired_test_matches_scipy_and_counts_signs():
    deltas = [0.01, -0.02, 0.03, 0.005, -0.001]
    from scipy import stats

    expected = stats.ttest_1samp(deltas, 0.0)
    result = ab._paired_test(deltas)
    assert result["n"] == 5
    assert result["positive"] == 3
    assert result["negative"] == 2
    assert result["mean"] == pytest.approx(sum(deltas) / 5)
    assert result["t"] == pytest.approx(float(expected.statistic))
    assert result["p_t"] == pytest.approx(float(expected.pvalue))
    assert result["p_sign"] == pytest.approx(1.0)


def test_paired_test_reports_what_it_could_have_detected():
    """A null result is only readable next to the power that produced it.

    #115's answer was wrong because an underpowered null was read as a verdict;
    an effect far below mde_80 means the run did not settle the question.
    """
    from scipy import stats

    small = ab._paired_test([0.001, -0.001] * 6)
    large = ab._paired_test([0.001, -0.001] * 60)
    assert small["alpha"] == pytest.approx(ab.NOMINAL_ALPHA / ab.PRIMARY_TESTS)
    expected = (stats.norm.ppf(1 - small["alpha"] / 2) + stats.norm.ppf(0.80)) * small["se"]
    assert small["mde_80"] == pytest.approx(expected)
    # Ten times the pairs: a smaller true effect becomes detectable.
    assert large["mde_80"] < small["mde_80"]
    # And the run measured nothing anywhere near it.
    assert small["mde_80"] > abs(small["mean"])


def test_result_carries_the_vectors_the_tests_ran_on(con, monkeypatch):
    monkeypatch.setattr(backtest, "MIN_TRAIN_ROWS", 500)
    seed_planted(con)
    result = ab.run_ab(
        con, arms=_arms(), strategy_name="baseline_gbm_v1", months=2, top_n=5, seeds=[42]
    )
    comparison = result["comparisons"]["with"]
    assert len(comparison["auc_deltas_by_fold"]) == comparison["auc_paired_by_fold"]["n"]
    assert (
        len(comparison["precision_deltas_by_day"])
        == comparison["precision_paired_by_day"]["n"]
        == result["test_days"]
    )
    for arm in ("without", "with"):
        assert len(result["arms"][arm]["precision_by_day"]) == result["test_days"]
    recomputed = [
        result["arms"]["with"]["precision_by_day"][day]
        - result["arms"]["without"]["precision_by_day"][day]
        for day in comparison["precision_deltas_by_day"]
    ]
    assert recomputed == pytest.approx(list(comparison["precision_deltas_by_day"].values()))


def test_paired_test_refuses_to_call_a_constant_difference_significant():
    """Zero paired variance is an undefined t, not an infinitely certain one."""
    result = ab._paired_test([0.01] * 8)
    assert result["t"] is None
    assert result["p_t"] is None
    assert result["positive"] == 8
    assert result["p_sign"] == pytest.approx(2 / 2**8)


def test_paired_test_excludes_ties_from_the_sign_test():
    result = ab._paired_test([0.0, 0.0, 0.01, -0.01, 0.02])
    assert result["zeros"] == 2
    assert result["p_sign"] == pytest.approx(1.0)  # 2 of 3, not 2 of 5
