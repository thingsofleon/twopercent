import numpy as np

from twopercent import calibration


def test_bin_indices_are_epsilon_guarded_at_edges():
    # Adversarial numeric bins (CLAUDE.md): a round-number bin test is a false
    # all-clear. Values exactly on a 0.1 edge must land in the bucket they name,
    # not one below through float representation error (0.3*10 == 2.9999...96).
    probs = np.array([0.0, 0.1, 0.3, 0.7, 0.9, 1.0, 0.29999999])
    idx = calibration._bin_indices(probs, 10)
    assert idx.tolist() == [0, 1, 3, 7, 9, 9, 2]  # 1.0 clips into the last bucket


def test_reliability_perfect_model_is_a_diagonal_with_bss_one():
    # A perfect model: predicts 1.0 for reachers, 0.0 for non-reachers. Reliability
    # is exactly the diagonal (slope 1) and Brier is 0, so BSS is exactly 1.
    y = np.array([1] * 30 + [0] * 70)
    p = y.astype(float)
    d = np.array(["2026-06-01"] * 100)  # one day, base rate 0.30

    table = calibration.reliability_table(p, y, 10)
    top, bottom = table[9], table[0]
    assert top["count"] == 30 and top["mean_pred"] == 1.0 and top["reach_rate"] == 1.0
    assert bottom["count"] == 70 and bottom["mean_pred"] == 0.0 and bottom["reach_rate"] == 0.0
    assert all(table[b]["count"] == 0 for b in range(1, 9))  # nothing in between

    assert calibration.calibration_slope(table) == 1.0
    skill = calibration.brier_skill_score(p, y, d)
    assert skill["brier"] == 0.0
    assert skill["bss"] == 1.0


def test_constant_half_prediction_on_low_base_has_nonpositive_bss():
    # Predict 0.5 for everything on a 30% base rate: the probability adds no skill
    # over the base rate, so BSS must be ~0 or negative. Hand math: brier_model =
    # 0.25, brier_ref (base 0.3) = 0.3*0.7^2 + 0.7*0.3^2 = 0.21, BSS = 1 - 0.25/0.21.
    y = np.array([1] * 30 + [0] * 70)
    p = np.full(100, 0.5)
    d = np.array(["2026-06-01"] * 100)
    skill = calibration.brier_skill_score(p, y, d)
    assert abs(skill["brier"] - 0.25) < 1e-9
    assert abs(skill["brier_ref"] - 0.21) < 1e-9
    assert skill["bss"] < 0  # 1 - 0.25/0.21 ≈ -0.19


def test_bss_reference_is_the_daily_base_rate_not_the_global_one():
    # THE pin for the reference choice. Two days: base 0.2 and base 0.8; the model
    # predicts each day's own base rate. Against the DAILY reference the model IS
    # the reference, so BSS == 0. Against the (wrong) GLOBAL base rate of 0.5 it
    # would look skilful (BSS ≈ +0.36). The result must be 0, proving daily.
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    p = np.array([0.2] * 10 + [0.8] * 10)
    d = np.array(["A"] * 10 + ["B"] * 10)
    skill = calibration.brier_skill_score(p, y, d)
    assert abs(skill["brier"] - 0.16) < 1e-9
    assert abs(skill["brier_ref"] - 0.16) < 1e-9  # daily reference == model here
    assert abs(skill["bss"] - 0.0) < 1e-9
    # Guard against a regression to the global-base-rate reference (would be +0.36).
    assert skill["bss"] != 0.36


def test_regime_reports_stratify_days_by_base_rate():
    # Three clearly separated base-rate regimes (low ~0.1, medium ~0.5, high ~0.9),
    # three days each. Regime conditioning must produce all three strata, ordered,
    # each spanning its own base-rate band — a pooled table cannot hide a regime.
    rng = np.random.default_rng(0)
    probs, labels, dates = [], [], []
    for regime, base in (("lo", 0.1), ("md", 0.5), ("hi", 0.9)):
        for day in range(3):
            n = 50
            y = (rng.random(n) < base).astype(int)
            probs.extend(y * 0.0 + base)  # predict the regime's base rate
            labels.extend(y)
            dates.extend([f"{regime}{day}"] * n)
    reports = calibration.regime_reports(np.array(probs), np.array(labels), np.array(dates), 10)
    names = [r["name"] for r in reports]
    assert names == ["low", "medium", "high"]
    assert reports[0]["base_hi"] < reports[2]["base_lo"]  # bands don't overlap
    assert all(r["n_days"] == 3 for r in reports)


def test_calibration_report_shape_and_population_label():
    y = np.array([1] * 30 + [0] * 70)
    p = np.clip(y * 0.9 + 0.05, 0, 1)
    d = np.array(["2026-06-01"] * 60 + ["2026-06-02"] * 40)
    report = calibration.calibration_report(p, y, d, 10)
    assert report["population"] == "all_names_test"
    assert report["n"] == 100
    assert report["n_days"] == 2
    assert report["n_bins"] == 10
    assert len(report["reliability"]) == 10
    assert report["bss"] is not None
    # No NaN survives into the recorded JSON — empty buckets carry None, not 0.
    empties = [r for r in report["reliability"] if r["count"] == 0]
    assert all(r["mean_pred"] is None and r["reach_rate"] is None for r in empties)


def test_slope_none_without_spread():
    # A single occupied bucket has no predicted-prob spread — slope is undefined
    # and must be None, never a fabricated number.
    y = np.array([1, 0, 1, 0])
    p = np.array([0.55, 0.55, 0.55, 0.55])
    table = calibration.reliability_table(p, y, 10)
    assert calibration.calibration_slope(table) is None
