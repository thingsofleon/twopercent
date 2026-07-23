"""Tier-1 auto-search generator: bounded, deterministic, constructible configs."""

from twopercent import generate, strategies
from twopercent.strategies.xgb_gbm import ALLOWED_PARAMS


def test_family_is_bounded():
    # The import-time assert is the real guard; pin it here too so a future
    # grid widening trips a named test, not a surprise ImportError.
    assert generate.family_size() <= generate.MAX_AUTO_FAMILY
    assert generate.family_size() == len(generate.grid_configs())


def test_grid_is_deterministic():
    assert generate.grid_configs() == generate.grid_configs()


def test_grid_configs_are_unique_by_canonical_key():
    # canonical_params (5.0 == 5, order-independent) is the ledger's identity;
    # duplicates would waste GPU re-running the same trial under two labels.
    from twopercent.research import canonical_params

    keys = [(c["strategy"], canonical_params(c["params"])) for c in generate.grid_configs()]
    assert len(keys) == len(set(keys))


def test_every_config_is_registered_and_annotated():
    registered = set(strategies.names())
    for cfg in generate.grid_configs():
        assert cfg["strategy"] in registered
        assert cfg["params"], "empty params would just re-run the default config"
        assert cfg["note"].startswith("auto-search")


def test_generated_params_are_valid_for_their_strategy():
    for cfg in generate.grid_configs():
        if cfg["strategy"] == "xgb_gbm_v1":
            # xgb rejects unknown kwargs loudly — every generated key must be allowed.
            assert set(cfg["params"]) <= ALLOWED_PARAMS, cfg
        elif cfg["strategy"] == "baseline_gbm_v1":
            # HistGradientBoosting: constructing proves the kwargs are accepted
            # (CPU-only, safe in CI — no CUDA touched).
            strategies.get("baseline_gbm_v1", **cfg["params"])
