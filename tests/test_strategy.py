"""Exit-rule simulation (strategy.py): every rule at adversarial values, the
band invariants, and — end-to-end through the store — that the fill flag is the
GUARDED touch event (a glitch-suspect high never fills a limit order)."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from twopercent import scan, store, strategy, track


def test_hold_close_is_exactly_oc():
    assert strategy.pick_return_band("hold_close", -0.05, 0.0123, True) == (0.0123, 0.0123)
    assert strategy.pick_return_band("hold_close", -0.05, -0.0456, False) == (-0.0456, -0.0456)


def test_limit_fill_pays_exactly_two_percent_else_close():
    # Filled → exactly +0.02 regardless of how far past 2% the high ran or how
    # the day closed; not filled → the close return, including losses.
    assert strategy.pick_return_band("limit_2pct", -0.002, -0.03, True) == (0.02, 0.02)
    assert strategy.pick_return_band("limit_2pct", -0.002, 0.35, True) == (0.02, 0.02)
    assert strategy.pick_return_band("limit_2pct", -0.002, -0.0117, False) == (-0.0117, -0.0117)


def test_stop_boundary_epsilon_at_adversarial_open():
    # open=5.00, low=4.95 is a true −1% low but lands ABOVE −0.01 in double math
    # — the epsilon guard must still trigger the stop (CLAUDE.md: adversarial
    # values, not round ones).
    ol = (4.95 - 5.00) / 5.00
    assert ol > strategy.STOP_LEVEL  # documents the FP landing this guards
    assert strategy.stop_triggered(ol)
    assert strategy.pick_return_band("limit_stop", ol, 0.004, False) == (-0.01, -0.01)
    # A low clearly above the stop does NOT trigger.
    assert not strategy.stop_triggered(-0.0099)
    assert strategy.pick_return_band("limit_stop", -0.0099, 0.004, False) == (0.004, 0.004)


def test_limit_stop_band_cases():
    # Both triggered → the honest BAND: worst = stop-first (−1%), best =
    # limit-first (+2%). Only stop → −1% both. Only limit → +2% both.
    # Neither → the close, both.
    assert strategy.pick_return_band("limit_stop", -0.015, 0.001, True) == (-0.01, 0.02)
    assert strategy.pick_return_band("limit_stop", -0.015, 0.001, False) == (-0.01, -0.01)
    assert strategy.pick_return_band("limit_stop", -0.0009, 0.0203, True) == (0.02, 0.02)
    assert strategy.pick_return_band("limit_stop", -0.0009, 0.0123, False) == (0.0123, 0.0123)


def test_band_worst_never_exceeds_best_across_a_value_sweep():
    ols = [-0.1, -0.015, -0.0100000001, -0.01, -0.0099, -0.0009, 0.0]
    ocs = [-0.08, -0.0117, 0.0, 0.0199, 0.02, 0.0203, 0.19]
    trails = [None, -0.09, -0.01, 0.0, 0.02, 0.31]
    for strat in strategy.PNL_STRATEGIES:
        for ol in ols:
            for oc in ocs:
                for filled in (False, True):
                    for trail in trails:
                        if strat == "trailing" and trail is None:
                            continue  # covered by its own must-raise test
                        worst, best = strategy.pick_return_band(
                            strat, ol, oc, filled, None, None, trail
                        )
                        assert worst <= best, (strat, ol, oc, filled, trail)


def test_unknown_strategy_dies_loudly():
    with pytest.raises(ValueError, match="unknown exit-rule strategy"):
        strategy.pick_return_band("reach", -0.01, 0.0, True)
    with pytest.raises(ValueError, match="unknown exit-rule strategy"):
        strategy.pick_return_band("nonsense", -0.01, 0.0, True)


def test_trailing_without_an_intraday_replay_refuses_rather_than_guessing():
    """A path-dependent rule cannot be evaluated on a path with holes in it."""
    with pytest.raises(ValueError, match="must be excluded"):
        strategy.pick_return_band("trailing", -0.01, 0.0, True)

    # And the day is EXCLUDED from the window, not silently averaged around.
    days = [
        {"d": "a", "picks": [[1, 0.03, -0.02, 0.01, 1, None, None, -0.004]]},
        {"d": "b", "picks": [[1, 0.03, -0.02, 0.01, 1, None, None, None]]},
    ]
    s = strategy.summarize_strategy_days(days, 1, "trailing")
    assert s["excluded"] == 1 and s["clean"] == 1
    assert abs(s["gw"] - 0.996) < 1e-9  # only day "a" compounded


def _bar(symbol, open_, high, low, close, volume=1_000_000, date=dt.date(2026, 1, 5)):
    return {
        "symbol": symbol,
        "date": date,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close,
        "volume": volume,
    }


def test_fill_flag_is_the_guarded_touch_event_end_to_end(con):
    """A +2% touch exactly at the threshold (FP-adversarial open) FILLS; a
    glitch-suspect spike does NOT (a fake print can't fill your order). Runs the
    real path: prices → daily_returns → logged prediction → daily_rank_outcomes
    → summarize_strategy_days — the LIVE row's exact pipeline."""
    # 21 flat baseline bars give GLTCH's guard its prev_close/volume reference.
    dates = pd.bdate_range("2026-01-05", periods=22)
    rows = []
    for d in dates[:-1]:
        rows.append(_bar("BND", 5.00, 5.01, 4.99, 5.00, date=d.date()))
        rows.append(_bar("GLTCH", 100.0, 100.1, 99.9, 100.0, date=d.date()))
    target = dates[-1].date()
    # BND: high exactly +2.0% off a 5.00 open — lands a few ULPs below 0.02.
    rows.append(_bar("BND", 5.00, 5.10, 4.99, 5.01, date=target))
    # GLTCH: isolated +20% high, close unconfirmed, volume collapsed → glitch.
    rows.append(_bar("GLTCH", 100.0, 120.0, 99.0, 100.5, volume=100_000, date=target))
    store.upsert_prices(con, pd.DataFrame(rows))

    signal = dates[-2].date()
    store.save_predictions(
        con,
        "s",
        signal,
        pd.DataFrame({"symbol": ["BND", "GLTCH"], "prob": [0.9, 0.8], "rank": [1, 2]}),
        event=scan.TOUCH_EVENT,
    )
    frame = track.daily_rank_outcomes(con, "s", top_n=2)
    frame = frame[[pd.Timestamp(d).date() == target for d in frame["target_date"]]]
    by_rank = frame.set_index("rank")
    assert by_rank.loc[1, "hit"] == 1  # exact-boundary touch counts (epsilon)
    assert by_rank.loc[2, "hit"] == 0  # glitch-suspect high is NOT a fill

    picks = [
        [
            int(r.Index),
            round(float(r.high_return), 6),
            round(float(r.low_return), 6),
            round(float(r.oc_return), 6),
            int(r.hit),
        ]
        for r in by_rank.itertuples()
    ]
    s = strategy.summarize_strategy_days([{"d": str(target), "picks": picks}], 2, "limit_2pct")
    # BND fills at exactly +2%; GLTCH must fall back to its close (+0.5%) — a
    # fill there would pay the fake print. Day return = (0.02 + 0.005) / 2.
    assert abs(s["gw"] - (1 + (0.02 + 0.005) / 2)) < 1e-12
    assert s["gw"] == s["gb"]


def test_summarize_missing_and_corrupt_days_never_average_around():
    good = {"d": "a", "picks": [[1, 0.021, -0.002, 0.005, 1]]}
    missing = {"d": "b", "picks": [[1, None, None, None, 0]]}  # pre-upgrade row
    s = strategy.summarize_strategy_days([good, missing], 1, "hold_close")
    assert s["missing"] == 1 and s["clean"] == 1
    # An empty-picks day is corrupt, mirroring the reach summarizer.
    s2 = strategy.summarize_strategy_days([good, {"d": "c", "picks": []}], 1, "hold_close")
    assert s2["corrupt"] == 1 and s2["clean"] == 1
    # A NaN outcome value (impossible via JSON, guarded anyway): the guard is on
    # the band OUTPUTS — a NaN close under hold_close propagates and the day is
    # corrupt, never averaged around.
    nan_day = {"d": "n", "picks": [[1, 0.021, -0.002, float("nan"), 1]]}
    s3 = strategy.summarize_strategy_days([good, nan_day], 1, "hold_close")
    assert s3["corrupt"] == 1 and s3["clean"] == 1


def test_partially_corrupt_day_leaks_no_ghost_wins():
    """A day whose FIRST pick is a winner but whose second pick is NaN must
    contribute zero to the win-rate numerator: wins are committed only after
    the whole day validates, or the rate inflates (numerator without its
    denominator — reviewer finding on PR #77)."""
    good = {"d": "a", "picks": [[1, 0.001, -0.002, -0.004, 0]]}  # clean losing day
    partial = {
        "d": "p",
        "picks": [
            [1, 0.03, -0.001, 0.03, 1],  # winning pick, tallied first...
            [2, 0.02, -0.002, float("nan"), 1],  # ...then the day dies as corrupt
        ],
    }
    s = strategy.summarize_strategy_days([good, partial], 2, "hold_close")
    assert s["corrupt"] == 1 and s["clean"] == 1
    assert s["picks"] == 1  # only the clean day's pick counts
    assert s["ww"] == 0.0 and s["wb"] == 0.0  # the ghost win never leaked


def test_dropdown_choices_shape():
    keys = [k for k, _label, _enabled in strategy.STRATEGY_CHOICES]
    assert keys[0] == "reach"  # prediction quality stays the default view
    disabled = {k for k, _label, enabled in strategy.STRATEGY_CHOICES if not enabled}
    # Trailing is WITHDRAWN: the replay is real but its number carries three
    # measured, all-flattering biases (see the STRATEGY_CHOICES comment). The
    # rule stays in PNL_STRATEGIES so its math and lockstep keep being tested;
    # only the dropdown entry is greyed out.
    assert disabled == {"trailing"}
    assert set(strategy.PNL_STRATEGIES) == set(keys) - {"reach"}
