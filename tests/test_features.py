import numpy as np
import pandas as pd

from tests.conftest import seed_history
from twopercent.features import FEATURE_COLUMNS, feature_frame


def _varied(n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(rng.uniform(-0.03, 0.04, n))


def test_lookahead_canary(con):
    """Mutating every bar AFTER signal_date S must not change S's features.

    This is the executable form of the no-lookahead invariant (ROADMAP.md).
    """
    seed_history(con, {"AAA": _varied(60, 1), "BBB": _varied(60, 2)})
    before = feature_frame(con)
    dates = sorted(before["signal_date"].unique())
    cutoff = dates[len(dates) // 2]
    vec_before = before[before["signal_date"] == cutoff].set_index("symbol")[FEATURE_COLUMNS]

    con.execute(
        "UPDATE prices SET close = close * 3, high = high * 3, volume = volume * 7 WHERE date > ?",
        [cutoff],
    )
    after = feature_frame(con)
    vec_after = after[after["signal_date"] == cutoff].set_index("symbol")[FEATURE_COLUMNS]

    assert vec_before.equals(vec_after)  # features untouched by the future
    # ...while the label DID change (it is the future, explicitly):
    lbl_b = before[before["signal_date"] == cutoff].set_index("symbol")["did_2pct_next"]
    lbl_a = after[after["signal_date"] == cutoff].set_index("symbol")["did_2pct_next"]
    assert not lbl_b.equals(lbl_a)


def test_label_and_target_date_are_next_day(con):
    seed_history(con, {"AAA": [0.0] * 24 + [0.01, 0.03]})
    frame = feature_frame(con)
    # Signal row whose NEXT day moved +3%: label 1; the +1% day itself: label 0.
    frame = frame.set_index("signal_date")
    dates = sorted(frame.index)
    assert frame.loc[dates[-2], "did_2pct_next"] == 1  # next day is the +3%
    assert pd.isna(frame.loc[dates[-1], "did_2pct_next"])  # newest row has no future yet


def test_feature_math_hand_checked(con):
    ocs = [0.03] * 25  # constant +3% days
    seed_history(con, {"AAA": ocs})
    frame = feature_frame(con)
    row = frame.iloc[-1]
    assert (
        row["oc_return_today"] == np.float64(0.03).item()
        or abs(row["oc_return_today"] - 0.03) < 1e-12
    )
    assert row["cnt_2pct_20d"] == 20  # every day in the 20-day window was a 2% day
    assert abs(row["vol_20d"]) < 1e-12  # constant returns → zero volatility
    assert abs(row["volume_ratio"] - 1.0) < 1e-12  # constant volume
    assert row["breadth"] == 1.0 and row["market_heat"] == 1.0


def test_thin_history_dropped_loudly(con, caplog):
    seed_history(con, {"NEW": _varied(10, 3)})  # under MIN_HISTORY_DAYS
    frame = feature_frame(con)
    assert frame.empty
    assert "dropped" in caplog.text
