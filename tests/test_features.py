import datetime as dt

import numpy as np
import pandas as pd

from tests.conftest import seed_history
from twopercent import store
from twopercent.features import FEATURE_COLUMNS, feature_frame


def _varied(n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(rng.uniform(-0.03, 0.04, n))


def _seed_universe(con, sectors: dict[str, str]) -> None:
    df = pd.DataFrame(
        {
            "symbol": list(sectors),
            "name": [f"{s} Inc" for s in sectors],
            "market_cap": [1e9 * (i + 1) for i in range(len(sectors))],
            "sector": list(sectors.values()),
        }
    )
    store.upsert_universe(con, df, as_of=dt.date(2026, 7, 17))


def test_lookahead_canary(con):
    """Mutating every bar AFTER signal_date S must not change S's features.

    This is the executable form of the no-lookahead invariant (ROADMAP.md).
    Covers the sector features too: both symbols share a sector, so
    sector_breadth/sector_excess are live values, not incidental NaNs.
    """
    seed_history(con, {"AAA": _varied(60, 1), "BBB": _varied(60, 2)})
    _seed_universe(con, {"AAA": "Technology", "BBB": "Technology"})
    before = feature_frame(con)
    dates = sorted(before["signal_date"].unique())
    cutoff = dates[len(dates) // 2]
    vec_before = before[before["signal_date"] == cutoff].set_index("symbol")[FEATURE_COLUMNS]
    # The canary must actually exercise the sector features, not compare NaN to NaN.
    assert vec_before["sector_breadth"].notna().all()
    assert vec_before["sector_excess"].notna().all()

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


def test_sector_math_hand_checked(con):
    base = [0.0] * 24
    seed_history(
        con,
        {
            "AAA": base + [0.03],
            "BBB": base + [-0.01],
            "CCC": base + [0.01],
            "DDD": base + [0.02],
        },
    )
    _seed_universe(con, {"AAA": "Tech", "BBB": "Tech", "CCC": "Tech", "DDD": ""})
    frame = feature_frame(con)
    last = frame[frame["signal_date"] == frame["signal_date"].max()].set_index("symbol")

    # Tech on the last day: returns +3%, -1%, +1% → breadth 2/3, mean +1%.
    assert abs(last.loc["AAA", "sector_breadth"] - 2 / 3) < 1e-12
    assert abs(last.loc["AAA", "sector_excess"] - (0.03 - 0.01)) < 1e-9
    assert abs(last.loc["BBB", "sector_excess"] - (-0.01 - 0.01)) < 1e-9
    assert abs(last.loc["CCC", "sector_excess"] - (0.01 - 0.01)) < 1e-9

    # Empty sector → NaN for both, but the row itself is KEPT.
    assert "DDD" in last.index
    assert pd.isna(last.loc["DDD", "sector_breadth"])
    assert pd.isna(last.loc["DDD", "sector_excess"])

    # Earlier flat days: nobody in Tech was positive → breadth 0, excess 0.
    first = frame[frame["signal_date"] == frame["signal_date"].min()].set_index("symbol")
    assert first.loc["AAA", "sector_breadth"] == 0.0
    assert abs(first.loc["AAA", "sector_excess"]) < 1e-12


def test_sector_features_nan_when_no_universe(con, caplog):
    # Prices without any universe snapshot: sector features are NaN, rows kept,
    # and the total absence of sector data is warned about loudly.
    seed_history(con, {"AAA": _varied(30, 4)})
    frame = feature_frame(con)
    assert not frame.empty
    assert frame["sector_breadth"].isna().all()
    assert frame["sector_excess"].isna().all()
    assert "no sector data" in caplog.text


def test_thin_history_dropped_loudly(con, caplog):
    seed_history(con, {"NEW": _varied(10, 3)})  # under MIN_HISTORY_DAYS
    frame = feature_frame(con)
    assert frame.empty
    assert "dropped" in caplog.text
