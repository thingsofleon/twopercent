import datetime as dt

import numpy as np
import pandas as pd

from tests.conftest import seed_history
from twopercent import store
from twopercent.features import FEATURE_COLUMNS, METADATA_COLUMNS, feature_frame


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
    watched = FEATURE_COLUMNS + METADATA_COLUMNS  # metadata must be trailing-only too
    seed_history(con, {"AAA": _varied(60, 1), "BBB": _varied(60, 2)})
    _seed_universe(con, {"AAA": "Technology", "BBB": "Technology"})
    before = feature_frame(con)
    dates = sorted(before["signal_date"].unique())
    cutoff = dates[len(dates) // 2]
    vec_before = before[before["signal_date"] == cutoff].set_index("symbol")[watched]
    # The canary must actually exercise the sector features, not compare NaN to NaN.
    assert vec_before["sector_breadth"].notna().all()
    assert vec_before["sector_excess"].notna().all()

    # Mutating `high` on FUTURE bars is load-bearing now that the label and
    # cnt_2pct_20d are TOUCH events (open-to-high): a refactor that dropped `high`
    # from this UPDATE would stop exercising the touch path and the canary would
    # pass blind. Keep `high` here, and assert the touch feature is future-invariant.
    # EVERY price column is mutated. open/low arrived with #110 (range_20d reads
    # low, gap_prior reads open); adj_close arrived because it was the last one
    # left, and a feature reading LEAD(adj_close) was demonstrated to pass this
    # canary GREEN while leaking. A column absent from this UPDATE makes the
    # canary vacuous for anything reading it — it runs, passes, and proves
    # nothing. Mutate the whole table, not the columns today's features happen
    # to use.
    # DISTINCT multipliers, deliberately. Scaling open and high by the SAME
    # factor leaves (high - open) / open unchanged, so the touch event -- and
    # therefore the label -- would be invariant and the "label must change"
    # assertion below would fail; scaling them differently perturbs every ratio
    # these features read. The factors keep OHLC ordering valid (2*open <= 3*high
    # and 1.5*low <= 2*open for any valid bar), so daily_returns does not filter
    # the mutated rows out and leave the canary comparing nothing.
    con.execute(
        "UPDATE prices SET open = open * 2, high = high * 3, low = low * 1.5, "
        "close = close * 3, volume = volume * 7, adj_close = adj_close * 4 "
        "WHERE date > ?",
        [cutoff],
    )
    after = feature_frame(con)
    vec_after = after[after["signal_date"] == cutoff].set_index("symbol")[watched]

    assert vec_before.equals(vec_after)  # features untouched by the future
    # Explicit: the touch-count feature at the cutoff row uses only bars through
    # the cutoff, so tripling every future high must not move it (no lookahead).
    assert vec_before["cnt_2pct_20d"].equals(vec_after["cnt_2pct_20d"])
    # Explicit for the columns this UPDATE newly covers: tripling every future
    # open and low must not move the features that read them (#110).
    for col in ("range_20d", "gap_prior", "high_return_mean_20d", "days_since_2pct"):
        # Guarded like the sector features above: a column that is all-NaN would
        # make the equality below compare nothing and pass vacuously.
        assert vec_before[col].notna().any(), col
        assert vec_before[col].equals(vec_after[col]), col
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


def test_partial_sector_coverage_blank_sector_warns_with_counts(con, caplog):
    # 25 seeded days − 19 thin-history days = 6 feature rows per symbol.
    seed_history(con, {"AAA": _varied(25, 5), "BBB": _varied(25, 6), "CCC": _varied(25, 7)})
    _seed_universe(con, {"AAA": "Tech", "BBB": "Tech", "CCC": ""})
    frame = feature_frame(con)
    assert frame.loc[frame["symbol"] == "AAA", "sector_breadth"].notna().all()  # covered stay
    nan_rows = frame[frame["sector_breadth"].isna()]
    assert set(nan_rows["symbol"]) == {"CCC"} and len(nan_rows) == 6
    assert "6 feature rows across 1 symbols have NaN sector features" in caplog.text


def test_partial_sector_coverage_missing_symbols_warn_with_counts(con, caplog):
    seed_history(con, {"AAA": _varied(25, 5), "BBB": _varied(25, 6), "CCC": _varied(25, 7)})
    _seed_universe(con, {"AAA": "Tech"})  # BBB and CCC absent from the universe
    frame = feature_frame(con)
    nan_rows = frame[frame["sector_breadth"].isna()]
    assert set(nan_rows["symbol"]) == {"BBB", "CCC"}
    assert "12 feature rows across 2 symbols have NaN sector features" in caplog.text


def test_sector_features_nan_when_no_universe(con, caplog):
    # Prices without any universe snapshot: sector features are NaN, rows kept,
    # and the total absence of sector data is warned about loudly.
    seed_history(con, {"AAA": _varied(30, 4)})
    frame = feature_frame(con)
    assert not frame.empty
    assert frame["sector_breadth"].isna().all()
    assert frame["sector_excess"].isna().all()
    assert "no sector data" in caplog.text


def test_median_vol_20_is_trailing_metadata_not_a_feature(con):
    # Models must never train on it: it is metadata, not a feature.
    assert "median_vol_20" in METADATA_COLUMNS
    assert "median_vol_20" not in FEATURE_COLUMNS

    seed_history(con, {"AAA": _varied(30, 8)}, vary_volume=True)
    frame = feature_frame(con).set_index("signal_date")
    volumes = con.execute("SELECT volume FROM prices WHERE symbol = 'AAA' ORDER BY date").df()[
        "volume"
    ]
    rolling = volumes.rolling(20).median()
    # Each signal date's value is the median of the 20 bars ENDING there —
    # trailing by construction (checked mid-history, not just at the frame end).
    idx = sorted(frame.index)  # frame rows start at the 20th bar (thin history dropped)
    for pos, bar in ((0, 19), (5, 24), (10, 29)):
        assert frame.loc[idx[pos], "median_vol_20"] == rolling.iloc[bar]


def test_outcome_return_columns_are_label_side_never_features(con):
    """Quant-skeptic guardrail for the strategy explorer: oh/ol are OUTCOME
    quantities. The lookahead canary deliberately excludes label columns, so
    their absence from the trainable sets is pinned EXPLICITLY (the
    median_vol_20 pattern) — a refactor that promoted next_high_return or
    next_low_return into FEATURE_COLUMNS would be lookahead the canary cannot
    see."""
    for col in ("high_return", "low_return", "next_high_return", "next_low_return"):
        assert col not in FEATURE_COLUMNS, col
        assert col not in METADATA_COLUMNS, col

    # They ARE in the frame, as label-side values: hand-check that each equals
    # the NEXT day's open-to-high / open-to-low return (open is always 100.0,
    # conftest seeds high = max(open, close) * 1.001 and low = min * 0.999).
    seed_history(con, {"AAA": [0.03, 0.01] * 13})
    frame = feature_frame(con).set_index("signal_date")
    dates = sorted(frame.index)
    row = frame.loc[dates[-3]]  # its target day (bar 24, even index) closed +3%
    next_close = 100.0 * 1.03
    assert abs(row["next_high_return"] - (next_close * 1.001 - 100.0) / 100.0) < 1e-12
    assert abs(row["next_low_return"] - (100.0 * 0.999 - 100.0) / 100.0) < 1e-12
    assert abs(row["next_oc_return"] - 0.03) < 1e-12


def test_thin_history_dropped_loudly(con, caplog):
    seed_history(con, {"NEW": _varied(10, 3)})  # under MIN_HISTORY_DAYS
    frame = feature_frame(con)
    assert frame.empty
    assert "dropped" in caplog.text


def test_new_features_are_trailing_only_at_an_adversarial_boundary(con):
    """#110. gap_prior is the lookahead trap in this batch.

    It must read the SIGNAL day's open against the PRIOR close — never the
    TARGET day's open, which is unknown at the pre-open prediction moment and
    is precisely what a careless "gap" feature reaches for.
    """
    seed_history(con, {"AAA": _varied(40, 1), "BBB": _varied(40, 2)})
    _seed_universe(con, {"AAA": "Technology", "BBB": "Technology"})
    frame = feature_frame(con)
    mine = frame[frame["symbol"] == "AAA"].reset_index(drop=True)
    row = mine.iloc[len(mine) // 2]  # middle row: a next bar is guaranteed
    day = row["signal_date"]

    bars = con.execute("SELECT date, open, close FROM prices WHERE symbol='AAA' ORDER BY date").df()
    bars["date"] = pd.to_datetime(bars["date"])
    i = int(bars.index[bars["date"] == pd.Timestamp(day)][0])
    assert 0 < i < len(bars) - 1
    expected = (bars.loc[i, "open"] - bars.loc[i - 1, "close"]) / bars.loc[i - 1, "close"]

    assert abs(row["gap_prior"] - expected) < 1e-12
    # The target day's open would give a DIFFERENT number — proving the test
    # discriminates rather than passing on a coincidence.
    wrong = (bars.loc[i + 1, "open"] - bars.loc[i, "close"]) / bars.loc[i, "close"]
    assert abs(expected - wrong) > 1e-9


def test_feature_set_version_changes_with_the_columns():
    """A recorded benchmark is only comparable to another over the SAME
    features; the research done-ledger keys on this (#78)."""
    import twopercent.features as F

    base = F.feature_set_version()
    assert base == F.feature_set_version()  # stable

    original = F.FEATURE_COLUMNS
    try:
        F.FEATURE_COLUMNS = [*original, "a_new_feature"]
        assert F.feature_set_version() != base
        # Reordering alone must NOT trigger a pointless full re-sweep.
        F.FEATURE_COLUMNS = list(reversed(original))
        assert F.feature_set_version() == base
    finally:
        F.FEATURE_COLUMNS = original
