import datetime as dt

import numpy as np
import pandas as pd

from tests.conftest import seed_history, seed_intraday
from twopercent import store
from twopercent.features import (
    FEATURE_COLUMNS,
    INTRADAY_CLOSE_HOUR,
    INTRADAY_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    feature_frame,
)


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
    # INTRADAY_FEATURE_COLUMNS are watched even though they are NOT model inputs:
    # they are computed, they are the most leak-prone columns in the frame, and
    # keying this list on FEATURE_COLUMNS alone would have silently dropped them
    # from the canary the moment they were held back — making four rounds of
    # work on exactly this check vacuous.
    watched = FEATURE_COLUMNS + METADATA_COLUMNS + INTRADAY_FEATURE_COLUMNS
    seed_history(con, {"AAA": _varied(60, 1), "BBB": _varied(60, 2)})
    _seed_universe(con, {"AAA": "Technology", "BBB": "Technology"})
    # Intraday bars for EVERY seeded day, or the phase-2 features are all-NaN
    # and the equality below compares nothing — the canary would pass while
    # proving nothing about the newest and most leak-prone columns.
    all_days = [
        r[0] for r in con.execute("SELECT DISTINCT date FROM prices ORDER BY date").fetchall()
    ]
    seed_intraday(con, ["AAA", "BBB"], all_days)
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
    # intraday_prices too, as of #79 phase 2. Its own module docstring warned
    # that a leak from this table "would pass it vacuously" because the canary
    # mutated only `prices` — that warning is now discharged rather than
    # restated. Distinct multipliers again, for the same reason as above: equal
    # scaling leaves the intraday ratios these features read unchanged.
    con.execute(
        # Volume is scaled NON-UNIFORMLY (per bar) on purpose. A single
        # multiplier cancels in any volume RATIO: close_volume_share is
        # last_bar_volume / session_volume, so volume*7 leaves it identical and
        # a leak in that column passes undetected. Measured, not assumed — a
        # leaking close_volume_share read 0.175824 before AND after a uniform
        # scaling. Making the factor depend on the bar breaks the invariance.
        "UPDATE intraday_prices SET open = open * 2, high = high * 3, "
        "low = low * 1.5, close = close * 3, "
        "volume = volume * (3 + (EXTRACT(hour FROM ts) % 5)) WHERE date > ?",
        [cutoff],
    )
    # Mutating VALUES cannot detect a leak of session STRUCTURE. `bars` and
    # `has_close_bar` are selected columns in the intraday CTE, one keystroke
    # from a feature expression, and a feature reading them off the TARGET day
    # would sail through the update above unchanged (7.0 stays 7.0). Such a leak
    # would be genuinely predictive too — a complete session tracks liquidity
    # and activity, which track the outcome. So the future is also RESHAPED.
    con.execute(
        "DELETE FROM intraday_prices WHERE date > ? AND EXTRACT(hour FROM ts) = ?",
        [cutoff, INTRADAY_CLOSE_HOUR],
    )
    after = feature_frame(con)
    vec_after = after[after["signal_date"] == cutoff].set_index("symbol")[watched]

    assert vec_before.equals(vec_after)  # features untouched by the future
    # Explicit: the touch-count feature at the cutoff row uses only bars through
    # the cutoff, so tripling every future high must not move it (no lookahead).
    assert vec_before["cnt_2pct_20d"].equals(vec_after["cnt_2pct_20d"])
    # Explicit for the columns this UPDATE newly covers: tripling every future
    # open and low must not move the features that read them (#110).
    for col in (
        "range_20d",
        "gap_prior",
        "high_return_mean_20d",
        "days_since_2pct",
        # Phase 2 (#79): these read intraday_prices, the table the canary did
        # not mutate until now.
        "close_vwap_gap",
        "last_hour_drift",
        "intraday_vol",
        "close_volume_share",
    ):
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


def test_feature_set_version_of_the_shipped_set_is_pinned():
    """The 56 recorded research configs are keyed on this exact string.

    Changing FEATURE_COLUMNS is allowed and invalidates them on purpose; a
    refactor that changes the FINGERPRINT without changing the columns would
    invalidate them by accident, which is the same ledger loss with none of the
    intent. #115 held the four intraday features out precisely to keep it.
    """
    import twopercent.features as F

    assert F.feature_set_version() == "24bd854eae74"
    assert F.feature_set_version(F.FEATURE_COLUMNS) == F.feature_set_version()
    assert F.feature_set_version(F.FEATURE_COLUMNS + F.INTRADAY_FEATURE_COLUMNS) == "67f45c5ed724"


def _one_intraday_session(con, symbol, day, hours, *, interval="1h"):
    """Insert a session with EXACTLY the given hours, so the gate can be probed."""
    rows = pd.DataFrame(
        {
            "symbol": symbol,
            "ts": [dt.datetime.combine(day, dt.time(h, 30)) for h in hours],
            "date": day,
            "interval": interval,
            "open": [100.0 + h for h in hours],
            "high": [101.0 + h for h in hours],
            "low": [99.0 + h for h in hours],
            "close": [100.5 + h for h in hours],
            "volume": [1_000 * (h + 1) for h in hours],
        }
    )
    con.register("_sess", rows)
    con.execute("INSERT INTO intraday_prices SELECT * FROM _sess")
    con.unregister("_sess")


def test_intraday_gate_requires_the_closing_bar_not_merely_a_bar_count(con):
    """A session can clear MIN_INTRADAY_BARS and still have no 15:30 bar.

    1,948 real sessions do. For those, `last(close ORDER BY ts)` is the 13:30 or
    14:30 close, so last_hour_drift is not the last hour and close_volume_share
    divides by the wrong numerator -- precisely the fabricated shape the floor
    exists to prevent. The count is necessary, not sufficient.
    """
    seed_history(con, {"AAA": [0.01] * 30, "BBB": [0.01] * 30}, vary_volume=True)
    days = [r[0] for r in con.execute("SELECT DISTINCT date FROM prices ORDER BY date").fetchall()]
    day = days[-2]
    # Six bars — comfortably over the floor — but the session stops at 14:30.
    _one_intraday_session(con, "AAA", day, [9, 10, 11, 12, 13, 14])
    # A complete session — the only kind that yields features.
    _one_intraday_session(con, "BBB", day, [9, 10, 11, 12, 13, 14, 15])

    frame = feature_frame(con)
    # signal_date comes back as datetime64 while `day` is a datetime.date --
    # comparing the two directly matches NOTHING silently (the dtype trap this
    # project has hit three times). Normalise before selecting.
    sig = pd.to_datetime(frame["signal_date"]).dt.date
    cols = ["close_vwap_gap", "last_hour_drift", "intraday_vol", "close_volume_share"]
    aaa = frame[(frame["symbol"] == "AAA") & (sig == day)].iloc[0]
    bbb = frame[(frame["symbol"] == "BBB") & (sig == day)].iloc[0]

    assert aaa[cols].isna().all(), "6 bars with no closing bar must NOT produce features"
    assert bbb[cols].notna().all(), "a complete 7-bar session is usable"


def test_intraday_gate_boundary_is_adversarial_not_round(con):
    """Exactly the full session passes; one bar fewer does not. Both include
    15:30, so completeness is the only thing under test (CLAUDE.md)."""
    seed_history(con, {"AAA": [0.01] * 30, "BBB": [0.01] * 30}, vary_volume=True)
    days = [r[0] for r in con.execute("SELECT DISTINCT date FROM prices ORDER BY date").fetchall()]
    day = days[-2]
    _one_intraday_session(con, "AAA", day, [9, 10, 11, 13, 14, 15])  # a HOLE at 12:30
    _one_intraday_session(con, "BBB", day, [9, 10, 11, 12, 13, 14, 15])  # complete

    frame = feature_frame(con)
    sig = pd.to_datetime(frame["signal_date"]).dt.date
    cols = ["close_vwap_gap", "last_hour_drift", "intraday_vol", "close_volume_share"]
    aaa = frame[(frame["symbol"] == "AAA") & (sig == day)].iloc[0]
    bbb = frame[(frame["symbol"] == "BBB") & (sig == day)].iloc[0]
    assert aaa[cols].isna().all()
    assert bbb[cols].notna().all()


def test_intraday_features_read_only_1h_bars(con):
    """The `interval = '1h'` filter is load-bearing and was untested.

    25,099 real symbol-days hold more than one interval. Dropping the filter
    mixes 5m bars into the aggregate and corrupts every column silently -- a
    measured ANET session went from close_volume_share 0.254 to 0.055.
    """
    seed_history(con, {"AAA": [0.01] * 30}, vary_volume=True)
    days = [r[0] for r in con.execute("SELECT DISTINCT date FROM prices ORDER BY date").fetchall()]
    day = days[-2]
    _one_intraday_session(con, "AAA", day, [9, 10, 11, 12, 13, 14, 15])
    clean = feature_frame(con)
    clean_sig = pd.to_datetime(clean["signal_date"]).dt.date
    clean_row = clean[(clean["symbol"] == "AAA") & (clean_sig == day)].iloc[0]

    # Same day, a DENSE 5m record that would dominate any unfiltered aggregate.
    _one_intraday_session(con, "AAA", day, list(range(9, 16)), interval="5m")
    mixed = feature_frame(con)
    mixed_sig = pd.to_datetime(mixed["signal_date"]).dt.date
    mixed_row = mixed[(mixed["symbol"] == "AAA") & (mixed_sig == day)].iloc[0]

    for col in ["close_vwap_gap", "last_hour_drift", "intraday_vol", "close_volume_share"]:
        assert clean_row[col] == mixed_row[col] or (
            pd.isna(clean_row[col]) and pd.isna(mixed_row[col])
        ), f"{col} changed when 5m bars were added — the interval filter is not holding"


def test_missing_intraday_coverage_is_reported_not_silent(con, caplog):
    """A silently 70%-NaN feature column trains a model on something the live
    path will not have. The gate must SAY what it excluded (CLAUDE.md)."""
    seed_history(con, {"AAA": [0.01] * 30, "BBB": [0.01] * 30}, vary_volume=True)
    days = [r[0] for r in con.execute("SELECT DISTINCT date FROM prices ORDER BY date").fetchall()]
    # AAA gets a full record; BBB gets none at all — and one whole day is blank.
    seed_intraday(con, ["AAA"], days[:-3])
    with caplog.at_level("WARNING"):
        feature_frame(con)
    assert "intraday features missing" in caplog.text
    assert "NO usable 1h session for ANY symbol" in caplog.text


def test_full_intraday_coverage_reports_nothing(con, caplog):
    """The warning must be a NO-OP on complete data, or it becomes alarm fatigue."""
    seed_history(con, {"AAA": [0.01] * 30}, vary_volume=True)
    days = [r[0] for r in con.execute("SELECT DISTINCT date FROM prices ORDER BY date").fetchall()]
    seed_intraday(con, ["AAA"], days)
    with caplog.at_level("WARNING"):
        feature_frame(con)
    assert "intraday features missing" not in caplog.text


def test_the_structural_pre_1h_era_is_not_a_standing_warning(con, caplog):
    """Yahoo serves ~730 days of 1h while daily history reaches 2021 — roughly
    63% of the frame can NEVER have these features.

    A permanent 63% WARNING on every feature_frame() call is the same
    alarm-fatigue trap the doctor's half-day check just cost us: it trains the
    operator to ignore the line that also carries the real news. The structural
    era is INFO; only gaps INSIDE the covered era are a warning.
    """
    seed_history(con, {"AAA": [0.01] * 40}, vary_volume=True)
    days = [r[0] for r in con.execute("SELECT DISTINCT date FROM prices ORDER BY date").fetchall()]
    # 1h exists only for the recent tail — exactly the real-world shape.
    seed_intraday(con, ["AAA"], days[-10:])
    with caplog.at_level("INFO"):
        feature_frame(con)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert not any("intraday features missing" in r.getMessage() for r in warnings), (
        "a structurally impossible gap must not be a standing WARNING"
    )
    assert "structurally absent" in caplog.text
