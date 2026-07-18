import pandas as pd

from tests.conftest import seed_planted
from twopercent import predict


def test_liquidity_floor_excludes_thin_symbols_loudly(con, caplog):
    seed_planted(con, n_each=10)
    # THIN: median 20-bar volume well below the floor. EDGE: exactly at the
    # floor — "< 100,000" excludes, so an exact-floor median must stay
    # (adversarial boundary, not a round pass-through).
    con.execute("UPDATE prices SET volume = 50_000 WHERE symbol = 'RUN00'")
    con.execute("UPDATE prices SET volume = 100_000 WHERE symbol = 'RUN01'")

    with caplog.at_level("WARNING", logger="twopercent.predict"):
        result = predict.predict_for(con, "logreg_v1")

    symbols = set(result.scored["symbol"])
    assert "RUN00" not in symbols
    assert "RUN01" in symbols  # exactly at the floor is kept
    assert "RUN02" in symbols  # liquid runner untouched
    assert result.scored["rank"].tolist() == list(range(1, len(result.scored) + 1))

    excluded_warnings = [r.message for r in caplog.records if "excluded from ranking" in r.message]
    assert len(excluded_warnings) == 1
    assert "1 of 20 symbols excluded" in excluded_warnings[0]
    assert "RUN00" in excluded_warnings[0]

    saved = con.execute("SELECT DISTINCT symbol FROM predictions").df()
    assert "RUN00" not in set(saved["symbol"])
    assert "RUN01" in set(saved["symbol"])


def test_liquidity_floor_silent_when_all_liquid(con, caplog):
    seed_planted(con, n_each=10)
    with caplog.at_level("WARNING", logger="twopercent.predict"):
        result = predict.predict_for(con, "logreg_v1", save=False)
    assert len(result.scored) == 20
    assert not any("excluded from ranking" in r.message for r in caplog.records)


def test_liquidity_floor_uses_only_trailing_bars(con):
    # Walk-forward honesty: a symbol that turns thin AFTER the signal date must
    # not be excluded from a backfilled prediction for that date.
    seed_planted(con, n_each=10)
    dates = sorted(pd.bdate_range("2026-01-05", periods=100).date)
    signal = dates[-5]
    con.execute("UPDATE prices SET volume = 10 WHERE symbol = 'RUN00' AND date > ?", [signal])
    result = predict.predict_for(con, "logreg_v1", signal_date=signal, save=False)
    assert "RUN00" in set(result.scored["symbol"])
