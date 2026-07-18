import datetime as dt

import pandas as pd

from tests.conftest import seed_history
from twopercent import store, track

# 25 business days from 2026-01-05; last day is 2026-02-06.
OC = {
    "HIT1": [0.001] * 23 + [0.03, 0.05],  # 2% on the last two days
    "HIT2": [0.001] * 23 + [0.001, 0.021],  # 2% on the last day only
    "MISS": [0.001] * 25,  # never
}


def _seed(con):
    seed_history(con, OC)
    dates = sorted(pd.bdate_range("2026-01-05", periods=25).date)
    return dates


def _save(con, signal_date, ranked):
    store.save_predictions(
        con,
        "test_strat",
        signal_date,
        pd.DataFrame({"symbol": ranked, "prob": [0.9, 0.5, 0.1], "rank": [1, 2, 3]}),
    )


def test_scoring_math_and_target_resolution(con):
    dates = _seed(con)
    # Predict on the second-to-last day; outcome day is the last day, where
    # HIT2 (rank 1) and HIT1 (rank 2) moved 2%+ and MISS didn't.
    _save(con, dates[-2], ["HIT2", "HIT1", "MISS"])

    record = track.score_predictions(con, "test_strat", top_n=3)
    assert len(record.scored) == 1
    row = record.scored.iloc[0]
    assert pd.Timestamp(row["target_date"]).date() == dates[-1]
    assert row["hits"] == 2 and row["n_scored"] == 3
    assert abs(row["precision"] - 2 / 3) < 1e-12
    assert abs(row["base_rate"] - 2 / 3) < 1e-12  # 2 of 3 symbols moved that day
    assert record.pending == []


def test_top_n_restricts_scoring(con):
    dates = _seed(con)
    _save(con, dates[-2], ["MISS", "HIT1", "HIT2"])  # rank 1 is the miss
    record = track.score_predictions(con, "test_strat", top_n=1)
    row = record.scored.iloc[0]
    assert row["hits"] == 0 and row["n_scored"] == 1


def test_unscoreable_day_is_pending_not_dropped(con):
    dates = _seed(con)
    _save(con, dates[-1], ["HIT1", "HIT2", "MISS"])  # no next trading day ingested
    record = track.score_predictions(con, "test_strat", top_n=3)
    assert record.scored.empty
    assert record.pending == [dates[-1]]


def test_save_predictions_idempotent(con):
    _seed(con)
    day = dt.date(2026, 2, 5)
    _save(con, day, ["HIT1", "HIT2", "MISS"])
    _save(con, day, ["HIT1", "HIT2", "MISS"])
    count = con.execute("SELECT count(*) FROM predictions").fetchone()[0]
    assert count == 3


def test_save_predictions_rerun_replaces_whole_slice(con):
    # A re-run that scores FEWER symbols (liquidity floor kicked one out) must
    # not leave the dropped symbol behind as a phantom rank from the first save.
    day = dt.date(2026, 2, 5)

    def ranked(symbols: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": symbols,
                "prob": [1 - i / 100 for i in range(len(symbols))],
                "rank": range(1, len(symbols) + 1),
            }
        )

    twenty = [f"S{i:02d}" for i in range(20)]
    store.save_predictions(con, "s", day, ranked(twenty))
    store.save_predictions(con, "s", day, ranked(twenty[1:]))  # S00 now excluded

    rows = con.execute(
        "SELECT symbol, rank FROM predictions WHERE strategy = 's' AND signal_date = ? "
        "ORDER BY rank",
        [day],
    ).df()
    assert len(rows) == 19
    assert "S00" not in set(rows["symbol"])
    assert rows["rank"].tolist() == list(range(1, 20))  # contiguous, no phantom rank 1
