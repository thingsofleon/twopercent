"""Shared prediction logic: train a strategy, score a signal date, log results."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

import duckdb
import pandas as pd

from twopercent import store, strategies
from twopercent.features import feature_frame
from twopercent.track import COMPLETENESS_MIN_PRIOR_DATES

logger = logging.getLogger(__name__)

# Liquidity floor, applied at selection time only (issue #25): symbols whose
# trailing median volume (the median_vol_20 metadata column from features.py,
# 20 bars ending at signal_date) is below LIQUIDITY_MIN_MEDIAN_VOLUME shares
# are too thin to trade and are excluded from the ranking and the saved rows.
# The benchmark applies the same floor to its top-N selection (backtest.py)
# so reported precision matches the shipped product. Labels and training keep
# illiquid names — see predict_for's docstring for the asymmetry.
LIQUIDITY_WINDOW_BARS = 20
LIQUIDITY_MIN_MEDIAN_VOLUME = 100_000


@dataclass
class PredictResult:
    strategy: str
    signal_date: dt.date
    scored: pd.DataFrame  # feature columns + prob + rank, sorted by prob desc
    trained_rows: int


def _default_signal_date(con: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> dt.date:
    """The latest COMPLETE signal day in `frame` — never forecast from a
    provisional edge day (#65).

    The raw latest signal day may be an in-progress day carrying only a
    pre-market/partial bar (store.complete_trading_days judges it incomplete).
    Forecasting from it would build features from a fraction of the universe, so
    the default falls back to the latest complete day and WARNs loudly (naming
    the held-back date and its coverage vs the trailing-median norm). An EXPLICIT
    signal_date argument bypasses this entirely and is honored verbatim — backfill
    and testing must keep working, including deliberately predicting from a
    provisional day.

    Young-store regime: a date with fewer than COMPLETENESS_MIN_PRIOR_DATES prior
    trading dates cannot be judged (no baseline) and is treated complete — but if
    the latest signal day is USED while unjudgeable, that is WARNed loudly too, so
    a from-scratch backfill never silently forecasts from a genuinely-provisional
    edge day.
    """
    signal_dates = set(frame["signal_date"])
    raw_latest = max(signal_dates)
    coverage = store.trading_day_coverage(con)
    complete_dates = set(pd.to_datetime(coverage.loc[coverage["complete"], "date"]).dt.date)
    candidates = signal_dates & complete_dates
    if not candidates:
        logger.warning(
            "no complete signal day available (store too small to judge coverage) — "
            "defaulting to latest signal day %s",
            raw_latest,
        )
        return raw_latest
    chosen = max(candidates)
    if chosen != raw_latest:
        held = coverage[pd.to_datetime(coverage["date"]).dt.date == raw_latest]
        cov = ""
        if not held.empty:
            row = held.iloc[0]
            cov = (
                f" ({int(row['bar_count'])} valid bars vs trailing-median "
                f"{float(row['trailing_median']):.0f})"
            )
        logger.warning(
            "latest signal day %s held back as INCOMPLETE%s — forecasting from the "
            "latest COMPLETE signal day %s instead (never predict from a provisional "
            "pre-market bar)",
            raw_latest,
            cov,
            chosen,
        )
        return chosen
    # chosen == raw_latest: either genuinely complete, or too young to judge.
    latest_row = coverage[pd.to_datetime(coverage["date"]).dt.date == raw_latest]
    if not latest_row.empty:
        prior_days = int(latest_row.iloc[0]["prior_days"])
        if prior_days < COMPLETENESS_MIN_PRIOR_DATES:
            logger.warning(
                "latest signal day %s used but UNJUDGEABLE for completeness (only %d prior "
                "trading day(s), need %d) — store too young to assess coverage; forecasting "
                "from it without a completeness check",
                raw_latest,
                prior_days,
                COMPLETENESS_MIN_PRIOR_DATES,
            )
    return chosen


def predict_for(
    con: duckdb.DuckDBPyConnection,
    strategy_name: str,
    signal_date: dt.date | None = None,
    save: bool = True,
    strategy_params: dict | None = None,
) -> PredictResult:
    """Score every symbol for the trading day after `signal_date`.

    Defaults to the latest COMPLETE date in the store (never a provisional
    in-progress edge day — see _default_signal_date, #65). For a PAST signal
    date (track-record backfill), training uses only outcomes with
    target_date <= signal_date — what was knowable at that day's close —
    so backfilled predictions stay walk-forward honest.

    `strategy_params` (default None → the strategy's default config) is passed
    to the strategy constructor, so a parameterized challenger can be scored
    with the same walk-forward machinery as the champion. The training filter
    is unchanged (target_date <= signal_date), so params never buy lookahead.

    Liquidity floor (selection time ONLY): after scoring, symbols whose
    median_vol_20 (trailing median volume over the LIQUIDITY_WINDOW_BARS
    bars ending at signal_date, computed in features.py) is below
    LIQUIDITY_MIN_MEDIAN_VOLUME shares are excluded from the ranking AND
    the saved rows — too thin to trade at any size. This is deliberately
    asymmetric: labels and training keep illiquid names (their moves are
    real observations that would otherwise bias the base rate), so trained
    probabilities include names the ranking will never surface. The window
    is strictly trailing — no lookahead — and the benchmark's top-N
    selection applies the same floor (backtest.py), so reported precision
    describes the list this function actually ships.
    """
    frame = feature_frame(con)
    if frame.empty:
        raise ValueError("no feature rows — is the store ingested?")
    frame = frame.assign(
        signal_date=pd.to_datetime(frame["signal_date"]).dt.date,
        target_date=pd.to_datetime(frame["target_date"]).dt.date,
    )
    signal_date = signal_date or _default_signal_date(con, frame)

    rows = frame[frame["signal_date"] == signal_date]
    if rows.empty:
        raise ValueError(f"no feature rows for signal date {signal_date}")
    train = frame[frame["did_2pct_next"].notna() & (frame["target_date"] <= signal_date)]
    if train.empty:
        raise ValueError("no labeled history to train on — ingest more data")

    strategy = strategies.get(strategy_name, **(strategy_params or {}))
    strategy.fit(train)
    scored = rows.assign(prob=strategy.predict_proba(rows))

    liquid = scored["median_vol_20"] >= LIQUIDITY_MIN_MEDIAN_VOLUME  # NaN median -> excluded
    if (~liquid).any():
        logger.warning(
            "%d of %d symbols excluded from ranking and saved predictions: median "
            "volume over last %d bars < %d shares (labels/training keep them): %s",
            int((~liquid).sum()),
            len(scored),
            LIQUIDITY_WINDOW_BARS,
            LIQUIDITY_MIN_MEDIAN_VOLUME,
            ", ".join(sorted(scored.loc[~liquid, "symbol"])[:20]),
        )
    scored = (
        scored[liquid]
        .sort_values("prob", ascending=False, kind="mergesort")  # stable: ties match the referee
        .reset_index(drop=True)
    )
    scored["rank"] = range(1, len(scored) + 1)
    if save:
        store.save_predictions(con, strategy_name, signal_date, scored)
    logger.info(
        "predicted %s for day after %s: %d symbols, trained on %d rows",
        strategy_name,
        signal_date,
        len(scored),
        len(train),
    )
    return PredictResult(strategy_name, signal_date, scored, len(train))
