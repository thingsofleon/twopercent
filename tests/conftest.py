from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from twopercent import issues, store


@pytest.fixture(autouse=True)
def isolate_cwd(tmp_path, monkeypatch):
    """Run every test in its own empty directory.

    Several library paths resolve against the CALLER's working directory —
    champion.json, research/queue.json, research/shadow.json, the DuckDB store.
    A test that omits one of them writes into the developer's checkout instead
    of failing, which is how `pytest` from the repo root came to overwrite the
    live dashboard.html with fixture output while reporting green (#81).

    Isolating the CWD makes that whole class of accident land in a temp dir.
    It is the backstop, not the fix: library code must still not default to a
    CWD-relative path (see routine.run).
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def no_real_gh(monkeypatch):
    """No test may reach the real `gh` CLI. Applies to the WHOLE suite.

    This guard existed, but only inside tests/test_research.py — so every other
    module could shell out for real. It did: a routine-score fixture drove the
    degradation detector into its issue-filing branch and pytest FILED #70
    against the live repo, with a title and numbers that then read as a genuine
    model-decay alert. The same accident produced #43/#44, and the fix for those
    was scoped to one file, which is why it happened again.

    Any subprocess call not intercepted by an explicit spy is a test bug and
    must blow up rather than reach the network (#99).
    """

    real_run = issues.subprocess.run

    def guard(args, **kw):
        # Targeted at `gh` ONLY. issues.subprocess IS the global module, so
        # blocking every call would also break the node-executed lockstep test
        # and joblib's cpu probe — and a guard that breaks unrelated tests gets
        # weakened or removed, which is how this hole reopened last time.
        argv = args if isinstance(args, (list, tuple)) else [args]
        if argv and str(argv[0]) == "gh":
            raise AssertionError(f"test reached the real gh CLI: {list(argv)}")
        return real_run(args, **kw)

    monkeypatch.setattr(issues.subprocess, "run", guard)


@pytest.fixture
def con(tmp_path):
    return store.connect(tmp_path / "test.duckdb")


@pytest.fixture
def screener_rows():
    """Canned NASDAQ screener payload rows, deliberately messy."""
    return [
        {
            "symbol": "NVDA",
            "name": "NVIDIA Corporation Common Stock",
            "marketCap": "4,974,496,340,000",
        },
        {"symbol": "AAPL", "name": "Apple Inc. Common Stock", "marketCap": "4,853,994,909,728"},
        {"symbol": "SMALL", "name": "Small Co Common Stock", "marketCap": "1,000,000"},
        {"symbol": "TINY", "name": "Tiny Co Common Stock", "marketCap": "500,000"},
        {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust", "marketCap": "600,000,000,000"},
        {"symbol": "FOO.W", "name": "Foo Inc Warrant", "marketCap": "10,000,000"},
        {"symbol": "NOCAP", "name": "No Cap Inc Common Stock", "marketCap": ""},
        {"symbol": "AAPL", "name": "Apple Inc. Common Stock", "marketCap": "4,853,994,909,728"},
        {
            "symbol": "BRK/B",
            "name": "Berkshire Hathaway Class B Common Stock",
            "marketCap": "1,100,000,000,000",
        },
    ]


def seed_history(
    con,
    oc_returns: dict[str, list[float]],
    start="2026-01-05",
    vary_volume: bool = False,
    high_returns: dict[str, list[float]] | None = None,
) -> pd.DataFrame:
    """Seed prices for symbols with exact open-to-close returns per business day.

    open is always 100.0, so close = 100 * (1 + oc). vary_volume avoids
    constant feature columns (sklearn's binner rejects single-valued columns).

    By default high sits 0.1% above max(open, close), so the intraday REACH
    (open-to-high) barely exceeds the close — a close-based and touch-based event
    coincide. Pass `high_returns[symbol]` (per-day (high-open)/open) to seed bars
    that TOUCH +2% intraday while closing below it — the Stage-A case that
    separates the touch label from the old close label. Highs are clamped to keep
    OHLC valid (high >= max(open, close))."""
    from twopercent import store

    frames = []
    for symbol, ocs in oc_returns.items():
        n = len(ocs)
        dates = pd.bdate_range(start, periods=n)
        opens = np.full(n, 100.0)
        closes = opens * (1 + np.asarray(ocs))
        if high_returns is not None and symbol in high_returns:
            seeded_high = opens * (1 + np.asarray(high_returns[symbol]))
            highs = np.maximum(seeded_high, np.maximum(opens, closes))
        else:
            highs = np.maximum(opens, closes) * 1.001
        volume = 1_000_000 + (np.arange(n) % 17) * 1_000 if vary_volume else 1_000_000
        frames.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates.date,
                    "open": opens,
                    "high": highs,
                    "low": np.minimum(opens, closes) * 0.999,
                    "close": closes,
                    "adj_close": closes,
                    "volume": volume,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    store.upsert_prices(con, df)
    return df


# Slight deterministic variation keeps every feature column multi-valued
# (sklearn's binner rejects constant columns).
RUNNER_OC = [0.03 + 0.001 * (i % 5) for i in range(100)]  # +3.0–3.4% every day
FLAT_OC = [0.002 + 0.001 * (i % 3) for i in range(100)]  # +0.2–0.4%, never 2%


def seed_planted(con, n_each: int = 30, universe_symbols: list[str] | None = None) -> list[str]:
    """Planted-signal history: RUN* symbols do +2% every day, FLT* never do.

    universe_symbols restricts which symbols get a universe row (default all);
    omitted symbols flow NULL log_mcap through the features LEFT JOIN.
    """
    data = {}
    for i in range(n_each):
        data[f"RUN{i:02d}"] = RUNNER_OC
        data[f"FLT{i:02d}"] = FLAT_OC
    seed_history(con, data, vary_volume=True)
    symbols = list(data) if universe_symbols is None else universe_symbols
    store.upsert_universe(
        con,
        pd.DataFrame(
            {
                "symbol": symbols,
                "name": symbols,
                "market_cap": [1e9 * (i + 1) for i in range(len(symbols))],
                # One shared sector: runners and flats mix, so sector features
                # vary per row (an all-NaN column crashes HistGBM's binner).
                "sector": ["Tech"] * len(symbols),
            }
        ),
        as_of=pd.Timestamp("2026-06-01").date(),
    )
    return list(data)


def make_yf_frame(symbols: list[str], days: int = 5, start_price: float = 100.0) -> pd.DataFrame:
    """Synthetic yf.download output: MultiIndex (ticker, field) columns."""
    dates = pd.bdate_range("2026-01-05", periods=days)
    frames = {}
    for i, sym in enumerate(symbols):
        base = start_price * (1 + i)
        opens = np.linspace(base, base * 1.05, days)
        closes = opens * 1.01
        frames[sym] = pd.DataFrame(
            {
                "Open": opens,
                "High": closes * 1.01,
                "Low": opens * 0.99,
                "Close": closes,
                "Adj Close": closes * 0.98,
                "Volume": np.full(days, 1_000_000.0),
            },
            index=dates,
        )
    return pd.concat(frames, axis=1)
