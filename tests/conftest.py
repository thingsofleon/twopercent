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


@pytest.fixture(autouse=True)
def no_real_yfinance(monkeypatch):
    """No test may reach the real market-data provider. Applies suite-wide.

    Same shape as no_real_gh, and added for the same reason: the routine's
    score mode gained an intraday capture step, no routine test stubbed it, and
    the suite quietly began issuing live yfinance downloads on every run —
    slow, flaky, dependent on the network, and hammering a provider this
    project already has rate-limit issues with. Offline unit tests against
    canned payloads plus `@pytest.mark.live` smoke tests is the documented
    pattern (CLAUDE.md); this makes departing from it fail loudly.
    """
    import yfinance

    def forbidden(*a, **kw):
        raise AssertionError(
            "test reached the real yfinance API — inject a downloader or stub "
            "the ingest step; live calls belong in @pytest.mark.live smoke tests"
        )

    monkeypatch.setattr(yfinance, "download", forbidden)


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


def seed_intraday(con, symbols: list[str], dates, interval: str = "1h", bars: int = 7):
    """Seed a full intraday session per symbol-day, shaped so every phase-2
    feature is LIVE (non-NaN) and VARIES across symbols and days.

    Without this the canary compares NaN to NaN for the intraday features and
    passes vacuously — the exact failure the notna() guards exist to catch. The
    per-bar drift and volume ramp differ by symbol and day so close_vwap_gap,
    last_hour_drift, intraday_vol and close_volume_share are all multi-valued
    (a constant column crashes sklearn's binner; see CLAUDE.md).
    """
    import datetime as _dt

    from twopercent import store as _store

    rows = []
    for si, sym in enumerate(symbols):
        for di, day in enumerate(dates):
            base = 100.0 + si
            for b in range(bars):
                drift = 0.002 * (b + 1) * (1 + 0.1 * si) * (1 + 0.05 * (di % 3))
                o = base * (1 + drift)
                c = base * (1 + drift + 0.001 * ((b + si) % 3 + 1))
                rows.append(
                    {
                        "symbol": sym,
                        "ts": _dt.datetime.combine(day, _dt.time(9, 30))
                        + _dt.timedelta(minutes=60 * b),
                        "date": day,
                        "interval": interval,
                        "open": o,
                        "high": max(o, c) * 1.002,
                        "low": min(o, c) * 0.998,
                        "close": c,
                        "volume": 10_000 + 1_000 * b + 100 * si + 10 * (di % 5),
                    }
                )
    frame = pd.DataFrame(rows)
    _store.connect  # noqa: B018 - keep the import meaningful for readers
    con.register("_seed_intraday", frame)
    con.execute("INSERT INTO intraday_prices SELECT * FROM _seed_intraday")
    con.unregister("_seed_intraday")
    return frame


# Slight deterministic variation keeps every feature column multi-valued
# (sklearn's binner rejects constant columns).
RUNNER_OC = [0.03 + 0.001 * (i % 5) for i in range(100)]  # +3.0–3.4% every day
FLAT_OC = [0.002 + 0.001 * (i % 3) for i in range(100)]  # +0.2–0.4%, never 2%


def seed_planted(
    con,
    n_each: int = 30,
    universe_symbols: list[str] | None = None,
    with_intraday: bool = True,
) -> list[str]:
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
    if with_intraday:
        # Phase-2 intraday features are all-NaN without this, so the strategies
        # drop them loudly and a test named "silent when all features observed"
        # would be asserting the opposite of its name. Off only for tests that
        # deliberately exercise the dropped-column path.
        days = [
            r[0] for r in con.execute("SELECT DISTINCT date FROM prices ORDER BY date").fetchall()
        ]
        seed_intraday(con, list(data), days)
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
