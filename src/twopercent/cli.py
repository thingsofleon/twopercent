"""Command-line interface: `twopercent universe`, `twopercent ingest`."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import duckdb
import pandas as pd
import typer

from twopercent import doctor as doctor_mod
from twopercent import ingest as ingest_mod
from twopercent import scan as scan_mod
from twopercent import store, universe

app = typer.Typer(help="Scanner + predictor for +2% open-to-close US stock moves.")

DbOption = typer.Option(store.DEFAULT_DB_PATH, "--db", help="Path to the DuckDB file.")
OutOption = typer.Option(Path("dashboard.html"), "--out", help="Output HTML path.")


@app.command("universe")
def universe_cmd(
    refresh: bool = typer.Option(False, "--refresh", help="Fetch a fresh snapshot."),
    top_n: int = typer.Option(universe.TOP_N, help="Universe size."),
    db: Path = DbOption,
) -> None:
    """Show or refresh the ticker universe."""
    con = store.connect(db)
    if refresh:
        df = universe.refresh_universe(top_n=top_n)
        n = store.upsert_universe(con, df, as_of=dt.date.today())
        typer.echo(f"Universe refreshed: {n} symbols as of {dt.date.today()}")
    else:
        df = store.latest_universe(con)
        if df.empty:
            typer.echo("No universe stored yet — run with --refresh.")
            raise typer.Exit(1)
        typer.echo(f"{len(df)} symbols as of {df['as_of'].iloc[0]}. Top 10 by market cap:")
    for _, row in df.head(10).iterrows():
        typer.echo(f"  {row['symbol']:<6} {row['market_cap']:>18,.0f}  {row['name'][:50]}")


@app.command("scan")
def scan_cmd(
    date: str = typer.Option(
        None, "--date", help="Trading day, YYYY-MM-DD (default: latest in store)."
    ),
    threshold: float = typer.Option(
        scan_mod.DEFAULT_THRESHOLD * 100, help="Move threshold in percent."
    ),
    limit: int = typer.Option(50, help="Max rows to print."),
    db: Path = DbOption,
) -> None:
    """List tickers that moved +THRESHOLD% open-to-close on a day."""
    try:
        con = store.connect(db)
    except duckdb.IOException:
        typer.echo(f"Database {db} is locked by another process (ingest running?). Try again.")
        raise typer.Exit(1) from None
    if date is not None:
        try:
            target = dt.date.fromisoformat(date)
        except ValueError:
            typer.echo(f"Invalid --date {date!r}: expected YYYY-MM-DD.")
            raise typer.Exit(2) from None
    else:
        target = scan_mod.latest_price_date(con)
        if target is None:
            typer.echo("Store has no price data — run `twopercent ingest` first.")
            raise typer.Exit(1)

    raw = scan_mod.price_count_on(con, target)
    if raw == 0:
        typer.echo(f"No price data for {target} (weekend, holiday, or not ingested).")
        raise typer.Exit(1)
    usable = scan_mod.returns_count_on(con, target)
    if usable < raw:
        typer.echo(f"warning: {raw - usable} rows on {target} excluded (invalid/missing open)")

    movers = scan_mod.daily_movers(con, date=target, threshold=threshold / 100)
    typer.echo(f"{len(movers)} tickers moved +{threshold:g}% open-to-close on {target}:")
    for i, row in enumerate(movers.head(limit).itertuples(), start=1):
        name = row.name if isinstance(row.name, str) else "?"
        volume = f"{int(row.volume):,}" if pd.notna(row.volume) else "?"
        typer.echo(
            f"  {i:>3}. {row.symbol:<6} {row.oc_return * 100:>6.2f}%  "
            f"close {row.close:>9.2f}  vol {volume:>12}  {name[:45]}"
        )
    if len(movers) > limit:
        typer.echo(f"  ... and {len(movers) - limit} more (raise --limit to see them)")


@app.command("benchmark")
def benchmark_cmd(
    strategy: str = typer.Argument(None, help="Strategy name (default: champion)."),
    months: int = typer.Option(12, help="Test months (walk-forward, monthly retrain)."),
    top: int = typer.Option(20, help="Daily top-N for precision@N."),
    db: Path = DbOption,
) -> None:
    """Walk-forward benchmark of a strategy; records an experiments row."""
    from twopercent import backtest, champion, strategies

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    name = strategy or champion.get_champion()
    if name not in strategies.names():
        typer.echo(f"Unknown strategy {name!r}. Available: {', '.join(strategies.names())}")
        raise typer.Exit(2)
    con = store.connect(db)
    metrics = backtest.run_benchmark(con, name, months=months, top_n=top)
    typer.echo(f"Benchmark {name} over last {months} months (top-{top} daily):")
    for key, value in metrics.items():
        typer.echo(f"  {key:>15}: {value}")


LIFT_NOISE_BAND = 0.1
_NOISE_BAND_EPSILON = 1e-9  # FP: 2.05 - 1.95 == 0.0999...987, yet is a true 0.1 gap


def _compare_verdict(strat_a: str, lift_a: float | None, strat_b: str, lift_b: float | None) -> str:
    """One-line verdict on lift; refuses to crown a winner inside the noise band."""
    if lift_a is None or lift_b is None:
        return "Winner on lift: undecided (lift unavailable for at least one strategy)"
    if lift_a == lift_b:
        return f"Winner on lift: tie at {lift_a}"
    if abs(lift_a - lift_b) < LIFT_NOISE_BAND - _NOISE_BAND_EPSILON:
        return (
            "Winner on lift: within noise — no meaningful difference "
            "(same-window comparison; repeated trials against the same months "
            "inflate the best result)"
        )
    winner = strat_a if lift_a > lift_b else strat_b
    return f"Winner on lift: {winner} ({max(lift_a, lift_b)} vs {min(lift_a, lift_b)})"


@app.command("compare")
def compare_cmd(
    strat_a: str = typer.Argument(..., help="First strategy name."),
    strat_b: str = typer.Argument(..., help="Second strategy name."),
    months: int = typer.Option(12, help="Test months (walk-forward, monthly retrain)."),
    top: int = typer.Option(20, help="Daily top-N for precision@N."),
    db: Path = DbOption,
) -> None:
    """Benchmark two strategies on identical folds and compare their metrics."""
    from twopercent import backtest, strategies

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in (strat_a, strat_b):
        if name not in strategies.names():
            typer.echo(f"Unknown strategy {name!r}. Available: {', '.join(strategies.names())}")
            raise typer.Exit(2)
    con = store.connect(db)
    results = {
        name: backtest.run_benchmark(con, name, months=months, top_n=top)
        for name in (strat_a, strat_b)
    }

    width = max(len(strat_a), len(strat_b), 10)
    typer.echo(f"Compare over last {months} months (top-{top} daily, identical folds):")
    typer.echo(f"  {'metric':>15}  {strat_a:>{width}}  {strat_b:>{width}}")
    for key in results[strat_a]:
        a, b = results[strat_a][key], results[strat_b][key]
        typer.echo(f"  {key:>15}  {a!s:>{width}}  {b!s:>{width}}")

    typer.echo(
        _compare_verdict(strat_a, results[strat_a]["lift"], strat_b, results[strat_b]["lift"])
    )


@app.command("predict")
def predict_cmd(
    strategy: str = typer.Option(None, help="Strategy name (default: champion)."),
    date: str = typer.Option(
        None, "--date", help="Signal date YYYY-MM-DD (default: latest; past dates backfill)."
    ),
    top: int = typer.Option(20, help="How many candidates to print."),
    save: bool = typer.Option(True, help="Log predictions for track-record scoring."),
    db: Path = DbOption,
) -> None:
    """Rank tickers by probability of a +2% open-to-close move next trading day."""
    from twopercent import champion
    from twopercent.predict import predict_for

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    name = strategy or champion.get_champion()
    con = store.connect(db)
    signal_date = None
    if date is not None:
        try:
            signal_date = dt.date.fromisoformat(date)
        except ValueError:
            typer.echo(f"Invalid --date {date!r}: expected YYYY-MM-DD.")
            raise typer.Exit(2) from None
    try:
        result = predict_for(con, name, signal_date=signal_date, save=save)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from None

    uni = store.latest_universe(con).set_index("symbol")["name"]
    typer.echo(
        f"Top {top} candidates for the trading day after {result.signal_date} "
        f"(strategy: {name}, trained on {result.trained_rows:,} rows"
        f"{', logged' if save else ''}):"
    )
    for row in result.scored.head(top).itertuples():
        company = str(uni.get(row.symbol, "?"))[:40]
        typer.echo(f"  {row.rank:>3}. {row.symbol:<6} p={row.prob:0.3f}  {company}")


@app.command("dashboard")
def dashboard_cmd(
    out: Path = OutOption,
    strategy: str = typer.Option(None, help="Strategy name (default: champion)."),
    top: int = typer.Option(20, help="Candidates to show / score."),
    db: Path = DbOption,
) -> None:
    """Generate the static HTML dashboard (candidates + track record)."""
    from twopercent import champion, dashboard

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    name = strategy or champion.get_champion()
    con = store.connect(db)
    try:
        path = dashboard.render(con, name, str(out), top=top)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1) from None
    typer.echo(f"Dashboard written to {path}")


@app.command("routine")
def routine_cmd(
    out: Path = OutOption,
    top: int = typer.Option(20, help="Candidates for dashboard/scoring."),
    db: Path = DbOption,
) -> None:
    """Run the morning cycle: doctor gate, ingest, predict, dashboard, summary.

    Exit codes: 0 clean, 1 degraded (ran with warnings), 2 failed/aborted.
    """
    from twopercent import routine as routine_mod

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = routine_mod.run(db_path=db, out_path=str(out), top=top)
    for line in report.summary_lines():
        typer.echo(line)
    raise typer.Exit(report.exit_code)


@app.command("experiments")
def experiments_cmd(
    limit: int = typer.Option(10, help="How many recent runs to show."),
    db: Path = DbOption,
) -> None:
    """List recent benchmark runs from the experiments table."""
    df = store.list_experiments(store.connect(db), limit=limit)
    if df.empty:
        typer.echo("No experiments recorded yet — run `twopercent benchmark`.")
        raise typer.Exit(0)
    for row in df.itertuples():
        typer.echo(f"#{row.id} {row.run_ts:%Y-%m-%d %H:%M} {row.strategy} {row.metrics}")


@app.command("doctor")
def doctor_cmd(
    stale_days: int = typer.Option(
        doctor_mod.DEFAULT_STALE_DAYS,
        help="Flag symbols whose last bar is more than this many trading days "
        "behind the store max.",
    ),
    examples: int = typer.Option(10, help="Worst examples to print per check."),
    db: Path = DbOption,
) -> None:
    """Data-quality checks over the price store; exit 1 if any check finds problems."""
    try:
        con = store.connect(db)
    except duckdb.IOException:
        typer.echo(f"Database {db} is locked by another process (ingest running?). Try again.")
        raise typer.Exit(1) from None
    if scan_mod.latest_price_date(con) is None:
        typer.echo("Store has no price data — run `twopercent ingest` first.")
        raise typer.Exit(1)

    report = doctor_mod.run(con, stale_days=stale_days)
    typer.echo(f"Doctor report for {db}")
    for line in doctor_mod.format_report(report, examples=examples):
        typer.echo(line)
    if not report.ok:
        typer.echo(f"doctor: {report.problem_count} problems found — store needs attention")
        raise typer.Exit(1)
    typer.echo("doctor: all checks passed")


@app.command("ingest")
def ingest_cmd(
    years: float = typer.Option(5.0, help="Years of history to download."),
    symbols: str = typer.Option(
        None, help="Comma-separated symbol override (default: stored universe)."
    ),
    batch_size: int = typer.Option(ingest_mod.BATCH_SIZE, help="Symbols per yfinance batch."),
    db: Path = DbOption,
) -> None:
    """Download daily OHLCV into the local store."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    con = store.connect(db)
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    else:
        # Union across all snapshots, so symbols churning around the rank-3000
        # boundary keep their histories current.
        symbol_list = store.all_universe_symbols(con)
        if not symbol_list:
            typer.echo("No universe stored — run `twopercent universe --refresh` first.")
            raise typer.Exit(1)

    result = ingest_mod.ingest(con, symbol_list, years=years, batch_size=batch_size)
    typer.echo(
        f"Ingest done: {result.rows_written} rows written, "
        f"{len(result.symbols_ok)} ok, {len(result.symbols_skipped)} already current, "
        f"{len(result.symbols_failed)} failed. Store now has "
        f"{store.price_row_count(con):,} price rows."
    )
    if result.symbols_failed:
        typer.echo(
            f"Failed: {', '.join(result.symbols_failed[:20])}"
            + (" ..." if len(result.symbols_failed) > 20 else "")
        )
