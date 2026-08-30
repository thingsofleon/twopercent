"""Command-line interface: `twopercent universe`, `twopercent ingest`."""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import duckdb
import pandas as pd
import typer

from twopercent import doctor as doctor_mod
from twopercent import ingest as ingest_mod
from twopercent import intraday as intraday_mod
from twopercent import scan as scan_mod
from twopercent import store, universe
from twopercent.compare import compare_verdict as _compare_verdict

app = typer.Typer(help="Scanner + predictor for +2% open-to-close US stock moves.")

DbOption = typer.Option(store.DEFAULT_DB_PATH, "--db", help="Path to the DuckDB file.")
OutOption = typer.Option(Path("dashboard.html"), "--out", help="Output HTML path.")
AbAddOption = typer.Option(
    None, "--add", help="Column the with-arm adds on top of FEATURE_COLUMNS (repeatable)."
)
AbOutOption = typer.Option(None, "--out", help="Write the full result as JSON here.")
QueueOption = typer.Option(
    Path("research/queue.json"), "--queue", help="Experiment queue JSON (edited via PR)."
)


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
    record: bool = typer.Option(
        True,
        "--record/--no-record",
        help="Record an experiments row. Use --no-record for diagnostic reruns "
        "(e.g. degradation investigation) so they never pollute the table the "
        "referee and auto-issues quote.",
    ),
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
    metrics = backtest.run_benchmark(con, name, months=months, top_n=top, record=record)
    typer.echo(f"Benchmark {name} over last {months} months (top-{top} daily):")
    for key, value in metrics.items():
        typer.echo(f"  {key:>15}: {value}")


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


@app.command("ab")
def ab_cmd(
    add: list[str] = AbAddOption,
    strategy: str = typer.Option(None, help="Strategy name (default: champion)."),
    months: int = typer.Option(12, help="Test months (walk-forward, monthly retrain)."),
    top: int = typer.Option(20, help="Daily top-N for precision@N."),
    seeds: str = typer.Option("42,43,44", help="Comma-separated seeds, averaged within fold/day."),
    train_start: str = typer.Option(
        None,
        help="Drop rows with target_date before this ISO date — confines BOTH arms to an era "
        "where the added columns are observed. Makes the run incomparable to the benchmark.",
    ),
    out: Path = AbOutOption,
    db: Path = DbOption,
) -> None:
    """Paired A/B of a feature set against FEATURE_COLUMNS. Records nothing."""
    from twopercent import ab, champion, features, strategies

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not add:
        held = sorted(strategies.SELECTABLE_FEATURE_COLUMNS - set(features.FEATURE_COLUMNS))
        typer.echo(
            "--add is required (at least one column). Selectable but not currently shipped: "
            + (", ".join(held) or "none")
        )
        raise typer.Exit(2)
    name = strategy or champion.get_champion()
    if name not in strategies.names():
        typer.echo(f"Unknown strategy {name!r}. Available: {', '.join(strategies.names())}")
        raise typer.Exit(2)
    try:
        seed_values = [int(s) for s in seeds.split(",") if s.strip()]
    except ValueError:
        typer.echo(f"--seeds must be comma-separated integers, got {seeds!r}")
        raise typer.Exit(2) from None
    if not seed_values:
        typer.echo("--seeds is empty")
        raise typer.Exit(2)
    start = dt.date.fromisoformat(train_start) if train_start else None

    con = store.connect(db)
    result = ab.run_ab(
        con,
        arms={
            "without": list(features.FEATURE_COLUMNS),
            "with": list(features.FEATURE_COLUMNS) + list(add),
        },
        strategy_name=name,
        months=months,
        top_n=top,
        seeds=seed_values,
        train_start=start,
    )
    typer.echo(ab.format_report(result))
    if out:
        out.write_text(json.dumps(result, indent=2, default=str))
        typer.echo(f"wrote {out}")


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
    mode: str = typer.Option(
        "predict",
        "--mode",
        help="predict = pre-open cycle (default); score = post-close scoring, "
        "degradation detector, auto-filed investigation issue.",
    ),
    out: Path = OutOption,
    top: int = typer.Option(20, help="Candidates for dashboard/scoring."),
    db: Path = DbOption,
) -> None:
    """Run the daily cycle: doctor gate, ingest, then predict (pre-open) or score (post-close).

    Exit codes: 0 clean, 1 degraded (ran with warnings), 2 failed/aborted.
    """
    from twopercent import routine as routine_mod

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if mode not in ("predict", "score"):
        typer.echo(f"Invalid --mode {mode!r}: expected 'predict' or 'score'.")
        raise typer.Exit(2)
    report = routine_mod.run(db_path=db, out_path=str(out), top=top, mode=mode)
    for line in report.summary_lines():
        typer.echo(line)
    raise typer.Exit(report.exit_code)


@app.command("research")
def research_cmd(
    budget: int = typer.Option(8, "--budget", min=1, help="Max experiments to run tonight."),
    queue: Path = QueueOption,
    db: Path = DbOption,
) -> None:
    """Overnight research loop: benchmark queued configs, flag champion beaters.

    Runs only between 16:30 and 05:00 America/Denver (clear of market hours and
    the routine runs). Exit codes: 0 clean, 1 some experiments failed, queue
    entries were malformed, or the queue is exhausted (empty or all-recorded —
    refill research/queue.json), 2 the runner itself failed.
    """
    from twopercent import research as research_mod

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = research_mod.run(db_path=db, budget=budget, queue_path=queue)
    except Exception as exc:
        logging.getLogger(__name__).exception("research runner crashed")
        typer.echo(f"research: FAIL — runner crashed: {exc}")
        raise typer.Exit(2) from exc
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
    repair_splits: bool = typer.Option(
        False,
        "--repair-splits",
        help="Delete split-artifact bars (extreme oc_return with open off-scale "
        "vs prior close) before running the checks. Without this flag the "
        "doctor is read-only.",
    ),
    db: Path = DbOption,
) -> None:
    """Data-quality checks over the price store; exit 1 if any check finds problems."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        con = store.connect(db)
    except duckdb.IOException:
        typer.echo(f"Database {db} is locked by another process (ingest running?). Try again.")
        raise typer.Exit(1) from None
    if scan_mod.latest_price_date(con) is None:
        typer.echo("Store has no price data — run `twopercent ingest` first.")
        raise typer.Exit(1)

    if repair_splits:
        removed = doctor_mod.repair_splits(con)
        if removed.empty:
            typer.echo("repair-splits: no split artifacts found")
        else:
            typer.echo(f"repair-splits: deleted {len(removed)} split-artifact bars:")
            for row in removed.itertuples():
                typer.echo(
                    f"    {row.symbol:<8} {row.date:%Y-%m-%d} oc_return {row.oc_return:+.1%}"
                )

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


@app.command("intraday")
def intraday_cmd(
    days: int = typer.Option(
        0,
        "--days",
        help="Calendar days back to fetch. 0 (default) uses the interval's own "
        "measured retention window; longer is clamped.",
    ),
    top: int = typer.Option(20, "--top", help="Pick ranks whose symbols to fetch (5m/1m)."),
    interval: str = typer.Option(
        intraday_mod.INTERVAL, "--interval", help="Bar interval: 5m, 1m or 1h."
    ),
    db: Path = DbOption,
) -> None:
    """Fetch intraday bars: 5m/1m for the model's picks, 1h for the universe.

    5m and 1m are picks-only by design — the explorer needs the ordering of the
    +2% limit and the -1% stop on days the model traded. 1h is universe-wide
    because it feeds FEATURES (#79 phase 2), which must exist for every symbol
    the model scores, not just the ones it already liked.

    This command is the ONLY supported way to build or refresh 1h history: run
    it with `--interval 1h` after a universe refresh. Exit codes: 0 clean,
    1 ran with gaps (some symbols returned nothing), 2 failed.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    con = store.connect(db)
    end = dt.date.today() + dt.timedelta(days=1)
    # Default to the INTERVAL'S OWN window, not 5m's: --days defaulted to the 5m
    # constant, so `--interval 1h` silently fetched 60 of its 700 available days.
    start = end - dt.timedelta(days=days or intraday_mod.spec(interval)["lookback"])
    symbols, scope = intraday_mod.capture_symbols(con, interval, top_n=top)
    if not symbols:
        typer.echo(
            f"No {scope} symbols found — run `twopercent predict`, `benchmark` or `universe` first."
        )
        raise typer.Exit(2)
    typer.echo(f"Fetching {interval} bars for {len(symbols)} {scope} symbol(s), {start} .. {end}")
    result = intraday_mod.ingest(con, symbols, start, end, interval=interval)
    typer.echo(result.summary())
    if result.batches_failed:
        raise typer.Exit(2)
    raise typer.Exit(0 if result.ok else 1)


@app.command("intraday-validate")
def intraday_validate_cmd(
    db: Path = DbOption,
) -> None:
    """Check 5m exit-path verdicts against 1m ground truth.

    The 5m orderings are derivationally sound — the contiguity gate means no
    missing bar could have carried the other trigger earlier — but that is a
    claim about the method, not a measurement. 1m is the only finer evidence
    Yahoo serves, and it expires in ~30 days, so this is worth running while
    the cover exists. Exit 1 if any verdict is inverted.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    con = store.connect(db)
    pairs = con.execute(
        "SELECT DISTINCT symbol, date FROM intraday_prices WHERE interval = '1m'"
    ).df()
    result = intraday_mod.validate_against_1m(con, pairs)
    typer.echo(result.summary())
    raise typer.Exit(1 if result.disagreed else 0)


@app.command("paper")
def paper_cmd(
    basket: int = typer.Option(
        0, "--basket", help="Equal-weight names per day (default: the recorded top-20)."
    ),
    db: Path = DbOption,
) -> None:
    """Net P&L of the FORWARD-ONLY paper record, at several cost levels.

    Shows a sensitivity grid rather than one net number: the cost model is an
    estimate, and a single figure invites trusting it. The breakeven line is the
    useful one — it turns "is the edge real?" into "is the edge bigger than the
    spread?", which is a question about the market, not the model.
    """
    from twopercent import paper as paper_mod

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    con = store.connect(db)
    from twopercent import champion

    name = champion.get_champion()
    # Default to what is RECORDED, not to the flattering corner. The library
    # default was fixed to PAPER_TOP_N; this surface kept 5, which is the one
    # the user actually reads — and at basket 5 the same record shows a
    # breakeven of 72bps against 20bps at the basket the detector reports.
    basket = basket or paper_mod.PAPER_TOP_N
    table = paper_mod.report(con, name, basket=basket)
    if table.empty:
        typer.echo(
            "No paper trades recorded yet. The ledger is forward-only: it fills in "
            "as the score run completes each day."
        )
        raise typer.Exit(0)
    typer.echo(f"Paper record — {paper_mod.RULE}, top-{basket}, {name}")
    typer.echo(table.to_string(index=False))
    days = int(table["days"].iloc[0])
    if days < paper_mod.MIN_REPORT_DAYS:
        typer.echo(
            f"\n{days} day(s) recorded — below the {paper_mod.MIN_REPORT_DAYS}-day floor. "
            "Growth and breakeven are NOT reported yet: on a handful of days they read "
            "as a finding and are noise. The ledger is forward-only, so this fills at "
            "one trading day per day."
        )
        raise typer.Exit(0)
    be = paper_mod.breakeven_bps(con, name, basket=basket)
    typer.echo(
        f"\nBreakeven round-trip cost: {be} bps"
        + ("  (gross edge is negative — no cost makes this profitable)" if be == 0 else "")
    )
    typer.echo("\nBreakeven by basket (the basket is a free parameter — see the whole curve):")
    typer.echo(paper_mod.basket_sweep(con, name).to_string(index=False))
