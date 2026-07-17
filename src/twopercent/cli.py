"""Command-line interface: `twopercent universe`, `twopercent ingest`."""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import typer

from twopercent import ingest as ingest_mod
from twopercent import store, universe

app = typer.Typer(help="Scanner + predictor for +2% open-to-close US stock moves.")

DbOption = typer.Option(store.DEFAULT_DB_PATH, "--db", help="Path to the DuckDB file.")


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
        uni = store.latest_universe(con)
        if uni.empty:
            typer.echo("No universe stored — run `twopercent universe --refresh` first.")
            raise typer.Exit(1)
        symbol_list = uni["symbol"].tolist()

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
