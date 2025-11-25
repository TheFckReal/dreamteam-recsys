import os
from pathlib import Path
from tempfile import TemporaryFile
from typing import Annotated, Optional

from loguru import logger
import polars as pl
from tqdm import tqdm
import typer

from src.config import RAW_DATA_DIR

app = typer.Typer()


@app.command()
def download(
    output_dir: Path = typer.Option(
        RAW_DATA_DIR, help="Output directory for the dataset.", file_okay=False
    ),
    token: Annotated[
        Optional[str],
        typer.Option(
            help="Hugging Face token. If not provided, "
            "will be taken from HF_TOKEN environment variable.",
            envvar="HF_TOKEN",
        ),
    ] = None,
):
    from huggingface_hub import snapshot_download

    if token is None:
        token = os.getenv("HF_TOKEN")
    if token is None:
        raise ValueError("HF_TOKEN is not set")
    logger.info("Downloading dataset to {output_dir}...", output_dir=output_dir)

    snapshot_download(
        repo_id="t-tech/T-ECD",
        repo_type="dataset",
        allow_patterns="dataset/small/",
        local_dir=output_dir,
        token=token,
    )
    logger.success("Dataset downloaded.")


@app.command("add-marketplace-dates", help="Add marketplace events dates to the dataset")
def add_marketplace_events_dates(
    events_dir: Path = typer.Option(
        RAW_DATA_DIR / "dataset" / "small" / "marketplace" / "events",
        help="Path to the marketplace events directory.",
        file_okay=False,
        exists=True,
    ),
):
    logger.info("Adding marketplace events dates...")
    event_files = sorted(events_dir.glob("*.pq"))
    if not event_files:
        logger.exception("No event files found.")
        raise FileNotFoundError(f"No event files found in {events_dir}")
    for event_file in tqdm(event_files):
        day_number = int(event_file.stem)
        event_df = pl.scan_parquet(event_file)
        event_df = event_df.with_columns(pl.lit(day_number, dtype=pl.Int32).alias("day"))
        with TemporaryFile() as temp_file:
            event_df.sink_parquet(temp_file)
            temp_file.seek(0)
            event_df = pl.scan_parquet(temp_file)
            event_df.sink_parquet(event_file)
    logger.success(
        "Marketplace events dates added. Saved to {events_dir} with new column 'day'.",
        events_dir=events_dir,
    )


if __name__ == "__main__":
    app()
