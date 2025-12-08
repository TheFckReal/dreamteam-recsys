import os
from pathlib import Path
from tempfile import TemporaryFile
from typing import Annotated, Literal, Optional, TypeVar

from loguru import logger
import polars as pl
from tqdm import tqdm
import typer

from src.config import RAW_DATA_DIR

app = typer.Typer()

PolarsFrame = TypeVar("PolarsFrame", pl.DataFrame, pl.LazyFrame)


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


def create_target(
    events_df: PolarsFrame,
    target_type: Literal["log_target", "sqrt_target", "unproccessed", "multiclass"],
) -> PolarsFrame:
    """
    Create target for the dataset.

    Parameters
    ----------
    events_df : PolarsFrame
        Input dataframe with events.
    target_type : Literal["log_target", "sqrt_target", "unproccessed", "multiclass"]
        Type of target encoding to apply.

    Returns
    -------
    PolarsFrame
        Dataframe with created target column(s).
    """
    target_process_expr = {
        "log_target": pl.col("len").log1p(),
        "sqrt_target": pl.col("len").sqrt(),
        "unproccessed": pl.col("len"),
    }
    actions_count = events_df.group_by("action_type").agg(len=pl.len()).lazy()
    _events_df = events_df.lazy()
    if target_type == "multiclass":
        result_df = _events_df.join(actions_count, on="action_type").with_columns(
            [
                (pl.col("action_type") == "view").cast(pl.Int8).alias("target_view"),
                (pl.col("action_type") == "clickout").cast(pl.Int8).alias("target_clickout"),
                (pl.col("action_type") == "like").cast(pl.Int8).alias("target_like"),
                (pl.col("action_type") == "click").cast(pl.Int8).alias("target_click"),
            ]
        )
    else:
        result_df = _events_df.join(
            actions_count.with_columns(target=target_process_expr[target_type], on="action_type")
        )
    if isinstance(events_df, pl.LazyFrame):
        return result_df
    else:
        return result_df.collect(engine="streaming")


def global_temporal_split(
    df: PolarsFrame, test_size: int | float = 1, date_column: str = "day"
) -> tuple[PolarsFrame, PolarsFrame]:
    """
    Split the dataset into training and test parts based on a global temporal boundary.
    One day between the test and training parts is ignored (gap).

    Parameters
    ----------
    df : PolarsFrame
        Dataset to split.
    test_size : int | float, optional
        Number of days in the test part or fraction of the total number of days.
        Default is 1.
    date_column : str, optional
        Name of the column with dates. Default is "day".

    Returns
    -------
    tuple[PolarsFrame, PolarsFrame]
        Tuple of two datasets: training and test parts.
    """
    _df = df.lazy()
    min_day, max_day = (
        _df.select(
            pl.col(date_column).min().alias("min_day"), pl.col(date_column).max().alias("max_day")
        )
        .collect(engine="streaming")
        .row(0)
    )
    days_all = (max_day - min_day) + 1
    if isinstance(test_size, float):
        test_size = int(days_all * test_size)
        if test_size == 0:
            test_size += 1
        cut_day = max_day - test_size
    else:
        cut_day = max_day - test_size

    if cut_day - 1 < min_day or cut_day + 1 > max_day:
        raise ValueError(
            f"Test size is too large. Test size: {test_size}, min day: {min_day}, max day: {max_day}, cut day: {cut_day}"
        )

    train_df = df.filter(pl.col(date_column) < cut_day)
    test_df = df.filter(pl.col(date_column) > cut_day)

    return train_df, test_df


def split_cold_start(
    train_df: PolarsFrame, test_df: PolarsFrame, user_col: str = "user_id"
) -> tuple[PolarsFrame, PolarsFrame]:
    """
    Split test data into cold-start and non-cold-start subsets by users.

    Parameters
    ----------
    train_df : PolarsFrame
        Training data. Used to determine which users are already known to the model.
    test_df : PolarsFrame
        Test data that will be split into cold-start and non-cold-start parts.
    user_col : str, optional
        Name of the column containing user identifiers, by default "user_id".

    Returns
    -------
    tuple[PolarsFrame, PolarsFrame]
        A tuple of two LazyFrames:

        - first element : test subset with cold-start users
          (users present in `test_df` but not in `train_df`);
        - second element : test subset with non-cold-start users
          (users present both in `train_df` and `test_df`).
    """
    _test_df = test_df.lazy()
    _train_df = train_df.lazy()
    cold_start_users = _test_df.select(pl.col(user_col).unique()).join(
        _train_df.select(pl.col(user_col).unique()), on=user_col, how="anti"
    )
    return _test_df.join(cold_start_users, on=user_col), _test_df.join(
        cold_start_users, on=user_col, how="anti"
    )


def ndcg_at_k(
    user_based_df: pl.DataFrame,
    relevancy_col: str,
    true_items_col: str,
    predicted_items_col: str,
    predicted_score_col: str,
    top_k: int = 15,
) -> pl.DataFrame:
    """
    Computes user-based NDCG@k for graded relevance in a recommendation setting.

    Parameters
    ----------
    user_based_df : pl.DataFrame
        Dataframe with user data. Each row must contain user and its lists with: truth
        ground items, their relevancy estimation and model prediction score.
    relevancy_col : str
        Column name contains list of relevancy estimations ground
        truth items (pl.List[float]) for user. Elements order must match `true_items_col`.
    true_items_col : str
        Column name of ground truth items with which user had interactions (pl.List[str]). Relevancy
        of these items must be set in `relevancy_col` respectively.
    predicted_items_col : str
        Columns name with predicted items (pl.List[str]). Must be set in order matches
        `predicted_score_col`.
    predicted_score_col : str
        Columns name with predicted scores for items in `predicted_items_col` (pl.List[float]).
        Used to sort predictions in descending order.
    top_k : int, optional
        Top k elements to calculate `@k` metric.

    Returns
    -------
    pl.DataFrame
        Columns:
        - ``user_id`` : user identifier;
        - ``ndcg`` : NDCG@k for current user.

    Notes
    -----
    For each user, the function:
    1. Aggregates relevancies for ground-truth items by taking the maximum value for each item.
    2. Joins predicted items with their ground-truth relevancies.
    3. Computes DCG@k using the order induced by the model (sorting by score).
    4. Computes IDCG@k using the ideal order (sorting by ground-truth relevancy).
    5. Returns NDCG@k = DCG@k / IDCG@k, or 0.0 if IDCG@k = 0.
    """
    user_ids = []
    ndcgs = []
    for row in user_based_df.iter_rows(named=True):
        true_items = pl.DataFrame(
            {"truth_items": row[true_items_col], "relevancy": row[relevancy_col]}
        )
        true_items = true_items.group_by("truth_items").agg(
            pl.col("relevancy").max()
        )  # Берем максимальную релевантность для товара
        predictions = (
            pl.DataFrame(
                {"predicted_items": row[predicted_items_col], "score": row[predicted_score_col]}
            )
            .join(
                true_items,
                left_on="predicted_items",
                right_on="truth_items",
                coalesce=True,
                how="left",
            )
            .fill_null(0)
        )
        idcg = (
            predictions.select("relevancy")
            .sort("relevancy", descending=True)
            .head(top_k)
            .select((pl.col("relevancy") / (pl.row_index() + 2).log(2)).sum())
            .item()
        )
        dcg = (
            predictions.select("score", "relevancy")
            .sort("score", descending=True)
            .head(top_k)
            .select((pl.col("relevancy") / (pl.row_index() + 2).log(2)).sum())
            .item()
        )
        user_ids.append(row["user_id"])
        ndcgs.append(0.0 if idcg == 0 else dcg / idcg)
    return pl.DataFrame({"user_id": user_ids, "ndcg": ndcgs})


def get_last_k_user_interactions(
    events_df: PolarsFrame,
    last_k: int | None = 30,
    date_column: str = "day",
    timestamp_column: str = "timestamp",
    user_column: str = "user_id",
    acceptable_action: list[str] | None = None,
) -> PolarsFrame:
    """
    Get last k user interactions.

    Parameters
    ----------
    events_df : PolarsFrame
        DataFrame containing user events.
    last_k : int | None, optional
        Number of most recent interactions to keep. If None, keeps all. Default is 30.
    date_column : str, optional
        Name of the date column. Default is "day".
    timestamp_column : str, optional
        Name of the timestamp column. Default is "timestamp".
    user_column : str, optional
        Name of the user ID column. Default is "user_id".
    acceptable_action : list[str] | None, optional
        List of acceptable action types. Default is None.

    Returns
    -------
    PolarsFrame
        DataFrame with last k user interactions grouped by user.
    """
    if acceptable_action is None:
        acceptable_action = ["view", "clickout", "like", "click"]
    return (
        events_df.filter(pl.col("action_type").is_in(set(acceptable_action)))
        .group_by(user_column)
        .agg(
            pl.all().sort_by(date_column, timestamp_column).tail(last_k)
            if last_k is not None
            else pl.all().sort_by(date_column, timestamp_column)
        )
    )


if __name__ == "__main__":
    app()
