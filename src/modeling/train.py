from pathlib import Path
import pickle

from loguru import logger
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import typer

from src.config import SVD_DIR
from src.dataset import create_target, global_temporal_split

app = typer.Typer()


def create_mappings(df: pl.DataFrame):
    unique_users = df["user_id"].unique().to_list()
    unique_items = df["item_id"].unique().to_list()

    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}

    return user_to_idx, item_to_idx


@app.command()
def main(
    events_path: Path = typer.Option(..., help="Путь к файлу с событиями"),
    artifacts_dir: Path = typer.Option(SVD_DIR, help="Папка для сохранения артефактов модели"),
    n_components: int = typer.Option(20, help="Количество компонент для SVD"),
):
    logger.info(f"Загрузка данных из {events_path}...")
    try:
        events = pl.read_parquet(events_path)
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
        raise

    if "day" not in events.columns:
        logger.error(
            "В данных нет колонки 'day'. Сначала выполните: python -m src.dataset add-marketplace-dates ..."
        )
        raise ValueError("Missing 'day' column")

    train_df, _ = global_temporal_split(events, test_size=1)

    train_df = create_target(train_df, target_type="log_target")

    user_to_idx, item_to_idx = create_mappings(train_df)

    users_mapped = [user_to_idx[uid] for uid in train_df["user_id"].to_list()]
    items_mapped = [item_to_idx[iid] for iid in train_df["item_id"].to_list()]

    values = train_df["target"].to_numpy().astype(np.float32)

    R_sparse = csr_matrix(
        (values, (users_mapped, items_mapped)), shape=(len(user_to_idx), len(item_to_idx))
    )

    logger.info(f"Обучение SVD (размерность вектора: {n_components})...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)

    user_features = svd.fit_transform(R_sparse)
    item_features = svd.components_.T

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Сохранение файлов в {artifacts_dir}...")

    artifacts = {
        "user_features.pkl": user_features,
        "item_features.pkl": item_features,
        "user_to_idx.pkl": user_to_idx,
        "item_to_idx.pkl": item_to_idx,
    }

    for name, obj in artifacts.items():
        with open(artifacts_dir / name, "wb") as f:
            pickle.dump(obj, f)

    logger.success("Обучение завершено.")


if __name__ == "__main__":
    app()
