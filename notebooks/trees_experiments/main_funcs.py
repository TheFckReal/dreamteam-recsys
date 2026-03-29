# в файле описаны основные функции, используемые в работе, для импорта в ноутбук
import polars as pl
import numpy as np


def global_temporal_split(
    df: pl.LazyFrame, test_size: int | float = 1, date_column: str = "day"
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Разделяет датасет на обучающую и тестовую части на основе глобальной временной границы. 1 день между тестовой и обучающей частью игнорируется.

    Args:
        df: Датасет для разделения
        date_column: Имя столбца с датами
        test_size: Количество дней в тестовой части или доля от общего количества дней

    Returns:
        Кортеж из двух датасетов: обучающая и тестовая части
    """
    min_day, max_day = (
        df.select(
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



def split_cold_start(train_df: pl.LazyFrame, test_df: pl.LazyFrame, user_col: str = "user_id"):
    """
    Split test data into cold-start and non-cold-start subsets by users.

    Parameters
    ----------
    train_df : pl.LazyFrame
        Training data. Used to determine which users are already known to the model.
    test_df : pl.LazyFrame
        Test data that will be split into cold-start and non-cold-start parts.
    user_col : str, optional
        Name of the column containing user identifiers, by default "user_id".

    Returns
    -------
    tuple[pl.LazyFrame, pl.LazyFrame]
        A tuple of two LazyFrames:

        - first element : test subset with cold-start users
          (users present in `test_df` but not in `train_df`);
        - second element : test subset with non-cold-start users
          (users present both in `train_df` and `test_df`).
    """
    cold_start_users = test_df.select(pl.col(user_col).unique()).join(
        train_df.select(pl.col(user_col).unique()), on=user_col, how="anti"
    )
    return test_df.join(cold_start_users, on=user_col), test_df.join(
        cold_start_users, on=user_col, how="anti"
    )


def ndcg_at_k(
    user_based_df: pl.DataFrame,
    relevancy_col: str,
    true_items_col: str,
    predicted_items_col: str,
    predicted_score_col: str,
    top_k: int = 15,
):
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


def calculate_metrics(df, k):
    """
    Расчет Precision@k и Recall@k
    df: датафрейм с колонками predicted_items и true_items
    k: количество рекомендаций (TOPK)
    """
    # Обрезаем список предсказаний до k элементов
    top_k_preds = pl.col("predicted_items").list.head(k)
    
    # Ищем пересечение обрезанного списка с правдой
    hits_expr = top_k_preds.list.set_intersection(pl.col("true_items")).list.len()
    
    # Вычисляем метрики одной командой select
    metrics = df.select(
        # Precision = (кол-во попаданий / k), берем среднее по всем юзерам
        (hits_expr / k).mean().alias('precision'),
        
        # Recall = (кол-во попаданий / длину реального списка), берем среднее
        (hits_expr / pl.col('true_items').list.len()).mean().alias('recall')
    )
    
    precision_val = metrics['precision'].item()
    recall_val = metrics['recall'].item()

    return precision_val, recall_val


def calculate_regression_metrics_vectorized(
    test_data,
    user_features: np.ndarray,
    item_features: np.ndarray,
    user_to_idx: dict,
    item_to_idx: dict,
    target_col: str = "log_target",
) -> dict:
    
    # Фильтруем только известные пары user-item
    test_filtered = test_data.filter(
        pl.col("user_id").is_in(list(user_to_idx.keys())) &
        pl.col("item_id").is_in(list(item_to_idx.keys()))
    )
    
    # Получаем индексы
    user_ids = test_filtered["user_id"].to_list()
    item_ids = test_filtered["item_id"].to_list()
    true_scores = test_filtered[target_col].to_numpy()
    
    user_idxs = np.array([user_to_idx[uid] for uid in user_ids])
    item_idxs = np.array([item_to_idx[iid] for iid in item_ids])
    
    # Вычисление скоров
    # pred_score[i] = user_features[user_idx[i]] @ item_features[:, item_idx[i]]
    pred_scores = np.sum(
        user_features[user_idxs] * item_features[:, item_idxs].T, 
        axis=1
    )
    
    # Метрики
    errors = true_scores - pred_scores
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    
    return {
        "rmse": rmse,
        "mae": mae,
        "n_samples": len(true_scores),
    }

