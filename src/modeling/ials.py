from __future__ import annotations

import implicit
from implicit.als import AlternatingLeastSquares
import implicit.cpu.als
import implicit.gpu.als
from loguru import logger
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from src.dataset import ndcg_at_k, precision_recall_at_k


class IALSRecommender:
    """
    Обёртка над ``implicit.als.AlternatingLeastSquares`` для задачи рекомендаций.

    Parameters
    ----------
    factors : int
        Количество латентных факторов.
    regularization : float
        Коэффициент L2-регуляризации.
    alpha : float
        Множитель уверенности для положительных взаимодействий.
    iterations : int
        Количество итераций ALS.
    random_state : int или None
        Сид для воспроизводимости.
    use_gpu : bool или None
        Использовать GPU для обучения. Если ``None`` — автоматически
        выбирается GPU при наличии CUDA, иначе CPU.
    """

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.01,
        alpha: float = 40.0,
        iterations: int = 15,
        random_state: int | None = 42,
        use_gpu: bool | None = None,
    ):
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.random_state = random_state
        self.use_gpu = implicit.gpu.HAS_CUDA if use_gpu is None else use_gpu

        self.model: (
            implicit.gpu.als.AlternatingLeastSquares
            | implicit.cpu.als.AlternatingLeastSquares
            | None
        ) = None
        self.user_items: csr_matrix | None = None
        self.user_to_idx: dict = {}
        self.idx_to_user: dict = {}
        self.item_to_idx: dict = {}
        self.idx_to_item: dict = {}

    def prepare_data(
        self,
        train_data: pl.LazyFrame,
        target_col: str = "log_target",
        min_interactions: int = 5,
    ) -> None:
        """
        Подготавливает CSR-матрицу взаимодействий и маппинги user/item.

        Parameters
        ----------
        train_data : pl.LazyFrame
            Обучающий датафрейм с колонками ``user_id``, ``item_id`` и таргетом.
        target_col : str
            Имя колонки с таргетом (используется как confidence weight).
        min_interactions : int
            Минимальное число взаимодействий для фильтрации пользователей и айтемов.
        """
        df = train_data.select("user_id", "item_id", target_col).collect(engine="streaming")

        # Агрегируем по паре (user, item): суммируем веса всех взаимодействий.
        # Confidence = sum(action_weight) по всем событиям пары — отражает и тип
        # действия (like > click > view), и количество повторений.
        df = df.group_by("user_id", "item_id").agg(pl.col(target_col).sum())

        user_counts = df.group_by("user_id").agg(pl.len().alias("cnt"))
        item_counts = df.group_by("item_id").agg(pl.len().alias("cnt"))

        valid_users = user_counts.filter(pl.col("cnt") >= min_interactions).select("user_id")
        valid_items = item_counts.filter(pl.col("cnt") >= min_interactions).select("item_id")

        df = df.join(valid_users, on="user_id").join(valid_items, on="item_id")
        logger.info(f"Уникальных пар (user, item) после фильтрации: {df.height}")

        user_ids_unique = df.select("user_id").unique().with_row_index("user_idx")
        item_ids_unique = df.select("item_id").unique().with_row_index("item_idx")

        self.user_to_idx = dict(
            zip(user_ids_unique["user_id"].to_list(), user_ids_unique["user_idx"].to_list())
        )
        self.item_to_idx = dict(
            zip(item_ids_unique["item_id"].to_list(), item_ids_unique["item_idx"].to_list())
        )
        self.idx_to_user = {v: k for k, v in self.user_to_idx.items()}
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}

        df_indexed = df.join(user_ids_unique, on="user_id").join(item_ids_unique, on="item_id")

        user_indices = df_indexed["user_idx"].cast(pl.Int32()).to_numpy()
        item_indices = df_indexed["item_idx"].cast(pl.Int32()).to_numpy()
        values = df_indexed[target_col].cast(pl.Float32()).to_numpy()

        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)

        self.user_items = csr_matrix(
            (values, (user_indices, item_indices)),
            shape=(num_users, num_items),
        )
        logger.info(f"CSR-матрица: {num_users} users x {num_items} items")

    def fit(self) -> None:
        """
        Обучает модель iALS на подготовленной CSR-матрице.

        Raises
        ------
        RuntimeError
            Если ``prepare_data`` не был вызван до ``fit``.
        """
        if self.user_items is None:
            raise RuntimeError("Сначала вызовите prepare_data()")

        use_gpu = self.use_gpu and implicit.gpu.HAS_CUDA
        if self.use_gpu and not implicit.gpu.HAS_CUDA:
            logger.warning(
                "GPU запрошен, но CUDA недоступна (implicit собран без GPU-поддержки). "
                "Переключаюсь на CPU."
            )
        logger.info(f"Обучение iALS на {'GPU' if use_gpu else 'CPU'}")

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=self.random_state,
            num_threads=0,
            use_gpu=use_gpu,
        )
        # implicit использует собственный пул потоков через OpenMP.
        # Ограничиваем BLAS до 1 потока, чтобы избежать oversubscription.
        with threadpool_limits(limits=1, user_api="blas"):
            self.model.fit(self.user_items, show_progress=True)
        logger.info("Модель iALS обучена")

    def recommend_batch(
        self,
        user_idxs: np.ndarray,
        top_k: int = 15,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Генерирует top-k рекомендации для батча пользователей.

        Делегирует все пользователи в ``model.recommend()`` за один вызов.
        Внутри implicit использует Cython + OpenMP — параллелит и скоринг,
        и top-k по всем ядрам, не создавая полный ``(n_users, n_items)`` массив.

        Parameters
        ----------
        user_idxs : np.ndarray
            Массив индексов пользователей (по внутренней нумерации).
        top_k : int
            Количество рекомендаций на пользователя.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(item_ids, scores)`` — оба массива формы ``(len(user_idxs), top_k)``.
        """
        if self.model is None or self.user_items is None:
            raise RuntimeError("Модель не обучена")

        k = min(top_k, self.model.item_factors.shape[0])
        ids, scores = self.model.recommend(
            user_idxs,
            self.user_items[user_idxs],
            N=k,
            filter_already_liked_items=True,
        )
        return ids, scores


def run_ials_experiment(
    train_data: pl.LazyFrame,
    test_data: pl.LazyFrame,
    target_col: str = "log_target",
    factors: int = 64,
    regularization: float = 0.01,
    alpha: float = 40.0,
    iterations: int = 15,
    top_k: int = 15,
    min_interactions: int = 5,
    use_gpu: bool | None = None,
) -> dict:
    """
    Запускает полный эксперимент iALS: подготовка, обучение, оценка.

    Parameters
    ----------
    train_data : pl.LazyFrame
        Обучающий датафрейм (должен содержать ``user_id``, ``item_id``, ``target_col``).
    test_data : pl.LazyFrame
        Тестовый датафрейм.
    target_col : str
        Колонка с таргетом (``log_target`` / ``sqrt_target`` / ``target``).
    factors : int
        Количество латентных факторов.
    regularization : float
        Коэффициент L2-регуляризации.
    alpha : float
        Множитель уверенности для положительных примеров.
    iterations : int
        Количество итераций ALS.
    top_k : int
        Количество рекомендаций (top-k).
    min_interactions : int
        Минимальное число взаимодействий для фильтрации.
    use_gpu : bool или None
        Использовать GPU. Если ``None`` — автоматический выбор.

    Returns
    -------
    dict
        Словарь с метриками и гиперпараметрами эксперимента:
        ``target``, ``factors``, ``regularization``, ``alpha``, ``iterations``,
        ``top_k``, ``ndcg``, ``precision``, ``recall``, ``rmse``, ``mae``.
    """
    print(
        f"iALS: target={target_col}, factors={factors}, reg={regularization}, "
        f"alpha={alpha}, iter={iterations}, top_k={top_k}"
    )

    rec = IALSRecommender(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        use_gpu=use_gpu,
    )
    print("Подготовка данных...")
    rec.prepare_data(train_data, target_col=target_col, min_interactions=min_interactions)
    print("Обучение модели...")
    rec.fit()
    print("Обучение модели завершено")
    test_df = test_data.select("user_id", "item_id", target_col).collect(engine="streaming")
    known_users = set(rec.user_to_idx.keys())
    known_items = set(rec.item_to_idx.keys())
    test_filtered = test_df.filter(pl.col("user_id").is_in(known_users))
    print(f"Пользователей в тесте после фильтрации: {test_filtered.select('user_id').n_unique()}")

    user_test_truth = test_filtered.group_by("user_id").agg(
        pl.col("item_id").alias("true_items"),
        pl.col(target_col).alias("relevancy"),
    )

    test_user_ids = user_test_truth["user_id"].to_numpy()
    test_user_idxs = np.array([rec.user_to_idx[uid] for uid in test_user_ids])

    item_idx_matrix, score_matrix = rec.recommend_batch(test_user_idxs, top_k=top_k)

    predicted_items_list = [[rec.idx_to_item[idx] for idx in row] for row in item_idx_matrix]
    predicted_scores_list = [row.tolist() for row in score_matrix]

    prediction_df = pl.DataFrame(
        {
            "user_id": test_user_ids,
            "predicted_items": predicted_items_list,
            "predicted_scores": predicted_scores_list,
        }
    )
    evaluation_df = user_test_truth.join(prediction_df, on="user_id")

    print("Сбор метрик...")
    ndcg_results = ndcg_at_k(
        evaluation_df,
        relevancy_col="relevancy",
        true_items_col="true_items",
        predicted_items_col="predicted_items",
        predicted_score_col="predicted_scores",
        top_k=top_k,
    )
    mean_ndcg = ndcg_results["ndcg"].mean()

    precision, recall = precision_recall_at_k(
        evaluation_df,
        predicted_items_col="predicted_items",
        true_items_col="true_items",
        top_k=top_k,
    )

    # RMSE / MAE: скалярное произведение user_factors и item_factors vs таргет
    test_for_rmse = test_filtered.filter(pl.col("item_id").is_in(known_items))

    assert rec.model is not None
    user_factors = rec.model.user_factors
    item_factors = rec.model.item_factors

    rmse_user_ids = test_for_rmse["user_id"].to_list()
    rmse_item_ids = test_for_rmse["item_id"].to_list()
    true_scores = test_for_rmse[target_col].to_numpy()

    u_idxs = np.array([rec.user_to_idx[uid] for uid in rmse_user_ids])
    i_idxs = np.array([rec.item_to_idx[iid] for iid in rmse_item_ids])

    n = len(u_idxs)
    batch = 500_000
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    for start in range(0, n, batch):
        end = min(start + batch, n)
        pred = np.sum(user_factors[u_idxs[start:end]] * item_factors[i_idxs[start:end]], axis=1)
        diff = true_scores[start:end] - pred
        sq_err_sum += float(np.sum(diff**2))
        abs_err_sum += float(np.sum(np.abs(diff)))

    rmse = float(np.sqrt(sq_err_sum / n))
    mae = float(abs_err_sum / n)

    return {
        "target": target_col,
        "factors": factors,
        "regularization": regularization,
        "alpha": alpha,
        "iterations": iterations,
        "top_k": top_k,
        "ndcg": mean_ndcg,
        "precision": precision,
        "recall": recall,
        "rmse": rmse,
        "mae": mae,
    }
