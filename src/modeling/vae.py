from __future__ import annotations

from dataclasses import dataclass
import time

from loguru import logger
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset import ndcg_at_k, precision_recall_at_k


@dataclass
class MultiVAEConfig:
    """Гиперпараметры Mult-VAE.

    Parameters
    ----------
    num_items : int
        Размер выходного слоя — мощность каталога.
    encoder_dims : list[int]
        Скрытые размерности энкодера от входа к латенту, не включая ``num_items``.
        Последний элемент — размер ``z``. Например, ``[600, 200]`` означает
        ``num_items -> 600 -> 2*200`` (последний слой удваивается под mu/logvar).
    dropout : float
        Дропаут на входной нормированной user-history.
    beta : float
        Максимальный вес KL-члена. Используется в ``annealing``-обновлении.
    """

    num_items: int
    encoder_dims: list[int]
    dropout: float = 0.5
    beta: float = 0.2


class MultiVAE(nn.Module):
    """Mult-VAE: симметричный энкодер/декодер с softmax-выходом по items."""

    def __init__(self, config: MultiVAEConfig) -> None:
        super().__init__()
        self.config = config

        # Размерности: вход -> encoder_dims[:-1] -> 2*encoder_dims[-1] (mu, logvar)
        # Декодер симметричен: encoder_dims[-1] -> encoder_dims[:-1][::-1] -> num_items
        latent_dim = config.encoder_dims[-1]
        enc_hidden = config.encoder_dims[:-1]

        enc_layers: list[nn.Module] = []
        prev = config.num_items
        for h in enc_hidden:
            enc_layers.append(nn.Linear(prev, h))
            prev = h
        enc_layers.append(nn.Linear(prev, 2 * latent_dim))
        self.encoder_layers = nn.ModuleList(enc_layers)

        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for h in reversed(enc_hidden):
            dec_layers.append(nn.Linear(prev, h))
            prev = h
        dec_layers.append(nn.Linear(prev, config.num_items))
        self.decoder_layers = nn.ModuleList(dec_layers)

        self.drop = nn.Dropout(config.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in list(self.encoder_layers) + list(self.decoder_layers):
            if isinstance(layer, nn.Linear):
                # Truncated normal init из оригинальной реализации Mult-VAE.
                nn.init.xavier_normal_(layer.weight)
                nn.init.normal_(layer.bias, std=1e-3)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # L2-нормализация user-history — стандартный прием Mult-VAE.
        h = F.normalize(x, p=2, dim=1)
        h = self.drop(h)
        for i, layer in enumerate(self.encoder_layers):
            h = layer(h)
            if i != len(self.encoder_layers) - 1:
                h = torch.tanh(h)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for i, layer in enumerate(self.decoder_layers):
            h = layer(h)
            if i != len(self.decoder_layers) - 1:
                h = torch.tanh(h)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar


def multivae_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ELBO Mult-VAE: multinomial log-likelihood + ``beta * KL``.

    Negative log-likelihood: ``-mean_u sum_i target_ui * log_softmax(logits)_ui``.
    KL: closed-form для ``N(mu, sigma) || N(0, I)``.
    """
    log_softmax = F.log_softmax(logits, dim=1)
    neg_ll = -(log_softmax * target).sum(dim=1).mean()
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return neg_ll + beta * kld, neg_ll, kld


class MultiVAERecommender:
    """Высокоуровневая обертка над ``MultiVAE``.

    По духу повторяет :class:`src.modeling.ials.IALSRecommender`: хранит
    маппинги user/item, sparse user-history и обученные веса, умеет генерить
    top-k рекомендации батчами.

    Parameters
    ----------
    encoder_dims : list[int]
        Архитектура энкодера (см. :class:`MultiVAEConfig`). По умолчанию
        ``[600, 200]`` — конфиг из оригинальной статьи.
    dropout : float
        Дропаут входной user-history.
    beta : float
        Максимальный вес KL-члена.
    device : str or None
        Имя torch-устройства. ``None`` — авто (``cuda`` если доступна).
    random_state : int or None
        Сид для воспроизводимости инициализации модели.
    """

    def __init__(
        self,
        encoder_dims: list[int] | None = None,
        dropout: float = 0.5,
        beta: float = 0.2,
        device: str | None = None,
        random_state: int | None = 42,
    ) -> None:
        self.encoder_dims = encoder_dims or [600, 200]
        self.dropout = dropout
        self.beta = beta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state

        self.model: MultiVAE | None = None
        self.user_items: csr_matrix | None = None
        self.user_to_idx: dict = {}
        self.idx_to_user: dict = {}
        self.item_to_idx: dict = {}
        self.idx_to_item: dict = {}

    # ------------------------------------------------------------------
    # Подготовка маппингов и user-history
    # ------------------------------------------------------------------

    def set_vocab(
        self,
        user_to_idx: dict,
        item_to_idx: dict,
    ) -> None:
        """Устанавливает маппинги user/item, построенные внешним пайплайном."""
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.idx_to_user = {v: k for k, v in user_to_idx.items()}
        self.idx_to_item = {v: k for k, v in item_to_idx.items()}

    def set_user_items(self, user_items: csr_matrix) -> None:
        """Сохраняет sparse user-history (для ``filter_already_liked_items``)."""
        self.user_items = user_items.astype(np.float32)

    def prepare_data(
        self,
        train_data: pl.LazyFrame,
        target_col: str = "log_target",
        min_interactions: int = 5,
        binary: bool = True,
    ) -> None:
        """Готовит CSR-матрицу и маппинги user/item из событий.

        Повторяет API :meth:`src.modeling.ials.IALSRecommender.prepare_data`,
        чтобы обе модели запускались из одного ноутбука одинаково.

        Parameters
        ----------
        train_data : pl.LazyFrame
            События с колонками ``user_id``, ``item_id``, ``target_col``.
        target_col : str
            Имя колонки с весом взаимодействия.
        min_interactions : int
            Минимум уникальных items на пользователя (и наоборот).
        binary : bool
            ``True`` — бинаризовать матрицу (как в оригинальном Mult-VAE).
            ``False`` — сохранить взвешенный сигнал из ``target_col``.
        """
        df = train_data.select("user_id", "item_id", target_col).collect(engine="streaming")
        df = df.group_by("user_id", "item_id").agg(pl.col(target_col).sum())

        user_counts = df.group_by("user_id").agg(pl.len().alias("cnt"))
        item_counts = df.group_by("item_id").agg(pl.len().alias("cnt"))
        valid_users = user_counts.filter(pl.col("cnt") >= min_interactions).select("user_id")
        valid_items = item_counts.filter(pl.col("cnt") >= min_interactions).select("item_id")
        df = df.join(valid_users, on="user_id").join(valid_items, on="item_id")
        logger.info("Уникальных пар (user, item) после фильтрации: {}", df.height)

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
        rows = df_indexed["user_idx"].cast(pl.Int32).to_numpy()
        cols = df_indexed["item_idx"].cast(pl.Int32).to_numpy()
        if binary:
            vals = np.ones(len(rows), dtype=np.float32)
        else:
            vals = df_indexed[target_col].cast(pl.Float32).to_numpy()

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        self.user_items = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items))
        logger.info("CSR-матрица: {} users x {} items", n_users, n_items)

    # ------------------------------------------------------------------
    # In-memory обучение (для ноутбуков и hyperopt)
    # ------------------------------------------------------------------

    def fit(
        self,
        epochs: int = 30,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        total_anneal_steps: int = 200_000,
        kl_schedule: str = "linear",
        use_lr_scheduler: bool = True,
        show_progress: bool = True,
    ) -> None:
        """Обучает Mult-VAE на подготовленной CSR-матрице.

        Это in-memory-вариант (для ноутбуков и hyperopt). Для OOM-обучения
        с memory-mapped Arrow есть отдельный пайплайн в
        :mod:`src.modeling.train_vae`.
        """
        if self.user_items is None:
            raise RuntimeError("Сначала вызовите prepare_data()")

        from src.modeling.train_vae import KLAnnealer

        if self.model is None:
            self.build_model()
        assert self.model is not None
        user_items = self.user_items
        model = self.model
        n_users = user_items.shape[0]  # type: ignore[union-attr]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        kl_annealer = KLAnnealer(
            beta_max=self.beta,
            total_anneal_steps=total_anneal_steps,
            schedule=kl_schedule,
        )

        steps_per_epoch = (n_users + batch_size - 1) // batch_size
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        if use_lr_scheduler:
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(steps_per_epoch * epochs, 1),
                eta_min=lr / 100.0,
            )

        rng = np.random.default_rng(self.random_state)

        epoch_iter = (
            tqdm(range(1, epochs + 1), desc="Эпохи", unit="epoch")
            if show_progress
            else range(1, epochs + 1)
        )
        for epoch in epoch_iter:
            model.train()
            order = rng.permutation(n_users)
            ep_loss = ep_nll = ep_kld = 0.0
            n_batches = 0
            t0 = time.time()

            batch_indices = range(0, n_users, batch_size)
            inner_iter = (
                tqdm(
                    batch_indices,
                    total=steps_per_epoch,
                    desc=f"Epoch {epoch:02d}",
                    unit="batch",
                    leave=False,
                )
                if show_progress
                else batch_indices
            )
            for start in inner_iter:
                idxs = order[start : start + batch_size]
                # Нарезаем sparse строки и материализуем dense только для текущего батча.
                history = torch.from_numpy(user_items[idxs].toarray().astype(np.float32)).to(
                    self.device, non_blocking=True
                )

                current_beta = kl_annealer.step()
                optimizer.zero_grad(set_to_none=True)
                logits, mu, logvar = model(history)
                # Локальный импорт во избежание мусора в namespace модуля.
                from src.modeling.vae import multivae_loss as _ml

                loss, nll, kld = _ml(logits, history, mu, logvar, current_beta)
                loss.backward()
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()

                ep_loss += float(loss.item())
                ep_nll += float(nll.item())
                ep_kld += float(kld.item())
                n_batches += 1
                if show_progress and isinstance(inner_iter, tqdm):
                    inner_iter.set_postfix(
                        loss=f"{loss.item():.3f}",
                        kld=f"{kld.item():.3f}",
                        beta=f"{current_beta:.3f}",
                    )

            dt = time.time() - t0
            logger.info(
                "Epoch {:02d}: loss={:.4f} nll={:.4f} kld={:.4f} beta={:.4f} time={:.1f}s",
                epoch,
                ep_loss / max(n_batches, 1),
                ep_nll / max(n_batches, 1),
                ep_kld / max(n_batches, 1),
                kl_annealer.last_value,
                dt,
            )

    # ------------------------------------------------------------------
    # Инициализация модели
    # ------------------------------------------------------------------

    def build_model(self) -> MultiVAE:
        """Создает ``MultiVAE`` с актуальным ``num_items``."""
        if not self.item_to_idx:
            raise RuntimeError("Сначала вызовите set_vocab() с непустым item_to_idx")
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        config = MultiVAEConfig(
            num_items=len(self.item_to_idx),
            encoder_dims=list(self.encoder_dims),
            dropout=self.dropout,
            beta=self.beta,
        )
        self.model = MultiVAE(config).to(self.device)
        logger.info(
            "MultiVAE собрана: num_items={}, dims={}, device={}",
            config.num_items,
            config.encoder_dims,
            self.device,
        )
        return self.model

    # ------------------------------------------------------------------
    # Инференс
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_batch(self, history_batch: torch.Tensor) -> np.ndarray:
        """Возвращает логиты ``(B, num_items)`` для батча user-history."""
        self._ensure_ready()
        assert self.model is not None
        self.model.eval()
        x = history_batch.to(self.device)
        logits, _, _ = self.model(x)
        return logits.cpu().numpy()

    @torch.no_grad()
    def recommend_batch(
        self,
        user_idxs: np.ndarray,
        top_k: int = 15,
        filter_already_liked_items: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k рекомендации для батча пользователей.

        Parameters
        ----------
        user_idxs : np.ndarray
            Массив индексов пользователей по внутренней нумерации.
        top_k : int
            Сколько рекомендаций возвращать на пользователя.
        filter_already_liked_items : bool
            Маскировать ли уже виденные user-history items (требует ``set_user_items``).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(item_idxs, scores)`` — оба массива формы ``(len(user_idxs), top_k)``.
        """
        self._ensure_ready()
        assert self.model is not None
        if self.user_items is None:
            raise RuntimeError("Сначала вызовите set_user_items() с user-history матрицей")

        history = torch.from_numpy(self.user_items[user_idxs].toarray().astype(np.float32))
        logits = self.score_batch(history)

        if filter_already_liked_items:
            # Маскируем уже виденные items, чтобы не рекомендовать их снова.
            seen = self.user_items[user_idxs].toarray() > 0
            logits[seen] = -np.inf

        k = min(top_k, logits.shape[1])
        # argpartition — O(n) до сортировки top-k; быстрее, чем full argsort.
        part = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(logits.shape[0])[:, None]
        part_scores = logits[rows, part]
        # Финально сортируем top-k по score внутри каждого пользователя.
        order = np.argsort(-part_scores, axis=1)
        item_idxs = part[rows, order]
        scores = part_scores[rows, order]
        return item_idxs, scores

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------

    def save(self, path) -> None:
        """Сохраняет веса, конфиг и маппинги одним torch-чекпоинтом."""
        from pathlib import Path

        self._ensure_ready()
        assert self.model is not None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "config": {
                "encoder_dims": self.encoder_dims,
                "dropout": self.dropout,
                "beta": self.beta,
            },
            "user_to_idx": self.user_to_idx,
            "item_to_idx": self.item_to_idx,
        }
        torch.save(payload, path)
        logger.info("MultiVAE сохранена в {}", path)

    def load(self, path, device: str | None = None) -> None:
        """Загружает чекпоинт, восстанавливает архитектуру и маппинги."""
        from pathlib import Path

        path = Path(path)
        # weights_only=False — допускаем dict с пользовательскими маппингами;
        # доверяем только собственным артефактам.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.encoder_dims = payload["config"]["encoder_dims"]
        self.dropout = payload["config"]["dropout"]
        self.beta = payload["config"]["beta"]
        self.set_vocab(payload["user_to_idx"], payload["item_to_idx"])
        if device is not None:
            self.device = device
        self.build_model()
        assert self.model is not None
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        logger.info("MultiVAE загружена из {}", path)

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> None:
        if self.model is None:
            raise RuntimeError("Модель не инициализирована: вызовите build_model() или load()")


# ---------------------------------------------------------------------------
# Полный эксперимент (для ноутбуков / Optuna), API повторяет run_ials_experiment
# ---------------------------------------------------------------------------


def run_vae_experiment(
    train_data: pl.LazyFrame,
    test_data: pl.LazyFrame,
    target_col: str = "log_target",
    *,
    encoder_dims: list[int] | None = None,
    dropout: float = 0.5,
    beta: float = 0.2,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    total_anneal_steps: int = 200_000,
    kl_schedule: str = "linear",
    binary: bool = True,
    top_k: int = 15,
    min_interactions: int = 5,
    device: str | None = None,
    random_state: int | None = 42,
    show_progress: bool = True,
) -> dict:
    """Полный эксперимент Mult-VAE: подготовка → обучение → оценка.

    API подобрано идентичным :func:`src.modeling.ials.run_ials_experiment`,
    чтобы оба алгоритма прогонялись из одного ноутбука одинаково.

    Parameters
    ----------
    train_data : pl.LazyFrame
        События для обучения (``user_id``, ``item_id``, ``target_col``).
    test_data : pl.LazyFrame
        События для оценки (та же схема).
    target_col : str
        Колонка с весом взаимодействия.
    encoder_dims : list[int]
        Архитектура энкодера (последний элемент — латент). Default ``[600, 200]``.
    dropout, beta : float
        Гиперпараметры Mult-VAE (см. :class:`MultiVAEConfig`).
    epochs, batch_size, lr, weight_decay, total_anneal_steps, kl_schedule :
        Параметры тренировочного цикла.
    binary : bool
        Бинаризовать матрицу взаимодействий (как в оригинальном Mult-VAE).
    top_k : int
        Размер top-k для NDCG / Precision / Recall.
    min_interactions : int
        Фильтр редких пользователей/айтемов (по числу пар).
    device : str or None
        ``cpu`` / ``cuda`` / ``None`` (auto).
    random_state : int or None
        Seed.
    show_progress : bool
        Прогресс-бары tqdm.

    Returns
    -------
    dict
        Метрики и гиперпараметры эксперимента: ``target``, ``encoder_dims``,
        ``dropout``, ``beta``, ``epochs``, ``batch_size``, ``lr``, ``top_k``,
        ``ndcg``, ``precision``, ``recall``, ``rmse``, ``mae``.
    """
    print(
        f"Mult-VAE: target={target_col}, dims={encoder_dims or [600, 200]}, "
        f"dropout={dropout}, beta={beta}, epochs={epochs}, bs={batch_size}, top_k={top_k}"
    )

    rec = MultiVAERecommender(
        encoder_dims=encoder_dims,
        dropout=dropout,
        beta=beta,
        device=device,
        random_state=random_state,
    )
    print("Подготовка данных...")
    rec.prepare_data(
        train_data,
        target_col=target_col,
        min_interactions=min_interactions,
        binary=binary,
    )
    print("Обучение модели...")
    rec.fit(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        total_anneal_steps=total_anneal_steps,
        kl_schedule=kl_schedule,
        show_progress=show_progress,
    )
    print("Обучение модели завершено")

    # --- подготовка test ground truth ----------------------------------
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

    # --- top-k рекомендации батчами (чтобы не материализовать все логиты) ---
    print("Сбор top-k рекомендаций...")
    BATCH = 1024
    item_idx_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    rec_iter = range(0, len(test_user_idxs), BATCH)
    if show_progress:
        rec_iter = tqdm(list(rec_iter), desc="recommend_batch", unit="chunk")
    for start in rec_iter:
        chunk = test_user_idxs[start : start + BATCH]
        ids, scores = rec.recommend_batch(chunk, top_k=top_k)
        item_idx_chunks.append(ids)
        score_chunks.append(scores)
    item_idx_matrix = np.concatenate(item_idx_chunks, axis=0)
    score_matrix = np.concatenate(score_chunks, axis=0)

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

    # --- RMSE / MAE: VAE-логиты vs target_col на парах из теста --------
    test_for_rmse = test_filtered.filter(pl.col("item_id").is_in(known_items))
    rmse_user_ids = test_for_rmse["user_id"].to_list()
    rmse_item_ids = test_for_rmse["item_id"].to_list()
    true_scores = test_for_rmse[target_col].to_numpy().astype(np.float32)

    rmse_user_idxs = np.array([rec.user_to_idx[uid] for uid in rmse_user_ids])
    rmse_item_idxs = np.array([rec.item_to_idx[iid] for iid in rmse_item_ids])

    n = len(rmse_user_idxs)
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    chunk = 4096
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        u_chunk = rmse_user_idxs[start:end]
        i_chunk = rmse_item_idxs[start:end]
        unique_u, inv = np.unique(u_chunk, return_inverse=True)
        assert rec.user_items is not None
        history = torch.from_numpy(rec.user_items[unique_u].toarray().astype(np.float32))
        logits = rec.score_batch(history)
        pred = logits[inv, i_chunk]
        diff = true_scores[start:end] - pred
        sq_err_sum += float(np.sum(diff**2))
        abs_err_sum += float(np.sum(np.abs(diff)))
    rmse = float(np.sqrt(sq_err_sum / max(n, 1)))
    mae = float(abs_err_sum / max(n, 1))

    return {
        "target": target_col,
        "encoder_dims": list(rec.encoder_dims),
        "dropout": dropout,
        "beta": beta,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "binary": binary,
        "top_k": top_k,
        "ndcg": mean_ndcg,
        "precision": precision,
        "recall": recall,
        "rmse": rmse,
        "mae": mae,
    }
