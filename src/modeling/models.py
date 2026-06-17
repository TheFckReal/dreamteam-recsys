from __future__ import annotations

import random
from time import sleep
from typing import TYPE_CHECKING, Callable, Dict, Literal

from loguru import logger
import polars as pl

from src.modeling.interface import InferenceModel
from src.modeling.sharing import InferenceData
from src.modeling.vae import MultiVAERecommender
from src.config import SVD_DIR, IALS_DIR, VAE_DIR

import pickle
import numpy as np
import torch
from pathlib import Path

if TYPE_CHECKING:
    # IALSRecommender тянет за собой ``implicit``. На Windows системе он не устанавливается напрямую
    # поэтому импортируем его лениво.
    from src.modeling.ials import IALSRecommender

# --- Registry Infrastructure ---

# Реестр моделей хранит фабричные функции для создания экземпляров моделей.
# Это позволяет добавлять новые модели без изменения кода сервиса (паттерн Registry).
_MODELS_REGISTRY: Dict[str, Callable[[], InferenceModel]] = {}


def register_model(key: str):
    """
    Decorator to register a model factory in the registry.
    Usage: @register_model("model_name")
    """
    # Декоратор для регистрации фабрики модели в реестре.

    def decorator(factory_func: Callable[[], InferenceModel]):
        _MODELS_REGISTRY[key] = factory_func
        return factory_func

    return decorator


def create_model(model_key: str) -> InferenceModel:
    """
    Factory method to create a model instance by key.
    """
    # Фабричный метод для создания экземпляра модели по ключу.
    factory = _MODELS_REGISTRY.get(model_key)
    if not factory:
        raise ValueError(
            f"Model '{model_key}' not found. Available: {list(_MODELS_REGISTRY.keys())}"
        )
    return factory()


ModelsType = Literal["dummy", "svd_v1", "ials_v1", "vae_v1"]


# --- Model Implementations ---


class DummyModel(InferenceModel):
    """
    Example model implementation (stub) for testing purposes.
    """

    # Пример реализации модели (заглушка) для тестов.

    def __init__(self, fail_probability: float = 0.2, load_time: float = 3):
        super().__init__()
        self.fail_probability = fail_probability
        self.load_time = load_time
        self._is_loaded = False

    def _predict_single(self, data: InferenceData) -> float:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded")
        if random.random() < self.fail_probability:
            raise ValueError("Model failed to predict (simulated error)")
        return 0.5

    def _predict_dataframe(self, data: pl.DataFrame) -> pl.Series:
        if not self._is_loaded:
            raise RuntimeError("Model is not loaded")
        return pl.Series([0.5] * len(data))

    def loads(self) -> None:
        """Simulates loading weights from disk/network."""
        if self._is_loaded:
            logger.warning("Model is already loaded")
            return
        sleep(self.load_time)
        self._is_loaded = True


# --- Factory Implementations ---


@register_model("dummy")
def _create_dummy_model() -> InferenceModel:
    """
    Factory for DummyModel.
    Here you can inject configuration from environment variables or config files.
    """
    # Example: prob = float(os.getenv("DUMMY_FAIL_PROB", 0.1))
    return DummyModel(fail_probability=0.1, load_time=1.0)


class SVDModel(InferenceModel):
    def __init__(self):
        super().__init__()
        self.user_features = None
        self.item_features = None
        self.user_to_idx = None
        self.item_to_idx = None
        self._is_loaded = False
        self.artifacts_path = Path(SVD_DIR)

    def loads(self) -> None:
        """Загрузка весов."""
        if self._is_loaded:
            return

        logger.info("Загрузка SVD модели...")
        try:
            with open(self.artifacts_path / "user_features.pkl", "rb") as f:
                self.user_features = pickle.load(f)

            with open(self.artifacts_path / "item_features.pkl", "rb") as f:
                self.item_features = pickle.load(f)

            with open(self.artifacts_path / "user_to_idx.pkl", "rb") as f:
                self.user_to_idx = pickle.load(f)

            with open(self.artifacts_path / "item_to_idx.pkl", "rb") as f:
                self.item_to_idx = pickle.load(f)

            self._is_loaded = True
            logger.info("SVD модель успешно загружена!")
        except Exception as e:
            logger.error(f"Ошибка загрузки SVD: {e}")
            raise e

    def _predict_single(self, data: InferenceData) -> float:
        """Предсказание Score для пары (user_id, item_id)."""
        if not self._is_loaded:
            raise RuntimeError("SVD Model is not loaded")

        u_idx = self.user_to_idx.get(data.user_id)
        i_idx = self.item_to_idx.get(data.item_id)

        if u_idx is None or i_idx is None:
            return 0.0

        u_vec = self.user_features[u_idx]
        i_vec = self.item_features[i_idx]

        score = np.dot(u_vec, i_vec)

        return float(score)

    def _predict_dataframe(self, data: pl.DataFrame) -> pl.Series:
        raise NotImplementedError("Batch prediction is not implemented for SVDModel")


class IALSModel(InferenceModel):
    def __init__(self):
        super().__init__()
        self.recommender: IALSRecommender | None = None
        self._is_loaded = False
        self.artifacts_path = Path(IALS_DIR)

    def loads(self) -> None:
        """Загрузка весов."""
        if self._is_loaded:
            return

        logger.info("Загрузка IALS модели...")
        try:
            # Импорт модуля делается лениво, чтобы остальные модели
            # (Dummy/SVD/VAE) могли работать без установленного ``implicit``.
            import src.modeling.ials  # noqa: F401

            with open(self.artifacts_path / "model.pkl", "rb") as f:
                self.recommender = pickle.load(f)

            self._is_loaded = True
            logger.info("IALS модель успешно загружена!")
        except Exception as e:
            logger.error(f"Ошибка загрузки IALS: {e}")
            raise e

    def _ensure_ready(self) -> IALSRecommender:
        if not self._is_loaded or self.recommender is None or self.recommender.model is None:
            raise RuntimeError("IALS Model is not loaded")
        return self.recommender

    def _predict_single(self, data: InferenceData) -> float:
        rec = self._ensure_ready()
        assert rec.model is not None

        u_idx = rec.user_to_idx.get(data.user_id)
        i_idx = rec.item_to_idx.get(data.item_id)

        if u_idx is None or i_idx is None:
            return 0.0

        score = np.dot(rec.model.user_factors[u_idx], rec.model.item_factors[i_idx])
        return float(score)

    def _predict_dataframe(self, data: pl.DataFrame) -> pl.Series:
        rec = self._ensure_ready()
        assert rec.model is not None

        user_ids = data["user_id"].to_list()
        item_ids = data["item_id"].to_list()

        user_factors = rec.model.user_factors
        item_factors = rec.model.item_factors

        u_idxs = [rec.user_to_idx.get(uid) for uid in user_ids]
        i_idxs = [rec.item_to_idx.get(iid) for iid in item_ids]

        scores = np.zeros(len(user_ids), dtype=np.float32)
        valid = np.array([(u is not None and i is not None) for u, i in zip(u_idxs, i_idxs)])
        if valid.any():
            valid_u = np.array([u for u, v in zip(u_idxs, valid) if v])
            valid_i = np.array([i for i, v in zip(i_idxs, valid) if v])
            scores[valid] = np.sum(user_factors[valid_u] * item_factors[valid_i], axis=1)

        return pl.Series("score", scores)


@register_model("svd_v1")
def _create_svd_model() -> InferenceModel:
    return SVDModel()


@register_model("ials_v1")
def _create_ials_model() -> InferenceModel:
    return IALSModel()


class VAEModel(InferenceModel):
    """Inference-обёртка над Mult-VAE.

    Артефакты загружаются из ``VAE_DIR``:
    - ``model.pt`` — torch-чекпоинт (state_dict + config + маппинги);
    - ``user_items.npz`` — sparse user-history (для скоринга и фильтрации
      уже виденных айтемов).
    """

    def __init__(self) -> None:
        super().__init__()
        self.recommender: MultiVAERecommender | None = None
        self._is_loaded = False
        self.artifacts_path = Path(VAE_DIR)

    def loads(self) -> None:
        if self._is_loaded:
            return

        logger.info("Загрузка Mult-VAE модели...")
        try:
            from scipy.sparse import load_npz

            rec = MultiVAERecommender()
            rec.load(self.artifacts_path / "model.pt")
            user_items = load_npz(self.artifacts_path / "user_items.npz")
            rec.set_user_items(user_items)
            self.recommender = rec
            self._is_loaded = True
            logger.info("Mult-VAE модель успешно загружена.")
        except Exception as e:
            logger.error(f"Ошибка загрузки Mult-VAE: {e}")
            raise

    def _ensure_ready(self) -> MultiVAERecommender:
        if not self._is_loaded or self.recommender is None or self.recommender.model is None:
            raise RuntimeError("VAE Model is not loaded")
        return self.recommender

    def _score_pair(self, rec: MultiVAERecommender, user_id: int, item_id: str) -> float:
        u_idx = rec.user_to_idx.get(user_id)
        i_idx = rec.item_to_idx.get(item_id)
        if u_idx is None or i_idx is None:
            return 0.0
        assert rec.user_items is not None
        history = rec.user_items[u_idx].toarray().astype(np.float32)
        history_t = torch.from_numpy(history)
        logits = rec.score_batch(history_t)
        return float(logits[0, i_idx])

    def _predict_single(self, data: InferenceData) -> float:
        rec = self._ensure_ready()
        return self._score_pair(rec, data.user_id, str(data.item_id))

    def _predict_dataframe(self, data: pl.DataFrame) -> pl.Series:
        rec = self._ensure_ready()
        assert rec.user_items is not None

        user_ids = data["user_id"].to_list()
        item_ids = [str(i) for i in data["item_id"].to_list()]

        # Группируем запросы по user_id, чтобы вызывать форвард VAE один раз
        # на пользователя — иначе ``_predict_single`` для каждой пары
        # пересчитывает энкодер на одной и той же истории.
        unique_users = list({uid: None for uid in user_ids}.keys())
        u_idxs = [rec.user_to_idx.get(uid) for uid in unique_users]
        valid_mask = np.array([u is not None for u in u_idxs])
        valid_users = [unique_users[i] for i in np.where(valid_mask)[0]]
        valid_idxs = np.array([u for u in u_idxs if u is not None], dtype=np.int64)

        # Логиты по всем айтемам для валидных пользователей.
        if len(valid_idxs) > 0:
            history = rec.user_items[valid_idxs].toarray().astype(np.float32)
            logits = rec.score_batch(torch.from_numpy(history))
            user_to_logits = dict(zip(valid_users, logits))
        else:
            user_to_logits = {}

        scores = np.zeros(len(user_ids), dtype=np.float32)
        for k, (uid, iid) in enumerate(zip(user_ids, item_ids)):
            row = user_to_logits.get(uid)
            if row is None:
                continue
            i_idx = rec.item_to_idx.get(iid)
            if i_idx is None:
                continue
            scores[k] = row[i_idx]

        return pl.Series("score", scores)


@register_model("vae_v1")
def _create_vae_model() -> InferenceModel:
    return VAEModel()
