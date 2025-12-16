import random
from time import sleep
from typing import Callable, Dict, Literal

from loguru import logger
import polars as pl

from src.modeling.interface import InferenceModel
from src.modeling.sharing import InferenceData

import pickle
import numpy as np
from pathlib import Path

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


ModelsType = Literal["dummy", "svd_v1"]


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
        # Путь к артефактам (папка, куда ты положила .pkl файлы)
        self.artifacts_path = Path("src/artifacts")

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
        i_vec = self.item_features[:, i_idx]

        score = np.dot(u_vec, i_vec)

        return float(score)

    def _predict_dataframe(self, data: pl.DataFrame) -> pl.Series:

        raise NotImplementedError("Batch prediction is not implemented for SVDModel")

@register_model("svd_v1")
def _create_svd_model() -> InferenceModel:

    return SVDModel()

