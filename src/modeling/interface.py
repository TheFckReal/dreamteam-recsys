from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, List, Union, cast, overload

import polars as pl

from src.modeling.sharing import InferenceData

PredictionInput = Union[InferenceData, pl.DataFrame, Iterable[InferenceData]]
PredictionOutput = Union[float, List[float], pl.Series]


class InferenceModel(ABC):
    """
    Abstract Base Class (interface) for all models.
    Defines a unified contract for loading weights and making predictions.
    """

    # Базовый абстрактный класс (интерфейс) для всех моделей.
    # Определяет единый контракт для загрузки весов и предсказания.

    def __init__(
        self,
    ):
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @is_loaded.setter
    def is_loaded(self) -> None:
        raise ValueError("is_loaded is a read-only property")

    @overload
    def predict(self, data: InferenceData) -> float: ...
    @overload
    def predict(self, data: pl.DataFrame) -> pl.Series: ...
    @overload
    def predict(self, data: Iterable[InferenceData]) -> Iterable[float]: ...

    def predict(self, data: PredictionInput) -> Any:
        """
        Universal prediction method.
        Automatically determines input type (single object, DataFrame, or batch)
        and calls the corresponding internal method.
        """
        # Универсальный метод предсказания.
        # Автоматически определяет тип входных данных (единичный объект, DataFrame или batch)
        # и вызывает соответствующий внутренний метод (_predict_single, _predict_dataframe, _predict_batch).
        if isinstance(data, InferenceData):
            return self._predict_single(data)
        elif isinstance(data, pl.DataFrame):
            return self._predict_dataframe(data)
        elif isinstance(data, Iterable):
            return self._predict_batch(cast(Iterable[InferenceData], data))
        else:
            raise TypeError(f"Invalid input type: {type(data)}")

    @abstractmethod
    def _predict_single(self, data: InferenceData) -> float:
        """
        Internal method to predict for a single item.
        """
        pass

    def _predict_batch(self, data: Iterable[InferenceData]) -> List[float]:
        """
        Internal method to predict for a batch of items.
        Default implementation iterates over _predict_single.
        Override this for vectorized implementations if possible.
        """
        return [self._predict_single(item) for item in data]

    @abstractmethod
    def _predict_dataframe(self, data: pl.DataFrame) -> pl.Series:
        """
        Internal method to predict for a Polars DataFrame.
        Should be optimized using vector operations.
        """
        pass

    @abstractmethod
    def loads(self) -> None:
        """
        Loads "heavy" resources (model weights, DB connections).
        Must be called explicitly before using the model.
        """
        # Метод для загрузки "тяжелых" ресурсов (весов модели, соединений с БД).
        # Должен вызываться явно перед использованием модели.
        pass
