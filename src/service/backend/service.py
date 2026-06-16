import json
import time
from typing import Any, Dict, List

from cachetools import LRUCache
from loguru import logger

import src.modeling.models as models
from src.modeling.sharing import InferenceData
from src.service.backend.database.history_repo import HistoryRepository


# Расчет метрик
def calculate_request_size(data: Dict[str, Any]) -> int:
    """Вычислить размер запроса в байтах"""
    try:
        # Считаем кол-во байтов json
        json_str = json.dumps(data)
        return len(json_str.encode("utf-8"))
    except Exception:
        return 0


def estimate_token_count(data: Dict[str, Any]) -> int:
    """Оценить количество токенов (упрощенная версия)"""
    total_tokens = 0

    if isinstance(data.get("item_id"), str):
        words = len(data["item_id"].split())
        total_tokens += int(words * 0.75)

    for key, value in data.items():
        if isinstance(value, str):
            words = len(value.split())
            total_tokens += int(words * 0.25)

    return total_tokens


class HistoryService:
    def __init__(self, repo: HistoryRepository | None = None):
        self._repo = repo or HistoryRepository()

    def log_request(self, record: dict) -> None:
        # Добавляем расчет метрик перед сохранением
        record["request_size"] = calculate_request_size(
            {
                "user_id": record.get("user_id"),
                "item_id": record.get("item_id"),
                "model_params": record.get("model_params"),
                "model_key": record.get("model_key"),
            }
        )
        record["token_count"] = estimate_token_count(
            {"item_id": record.get("item_id"), "model_params": record.get("model_params")}
        )

        self._repo.add_record(record)

    def get_all(self) -> List[dict]:
        return self._repo.list_all()

    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику запросов"""
        return self._repo.get_statistics()


class PredictionService:
    """
    Service responsible for prediction business logic.
    Manages model lifecycle (loading, caching) and prediction execution.
    """

    # Сервис, отвечающий за бизнес-логику предсказаний.
    # Управляет жизненным циклом моделей (загрузка, кэширование) и выполнением предсказаний.

    def __init__(self, history_service: HistoryService | None = None):
        # LRU кэш для хранения загруженных моделей в памяти.
        # maxsize=2 ограничивает количество одновременно загруженных моделей (экономия памяти).
        self._models_cache: LRUCache[models.ModelsType, models.InferenceModel] = LRUCache(
            maxsize=2
        )
        self._history_service = history_service or HistoryService()

    def make_prediction(self, data: InferenceData, model_key: models.ModelsType) -> dict:
        """
        Main entry point for making a prediction.
        """
        start = time.monotonic()  # замеряем время обработки запроса, чтоб после записать значение в БД для анализа статистики
        status = "ok"  # фиксируем статус запроса

        try:
            # Основной метод для выполнения предсказания.
            model = self._get_or_load_model(model_key)

            logger.info(f"Using model '{model_key}'")
            score = model.predict(data)

        except Exception:
            status = "error"  # обрабатываю ошибки выполнения, меняю статус запроса в случае ошибок предсказания
            logger.exception("Prediction failed")
            raise
        finally:
            duration_ms = int(
                (time.monotonic() - start) * 1000
            )  # вычисляю время обработки запроса

            record = {
                "user_id": data.user_id,
                "item_id": data.item_id,
                "model_params": data.model_params,
                "model_key": model_key,
                "status": status,  # ok или error
                "duration_ms": duration_ms,
            }
            self._history_service.log_request(record)

        return {
            "user_id": data.user_id,
            "item_id": data.item_id,
            "score": score,
            "model_key": model_key,
        }

    def make_recommendations(
        self, user_id: int, model_key: models.ModelsType, top_k: int = 15
    ) -> dict:
        """
        Формирует top-k рекомендаций для пользователя и логирует запрос в историю.
        """
        start = time.monotonic()
        status = "ok"
        recommendations: list[dict] = []

        try:
            model = self._get_or_load_model(model_key)

            logger.info(f"Recommending top-{top_k} with model '{model_key}' for user {user_id}")
            recs = model.recommend(user_id, top_k)
            recommendations = [
                {"rank": rank, "item_id": item_id, "score": float(score)}
                for rank, (item_id, score) in enumerate(recs, start=1)
            ]
        except Exception:
            status = "error"
            logger.exception("Recommendation failed")
            raise
        finally:
            duration_ms = int((time.monotonic() - start) * 1000)
            record = {
                "user_id": user_id,
                "item_id": f"top{top_k}",
                "model_params": {"top_k": top_k},
                "model_key": model_key,
                "status": status,
                "duration_ms": duration_ms,
            }
            self._history_service.log_request(record)

        return {
            "user_id": user_id,
            "model_key": model_key,
            "recommendations": recommendations,
        }

    def _get_or_load_model(self, model_key: models.ModelsType) -> models.InferenceModel:
        """
        Retrieves a model from cache or loads it if it's missing.
        """
        # Получает модель из кэша или загружает её, если она отсутствует.
        if model_key in self._models_cache:
            return self._models_cache[model_key]

        logger.info(f"Model '{model_key}' not in cache. initializing...")

        # 1. Создание экземпляра (Фабрика)
        model_instance = models.create_model(model_key)

        # 2. Загрузка весов (Жизненный цикл)
        # Явный вызов loads() здесь позволяет контролировать момент загрузки ресурсов.
        logger.info(f"Loading weights for '{model_key}'...")
        model_instance.loads()

        # 3. Кэширование
        self._models_cache[model_key] = model_instance

        return model_instance
