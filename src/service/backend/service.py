from cachetools import LRUCache
from loguru import logger

import src.modeling.models as models
from src.modeling.sharing import InferenceData


class PredictionService:
    """
    Service responsible for prediction business logic.
    Manages model lifecycle (loading, caching) and prediction execution.
    """

    # Сервис, отвечающий за бизнес-логику предсказаний.
    # Управляет жизненным циклом моделей (загрузка, кэширование) и выполнением предсказаний.

    def __init__(self):
        # LRU кэш для хранения загруженных моделей в памяти.
        # maxsize=2 ограничивает количество одновременно загруженных моделей (экономия памяти).
        self._models_cache: LRUCache[models.ModelsType, models.InferenceModel] = LRUCache(
            maxsize=2
        )

    def make_prediction(self, data: InferenceData, model_key: models.ModelsType) -> dict:
        """
        Main entry point for making a prediction.
        """
        # Основной метод для выполнения предсказания.
        model = self._get_or_load_model(model_key)

        logger.info(f"Using model '{model_key}' for subdomain '{data.subdomain}'")
        score = model.predict(data)

        return {
            "user_id": data.user_id,
            "item_id": data.item_id,
            "score": score,
            "model_key": model_key,
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
