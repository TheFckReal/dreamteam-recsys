from cachetools import LRUCache
from loguru import logger
import time
import json


import src.modeling.models as models
from src.modeling.sharing import InferenceData

from src.service.backend.database.history_repo import HistoryRepository
from typing import List, Dict, Any


# Расчет метрик
def calculate_request_size(data: Dict[str, Any]) -> int:
    """Вычислить размер запроса в байтах"""
    try:
        # Считаем кол-во байтов json
        json_str = json.dumps(data)
        return len(json_str.encode('utf-8'))
    except:
        return 0
    
def estimate_token_count(data: Dict[str, Any]) -> int:
    """Оценить количество токенов (упрощенная версия)"""
    total_tokens = 0
    
    if isinstance(data.get('item_id'), str):
        words = len(data['item_id'].split())
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
        record["request_size"] = calculate_request_size({
            "user_id": record.get("user_id"),
            "item_id": record.get("item_id"),
            "action_type": record.get("action_type"),
            "subdomain": record.get("subdomain"),
            "os": record.get("os"),
            "model_key": record.get("model_key")
        })
        record["token_count"] = estimate_token_count({
            "item_id": record.get("item_id"),
            "action_type": record.get("action_type"),
            "subdomain": record.get("subdomain")
        })
        
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
        start = time.monotonic() #замеряем время обработки запроса, чтоб после записать значение в БД для анализа статистики
        status = "ok" #фиксируем статус запроса

        try:
            # Основной метод для выполнения предсказания.
            model = self._get_or_load_model(model_key)

            logger.info(f"Using model '{model_key}' for subdomain '{data.subdomain}'")
            score = model.predict(data)

        except Exception as e:
            status = "error" #обрабатываю ошибки выполнения, меняю статус запроса в случае ошибок предсказания
            logger.exception("Prediction failed")
            raise
        finally:
            duration_ms = int((time.monotonic() - start) * 1000) #вычисляю время обработки запроса

            record = {
                "user_id": data.user_id,
                "item_id": data.item_id,
                "action_type": data.action_type,   
                "subdomain": data.subdomain,
                "os": data.os,                     
                "model_key": model_key,
                "status": status,                  #ok или error
                "duration_ms": duration_ms,
            }
            self._history_service.log_request(record)

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
    


