from functools import lru_cache

from src.service.backend.service import PredictionService
from src.service.backend.service import PredictionService, HistoryService

@lru_cache
def get_prediction_service() -> PredictionService:
    """
    Dependency Provider for PredictionService.
    Uses @lru_cache to implement Singleton (service is created once).
    """
    # Провайдер зависимости для PredictionService.
    # Использует @lru_cache для реализации Singleton (сервис создается один раз).
    service = PredictionService()
    return service

#Провайдер зависимости для HistoryService
@lru_cache
def get_history_service() -> HistoryService:
    return HistoryService()
