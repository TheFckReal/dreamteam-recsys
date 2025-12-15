from functools import lru_cache

from src.service.backend.service import PredictionService


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
