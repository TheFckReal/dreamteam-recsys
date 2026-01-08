from dataclasses import dataclass
from typing import Literal, Optional, Union


@dataclass
class InferenceData:
    """
    Data Transfer Object (DTO) for model input parameters.
    Used to standardize inputs across all models.
    """

    # Структура данных (DTO) для входных параметров модели.
    # Используется для стандартизации входов всех моделей.
    user_id: int
    item_id: Union[str, int]
    action_type: Literal["view", "click", "clickout", "like"]
    subdomain: Literal["u2i", "i2i", "catalog", "search", "other"]
    os: Literal["android", "ios", "other"]


@dataclass
class PredictionResult:
    """
    Data Transfer Object (DTO) for prediction results.
    """

    # Структура данных для результата предсказания.
    prediction: float
    metadata: Optional[dict] = None

@dataclass
class TopNRequestData:
    user_id: int
    n: int = 10  
    action_type: str = "view"
    subdomain: str = "u2i"
    os: str = "android"