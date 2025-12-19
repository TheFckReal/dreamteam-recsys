from typing import Annotated, Literal, Union

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
from datetime import datetime

from src.modeling.models import ModelsType
from src.modeling.sharing import InferenceData
import src.service.backend.dependencies as dependencies
from src.service.backend.service import PredictionService, HistoryService


# Создаем зависимость для сервиса
# Это позволяет использовать сервис в разных частях приложения
# и избежать создания новых экземпляров сервиса для каждого запроса
# Depends - это декоратор, который позволяет использовать зависимость в качестве параметра функции
# get_prediction_service - это функция, которая возвращает экземпляр сервиса
# PredictionServiceDep - это тип, который используется для обозначения зависимости
# Annotated - это тип, который используется для обозначения зависимости и позволяет использовать зависимость в качестве параметра функции
PredictionServiceDep = Annotated[PredictionService, Depends(dependencies.get_prediction_service)]
HistoryServiceDep = Annotated[HistoryService, Depends(dependencies.get_history_service)]

# Создаем Pydantic модель для входных данных
# Это необходимо для определения DTO (Data Transfer Object) для входных данных
# DTO - это объект, который используется для передачи данных между слоями приложения
# В данном случае, мы используем DTO для передачи данных от клиента к серверу
# и от сервера к клиенту
# DTO используется для того, чтобы упростить передачу данных и уменьшить количество кода
class ModelInput(BaseModel):
    """Data Transfer Object (DTO) for prediction request."""

    user_id: int
    item_id: Union[str, int]
    action_type: Literal["view", "click", "clickout", "like"]
    subdomain: Literal["u2i", "i2i", "catalog", "search", "other"]
    os: Literal["android", "ios", "other"]
    model_key: ModelsType


class PredictionResponse(BaseModel):
    """DTO for prediction response."""

    user_id: int
    item_id: Union[str, int]
    score: float
    model_key: ModelsType

#DTO для истории
class HistoryItem(BaseModel):
    id: int
    user_id: int
    item_id: Union[str, int]
    action_type: str
    subdomain: str
    os: str
    model_key: ModelsType
    status: str          # "ok" / "error"
    duration_ms: int
    created_at: datetime

app = FastAPI()
# Создаем экземпляр FastAPI
# FastAPI - это фреймворк для создания API серверов на Python
# Он позволяет создавать API сервера с помощью декораторов
# и автоматически генерировать OpenAPI спецификацию


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    """Custom handler for validation errors to return 400 Bad Request."""
    return JSONResponse(
        status_code=400,
        content={"detail": "bad request", "errors": exc.errors()},
    )


@app.post("/forward", response_model=PredictionResponse)
def predict(model_input: ModelInput, prediction_service: PredictionServiceDep):
    """
    Main endpoint for retrieving predictions.
    Delegates logic to PredictionService.
    """
    # Основной эндпоинт для получения предсказаний.
    # Делегирует логику сервису PredictionService.

    # Преобразование внешнего DTO во внутреннюю структуру данных (InferenceData)
    data = InferenceData(
        user_id=model_input.user_id,
        item_id=model_input.item_id,
        action_type=model_input.action_type,
        subdomain=model_input.subdomain,
        os=model_input.os,
    )

    try:
        result = prediction_service.make_prediction(data, model_input.model_key)
    except Exception as e:
        logger.error(f"Failed to make prediction: {e}")
        return JSONResponse(
            status_code=403,
            content={"detail": "модель не смогла обработать данные"},
        )
    
    return PredictionResponse(**result)


@app.get("/history", response_model=list[HistoryItem])
def get_history(history_service: HistoryServiceDep):
    """
    Эндпоинт history для получения всех запросов к /forward.
    """
    rows = history_service.get_all()
    return rows