from datetime import datetime
from typing import Annotated, Union

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict

from src.modeling.models import ModelsType
from src.modeling.sharing import InferenceData
from src.service.backend.database.database import Base, engine, get_db
from src.service.backend.database.tables import User
import src.service.backend.dependencies as dependencies
from src.service.backend.models import UserInDB
from src.service.backend.routers import authorization
from src.service.backend.security import get_password_hash
from src.service.backend.service import HistoryService, PredictionService

# Создаем таблицы при запуске (для простоты)
Base.metadata.create_all(bind=engine)

# Создаем зависимость для сервиса
PredictionServiceDep = Annotated[PredictionService, Depends(dependencies.get_prediction_service)]
HistoryServiceDep = Annotated[HistoryService, Depends(dependencies.get_history_service)]
# Создаем зависимости для пользователей
CurrentUser = Annotated[UserInDB, Depends(dependencies.get_current_active_user)]
AdminUser = Annotated[UserInDB, Depends(dependencies.get_current_admin_user)]


class BaseModelInput(BaseModel):
    """Data Transfer Object (DTO) for prediction request."""

    user_id: int
    item_id: Union[str, int]
    # action_type: Literal["view", "click", "clickout", "like"]
    # subdomain: Literal["u2i", "i2i", "catalog", "search", "other"]
    # os: Literal["android", "ios", "other"]
    model_key: ModelsType


class PredictionResponse(BaseModel):
    """DTO for prediction response."""

    user_id: int
    item_id: Union[str, int]
    score: float
    model_key: ModelsType


# DTO для истории
class HistoryItem(BaseModel):
    id: int
    user_id: int
    item_id: Union[str, int]
    model_params: dict | None
    model_key: ModelsType
    status: str  # "ok" / "error"
    duration_ms: int
    created_at: datetime


class StatisticsResponse(BaseModel):
    total_requests: int
    avg_duration_ms: float
    min_duration_ms: int
    max_duration_ms: int
    success_rate: float
    duration_quantiles: dict
    request_characteristics: dict
    by_model: list
    by_subdomain: list

    model_config = ConfigDict(extra="allow", protected_namespaces=())


app = FastAPI()

# Подключаем роутер авторизации
app.include_router(authorization.router)


# Функция для инициализации начальных данных
def init_db():
    db = next(get_db())
    if not db.query(User).filter(User.username == "admin").first():
        logger.info("Initializing DB with admin user...")
        admin = User(
            username="admin",
            email="admin@example.com",
            full_name="Admin User",
            hashed_password=get_password_hash("admin"),
            is_active=True,
            is_admin=True,
        )
        db.add(admin)
        db.commit()

    # Специальный тестовый пользователь для проверки ассистентам :3
    if not db.query(User).filter(User.username == "dreamer").first():
        logger.info("Initializing DB with test user...")
        user = User(
            username="dreamer",
            email="dreamer@example.com",
            full_name="Dreamer",
            hashed_password=get_password_hash("secret"),
            is_active=True,
            is_admin=False,
        )
        db.add(user)
        db.commit()


# Запускаем инициализацию при старте
init_db()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    """Custom handler for validation errors to return 400 Bad Request."""
    return JSONResponse(
        status_code=400,
        content={"detail": "bad request", "errors": exc.errors()},
    )


@app.post("/forward", response_model=PredictionResponse)
def predict(
    model_input: BaseModelInput,
    prediction_service: PredictionServiceDep,
    model_params: dict | None = None,
):
    """
    Главный эндпоинт для получения предсказаний.

    """

    data = InferenceData(
        user_id=model_input.user_id, item_id=model_input.item_id, model_params=model_params
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


@app.get("/admin/info")
def admin_info(current_user: AdminUser):
    """
    Example of an admin-only endpoint.
    Only accessible if is_admin=True in user DB.
    """
    return {"status": "admin_access_granted", "user": current_user.username}


app.include_router(authorization.router)


@app.get("/history", response_model=list[HistoryItem])
def get_history(history_service: HistoryServiceDep, current_user: CurrentUser):
    """
    Эндпоинт history для получения всех запросов к /forward.
    """
    logger.info(f"User {current_user.username} requested history")
    rows = history_service.get_all()
    return rows


@app.get("/stats", response_model=StatisticsResponse)
def get_statistics(history_service: HistoryServiceDep):
    stats = history_service.get_statistics()

    total = stats.get("total_requests", 0)
    success_count = stats.get("success_count", 0)

    return StatisticsResponse(
        total_requests=total,
        avg_duration_ms=float(stats.get("avg_duration", 0)),
        min_duration_ms=int(stats.get("min_duration", 0)),
        max_duration_ms=int(stats.get("max_duration", 0)),
        success_rate=success_count / total if total > 0 else 0,
        duration_quantiles=stats.get("duration_quantiles", {}),
        request_characteristics={
            "avg_request_size_bytes": float(stats.get("avg_request_size", 0)),
            "avg_token_count": float(stats.get("avg_token_count", 0)),
            "distinct_models": len([m for m in stats.get("by_model", []) if m]),
            "distinct_subdomains": len([s for s in stats.get("by_subdomain", []) if s]),
        },
        by_model=stats.get("by_model", []),
        by_subdomain=stats.get("by_subdomain", []),
    )
