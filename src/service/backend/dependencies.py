from functools import lru_cache
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from loguru import logger
from sqlalchemy.orm import Session

from src.config import ALGORITHM, SECRET_KEY
from src.service.backend.database.database import get_db
from src.service.backend.database.tables import User
from src.service.backend.models import TokenData, UserInDB
from src.service.backend.security import verify_password
from src.service.backend.service import HistoryService, PredictionService


# --- Prediction Service Dependency ---
@lru_cache
def get_prediction_service() -> PredictionService:
    """
    Dependency Provider for PredictionService.
    Uses @lru_cache to implement Singleton (service is created once).
    """
    service = PredictionService()
    return service


# Провайдер зависимости для HistoryService
@lru_cache
def get_history_service() -> HistoryService:
    return HistoryService()


# --- Authentication Dependencies ---

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_user(db: Session, username: str):
    """Retrieves user from database by username."""
    user = db.query(User).filter(User.username == username).first()
    if user:
        return UserInDB(
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            disabled=not user.is_active,
            hashed_password=user.hashed_password,
            is_admin=user.is_admin,
        )
    return None


def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        logger.warning(f"User {username} not found in database")
        return None
    if not verify_password(password, user.hashed_password):
        logger.warning(f"Password verification failed for user {username}")
        return None
    return user


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)], db: Annotated[Session, Depends(get_db)]
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.InvalidTokenError:
        raise credentials_exception

    if token_data.username is None:
        raise credentials_exception

    user = get_user(db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_current_admin_user(
    current_user: Annotated[UserInDB, Depends(get_current_active_user)],
):
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not enough privileges")
    return current_user
