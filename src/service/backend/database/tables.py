from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, func
from sqlalchemy.orm import Mapped, mapped_column

from src.service.backend.database.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(unique=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(unique=True, index=True, nullable=True)
    full_name: Mapped[Optional[str]] = mapped_column(nullable=True)
    hashed_password: Mapped[str]
    is_active: Mapped[bool] = mapped_column(default=True)
    is_admin: Mapped[bool] = mapped_column(default=False)


class History(Base):
    __tablename__ = "history"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    user_id: Mapped[int]
    item_id: Mapped[str]
    model_key: Mapped[str]
    model_params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str]
    duration_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    request_size: Mapped[int] = mapped_column(default=0)
    token_count: Mapped[int] = mapped_column(default=0)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
