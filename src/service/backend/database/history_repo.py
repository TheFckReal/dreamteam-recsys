from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import case, func

from src.service.backend.database.database import SessionLocal
from src.service.backend.database.tables import History


class HistoryRepository:
    def add_record(self, record: Dict[str, Any]) -> None:
        db = SessionLocal()
        try:
            history_item = History(
                user_id=record["user_id"],
                item_id=str(record["item_id"]),
                model_params=record["model_params"],
                model_key=record["model_key"],
                status=str(record["status"]),
                duration_ms=int(record["duration_ms"]),
                request_size=record.get("request_size", 0),
                token_count=int(record.get("token_count", 0)),
                created_at=datetime.now(),
            )
            db.add(history_item)
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def list_all(self) -> List[dict]:
        db = SessionLocal()
        try:
            items = db.query(History).order_by(History.created_at.desc()).all()
            return [
                {
                    "id": item.id,
                    "user_id": item.user_id,
                    "item_id": item.item_id,
                    "model_params": item.model_params,
                    "model_key": item.model_key,
                    "status": item.status,
                    "duration_ms": item.duration_ms,
                    "request_size": item.request_size,
                    "token_count": item.token_count,
                    "created_at": item.created_at,
                }
                for item in items
            ]
        finally:
            db.close()

    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику по всем запросам"""
        db = SessionLocal()
        try:
            # Метрики по времени выполнения
            stats_query = db.query(
                func.count(History.id).label("total_requests"),
                func.avg(History.duration_ms).label("avg_duration"),
                func.min(History.duration_ms).label("min_duration"),
                func.max(History.duration_ms).label("max_duration"),
                func.sum(case((History.status == "ok", 1), else_=0)).label("success_count"),
                func.sum(case((History.status == "error", 1), else_=0)).label("error_count"),
            )

            result = stats_query.first()
            if result is None:
                return {"total_requests": 0, "avg_duration": 0.0, "min_duration": 0,
                        "max_duration": 0, "success_count": 0, "error_count": 0,
                        "duration_quantiles": {"p50": 0, "p95": 0, "p99": 0},
                        "avg_request_size": 0.0, "avg_token_count": 0.0,
                        "distinct_models": 0, "by_model": [], "by_model_key": []}
            stats = {
                "total_requests": int(result.total_requests or 0),
                "avg_duration": float(result.avg_duration or 0),
                "min_duration": int(result.min_duration or 0),
                "max_duration": int(result.max_duration or 0),
                "success_count": int(result.success_count or 0),
                "error_count": int(result.error_count or 0),
            }

            # Квантили распределения времени выполнения
            durations = (
                db.query(History.duration_ms)
                .filter(History.duration_ms.isnot(None))
                .order_by(History.duration_ms)
                .all()
            )
            durations = [int(d[0]) for d in durations]

            # Статистики по размеру запросов
            size_stats_query = db.query(
                func.avg(History.request_size).label("avg_request_size"),
                func.avg(History.token_count).label("avg_token_count"),
                func.count(func.distinct(History.model_key)).label("distinct_models"),
            )
            size_result = size_stats_query.first()
            size_stats = {
                "avg_request_size": float((size_result.avg_request_size if size_result else None) or 0),
                "avg_token_count": float((size_result.avg_token_count if size_result else None) or 0),
                "distinct_models": int((size_result.distinct_models if size_result else None) or 0),
            }

            # Вычисляем квантили
            if durations:
                n = len(durations)
                stats["duration_quantiles"] = {
                    "p50": durations[int(n * 0.5)] if n > 0 else 0,
                    "p95": durations[int(n * 0.95)] if n > 1 else durations[-1],
                    "p99": durations[int(n * 0.99)] if n > 1 else durations[-1],
                }
            else:
                stats["duration_quantiles"] = {"p50": 0, "p95": 0, "p99": 0}

            # Объединяем статистики
            stats.update(size_stats)

            # Статистика по моделям
            by_model = (
                db.query(
                    History.model_key,
                    func.count(History.id).label("request_count"),
                    func.avg(History.duration_ms).label("avg_duration"),
                    func.avg(History.request_size).label("avg_request_size"),
                    func.avg(History.token_count).label("avg_token_count"),
                )
                .group_by(History.model_key)
                .all()
            )

            stats["by_model"] = [
                {
                    "model_key": row.model_key,
                    "request_count": int(row.request_count),
                    "avg_duration": float(row.avg_duration or 0),
                    "avg_request_size": float(row.avg_request_size or 0),
                    "avg_token_count": float(row.avg_token_count or 0),
                }
                for row in by_model
            ]

            # Статистика по поддоменам
            by_model_key = (
                db.query(
                    History.model_key,
                    func.count(History.id).label("request_count"),
                    func.avg(History.duration_ms).label("avg_duration"),
                )
                .group_by(History.model_key)
                .all()
            )

            stats["by_model_key"] = [
                {
                    "model_key": row.model_key,
                    "request_count": int(row.request_count),
                    "avg_duration": float(row.avg_duration or 0),
                }
                for row in by_model_key
            ]

            return stats
        finally:
            db.close()
