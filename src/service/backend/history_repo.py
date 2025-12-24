from typing import List, Dict, Any
from src.service.backend.history_db import get_connection

class HistoryRepository:
    def add_record(self, record: Dict[str, Any]) -> None:
        conn = get_connection()
        cur = conn.cursor() #объект типа Cursor, через который делаем запросы к БД
        cur.execute( #разбирает SQL строку, подставляет значения параметров и отдаёт команду движку SQLite на выполнение
#аргументы метода cur.execute("INSERT INTO history (...) VALUES (?, ?, ...)", params)

            """
            INSERT INTO history (
                user_id, item_id, action_type, subdomain, os, model_key, status, duration_ms, request_size, token_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["user_id"],
                str(record["item_id"]),
                record["action_type"],
                record["subdomain"],
                record["os"],
                record["model_key"],
                str(record["status"]),        
                int(record["duration_ms"]),           #время в мс для задачи статистики запросов 
                record.get("request_size", 0),        #размер запроса
                int(record.get("token_count", 0)),    #кол-во токенов
            ),
        )
        conn.commit() #сохраняем изменения, чтоб не потерять
        conn.close() 

    def list_all(self) -> List[dict]:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM history ORDER BY created_at DESC")
        rows = cur.fetchall() #получаем результат запроса
        conn.close()
        return [dict(row) for row in rows]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику по всем запросам"""
        conn = get_connection()
        cur = conn.cursor()
        
        # Метрики по времени выполнения
        cur.execute("""
            select 
                count(*) as total_requests,
                avg(duration_ms) as avg_duration,
                min(duration_ms) as min_duration,
                max(duration_ms) as max_duration,
                sum(CASE WHEN status = 'ok' THEN 1 ELSE 0 END) as success_count,
                sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count
            from history
        """)
        stats = dict(cur.fetchone())
        
        # Квантили распределения времени выполнения
        cur.execute("""
            SELECT 
                duration_ms as duration
            FROM history 
            WHERE duration_ms IS NOT NULL 
            ORDER BY duration_ms
        """)
        durations = [row[0] for row in cur.fetchall()]
        
        # Статистики по размеру запросов
        cur.execute("""
            SELECT 
                AVG(request_size) as avg_request_size,
                AVG(token_count) as avg_token_count,
                COUNT(DISTINCT model_key) as distinct_models,
                COUNT(DISTINCT subdomain) as distinct_subdomains
            FROM history
        """)
        size_stats = dict(cur.fetchone())
        
        conn.close()
        
        # Вычисляем квантили
        if durations:
            n = len(durations)
            stats["duration_quantiles"] = {
                "p50": durations[int(n * 0.5)] if n > 0 else 0,
                "p95": durations[int(n * 0.95)] if n > 1 else durations[-1],
                "p99": durations[int(n * 0.99)] if n > 1 else durations[-1]
            }
        else:
            stats["duration_quantiles"] = {"p50": 0, "p95": 0, "p99": 0}
        
        # Объединяем статистики
        stats.update(size_stats)
        
        # Дополнительные группировки
        conn = get_connection()
        cur = conn.cursor()
        
        # Статистика по моделям
        cur.execute("""
            SELECT 
                model_key,
                COUNT(*) as request_count,
                AVG(duration_ms) as avg_duration,
                AVG(request_size) as avg_request_size
            FROM history 
            GROUP BY model_key
        """)
        stats["by_model"] = [dict(row) for row in cur.fetchall()]
        
        # Статистика по поддоменам
        cur.execute("""
            SELECT 
                subdomain,
                COUNT(*) as request_count,
                AVG(duration_ms) as avg_duration
            FROM history 
            GROUP BY subdomain
        """)
        stats["by_subdomain"] = [dict(row) for row in cur.fetchall()]
        
        conn.close()
        
        return stats

