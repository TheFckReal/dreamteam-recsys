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
                user_id, item_id, action_type, subdomain, os, model_key, status, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["user_id"],
                str(record["item_id"]),
                record["action_type"],
                record["subdomain"],
                record["os"],
                record["model_key"],
                str(record["status"]),        
                int(record["duration_ms"]),   #время в мс для задачи статистики запросов 
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
