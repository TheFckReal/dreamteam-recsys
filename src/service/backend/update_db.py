from loguru import logger
from sqlalchemy import text

from src.service.backend.database.database import engine


def update_database():
    """Добавляем недостающие колонки в таблицу history"""

    with engine.connect() as conn:
        result = conn.execute(text("PRAGMA table_info(history)"))
        columns = [row[1] for row in result.fetchall()]

        new_columns = [("request_size", "INTEGER DEFAULT 0"), ("token_count", "INTEGER DEFAULT 0")]

        for column_name, column_type in new_columns:
            if column_name not in columns:
                try:
                    conn.execute(
                        text(f"ALTER TABLE history ADD COLUMN {column_name} {column_type}")
                    )
                    logger.info(f"Added column {column_name}")
                except Exception as e:
                    logger.error(f"Error adding column {column_name}: {e}")

        conn.commit()


if __name__ == "__main__":
    update_database()
