import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "history.db"

def update_database():
    """Добавляем недостающие колонки в таблицу history"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(history)")
    columns = [col[1] for col in cursor.fetchall()]
    
    new_columns = [
        ("request_size", "INTEGER DEFAULT 0"),
        ("token_count", "INTEGER DEFAULT 0")
    ]
    
    for column_name, column_type in new_columns:
        if column_name not in columns:
            try:
                cursor.execute(f"ALTER TABLE history ADD COLUMN {column_name} {column_type}")
            except sqlite3.Error as e:
                print(f"✗ Ошибка добавления колонки {column_name}: {e}")
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    update_database()