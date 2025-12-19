import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "history.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # чтобы получать dict-подобные строки
    return conn

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        #duration_ms INTEGER - время обработки запроса
        """
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            item_id TEXT NOT NULL,
            action_type TEXT NOT NULL,
            subdomain TEXT NOT NULL,
            os TEXT NOT NULL,
            model_key TEXT NOT NULL,
            status TEXT NOT NULL,
            duration_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("DB initialized")
    