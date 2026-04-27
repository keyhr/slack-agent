import sqlite3
import os

DB_PATH = os.getenv("DB_PATH", "/app/data/bot.db")


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_models (
                channel_id TEXT PRIMARY KEY,
                model      TEXT NOT NULL,
                updated_by TEXT,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bot_config (
                key        TEXT PRIMARY KEY,
                value      TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)


def get_model(channel_id: str) -> str | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT model FROM channel_models WHERE channel_id = ?",
            (channel_id,),
        ).fetchone()
    return row[0] if row else None


def set_model(channel_id: str, model: str, updated_by: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO channel_models (channel_id, model, updated_by)
            VALUES (?, ?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                model      = excluded.model,
                updated_by = excluded.updated_by,
                updated_at = datetime('now')
            """,
            (channel_id, model, updated_by),
        )


def get_config(key: str) -> str | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT value FROM bot_config WHERE key = ?", (key,)
        ).fetchone()
    return row[0] if row else None


def set_config(key: str, value: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO bot_config (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value      = excluded.value,
                updated_at = datetime('now')
            """,
            (key, value),
        )


def list_models() -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT channel_id, model, updated_by, updated_at FROM channel_models"
        ).fetchall()
    return [
        {"channel_id": r[0], "model": r[1], "updated_by": r[2], "updated_at": r[3]}
        for r in rows
    ]
