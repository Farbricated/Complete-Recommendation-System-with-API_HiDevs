"""
Database connection and schema management for the recommendation system.
Uses SQLite with a clean repository pattern.
"""

import sqlite3
import os
import threading
from contextlib import contextmanager

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "..", "recommendation.db"))

# Thread-local storage for connections
_local = threading.local()


def get_connection():
    """Get a thread-local database connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


@contextmanager
def get_db():
    """Context manager for database operations with automatic commit/rollback."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db():
    """Initialize the database schema."""
    conn = get_connection()
    cursor = conn.cursor()

    # Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id     TEXT PRIMARY KEY,
            username    TEXT NOT NULL UNIQUE,
            email       TEXT UNIQUE,
            skill_level TEXT CHECK(skill_level IN ('beginner','intermediate','advanced')) DEFAULT 'beginner',
            preferences TEXT,          -- JSON array of preferred categories
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_active DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Content (courses/articles) table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content (
            content_id   TEXT PRIMARY KEY,
            title        TEXT NOT NULL,
            description  TEXT,
            category     TEXT NOT NULL,
            difficulty   TEXT CHECK(difficulty IN ('beginner','intermediate','advanced')),
            tags         TEXT,          -- JSON array
            author       TEXT,
            duration_min INTEGER,
            rating       REAL DEFAULT 0.0,
            view_count   INTEGER DEFAULT 0,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Skills table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS skills (
            skill_id    TEXT PRIMARY KEY,
            name        TEXT NOT NULL UNIQUE,
            category    TEXT NOT NULL,
            description TEXT,
            parent_skill TEXT REFERENCES skills(skill_id),
            level       INTEGER DEFAULT 1   -- hierarchy depth
        )
    """)

    # Content-Skill mapping
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_skills (
            content_id TEXT REFERENCES content(content_id),
            skill_id   TEXT REFERENCES skills(skill_id),
            relevance  REAL DEFAULT 1.0,
            PRIMARY KEY (content_id, skill_id)
        )
    """)

    # User interactions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            interaction_id TEXT PRIMARY KEY,
            user_id        TEXT NOT NULL REFERENCES users(user_id),
            content_id     TEXT NOT NULL REFERENCES content(content_id),
            event_type     TEXT NOT NULL CHECK(event_type IN ('view','click','complete','rate','bookmark','skip')),
            rating         REAL,          -- 1-5 if rated
            duration_sec   INTEGER,       -- time spent
            timestamp      DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id     TEXT
        )
    """)

    # Request logs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS request_logs (
            log_id       TEXT PRIMARY KEY,
            user_id      TEXT,
            endpoint     TEXT NOT NULL,
            method       TEXT NOT NULL,
            status_code  INTEGER,
            response_ms  REAL,
            request_id   TEXT UNIQUE,
            timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create indexes for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_content ON interactions(content_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_category ON content(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON request_logs(timestamp)")

    conn.commit()
    print("✅ Database initialized successfully")


def close_connection():
    """Close the thread-local connection."""
    if hasattr(_local, "conn") and _local.conn:
        _local.conn.close()
        _local.conn = None