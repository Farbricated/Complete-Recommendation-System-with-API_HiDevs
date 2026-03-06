"""
Repository pattern for database access.
Each repository handles CRUD for one domain entity.
"""

import json
import uuid
from datetime import datetime
from data.database import get_db


# ---------------------------------------------------------------------------
# User Repository
# ---------------------------------------------------------------------------

class UserRepository:
    """Handles all user-related database operations."""

    def create(self, username: str, email: str = None, skill_level: str = "beginner",
               preferences: list = None) -> dict:
        user_id = str(uuid.uuid4())
        prefs_json = json.dumps(preferences or [])
        with get_db() as conn:
            conn.execute(
                """INSERT INTO users (user_id, username, email, skill_level, preferences)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, username, email, skill_level, prefs_json)
            )
        return self.get(user_id)

    def get(self, user_id: str) -> dict | None:
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            ).fetchone()
        if row:
            d = dict(row)
            d["preferences"] = json.loads(d.get("preferences") or "[]")
            return d
        return None

    def get_by_username(self, username: str) -> dict | None:
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            ).fetchone()
        if row:
            d = dict(row)
            d["preferences"] = json.loads(d.get("preferences") or "[]")
            return d
        return None

    def list_all(self) -> list[dict]:
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["preferences"] = json.loads(d.get("preferences") or "[]")
            result.append(d)
        return result

    def update_last_active(self, user_id: str):
        with get_db() as conn:
            conn.execute(
                "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?",
                (user_id,)
            )


# ---------------------------------------------------------------------------
# Content Repository
# ---------------------------------------------------------------------------

class ContentRepository:
    """Handles all content-related database operations."""

    def create(self, content_id: str, title: str, description: str, category: str,
               difficulty: str = "beginner", tags: list = None, author: str = None,
               duration_min: int = None, rating: float = 0.0) -> dict:
        tags_json = json.dumps(tags or [])
        with get_db() as conn:
            conn.execute(
                """INSERT INTO content (content_id, title, description, category, difficulty,
                   tags, author, duration_min, rating)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (content_id, title, description, category, difficulty,
                 tags_json, author, duration_min, rating)
            )
        return self.get(content_id)

    def get(self, content_id: str) -> dict | None:
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM content WHERE content_id = ?", (content_id,)
            ).fetchone()
        if row:
            d = dict(row)
            d["tags"] = json.loads(d.get("tags") or "[]")
            return d
        return None

    def list_all(self) -> list[dict]:
        with get_db() as conn:
            rows = conn.execute("SELECT * FROM content ORDER BY created_at").fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d.get("tags") or "[]")
            result.append(d)
        return result

    def get_by_category(self, category: str) -> list[dict]:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM content WHERE category = ? ORDER BY rating DESC",
                (category,)
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d.get("tags") or "[]")
            result.append(d)
        return result

    def get_popular(self, limit: int = 10) -> list[dict]:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM content ORDER BY view_count DESC, rating DESC LIMIT ?",
                (limit,)
            ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["tags"] = json.loads(d.get("tags") or "[]")
            result.append(d)
        return result

    def increment_views(self, content_id: str):
        with get_db() as conn:
            conn.execute(
                "UPDATE content SET view_count = view_count + 1 WHERE content_id = ?",
                (content_id,)
            )

    def add_skill(self, content_id: str, skill_id: str, relevance: float = 1.0):
        with get_db() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO content_skills (content_id, skill_id, relevance)
                   VALUES (?, ?, ?)""",
                (content_id, skill_id, relevance)
            )

    def get_skills(self, content_id: str) -> list[str]:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT skill_id FROM content_skills WHERE content_id = ? ORDER BY relevance DESC",
                (content_id,)
            ).fetchall()
        return [r["skill_id"] for r in rows]


# ---------------------------------------------------------------------------
# Interaction Repository
# ---------------------------------------------------------------------------

class InteractionRepository:
    """Handles user-content interaction storage and retrieval."""

    # Weight map for scoring
    EVENT_WEIGHTS = {
        "view": 1.0,
        "click": 1.5,
        "bookmark": 2.0,
        "complete": 3.0,
        "rate": 2.5,
        "skip": -0.5,
    }

    def record(self, user_id: str, content_id: str, event_type: str,
               rating: float = None, duration_sec: int = None,
               session_id: str = None) -> str:
        interaction_id = str(uuid.uuid4())
        with get_db() as conn:
            conn.execute(
                """INSERT INTO interactions
                   (interaction_id, user_id, content_id, event_type, rating, duration_sec, session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (interaction_id, user_id, content_id, event_type, rating, duration_sec, session_id)
            )
        return interaction_id

    def get_user_history(self, user_id: str, limit: int = 100) -> list[dict]:
        with get_db() as conn:
            rows = conn.execute(
                """SELECT * FROM interactions WHERE user_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (user_id, limit)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_user_item_matrix(self) -> dict:
        """Build user→{content_id: score} matrix from all interactions."""
        with get_db() as conn:
            rows = conn.execute(
                "SELECT user_id, content_id, event_type, rating FROM interactions"
            ).fetchall()
        matrix = {}
        for row in rows:
            uid, cid, etype, rating = row["user_id"], row["content_id"], row["event_type"], row["rating"]
            if uid not in matrix:
                matrix[uid] = {}
            weight = self.EVENT_WEIGHTS.get(etype, 1.0)
            if etype == "rate" and rating is not None:
                weight = (rating / 5.0) * 3.0
            matrix[uid][cid] = matrix[uid].get(cid, 0.0) + weight
        return matrix

    def get_content_interaction_counts(self) -> dict:
        """Return {content_id: interaction_count} for popularity scoring."""
        with get_db() as conn:
            rows = conn.execute(
                "SELECT content_id, COUNT(*) as cnt FROM interactions GROUP BY content_id"
            ).fetchall()
        return {r["content_id"]: r["cnt"] for r in rows}

    def get_user_interacted_content(self, user_id: str) -> set:
        """Return set of content IDs the user has already interacted with."""
        with get_db() as conn:
            rows = conn.execute(
                "SELECT DISTINCT content_id FROM interactions WHERE user_id = ?",
                (user_id,)
            ).fetchall()
        return {r["content_id"] for r in rows}


# ---------------------------------------------------------------------------
# Log Repository
# ---------------------------------------------------------------------------

class LogRepository:
    """Handles API request logging."""

    def log(self, endpoint: str, method: str, status_code: int,
            response_ms: float, request_id: str, user_id: str = None):
        log_id = str(uuid.uuid4())
        with get_db() as conn:
            conn.execute(
                """INSERT INTO request_logs
                   (log_id, user_id, endpoint, method, status_code, response_ms, request_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (log_id, user_id, endpoint, method, status_code, response_ms, request_id)
            )

    def get_metrics(self) -> dict:
        """Aggregate performance metrics from logs."""
        with get_db() as conn:
            stats = conn.execute(
                """SELECT endpoint,
                          COUNT(*) as total_requests,
                          AVG(response_ms) as avg_ms,
                          MIN(response_ms) as min_ms,
                          MAX(response_ms) as max_ms,
                          SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as errors
                   FROM request_logs
                   GROUP BY endpoint"""
            ).fetchall()
        return [dict(s) for s in stats]