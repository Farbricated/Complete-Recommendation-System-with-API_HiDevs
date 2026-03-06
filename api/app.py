"""
Flask REST API for the Recommendation System.
Endpoints: /recommend, /feedback, /health, /metrics, /similar
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import uuid
import logging
from functools import wraps
from flask import Flask, request, jsonify, g

from data.database import init_db
from data.repositories import UserRepository, ContentRepository, InteractionRepository, LogRepository
from engine.orchestrator import RecommendationOrchestrator

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("recommender_api")

# Shared singletons
orchestrator = RecommendationOrchestrator(cache_ttl=300)
user_repo    = UserRepository()
content_repo = ContentRepository()
interaction_repo = InteractionRepository()
log_repo     = LogRepository()

# Simple API key auth (in production use proper auth)
VALID_API_KEYS = {"dev-key-123", "test-key-456"}


# ---------------------------------------------------------------------------
# Middleware & Decorators
# ---------------------------------------------------------------------------

def require_api_key(f):
    """Decorator to enforce API key authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key") or request.args.get("api_key")
        if key not in VALID_API_KEYS:
            return jsonify({"error": "Unauthorized. Provide a valid X-API-Key header."}), 401
        return f(*args, **kwargs)
    return decorated


@app.before_request
def before_request():
    """Attach request metadata before each request."""
    g.start_time = time.time()
    g.request_id = str(uuid.uuid4())
    logger.info(f"→ {request.method} {request.path} [req={g.request_id}]")


@app.after_request
def after_request(response):
    """Log request completion and timing."""
    elapsed_ms = round((time.time() - g.start_time) * 1000, 2)
    response.headers["X-Request-ID"] = g.request_id
    response.headers["X-Response-Time-MS"] = str(elapsed_ms)

    user_id = None
    try:
        body = request.get_json(silent=True) or {}
        user_id = body.get("user_id") or request.args.get("user_id")
    except Exception:
        pass

    try:
        log_repo.log(
            endpoint=request.path,
            method=request.method,
            status_code=response.status_code,
            response_ms=elapsed_ms,
            request_id=g.request_id,
            user_id=user_id,
        )
    except Exception:
        pass

    logger.info(f"← {response.status_code} {request.path} [{elapsed_ms}ms]")
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    Returns system status and basic statistics.
    """
    stats = orchestrator.get_stats()
    return jsonify({
        "status": "healthy",
        "service": "recommendation-api",
        "version": "1.0.0",
        "stats": stats,
    }), 200


@app.route("/recommend", methods=["POST"])
@require_api_key
def recommend():
    """
    Generate personalized recommendations for a user.

    Body (JSON):
        user_id   (str, required)
        n         (int, optional, default=10)
        diversity (bool, optional, default=true)

    Returns:
        List of recommended content items with scores and explanations.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 422

    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "user_id is required"}), 422

    n         = min(int(data.get("n", 10)), 50)
    diversity = bool(data.get("diversity", True))

    result = orchestrator.recommend(user_id=user_id, n=n, diversity=diversity)

    if "error" in result:
        return jsonify(result), 404

    return jsonify(result), 200


@app.route("/feedback", methods=["POST"])
@require_api_key
def feedback():
    """
    Record user interaction/feedback on content.

    Body (JSON):
        user_id     (str, required)
        content_id  (str, required)
        event_type  (str, required) - one of: view, click, complete, rate, bookmark, skip
        rating      (float, optional) - 1-5 if event_type is 'rate'
        duration_sec(int, optional)

    Returns:
        Confirmation with interaction ID.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    required = ["user_id", "content_id", "event_type"]
    missing  = [f for f in required if not data.get(f)]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 422

    valid_events = {"view", "click", "complete", "rate", "bookmark", "skip"}
    if data["event_type"] not in valid_events:
        return jsonify({"error": f"event_type must be one of: {', '.join(valid_events)}"}), 422

    # Validate user and content exist
    user    = user_repo.get(data["user_id"])
    content = content_repo.get(data["content_id"])
    if not user:
        return jsonify({"error": f"User '{data['user_id']}' not found"}), 404
    if not content:
        return jsonify({"error": f"Content '{data['content_id']}' not found"}), 404

    rating      = float(data["rating"]) if data.get("rating") else None
    duration    = int(data["duration_sec"]) if data.get("duration_sec") else None
    session_id  = data.get("session_id", str(uuid.uuid4()))

    interaction_id = interaction_repo.record(
        user_id      = data["user_id"],
        content_id   = data["content_id"],
        event_type   = data["event_type"],
        rating       = rating,
        duration_sec = duration,
        session_id   = session_id,
    )

    # Increment view count on the content
    if data["event_type"] in ("view", "click"):
        content_repo.increment_views(data["content_id"])

    # Invalidate cache for this user
    orchestrator.invalidate_cache(data["user_id"])

    return jsonify({
        "message":        "Feedback recorded successfully",
        "interaction_id": interaction_id,
        "user_id":        data["user_id"],
        "content_id":     data["content_id"],
        "event_type":     data["event_type"],
    }), 201


@app.route("/metrics", methods=["GET"])
@require_api_key
def metrics():
    """
    Performance and usage metrics endpoint.
    Returns API stats and system information.
    """
    api_stats  = log_repo.get_metrics()
    sys_stats  = orchestrator.get_stats()

    return jsonify({
        "system":      sys_stats,
        "api_metrics": api_stats,
    }), 200


@app.route("/similar/<content_id>", methods=["GET"])
@require_api_key
def similar_items(content_id: str):
    """
    Return items similar to a given content item.

    Path param: content_id
    Query param: n (default=5)
    """
    n       = min(int(request.args.get("n", 5)), 20)
    content = content_repo.get(content_id)
    if not content:
        return jsonify({"error": f"Content '{content_id}' not found"}), 404

    similar = orchestrator.similar_items(content_id, n=n)
    return jsonify({
        "content_id":  content_id,
        "title":       content["title"],
        "similar":     similar,
        "count":       len(similar),
    }), 200


@app.route("/users", methods=["POST"])
@require_api_key
def create_user():
    """Create a new user."""
    data = request.get_json()
    if not data or not data.get("username"):
        return jsonify({"error": "username is required"}), 422
    try:
        user = user_repo.create(
            username    = data["username"],
            email       = data.get("email"),
            skill_level = data.get("skill_level", "beginner"),
            preferences = data.get("preferences", []),
        )
        return jsonify(user), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 409


@app.route("/users/<user_id>", methods=["GET"])
@require_api_key
def get_user(user_id: str):
    """Get user profile."""
    user = user_repo.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user), 200


@app.route("/content", methods=["GET"])
@require_api_key
def list_content():
    """List all content items."""
    category = request.args.get("category")
    if category:
        items = content_repo.get_by_category(category)
    else:
        items = content_repo.list_all()
    return jsonify({"items": items, "count": len(items)}), 200


# ---------------------------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def server_error(e):
    logger.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    init_db()
    print("🚀 Starting Recommendation API on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)