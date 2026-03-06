# Day 30 Capstone: Production Recommendation System

A complete, production-ready recommendation system microservice with REST API, SQLite database, hybrid recommendation engine, and comprehensive testing.

## Features

- **Hybrid Recommendation Engine**: Combines collaborative filtering, content-based filtering, and popularity signals
- **Cold Start Handling**: Smart recommendations for new users using profile matching
- **In-Memory Caching**: Thread-safe TTL cache with automatic invalidation
- **REST API**: Flask-based API with auth, request tracing, and structured logging
- **6-Table SQLite DB**: Users, content, skills, content_skills, interactions, request_logs
- **Evaluation Metrics**: Precision@5, Recall@5, NDCG@5
- **89 Unit Tests**: 100% pass rate across all modules

## Project Structure

```
day30_capstone/
├── data/
│   ├── database.py        # SQLite connection & schema init
│   ├── repositories.py    # Repository pattern (User, Content, Interaction, Log)
│   └── __init__.py
├── engine/
│   ├── orchestrator.py    # Main recommendation orchestrator + TTLCache
│   ├── similarity.py      # Cosine & Jaccard similarity computations
│   ├── candidate_gen.py   # CF, CB, popularity & cold-start candidates
│   ├── scorer.py          # Multi-signal scoring, diversity filter, explanations
│   └── evaluator.py       # Precision@K, Recall@K, NDCG@K metrics
├── api/
│   └── app.py             # Flask REST API (6 endpoints)
├── tests/
│   ├── test_data.py       # Data layer unit tests
│   ├── test_engine.py     # Engine unit tests
│   └── test_api.py        # API integration tests (pytest-compatible)
├── scripts/
│   ├── seed_data.py       # Populate DB (12 users, 22 content, 49 interactions)
│   └── evaluate.py        # Full evaluation with load test
├── run_tests.py           # Standalone test runner (no pytest required)
├── requirements.txt
└── README.md
```

## Setup & Run

### 1. Install Dependencies
```bash
pip install flask sqlalchemy numpy scikit-learn
# Note: cachetools is replaced by built-in TTLCache implementation
```

### 2. Seed the Database
```bash
python3 scripts/seed_data.py
```

### 3. Start the API
```bash
python3 api/app.py
# Server starts at http://localhost:5000
```

### 4. Run Tests
```bash
python3 run_tests.py
# → 89/89 passed (100.0%)
```

### 5. Run Evaluation
```bash
python3 scripts/evaluate.py
```

## API Endpoints

All endpoints (except `/health`) require the header `X-API-Key: dev-key-123`.

### GET /health
Returns system status and statistics. No authentication required.

```bash
curl http://localhost:5000/health
```

### POST /recommend
Generate personalized recommendations for a user.

```bash
curl -X POST http://localhost:5000/recommend \
  -H "X-API-Key: dev-key-123" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<uuid>", "n": 10, "diversity": true}'
```

Response includes: recommendations list, strategy used, latency, cold-start flag.

### POST /feedback
Record user interaction with content.

```bash
curl -X POST http://localhost:5000/feedback \
  -H "X-API-Key: dev-key-123" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "<uuid>", "content_id": "c01", "event_type": "complete", "rating": 4.5}'
```

Valid event types: `view`, `click`, `complete`, `rate`, `bookmark`, `skip`

### GET /metrics
Returns API performance statistics and system info.

```bash
curl http://localhost:5000/metrics -H "X-API-Key: dev-key-123"
```

### GET /similar/<content_id>
Returns items similar to the given content.

```bash
curl http://localhost:5000/similar/c01?n=5 -H "X-API-Key: dev-key-123"
```

### POST /users
Create a new user.

```bash
curl -X POST http://localhost:5000/users \
  -H "X-API-Key: dev-key-123" \
  -H "Content-Type: application/json" \
  -d '{"username": "newuser", "skill_level": "intermediate", "preferences": ["python", "ml"]}'
```

### GET /content
List all content (optional `?category=python` filter).

## Recommendation Strategy

The hybrid engine blends three signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| Collaborative Filtering | 40% | Items liked by similar users |
| Content-Based | 30% | Items similar to what user previously liked |
| Popularity | 15% | Trending content by interaction count |
| Quality | 10% | Content rating (1–5 stars) |
| Recency | 5% | Freshness bonus |

**Cold Start**: New users (no interaction history) receive recommendations based on skill level matching, category preferences, and global popularity.

## Evaluation Results

| Metric | Score |
|--------|-------|
| Precision@5 | 0.42 |
| Recall@5 | 0.82 |
| NDCG@5 | 0.63 |
| Avg Latency | < 5ms |
| Load Test (10 users) | ~1800 req/sec |
| SLA (<200ms) | ✅ |

## Database Schema

6 tables with proper foreign keys and indexes:
- **users**: User profiles with skill level and preferences
- **content**: Course/article metadata with ratings
- **skills**: Skill taxonomy with hierarchy support
- **content_skills**: Many-to-many content-skill mapping
- **interactions**: User event log (view/click/complete/rate/bookmark/skip)
- **request_logs**: API request audit trail