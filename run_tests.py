"""
Custom test runner — works without pytest.
Runs all unit and integration tests and reports results.
"""

import sys
import os
import time
import traceback
import tempfile
import json

# ── Engine tests (no DB needed) ─────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.similarity import (
    cosine_similarity_dict, jaccard_similarity,
    build_user_similarity_matrix, get_top_similar_users,
    content_to_feature_vector,
)
from engine.candidate_gen import (
    collaborative_filtering_candidates, content_based_candidates,
    popularity_candidates, cold_start_candidates, merge_candidates,
)
from engine.scorer import score_candidate, rank_candidates, apply_diversity_filter, generate_explanation
from engine.evaluator import precision_at_k, recall_at_k, ndcg_at_k, evaluate_system, f1_at_k

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = {"pass": 0, "fail": 0, "errors": []}


def test(name, condition, msg=""):
    if condition:
        print(f"  {PASS}  {name}")
        results["pass"] += 1
    else:
        detail = f" → {msg}" if msg else ""
        print(f"  {FAIL}  {name}{detail}")
        results["fail"] += 1
        results["errors"].append(name)


def approx_eq(a, b, tol=1e-6):
    return abs(a - b) < tol


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SIMILARITY TESTS")
print("="*60)

v = {"a": 1.0, "b": 2.0}
test("cosine_identical_vectors", approx_eq(cosine_similarity_dict(v, v), 1.0))
test("cosine_orthogonal", approx_eq(cosine_similarity_dict({"a": 1.0}, {"b": 1.0}), 0.0))
test("cosine_empty", cosine_similarity_dict({}, {"a": 1.0}) == 0.0)
test("cosine_partial_overlap", 0 < cosine_similarity_dict({"a":1,"b":1}, {"a":1,"c":1}) < 1)

s = {"a", "b", "c"}
test("jaccard_identical", approx_eq(jaccard_similarity(s, s), 1.0))
test("jaccard_disjoint", approx_eq(jaccard_similarity({"a"}, {"b"}), 0.0))
test("jaccard_partial", approx_eq(jaccard_similarity({"a","b"}, {"b","c"}), 1/3))
test("jaccard_empty", jaccard_similarity(set(), {"a"}) == 0.0)

matrix = {"u1": {"c1": 1.0, "c2": 2.0}, "u2": {"c1": 1.0, "c3": 1.0}, "u3": {"c4": 1.0}}
sim = build_user_similarity_matrix(matrix)
test("user_sim_symmetric", approx_eq(sim["u1"]["u2"], sim["u2"]["u1"]))
test("user_sim_no_overlap", approx_eq(sim["u1"]["u3"], 0.0))

sim_m = {"u1": {"u2": 0.9, "u3": 0.5}, "u2": {"u1": 0.9}}
top = get_top_similar_users("u1", sim_m, top_k=2)
test("top_similar_users_sorted", top[0][0] == "u2")

vec = content_to_feature_vector({"category":"python","difficulty":"beginner","tags":["oop"]})
test("feature_vector_category", "cat:python" in vec)
test("feature_vector_difficulty", "diff:beginner" in vec)
test("feature_vector_tag", "tag:oop" in vec)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  CANDIDATE GENERATION TESTS")
print("="*60)

ui = {"u1": {"c1": 3.0, "c2": 1.5}, "u2": {"c1": 2.5, "c3": 3.0}, "u3": {"c2": 2.0}}
user_sim = build_user_similarity_matrix(ui)
item_sim = {"c1": {"c2": 0.8, "c3": 0.6}, "c2": {"c1": 0.8}, "c3": {"c1": 0.6}}

cf = collaborative_filtering_candidates("u1", ui, user_sim, {"c1", "c2"})
test("CF_seen_excluded", "c1" not in cf and "c2" not in cf)
test("CF_new_user_empty", collaborative_filtering_candidates("u_new", ui, user_sim, set()) == {})

cb = content_based_candidates("u1", ui, item_sim, {"c1"})
test("CB_seen_excluded", "c1" not in cb)
test("CB_is_dict", isinstance(cb, dict))

counts = {"c1": 100, "c2": 50, "c3": 10}
pop = popularity_candidates(counts, {"c1"})
test("pop_seen_excluded", "c1" not in pop)
test("pop_normalized", all(0 <= s <= 1 for s in pop.values()))

user = {"user_id": "new", "skill_level": "beginner", "preferences": ["python"]}
all_content = [
    {"content_id": "c1", "category": "python", "difficulty": "beginner", "rating": 4.5},
    {"content_id": "c2", "category": "ml", "difficulty": "advanced", "rating": 3.0},
]
cs = cold_start_candidates(user, all_content, {"c1": 50, "c2": 10})
test("cold_start_prefers_matching", cs.get("c1", 0) > cs.get("c2", 0))

merged = merge_candidates({"c1": 1.0, "c2": 0.5}, {"c2": 0.5, "c3": 1.0}, weights=[1.0, 1.0])
test("merge_candidates_sum", approx_eq(merged["c2"], 1.0))

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SCORER TESTS")
print("="*60)

score = score_candidate("c1", {"rating": 5.0}, cf_score=1.0, cb_score=0.5, pop_score=0.3)
test("score_positive", score > 0)
test("score_zero_signals", approx_eq(score_candidate("c1", {}, 0, 0, 0), 0.0))

scored = [("c1", 0.9), ("c2", 0.8), ("c3", 0.7), ("c4", 0.6), ("c5", 0.5)]
lookup = {
    "c1":{"category":"python","rating":4.0}, "c2":{"category":"python","rating":3.5},
    "c3":{"category":"python","rating":4.0}, "c4":{"category":"python","rating":4.0},
    "c5":{"category":"ml","rating":4.0},
}
ranked = rank_candidates(dict(scored), lookup, top_n=3)
test("rank_top_n", len(ranked) == 3)
test("rank_ordered", ranked[0][1] >= ranked[1][1])

filtered = apply_diversity_filter(scored, lookup, max_per_category=2)
python_count = sum(1 for cid, _ in filtered if lookup[cid]["category"] == "python")
test("diversity_filter_max_cat", python_count <= 2)

expl = generate_explanation("c1","u1",{"category":"python","difficulty":"beginner"},0.8,0.3,0.1)
test("explanation_is_string", isinstance(expl, str) and len(expl) > 0)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  EVALUATION METRIC TESTS")
print("="*60)

test("precision@K_perfect",  approx_eq(precision_at_k(["c1","c2","c3"], {"c1","c2","c3"}, 3), 1.0))
test("precision@K_zero",     approx_eq(precision_at_k(["c1","c2","c3"], {"c4","c5"}, 3), 0.0))
test("precision@K_partial",  approx_eq(precision_at_k(["c1","c2","c3","c4","c5"], {"c1","c3"}, 5), 2/5))
test("precision@K_empty",    precision_at_k([], {"c1"}, 5) == 0.0)

test("recall@K_perfect",  approx_eq(recall_at_k(["c1","c2","c3"], {"c1","c2","c3"}, 3), 1.0))
test("recall@K_partial",  approx_eq(recall_at_k(["c1","c2"], {"c1","c2","c3","c4"}, 2), 0.5))
test("recall@K_empty",    recall_at_k([], {"c1"}, 5) == 0.0)

test("ndcg@K_perfect",  approx_eq(ndcg_at_k(["c1","c2","c3"], {"c1","c2","c3"}, 3), 1.0))
test("ndcg@K_zero",     approx_eq(ndcg_at_k(["c4","c5"], {"c1","c2"}, 2), 0.0))

# NDCG should be higher when relevant items are ranked first
ndcg_good = ndcg_at_k(["c1","c2","c3"], {"c1","c2"}, 3)
ndcg_bad  = ndcg_at_k(["c3","c1","c2"], {"c1","c2"}, 3)
test("ndcg_position_sensitive", ndcg_good > ndcg_bad)

metrics = evaluate_system(
    {"u1": ["c1","c2","c3","c4","c5"], "u2": ["c6","c1","c2","c3","c4"]},
    {"u1": {"c1","c3"}, "u2": {"c6"}},
    k=5
)
test("evaluate_system_keys",   all(k in metrics for k in ["precision@5","recall@5","ndcg@5"]))
test("evaluate_system_count",  metrics["users_evaluated"] == 2)
test("f1_half_half",           approx_eq(f1_at_k(0.5, 0.5), 0.5))
test("f1_zero_recall",         approx_eq(f1_at_k(0.5, 0.0), 0.0))

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  DATABASE & REPOSITORY TESTS")
print("="*60)

import tempfile
test_db = tempfile.mktemp(suffix="_test.db")
os.environ["DB_PATH"] = test_db

# Re-import to pick up new DB path
import importlib
import data.database as db_mod
db_mod.DB_PATH = test_db
if hasattr(db_mod._local, "conn"):
    db_mod._local.conn = None

from data.database import init_db, get_db
from data.repositories import UserRepository, ContentRepository, InteractionRepository

init_db()
ur = UserRepository()
cr = ContentRepository()
ir = InteractionRepository()

u = ur.create("testuser1", "t1@ex.com", "beginner", ["python"])
test("user_create", u["username"] == "testuser1")
test("user_preferences_list", isinstance(u["preferences"], list))

fetched = ur.get(u["user_id"])
test("user_get", fetched is not None and fetched["username"] == "testuser1")
test("user_get_missing", ur.get("bad-id") is None)

by_name = ur.get_by_username("testuser1")
test("user_get_by_username", by_name is not None)

c = cr.create("tc01", "Test Course", "Desc", "python", "beginner", ["py","test"], rating=4.5)
test("content_create", c["content_id"] == "tc01")
test("content_tags_list", isinstance(c["tags"], list) and "py" in c["tags"])

fetched_c = cr.get("tc01")
test("content_get", fetched_c is not None)
test("content_get_missing", cr.get("bad") is None)

cat_results = cr.get_by_category("python")
test("content_by_category", all(r["category"] == "python" for r in cat_results))

cr.increment_views("tc01")
test("content_increment_views", cr.get("tc01")["view_count"] == 1)

with get_db() as conn:
    conn.execute("INSERT OR IGNORE INTO skills (skill_id, name, category) VALUES ('sk-test','TestSkill','test')")
cr.add_skill("tc01", "sk-test", 0.9)
skills = cr.get_skills("tc01")
test("content_skills", "sk-test" in skills)

iid = ir.record(u["user_id"], "tc01", "complete", rating=5.0)
test("interaction_record", isinstance(iid, str) and len(iid) > 0)

history = ir.get_user_history(u["user_id"])
test("user_history", len(history) >= 1)

seen = ir.get_user_interacted_content(u["user_id"])
test("user_seen_content", "tc01" in seen)

matrix = ir.get_user_item_matrix()
test("user_item_matrix", isinstance(matrix, dict))

counts = ir.get_content_interaction_counts()
test("interaction_counts", isinstance(counts, dict))

test("event_weights_complete>view", InteractionRepository.EVENT_WEIGHTS["complete"] > InteractionRepository.EVENT_WEIGHTS["view"])
test("event_weights_skip_negative", InteractionRepository.EVENT_WEIGHTS["skip"] < 0)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  ORCHESTRATOR TESTS")
print("="*60)

from engine.orchestrator import RecommendationOrchestrator

# Use main DB (seeded)
main_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recommendation.db")
if os.path.exists(main_db):
    os.environ["DB_PATH"] = main_db
    db_mod.DB_PATH = main_db
    db_mod._local.conn = None  # force reconnect

    orch = RecommendationOrchestrator(cache_ttl=60)

    # Get a real user
    all_users = ur.list_all() if not os.path.exists(main_db) else UserRepository().list_all()
    all_users = UserRepository().list_all()
    warm_user = next((u for u in all_users if u["username"] == "alice_dev"), None)
    cold_user = next((u for u in all_users if u["username"] == "grace_new"), None)

    if warm_user:
        res = orch.recommend(warm_user["user_id"], n=5)
        test("orch_warm_returns_recs", len(res.get("recommendations", [])) > 0)
        test("orch_warm_has_latency", "latency_ms" in res)
        test("orch_warm_has_strategy", "strategy" in res)
        recs = res.get("recommendations", [])
        if recs:
            test("orch_rec_has_explanation", "explanation" in recs[0])
            test("orch_rec_has_score", "score" in recs[0])

        # Test caching
        res2 = orch.recommend(warm_user["user_id"], n=5)
        test("orch_cache_hit", res2.get("from_cache") is True)

    if cold_user:
        res = orch.recommend(cold_user["user_id"], n=5)
        test("orch_cold_start", res.get("is_cold_start") is True)
        test("orch_cold_returns_recs", len(res.get("recommendations", [])) > 0)

    # Test invalid user
    err_res = orch.recommend("nonexistent-user-id", n=5)
    test("orch_invalid_user_error", "error" in err_res)

    # Stats
    stats = orch.get_stats()
    test("orch_stats_has_users", "total_users" in stats)
else:
    print("  ⚠️  Main DB not found — skipping orchestrator tests (run seed_data.py first)")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  API INTEGRATION TESTS")
print("="*60)

if os.path.exists(main_db):
    os.environ["DB_PATH"] = main_db
    db_mod.DB_PATH = main_db
    db_mod._local.conn = None

    import importlib
    import api.app as api_mod
    importlib.reload(api_mod)
    api_mod.app.config["TESTING"] = True
    client = api_mod.app.test_client()
    HEADERS = {"X-API-Key": "dev-key-123", "Content-Type": "application/json"}

    # Get a real user ID
    all_users = UserRepository().list_all()
    alice = next((u for u in all_users if u["username"] == "alice_dev"), None)
    uid = alice["user_id"] if alice else None

    # Health
    r = client.get("/health")
    test("api_health_200", r.status_code == 200)
    test("api_health_status", r.get_json().get("status") == "healthy")

    # Auth
    if uid:
        r = client.post("/recommend", json={"user_id": uid})
        test("api_recommend_401_no_key", r.status_code == 401)

        r = client.post("/recommend", json={"user_id": uid}, headers=HEADERS)
        test("api_recommend_200", r.status_code == 200)
        data = r.get_json()
        test("api_recommend_has_recs", "recommendations" in data)
        test("api_recommend_has_latency", "latency_ms" in data)

        r = client.post("/recommend", json={"user_id": "bad-user"}, headers=HEADERS)
        test("api_recommend_404_bad_user", r.status_code == 404)

        r = client.post("/recommend", json={}, headers=HEADERS)
        test("api_recommend_422_missing_uid", r.status_code == 422)

        all_content = ContentRepository().list_all()
        cid = all_content[0]["content_id"] if all_content else None

        if cid:
            r = client.post("/feedback",
                            json={"user_id": uid, "content_id": cid, "event_type": "view"},
                            headers=HEADERS)
            test("api_feedback_201", r.status_code == 201)
            test("api_feedback_has_interaction_id", "interaction_id" in r.get_json())

            r = client.post("/feedback",
                            json={"user_id": uid, "content_id": cid, "event_type": "BAD"},
                            headers=HEADERS)
            test("api_feedback_422_bad_event", r.status_code == 422)

            r = client.get(f"/similar/{cid}", headers=HEADERS)
            test("api_similar_200", r.status_code == 200)
            test("api_similar_has_similar", "similar" in r.get_json())

    r = client.get("/metrics", headers=HEADERS)
    test("api_metrics_200", r.status_code == 200)
    test("api_metrics_has_system", "system" in r.get_json())

    r = client.get("/content", headers=HEADERS)
    test("api_content_list_200", r.status_code == 200)

    r = client.post("/users",
                    json={"username": "test_runner_user", "skill_level": "intermediate"},
                    headers=HEADERS)
    test("api_create_user_201", r.status_code in (201, 409))  # 409 if already exists

    r = client.get("/users/bad-id", headers=HEADERS)
    test("api_get_user_404", r.status_code == 404)

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
total = results["pass"] + results["fail"]
pct = round(results["pass"] / total * 100, 1) if total > 0 else 0
print(f"  RESULTS: {results['pass']}/{total} passed ({pct}% coverage)")
if results["errors"]:
    print(f"  Failed: {', '.join(results['errors'])}")
print("="*60 + "\n")

sys.exit(0 if results["fail"] == 0 else 1)
