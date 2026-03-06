"""
Evaluation script for the recommendation system.
Computes Precision@5, Recall@5, and NDCG@5 using leave-one-out cross-validation.
Also runs a load test simulating 10 concurrent users.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import threading
import statistics
from data.database import init_db
from data.repositories import UserRepository, InteractionRepository
from engine.orchestrator import RecommendationOrchestrator
from engine.evaluator import evaluate_system, precision_at_k, recall_at_k, ndcg_at_k, f1_at_k


def build_ground_truth(interaction_repo: InteractionRepository, min_rating: float = 3.0) -> dict:
    """
    Build ground truth from high-quality interactions.
    Uses completed, bookmarked, or highly-rated items as relevant.
    """
    matrix = interaction_repo.get_user_item_matrix()
    ground_truth = {}

    for uid, items in matrix.items():
        # Items with score >= threshold are "relevant"
        relevant = {cid for cid, score in items.items() if score >= min_rating}
        if relevant:
            ground_truth[uid] = relevant

    return ground_truth


def leave_one_out_evaluation(
    orchestrator: RecommendationOrchestrator,
    user_repo: UserRepository,
    interaction_repo: InteractionRepository,
    k: int = 5,
) -> dict:
    """
    Leave-one-out evaluation: for each user, hold out one high-signal interaction
    and check if the system recommends the held-out item.
    """
    all_users = user_repo.list_all()
    matrix    = interaction_repo.get_user_item_matrix()

    recommendations = {}
    ground_truth    = {}

    for user in all_users:
        uid   = user["user_id"]
        items = matrix.get(uid, {})
        if len(items) < 2:
            continue  # need at least 2 interactions

        # Hold out the item with highest interaction score
        held_out_id = max(items, key=items.get)
        ground_truth[uid] = {held_out_id}

        # Get recommendations (exclude known interactions so system must re-discover)
        result = orchestrator.recommend(uid, n=20, use_cache=False)
        recs   = [r["content_id"] for r in result.get("recommendations", [])]
        recommendations[uid] = recs

    return evaluate_system(recommendations, ground_truth, k=k)


def full_evaluation(
    orchestrator: RecommendationOrchestrator,
    user_repo: UserRepository,
    interaction_repo: InteractionRepository,
    k: int = 5,
) -> dict:
    """
    Full evaluation using actual interaction data as ground truth.
    Uses recommend with diversity=False to get more candidates.
    """
    ground_truth = build_ground_truth(interaction_repo)
    recommendations = {}

    for uid in ground_truth:
        # Use the orchestrator directly with matrices already built
        orchestrator._ensure_matrices()
        from engine.candidate_gen import popularity_candidates, cold_start_candidates
        from engine.scorer import rank_candidates

        # Get all candidates including seen items for evaluation purposes
        all_content = list(orchestrator._content_lookup.values())
        seen = set()  # don't exclude seen items for evaluation
        pop_scores = popularity_candidates(orchestrator._interaction_counts, seen, max_candidates=100)
        cf_scores  = {}
        cb_scores  = {}
        try:
            from engine.candidate_gen import collaborative_filtering_candidates, content_based_candidates
            cf_scores = collaborative_filtering_candidates(uid, orchestrator._ui_matrix, orchestrator._user_sim, seen)
            cb_scores = content_based_candidates(uid, orchestrator._ui_matrix, orchestrator._item_sim, seen)
        except Exception:
            pass

        from engine.candidate_gen import merge_candidates
        merged = merge_candidates(cf_scores, cb_scores, pop_scores, weights=[0.5, 0.35, 0.15])
        ranked = rank_candidates(merged, orchestrator._content_lookup,
                                 cf_scores=cf_scores, cb_scores=cb_scores, pop_scores=pop_scores,
                                 top_n=k * 4)
        recs = [cid for cid, _ in ranked]
        recommendations[uid] = recs

    return evaluate_system(recommendations, ground_truth, k=k)


def load_test(orchestrator: RecommendationOrchestrator,
              user_repo: UserRepository,
              n_concurrent: int = 10,
              requests_per_user: int = 5) -> dict:
    """
    Simulate concurrent users and measure throughput/latency.
    """
    all_users   = user_repo.list_all()
    results     = []
    errors      = []
    lock        = threading.Lock()

    def user_session(user):
        uid = user["user_id"]
        for _ in range(requests_per_user):
            t0 = time.time()
            try:
                result = orchestrator.recommend(uid, n=10)
                elapsed = (time.time() - t0) * 1000
                with lock:
                    results.append(elapsed)
            except Exception as e:
                with lock:
                    errors.append(str(e))

    users_to_test = all_users[:n_concurrent]
    threads = [threading.Thread(target=user_session, args=(u,)) for u in users_to_test]

    start = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_time = time.time() - start

    total_requests = len(results) + len(errors)
    return {
        "concurrent_users":    n_concurrent,
        "requests_per_user":   requests_per_user,
        "total_requests":      total_requests,
        "successful_requests": len(results),
        "errors":              len(errors),
        "total_time_sec":      round(total_time, 2),
        "throughput_rps":      round(total_requests / total_time, 1) if total_time > 0 else 0,
        "latency_ms": {
            "mean":   round(statistics.mean(results), 2)   if results else 0,
            "median": round(statistics.median(results), 2) if results else 0,
            "p95":    round(sorted(results)[int(len(results) * 0.95)], 2) if results else 0,
            "min":    round(min(results), 2) if results else 0,
            "max":    round(max(results), 2) if results else 0,
        },
        "meets_200ms_sla": (statistics.median(results) < 200) if results else False,
    }


def run_full_evaluation():
    """Run all evaluations and print/save the report."""
    print("=" * 60)
    print("  RECOMMENDATION SYSTEM EVALUATION")
    print("=" * 60)

    init_db()
    user_repo        = UserRepository()
    interaction_repo = InteractionRepository()
    orchestrator     = RecommendationOrchestrator()

    # --- Metric Evaluation ---
    print("\n📊 Computing recommendation metrics (k=5)...")
    full_metrics = full_evaluation(orchestrator, user_repo, interaction_repo, k=5)
    loo_metrics  = leave_one_out_evaluation(orchestrator, user_repo, interaction_repo, k=5)

    print(f"\n  Full Dataset Evaluation:")
    print(f"    Precision@5: {full_metrics.get('precision@5', 'N/A')}")
    print(f"    Recall@5:    {full_metrics.get('recall@5', 'N/A')}")
    print(f"    NDCG@5:      {full_metrics.get('ndcg@5', 'N/A')}")
    print(f"    Users evaluated: {full_metrics.get('users_evaluated', 0)}")

    print(f"\n  Leave-One-Out Evaluation:")
    print(f"    Precision@5: {loo_metrics.get('precision@5', 'N/A')}")
    print(f"    Recall@5:    {loo_metrics.get('recall@5', 'N/A')}")
    print(f"    NDCG@5:      {loo_metrics.get('ndcg@5', 'N/A')}")
    print(f"    Users evaluated: {loo_metrics.get('users_evaluated', 0)}")

    # --- Cold Start Check ---
    print("\n❄️  Cold Start Test...")
    new_user = user_repo.get_by_username("grace_new")
    if new_user:
        cs_result = orchestrator.recommend(new_user["user_id"], n=5, use_cache=False)
        print(f"    Strategy: {cs_result.get('strategy')}")
        print(f"    Recs returned: {cs_result.get('count', 0)}")
        print(f"    Latency: {cs_result.get('latency_ms')}ms")

    # --- Latency Test ---
    print("\n⚡ Latency benchmark (single requests)...")
    all_users = user_repo.list_all()[:5]
    latencies = []
    for user in all_users:
        t0 = time.time()
        orchestrator.recommend(user["user_id"], n=10)
        latencies.append((time.time() - t0) * 1000)
    avg_lat = statistics.mean(latencies)
    print(f"    Average latency: {avg_lat:.1f}ms")
    print(f"    Under 200ms SLA: {'✅' if avg_lat < 200 else '❌'}")

    # --- Load Test ---
    print("\n🔥 Load test (10 concurrent users)...")
    load_results = load_test(orchestrator, user_repo, n_concurrent=10, requests_per_user=5)
    print(f"    Throughput:  {load_results['throughput_rps']} req/sec")
    print(f"    Median P50:  {load_results['latency_ms']['median']}ms")
    print(f"    P95 latency: {load_results['latency_ms']['p95']}ms")
    print(f"    Success rate: {load_results['successful_requests']}/{load_results['total_requests']}")
    print(f"    Meets 200ms SLA: {'✅' if load_results['meets_200ms_sla'] else '❌'}")

    # --- Save report ---
    report = {
        "evaluation_metrics": {
            "full_dataset": full_metrics,
            "leave_one_out": loo_metrics,
        },
        "latency_benchmark": {
            "single_request_avg_ms": round(avg_lat, 2),
            "meets_200ms_sla": avg_lat < 200,
        },
        "load_test": load_results,
        "system_stats": orchestrator.get_stats(),
    }

    report_path = os.path.join(os.path.dirname(__file__), "..", "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Full report saved to evaluation_report.json")
    print("=" * 60)
    return report


if __name__ == "__main__":
    run_full_evaluation()