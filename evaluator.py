"""
Evaluation metrics for the recommendation system.
Implements Precision@K, Recall@K, and NDCG@K.
"""

import math
from typing import List, Set, Dict


def precision_at_k(recommended: List[str], relevant: Set[str], k: int = 5) -> float:
    """
    Precision@K: fraction of top-K recommendations that are relevant.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of ground-truth relevant item IDs
        k: Cutoff
    Returns:
        Precision@K in [0, 1]
    """
    if not recommended or not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / k


def recall_at_k(recommended: List[str], relevant: Set[str], k: int = 5) -> float:
    """
    Recall@K: fraction of relevant items found in top-K recommendations.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of ground-truth relevant item IDs
        k: Cutoff
    Returns:
        Recall@K in [0, 1]
    """
    if not recommended or not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def dcg_at_k(recommended: List[str], relevant: Set[str], k: int = 5) -> float:
    """Compute Discounted Cumulative Gain at K."""
    top_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(i + 1)
    return dcg


def ndcg_at_k(recommended: List[str], relevant: Set[str], k: int = 5) -> float:
    """
    NDCG@K: Normalized Discounted Cumulative Gain.

    Args:
        recommended: Ordered list of recommended item IDs
        relevant: Set of ground-truth relevant item IDs
        k: Cutoff
    Returns:
        NDCG@K in [0, 1]
    """
    if not recommended or not relevant:
        return 0.0
    # Ideal DCG: all relevant items at top positions
    ideal = list(relevant)[:k]
    idcg = dcg_at_k(ideal, relevant, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(recommended, relevant, k) / idcg


def evaluate_system(
    recommendations: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    k: int = 5
) -> dict:
    """
    Evaluate the full recommendation system across multiple users.

    Args:
        recommendations: {user_id: [ordered content_ids]}
        ground_truth: {user_id: {relevant content_ids}}
        k: Evaluation cutoff
    Returns:
        Dict with mean metrics and per-user details
    """
    results = {}
    p_scores, r_scores, n_scores = [], [], []

    for uid, recs in recommendations.items():
        relevant = ground_truth.get(uid, set())
        if not relevant:
            continue

        p = precision_at_k(recs, relevant, k)
        r = recall_at_k(recs, relevant, k)
        n = ndcg_at_k(recs, relevant, k)

        results[uid] = {"precision": p, "recall": r, "ndcg": n}
        p_scores.append(p)
        r_scores.append(r)
        n_scores.append(n)

    if not p_scores:
        return {"error": "No users with ground truth data", "users_evaluated": 0}

    return {
        f"precision@{k}": round(sum(p_scores) / len(p_scores), 4),
        f"recall@{k}":    round(sum(r_scores) / len(r_scores), 4),
        f"ndcg@{k}":      round(sum(n_scores) / len(n_scores), 4),
        "users_evaluated": len(p_scores),
        "per_user": results,
    }


def f1_at_k(precision: float, recall: float) -> float:
    """Compute F1 score from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)