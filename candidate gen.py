"""
Candidate generation strategies for recommendations.
Generates pools of candidate items using different signals.
"""

from typing import Dict, List, Set
from engine.similarity import get_top_similar_users, get_top_similar_items


def collaborative_filtering_candidates(
    user_id: str,
    user_item_matrix: Dict[str, Dict[str, float]],
    user_similarity_matrix: Dict[str, Dict[str, float]],
    seen_items: Set[str],
    top_k_users: int = 10,
    max_candidates: int = 50
) -> Dict[str, float]:
    """
    Collaborative filtering: recommend items liked by similar users.
    Returns {content_id: predicted_score}.
    """
    similar_users = get_top_similar_users(user_id, user_similarity_matrix, top_k=top_k_users)
    if not similar_users:
        return {}

    scores = {}
    total_sim = sum(sim for _, sim in similar_users)
    if total_sim == 0:
        return {}

    for neighbor_id, sim in similar_users:
        neighbor_items = user_item_matrix.get(neighbor_id, {})
        for cid, rating in neighbor_items.items():
            if cid in seen_items:
                continue
            scores[cid] = scores.get(cid, 0.0) + sim * rating

    # Normalize
    for cid in scores:
        scores[cid] /= total_sim

    # Return top candidates
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_items[:max_candidates])


def content_based_candidates(
    user_id: str,
    user_item_matrix: Dict[str, Dict[str, float]],
    item_similarity_matrix: Dict[str, Dict[str, float]],
    seen_items: Set[str],
    max_candidates: int = 50
) -> Dict[str, float]:
    """
    Content-based filtering: find items similar to what user liked.
    Returns {content_id: score}.
    """
    user_history = user_item_matrix.get(user_id, {})
    if not user_history:
        return {}

    # Get top-rated items from user history
    top_liked = sorted(user_history.items(), key=lambda x: x[1], reverse=True)[:5]

    candidates = {}
    for liked_id, user_score in top_liked:
        similar = item_similarity_matrix.get(liked_id, {})
        for cid, item_sim in similar.items():
            if cid in seen_items or cid == liked_id:
                continue
            combined = item_sim * user_score
            candidates[cid] = candidates.get(cid, 0.0) + combined

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_candidates[:max_candidates])


def popularity_candidates(
    interaction_counts: Dict[str, int],
    seen_items: Set[str],
    max_candidates: int = 50
) -> Dict[str, float]:
    """
    Popularity-based candidates (trending/most-viewed content).
    Returns {content_id: popularity_score}.
    """
    max_count = max(interaction_counts.values()) if interaction_counts else 1
    candidates = {}
    for cid, count in interaction_counts.items():
        if cid not in seen_items:
            candidates[cid] = count / max_count  # normalize to [0,1]

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_candidates[:max_candidates])


def cold_start_candidates(
    user: dict,
    all_content: List[dict],
    interaction_counts: Dict[str, int],
    max_candidates: int = 20
) -> Dict[str, float]:
    """
    Cold start strategy for new users with no interaction history.
    Uses user profile (skill_level, preferences) + popularity.
    """
    skill_level = user.get("skill_level", "beginner")
    preferences = user.get("preferences", [])
    max_count = max(interaction_counts.values(), default=1)

    scores = {}
    for content in all_content:
        score = 0.0

        # Boost matching skill level
        if content.get("difficulty") == skill_level:
            score += 2.0

        # Boost matching category preferences
        if content.get("category") in preferences:
            score += 3.0

        # Add popularity signal
        pop_score = interaction_counts.get(content["content_id"], 0) / max_count
        score += pop_score

        # Boost highly-rated content
        score += content.get("rating", 0.0) / 5.0

        if score > 0:
            scores[content["content_id"]] = score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_scores[:max_candidates])


def merge_candidates(*candidate_dicts, weights: list = None) -> Dict[str, float]:
    """
    Merge multiple candidate dictionaries with optional weights.
    Returns unified scoring dict.
    """
    if weights is None:
        weights = [1.0] * len(candidate_dicts)

    merged = {}
    for candidates, w in zip(candidate_dicts, weights):
        for cid, score in candidates.items():
            merged[cid] = merged.get(cid, 0.0) + score * w

    return merged