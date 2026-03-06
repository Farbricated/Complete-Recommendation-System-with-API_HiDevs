"""
Scoring and ranking module.
Combines multiple signals into a final recommendation score.
"""

from typing import Dict, List, Tuple


# Feature weights for the final scoring function
SCORE_WEIGHTS = {
    "collaborative": 0.40,
    "content_based": 0.30,
    "popularity":    0.15,
    "quality":       0.10,
    "recency":       0.05,
}


def score_candidate(
    content_id: str,
    content_meta: dict,
    cf_score: float = 0.0,
    cb_score: float = 0.0,
    pop_score: float = 0.0,
) -> float:
    """
    Compute a weighted final score for a single candidate item.

    Args:
        content_id: The item identifier
        content_meta: Content attributes (rating, difficulty, etc.)
        cf_score: Collaborative filtering score
        cb_score: Content-based score
        pop_score: Popularity score
    Returns:
        Final blended score in [0, ∞)
    """
    # Quality signal from content rating
    quality = content_meta.get("rating", 0.0) / 5.0

    final = (
        SCORE_WEIGHTS["collaborative"] * cf_score +
        SCORE_WEIGHTS["content_based"] * cb_score +
        SCORE_WEIGHTS["popularity"]    * pop_score +
        SCORE_WEIGHTS["quality"]       * quality
    )
    return round(final, 6)


def rank_candidates(
    candidate_scores: Dict[str, float],
    content_lookup: Dict[str, dict],
    cf_scores: Dict[str, float] = None,
    cb_scores: Dict[str, float] = None,
    pop_scores: Dict[str, float] = None,
    top_n: int = 10,
) -> List[Tuple[str, float]]:
    """
    Rank all candidates and return top-N.

    Args:
        candidate_scores: Pre-merged candidate scores
        content_lookup: {content_id: content_dict}
        cf_scores, cb_scores, pop_scores: Individual signal scores
        top_n: Number of recommendations to return
    Returns:
        Sorted list of (content_id, final_score) tuples
    """
    cf_scores  = cf_scores  or {}
    cb_scores  = cb_scores  or {}
    pop_scores = pop_scores or {}

    ranked = []
    for cid, merged_score in candidate_scores.items():
        meta = content_lookup.get(cid, {})
        final = score_candidate(
            cid, meta,
            cf_score  = cf_scores.get(cid, merged_score),
            cb_score  = cb_scores.get(cid, 0.0),
            pop_score = pop_scores.get(cid, 0.0),
        )
        ranked.append((cid, final))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def apply_diversity_filter(
    ranked: List[Tuple[str, float]],
    content_lookup: Dict[str, dict],
    max_per_category: int = 3
) -> List[Tuple[str, float]]:
    """
    Apply diversity filter to avoid showing too many items from one category.
    """
    category_count = {}
    filtered = []
    for cid, score in ranked:
        cat = content_lookup.get(cid, {}).get("category", "unknown")
        if category_count.get(cat, 0) < max_per_category:
            filtered.append((cid, score))
            category_count[cat] = category_count.get(cat, 0) + 1
    return filtered


def generate_explanation(
    content_id: str,
    user_id: str,
    content_meta: dict,
    cf_score: float,
    cb_score: float,
    pop_score: float,
    similar_users_count: int = 0,
    liked_similar: str = None,
) -> str:
    """Generate a human-readable explanation for a recommendation."""
    reasons = []

    if cf_score > 0.3:
        reasons.append(f"users with similar interests enjoyed this")
    if cb_score > 0.2 and liked_similar:
        reasons.append(f"similar to '{liked_similar}' which you liked")
    if pop_score > 0.5:
        reasons.append("trending content")

    category = content_meta.get("category", "this topic")
    difficulty = content_meta.get("difficulty", "")
    if difficulty:
        reasons.append(f"matches your {difficulty} level")

    if not reasons:
        reasons.append(f"popular in {category}")

    return "Recommended because: " + "; ".join(reasons) + "."