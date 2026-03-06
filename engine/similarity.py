"""
Similarity computation module.
Implements user-user and item-item similarity using cosine similarity.
"""

import numpy as np
from typing import Dict, List, Tuple


def cosine_similarity_dict(vec_a: dict, vec_b: dict) -> float:
    """Compute cosine similarity between two sparse vectors (dicts)."""
    if not vec_a or not vec_b:
        return 0.0
    keys = set(vec_a) & set(vec_b)
    if not keys:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in keys)
    norm_a = np.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = np.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def build_user_similarity_matrix(user_item_matrix: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Build user-user similarity matrix from interaction data.
    Returns {user_id: {other_user_id: similarity_score}}.
    """
    users = list(user_item_matrix.keys())
    similarity = {u: {} for u in users}

    for i, uid_a in enumerate(users):
        for uid_b in users[i + 1:]:
            sim = cosine_similarity_dict(user_item_matrix[uid_a], user_item_matrix[uid_b])
            similarity[uid_a][uid_b] = sim
            similarity[uid_b][uid_a] = sim

    return similarity


def build_item_similarity_matrix(item_profiles: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Build item-item similarity from content feature vectors.
    item_profiles: {content_id: {feature: weight}}
    """
    items = list(item_profiles.keys())
    similarity = {i: {} for i in items}

    for idx, cid_a in enumerate(items):
        for cid_b in items[idx + 1:]:
            sim = cosine_similarity_dict(item_profiles[cid_a], item_profiles[cid_b])
            similarity[cid_a][cid_b] = sim
            similarity[cid_b][cid_a] = sim

    return similarity


def get_top_similar_users(user_id: str, similarity_matrix: dict, top_k: int = 10) -> List[Tuple[str, float]]:
    """Return top-k most similar users (excluding self)."""
    if user_id not in similarity_matrix:
        return []
    sims = [(uid, score) for uid, score in similarity_matrix[user_id].items() if uid != user_id]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def get_top_similar_items(content_id: str, similarity_matrix: dict, top_k: int = 10) -> List[Tuple[str, float]]:
    """Return top-k most similar items to a given content item."""
    if content_id not in similarity_matrix:
        return []
    sims = [(cid, score) for cid, score in similarity_matrix[content_id].items()]
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def content_to_feature_vector(content: dict) -> dict:
    """
    Convert a content dict to a feature vector for similarity computation.
    Uses category, difficulty, and tags as features.
    """
    features = {}
    # Category feature
    features[f"cat:{content['category']}"] = 2.0
    # Difficulty feature
    features[f"diff:{content.get('difficulty', 'beginner')}"] = 1.0
    # Tag features
    for tag in content.get("tags", []):
        features[f"tag:{tag}"] = 1.0
    return features