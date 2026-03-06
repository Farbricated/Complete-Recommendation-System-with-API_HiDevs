"""
Recommendation Orchestrator — integrates all engine components.
Implements caching, cold-start handling, and explanation generation.
"""

import time
import threading
from typing import Dict, List, Optional


class TTLCache:
    """Simple thread-safe TTL cache (replaces cachetools.TTLCache)."""
    def __init__(self, maxsize: int, ttl: float):
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: dict = {}        # key → value
        self._expires: dict = {}     # key → expiry timestamp
        self._lock = threading.Lock()

    def _evict_expired(self):
        now = time.time()
        expired = [k for k, exp in self._expires.items() if exp <= now]
        for k in expired:
            self._data.pop(k, None)
            self._expires.pop(k, None)

    def __contains__(self, key):
        with self._lock:
            self._evict_expired()
            return key in self._data

    def __getitem__(self, key):
        with self._lock:
            self._evict_expired()
            return self._data[key]

    def __setitem__(self, key, value):
        with self._lock:
            self._evict_expired()
            if len(self._data) >= self.maxsize and key not in self._data:
                # Evict oldest entry
                oldest = min(self._expires, key=self._expires.get)
                self._data.pop(oldest, None)
                self._expires.pop(oldest, None)
            self._data[key] = value
            self._expires[key] = time.time() + self.ttl

    def __len__(self):
        with self._lock:
            self._evict_expired()
            return len(self._data)

    def __iter__(self):
        with self._lock:
            self._evict_expired()
            return iter(list(self._data.keys()))

    def keys(self):
        with self._lock:
            self._evict_expired()
            return list(self._data.keys())

    def pop(self, key, default=None):
        with self._lock:
            self._expires.pop(key, None)
            return self._data.pop(key, default)

    def clear(self):
        with self._lock:
            self._data.clear()
            self._expires.clear()

from data.repositories import UserRepository, ContentRepository, InteractionRepository
from engine.similarity import (
    build_user_similarity_matrix,
    build_item_similarity_matrix,
    content_to_feature_vector,
)
from engine.candidate_gen import (
    collaborative_filtering_candidates,
    content_based_candidates,
    popularity_candidates,
    cold_start_candidates,
    merge_candidates,
)
from engine.scorer import rank_candidates, apply_diversity_filter, generate_explanation


class RecommendationOrchestrator:
    """
    Central orchestrator for the recommendation system.
    Manages component lifecycle, caching, and request routing.
    """

    def __init__(self, cache_ttl: int = 300, cache_maxsize: int = 500):
        self.user_repo    = UserRepository()
        self.content_repo = ContentRepository()
        self.interaction_repo = InteractionRepository()

        # Recommendation cache: (user_id, n) → result
        self._rec_cache: TTLCache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        # Matrix caches
        self._ui_matrix: Dict = {}
        self._user_sim: Dict  = {}
        self._item_sim: Dict  = {}
        self._content_lookup: Dict = {}
        self._interaction_counts: Dict = {}
        self._last_matrix_build: float = 0
        self._matrix_ttl: float = 120  # rebuild matrices every 2 minutes
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Matrix Management
    # ------------------------------------------------------------------

    def _matrices_stale(self) -> bool:
        return time.time() - self._last_matrix_build > self._matrix_ttl

    def _rebuild_matrices(self):
        """Rebuild user-item, similarity matrices and content lookup."""
        all_content = self.content_repo.list_all()
        self._content_lookup = {c["content_id"]: c for c in all_content}

        # Item feature vectors for content-based similarity
        item_profiles = {
            cid: content_to_feature_vector(meta)
            for cid, meta in self._content_lookup.items()
        }

        self._ui_matrix          = self.interaction_repo.get_user_item_matrix()
        self._interaction_counts = self.interaction_repo.get_content_interaction_counts()
        self._user_sim           = build_user_similarity_matrix(self._ui_matrix)
        self._item_sim           = build_item_similarity_matrix(item_profiles)
        self._last_matrix_build  = time.time()

    def _ensure_matrices(self):
        with self._lock:
            if self._matrices_stale() or not self._content_lookup:
                self._rebuild_matrices()

    def invalidate_cache(self, user_id: str = None):
        """Invalidate recommendation cache (user-specific or full)."""
        if user_id:
            keys_to_del = [k for k in self._rec_cache if k[0] == user_id]
            for k in keys_to_del:
                self._rec_cache.pop(k, None)
        else:
            self._rec_cache.clear()

    # ------------------------------------------------------------------
    # Core Recommendation Logic
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        diversity: bool = True,
        use_cache: bool = True,
    ) -> dict:
        """
        Generate recommendations for a user.

        Args:
            user_id: Target user identifier
            n: Number of recommendations to return
            diversity: Apply diversity filter across categories
            use_cache: Return cached result if available
        Returns:
            Dict with recommendations list and metadata
        """
        start_time = time.time()
        cache_key = (user_id, n, diversity)

        # Cache hit
        if use_cache and cache_key in self._rec_cache:
            result = self._rec_cache[cache_key]
            result["from_cache"] = True
            result["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            return result

        # Fetch user
        user = self.user_repo.get(user_id)
        if user is None:
            return {"error": f"User '{user_id}' not found", "recommendations": []}

        self._ensure_matrices()
        self.user_repo.update_last_active(user_id)

        seen_items = self.interaction_repo.get_user_interacted_content(user_id)
        is_cold_start = user_id not in self._ui_matrix or not self._ui_matrix.get(user_id)

        # ---- Cold-start path ------------------------------------------------
        if is_cold_start:
            return self._cold_start_recommend(user, seen_items, n, start_time)

        # ---- Warm-user path -------------------------------------------------
        cf_scores  = collaborative_filtering_candidates(
            user_id, self._ui_matrix, self._user_sim, seen_items
        )
        cb_scores  = content_based_candidates(
            user_id, self._ui_matrix, self._item_sim, seen_items
        )
        pop_scores = popularity_candidates(self._interaction_counts, seen_items)

        merged = merge_candidates(
            cf_scores, cb_scores, pop_scores,
            weights=[0.5, 0.35, 0.15]
        )

        ranked = rank_candidates(
            merged, self._content_lookup,
            cf_scores=cf_scores, cb_scores=cb_scores, pop_scores=pop_scores,
            top_n=n * 2  # over-fetch for diversity filter
        )

        if diversity:
            ranked = apply_diversity_filter(ranked, self._content_lookup, max_per_category=3)

        ranked = ranked[:n]

        # Build response
        recommendations = []
        for cid, score in ranked:
            meta = self._content_lookup.get(cid, {})
            explanation = generate_explanation(
                cid, user_id, meta,
                cf_score  = cf_scores.get(cid, 0.0),
                cb_score  = cb_scores.get(cid, 0.0),
                pop_score = pop_scores.get(cid, 0.0),
            )
            recommendations.append({
                "content_id":  cid,
                "title":       meta.get("title", "Unknown"),
                "category":    meta.get("category", ""),
                "difficulty":  meta.get("difficulty", ""),
                "score":       round(score, 4),
                "explanation": explanation,
            })

        latency = round((time.time() - start_time) * 1000, 2)
        result = {
            "user_id":         user_id,
            "recommendations": recommendations,
            "strategy":        "hybrid",
            "is_cold_start":   False,
            "from_cache":      False,
            "latency_ms":      latency,
            "count":           len(recommendations),
        }
        self._rec_cache[cache_key] = result
        return result

    def _cold_start_recommend(self, user: dict, seen_items: set, n: int, start_time: float) -> dict:
        """Handle recommendation for users with no interaction history."""
        all_content = list(self._content_lookup.values())
        scores = cold_start_candidates(user, all_content, self._interaction_counts, max_candidates=n * 2)

        recommendations = []
        for cid, score in list(scores.items())[:n]:
            meta = self._content_lookup.get(cid, {})
            recommendations.append({
                "content_id":  cid,
                "title":       meta.get("title", "Unknown"),
                "category":    meta.get("category", ""),
                "difficulty":  meta.get("difficulty", ""),
                "score":       round(score, 4),
                "explanation": (
                    f"Recommended based on your {user.get('skill_level','beginner')} "
                    f"level and interests in {', '.join(user.get('preferences', ['general topics']))}."
                ),
            })

        latency = round((time.time() - start_time) * 1000, 2)
        return {
            "user_id":         user["user_id"],
            "recommendations": recommendations,
            "strategy":        "cold_start",
            "is_cold_start":   True,
            "from_cache":      False,
            "latency_ms":      latency,
            "count":           len(recommendations),
        }

    # ------------------------------------------------------------------
    # Similar Items
    # ------------------------------------------------------------------

    def similar_items(self, content_id: str, n: int = 5) -> List[dict]:
        """Return items similar to a given content item."""
        self._ensure_matrices()
        similar = self._item_sim.get(content_id, {})
        sorted_sim = sorted(similar.items(), key=lambda x: x[1], reverse=True)[:n]
        results = []
        for cid, sim_score in sorted_sim:
            meta = self._content_lookup.get(cid, {})
            results.append({
                "content_id":       cid,
                "title":            meta.get("title", "Unknown"),
                "category":         meta.get("category", ""),
                "similarity_score": round(sim_score, 4),
            })
        return results

    # ------------------------------------------------------------------
    # System Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Return system statistics for the metrics endpoint."""
        self._ensure_matrices()
        return {
            "total_users":     len(self.user_repo.list_all()),
            "total_content":   len(self._content_lookup),
            "total_interactions": sum(self._interaction_counts.values()),
            "matrix_users":    len(self._ui_matrix),
            "cache_size":      len(self._rec_cache),
            "cache_maxsize":   self._rec_cache.maxsize,
            "matrix_age_sec":  round(time.time() - self._last_matrix_build, 1),
        }