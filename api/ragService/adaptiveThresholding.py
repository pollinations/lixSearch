"""
Adaptive Thresholding Module for Dynamic Similarity and Cache Filtering

Implements adaptive threshold calculation based on query characteristics:
- H(q) = entropy of query embedding distribution
- V_s = variance of session embedding cluster  
- A(q) = estimated ambiguity score
- T(q) = query temporal sensitivity classifier

Dynamic threshold: τ(q) = α·H(q) + β·V_s + γ·A(q)

Replaces fixed thresholds 0.85, 0.90 with context-aware policies.
"""
import numpy as np
from typing import Dict, Optional, Tuple, List
from loguru import logger
import scipy.stats as stats
from datetime import datetime, timedelta

class AdaptiveThresholdCalculator:
    def __init__(self, alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.base_threshold = 0.70
        logger.info(f"[AdaptiveThreshold] Initialized with α={alpha}, β={beta}, γ={gamma}")
    
    def compute_embedding_entropy(self, embedding: np.ndarray) -> float:
        if embedding is None or len(embedding) == 0:
            return 0.5
        abs_emb = np.abs(embedding)
        emb_dist = abs_emb / (np.sum(abs_emb) + 1e-8)
        entropy = -np.sum(emb_dist * np.log(emb_dist + 1e-10))
        max_entropy = np.log(len(embedding))
        normalized_entropy = entropy / (max_entropy + 1e-8)
        return min(1.0, max(0.0, normalized_entropy))
    
    def compute_cluster_variance(self, session_embeddings: List[np.ndarray]) -> float:
        if not session_embeddings or len(session_embeddings) < 2:
            return 0.5
        embeddings_array = np.array(session_embeddings)
        distances = []
        for i in range(len(embeddings_array)):
            for j in range(i + 1, len(embeddings_array)):
                emb_i = embeddings_array[i] / (np.linalg.norm(embeddings_array[i]) + 1e-8)
                emb_j = embeddings_array[j] / (np.linalg.norm(embeddings_array[j]) + 1e-8)
                distance = 1.0 - np.dot(emb_i, emb_j)
                distances.append(distance)
        if not distances:
            return 0.5
        variance = np.var(distances)
        normalized_variance = min(1.0, variance * 2)
        return normalized_variance
    
    def compute_query_ambiguity(self, query_text: str) -> float:
        if not query_text:
            return 0.5
        query_lower = query_text.lower()
        words = query_text.split()
        word_count = len(words)
        length_ambiguity = 1.0 / (1.0 + word_count / 5.0)
        ambiguous_keywords = {
            'what', 'how', 'why', 'which', 'who', 'when', 'where',
            'maybe', 'possibly', 'could', 'might', 'seem', 'appear'
        }
        keyword_count = sum(1 for kw in ambiguous_keywords if kw in query_lower)
        keyword_ambiguity = keyword_count / max(1, word_count)
        ambiguity = 0.6 * length_ambiguity + 0.4 * keyword_ambiguity
        return min(1.0, max(0.0, ambiguity))
    
    def compute_temporal_sensitivity(self, 
                                    query_text: str,
                                    session_age_minutes: Optional[int] = None) -> float:
        query_lower = query_text.lower()
        time_sensitive_keywords = {
            'today', 'now', 'current', 'latest', 'recent', 'live',
            'breaking', 'news', 'stock', 'price', 'weather', 'real-time',
            'trending', 'today\'s', 'this week'
        }
        keyword_score = sum(
            1.0 for kw in time_sensitive_keywords 
            if kw in query_lower
        ) / len(time_sensitive_keywords)
        age_score = 0.0
        if session_age_minutes:
            age_score = min(1.0, session_age_minutes / 60.0)
        temporal_sensitivity = 0.7 * keyword_score + 0.3 * age_score
        return min(1.0, max(0.0, temporal_sensitivity))
    
    def compute_adaptive_threshold(self,
                                  query_embedding: np.ndarray,
                                  query_text: str,
                                  session_embeddings: Optional[List[np.ndarray]] = None,
                                  session_age_minutes: Optional[int] = None) -> float:
        entropy = self.compute_embedding_entropy(query_embedding)
        cluster_var = self.compute_cluster_variance(session_embeddings or [])
        ambiguity = self.compute_query_ambiguity(query_text)
        temporal = self.compute_temporal_sensitivity(query_text, session_age_minutes)
        base_score = (
            self.alpha * entropy + 
            self.beta * cluster_var + 
            self.gamma * ambiguity
        )
        temporal_adjustment = -0.10 * temporal
        threshold = self.base_threshold + base_score + temporal_adjustment
        threshold = min(0.95, max(0.65, threshold))
        logger.debug(
            f"[AdaptiveThreshold] τ(q)={threshold:.3f} "
            f"(H={entropy:.3f}, V_s={cluster_var:.3f}, A={ambiguity:.3f}, T={temporal:.3f})"
        )
        return threshold
    
    def get_diagnostic_info(self,
                           query_embedding: np.ndarray,
                           query_text: str,
                           session_embeddings: Optional[List[np.ndarray]] = None,
                           session_age_minutes: Optional[int] = None) -> Dict:
        entropy = self.compute_embedding_entropy(query_embedding)
        cluster_var = self.compute_cluster_variance(session_embeddings or [])
        ambiguity = self.compute_query_ambiguity(query_text)
        temporal = self.compute_temporal_sensitivity(query_text, session_age_minutes)
        threshold = self.compute_adaptive_threshold(
            query_embedding, query_text, session_embeddings, session_age_minutes
        )
        return {
            "adaptive_threshold": threshold,
            "embedding_entropy": entropy,
            "cluster_variance": cluster_var,
            "query_ambiguity": ambiguity,
            "temporal_sensitivity": temporal,
            "base_threshold": self.base_threshold,
            "weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma
            }
        }


class ThresholdAdapter:
    def __init__(self, calculator: AdaptiveThresholdCalculator):
        self.calculator = calculator
        self.hit_history: List[Tuple[float, bool]] = []
        self.performance_window = 100
        
    def record_performance(self, threshold_used: float, was_cache_hit: bool):
        self.hit_history.append((threshold_used, was_cache_hit))
        if len(self.hit_history) > self.performance_window:
            self.hit_history = self.hit_history[-self.performance_window:]
    
    def compute_optimal_threshold_band(self) -> Tuple[float, float]:
        if len(self.hit_history) < 10:
            return 0.70, 0.90
        bins = {}
        for threshold, hit in self.hit_history:
            bin_key = round(threshold, 2)
            if bin_key not in bins:
                bins[bin_key] = {"hits": 0, "total": 0}
            bins[bin_key]["total"] += 1
            if hit:
                bins[bin_key]["hits"] += 1
        best_threshold = max(
            bins.keys(),
            key=lambda t: bins[t]["hits"] / bins[t]["total"] if bins[t]["total"] > 0 else 0
        )
        lower = max(0.65, best_threshold - 0.10)
        upper = min(0.95, best_threshold + 0.15)
        logger.info(
            f"[ThresholdAdapter] Recommended band: [{lower:.3f}, {upper:.3f}] "
            f"based on {len(self.hit_history)} observations"
        )
        return lower, upper
    
    def get_performance_metrics(self) -> Dict:
        if not self.hit_history:
            return {"total_queries": 0, "hit_rate": 0.0}
        total = len(self.hit_history)
        hits = sum(1 for _, was_hit in self.hit_history if was_hit)
        hit_rate = hits / total if total > 0 else 0.0
        return {
            "total_queries": total,
            "hit_rate": hit_rate,
            "total_hits": hits,
            "total_misses": total - hits
        }
