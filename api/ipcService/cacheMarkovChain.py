"""
Cache Layer Markov Chain Model for Latency Analysis

Models retrieval as a Markov chain across 5 cache layers:
1. Conversation cache (in-memory, session-specific)
2. Semantic cache (URL-based, query similarity)
3. Session cache (session embeddings)
4. Global cache (persistent, cross-session)
5. Web (fallback)

Expected latency:
    E[L] = Σ_i P(hit_i)·L_i + P(miss_all)·L_web

Where:
- P(hit_i) = probability of hit at layer i
- L_i = latency of layer i
- L_web = web search latency
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict


class CacheLayer(Enum):
    CONVERSATION = "conversation"
    SEMANTIC = "semantic"
    SESSION = "session"
    GLOBAL = "global"
    WEB = "web"


@dataclass
class LayerLatency:
    avg_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    sample_count: int = 0
    
    def update(self, latency_ms: float):
        if self.sample_count == 0:
            self.avg_ms = latency_ms
            self.min_ms = latency_ms
            self.max_ms = latency_ms
        else:
            self.avg_ms = (self.avg_ms * self.sample_count + latency_ms) / (self.sample_count + 1)
            self.min_ms = min(self.min_ms, latency_ms)
            self.max_ms = max(self.max_ms, latency_ms)
        
        self.sample_count += 1


@dataclass
class LayerMetrics:
    layer: CacheLayer
    hit_count: int = 0
    miss_count: int = 0
    total_lookups: int = 0
    latencies: LayerLatency = field(default_factory=LayerLatency)
    
    @property
    def hit_probability(self) -> float:
        total = self.total_lookups
        if total == 0:
            return self._get_default_probability()
        return self.hit_count / total
    
    @property
    def miss_probability(self) -> float:
        return 1.0 - self.hit_probability
    
    def _get_default_probability(self) -> float:
        defaults = {
            CacheLayer.CONVERSATION: 0.30,
            CacheLayer.SEMANTIC: 0.25,
            CacheLayer.SESSION: 0.20,
            CacheLayer.GLOBAL: 0.15,
            CacheLayer.WEB: 1.0
        }
        return defaults.get(self.layer, 0.1)
    
    def get_default_latency(self) -> float:
        defaults = {
            CacheLayer.CONVERSATION: 5.0,
            CacheLayer.SEMANTIC: 15.0,
            CacheLayer.SESSION: 20.0,
            CacheLayer.GLOBAL: 50.0,
            CacheLayer.WEB: 2000.0
        }
        return defaults.get(self.layer, 100.0)
    
    def record_hit(self, latency_ms: float):
        self.hit_count += 1
        self.total_lookups += 1
        self.latencies.update(latency_ms)
    
    def record_miss(self, latency_ms: float = 0.0):
        self.miss_count += 1
        self.total_lookups += 1
        if latency_ms > 0:
            self.latencies.update(latency_ms)


class CacheMarkovChain:
    
    def __init__(self):
        self.layers: Dict[CacheLayer, LayerMetrics] = {
            layer: LayerMetrics(layer=layer)
            for layer in CacheLayer
        }
        self.transition_history: List[Tuple[CacheLayer, bool]] = []
        self.observation_window = 500
        
        logger.info("[CacheMarkovChain] Initialized with 5-layer cache model")
    
    def record_lookup(self,
                     layer: CacheLayer,
                     was_hit: bool,
                     latency_ms: float):
        metrics = self.layers[layer]
        
        if was_hit:
            metrics.record_hit(latency_ms)
        else:
            metrics.record_miss(latency_ms)
        
        self.transition_history.append((layer, was_hit))
        if len(self.transition_history) > self.observation_window:
            self.transition_history = self.transition_history[-self.observation_window:]
        
        logger.debug(
            f"[CacheMarkovChain] Recorded {layer.value} "
            f"({'HIT' if was_hit else 'MISS'}) in {latency_ms:.1f}ms"
        )
    
    def get_layer_metrics(self, layer: CacheLayer) -> LayerMetrics:
        return self.layers[layer]
    
    def compute_hit_probability(self, layer: CacheLayer) -> float:
        metrics = self.layers[layer]
        return metrics.hit_probability
    
    def compute_expected_latency(self) -> Dict:
        total_expected_latency = 0.0
        layer_contributions = {}
        
        prob_miss_previous = 1.0
        
        for i, layer in enumerate([
            CacheLayer.CONVERSATION,
            CacheLayer.SEMANTIC,
            CacheLayer.SESSION,
            CacheLayer.GLOBAL,
            CacheLayer.WEB
        ]):
            metrics = self.layers[layer]
            hit_prob = self.compute_hit_probability(layer)
            
            if layer == CacheLayer.WEB:
                prob_reach = prob_miss_previous
                contribution = prob_reach * metrics.latencies.avg_ms
            else:
                prob_reach = prob_miss_previous * 1.0
                contribution = prob_reach * hit_prob * metrics.latencies.avg_ms
                prob_miss_previous *= (1.0 - hit_prob)
            
            layer_contributions[layer.value] = {
                "latency_ms": metrics.latencies.avg_ms,
                "hit_probability": hit_prob,
                "probability_reach": prob_reach,
                "contribution_ms": contribution
            }
            
            total_expected_latency += contribution
        
        logger.debug(
            f"[CacheMarkovChain] E[L]={total_expected_latency:.1f}ms "
            f"(breakdown: {layer_contributions})"
        )
        
        return {
            "expected_latency_ms": total_expected_latency,
            "layer_contributions": layer_contributions,
            "all_miss_probability": prob_miss_previous
        }
    
    def compute_layer_utilization(self) -> Dict[str, float]:
        utilization = {}
        
        for layer in CacheLayer:
            metrics = self.layers[layer]
            utilization[layer.value] = {
                "total_lookups": metrics.total_lookups,
                "hit_count": metrics.hit_count,
                "miss_count": metrics.miss_count,
                "hit_rate": metrics.hit_probability,
                "avg_latency_ms": metrics.latencies.avg_ms
            }
        
        return utilization
    
    def predict_latency_percentiles(self) -> Dict[str, float]:
        samples = []
        
        for layer in [CacheLayer.CONVERSATION, CacheLayer.SEMANTIC,
                      CacheLayer.SESSION, CacheLayer.GLOBAL, CacheLayer.WEB]:
            metrics = self.layers[layer]
            hit_prob = self.compute_hit_probability(layer)
            
            hit_latencies = [metrics.latencies.avg_ms] * int(hit_prob * 1000)
            miss_latencies = [0.1] * int((1 - hit_prob) * 1000)
            
            samples.extend(hit_latencies)
            samples.extend(miss_latencies)
        
        if not samples:
            return {
                "p50_ms": 0.0,
                "p90_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0
            }
        
        samples = sorted(samples)
        
        return {
            "p50_ms": float(np.percentile(samples, 50)),
            "p90_ms": float(np.percentile(samples, 90)),
            "p95_ms": float(np.percentile(samples, 95)),
            "p99_ms": float(np.percentile(samples, 99)),
            "mean_ms": float(np.mean(samples))
        }
    
    def get_convergence_status(self) -> Dict:
        convergence = {}
        
        for layer in CacheLayer:
            metrics = self.layers[layer]
            total = metrics.total_lookups
            
            if total == 0:
                confidence = 0.0
            else:
                p = metrics.hit_probability
                confidence = min(1.0, total / 100.0)
            
            convergence[layer.value] = {
                "sample_count": total,
                "convergence_confidence": confidence,
                "estimated_error": np.sqrt(p * (1 - p) / max(1, total)) if total > 0 else 0.5
            }
        
        return convergence
    
    def plot_expected_latency_curve(self, sample_count_range: range = range(10, 501, 50)) -> Dict:
        curve = []
        
        for sample_count in sample_count_range:
            expected = self.compute_expected_latency()
            curve.append({
                "sample_count": sample_count,
                "expected_latency_ms": expected["expected_latency_ms"],
                "all_miss_probability": expected["all_miss_probability"]
            })
        
        return {
            "convergence_curve": curve,
            "observation_window": self.observation_window
        }
    
    def get_diagnostic_report(self) -> Dict:
        expected_latency = self.compute_expected_latency()
        percentiles = self.predict_latency_percentiles()
        utilization = self.compute_layer_utilization()
        convergence = self.get_convergence_status()
        
        return {
            "model_name": "CacheMarkovChain",
            "expected_latency": expected_latency,
            "latency_percentiles": percentiles,
            "layer_utilization": utilization,
            "convergence_status": convergence,
            "recommendation": self._generate_recommendation(expected_latency, utilization)
        }
    
    def _generate_recommendation(self,
                               expected_latency: Dict,
                               utilization: Dict) -> str:
        total_expected_ms = expected_latency["expected_latency_ms"]
        
        max_contribution = 0.0
        bottleneck_layer = None
        
        for layer_name, contrib in expected_latency["layer_contributions"].items():
            if contrib["contribution_ms"] > max_contribution:
                max_contribution = contrib["contribution_ms"]
                bottleneck_layer = layer_name
        
        if bottleneck_layer == "web":
            return (
                f"Web search is bottleneck ({max_contribution:.0f}ms contribution). "
                "Consider: improved caching, parallel search, or query optimization."
            )
        elif bottleneck_layer:
            hit_rate = utilization[bottleneck_layer]["hit_rate"]
            if hit_rate < 0.3:
                return (
                    f"{bottleneck_layer} cache has low hit rate ({hit_rate:.1%}). "
                    "Consider: expanding cache size or adjusting similarity thresholds."
                )
        
        if total_expected_ms > 1000:
            return f"Latency high ({total_expected_ms:.0f}ms). Multi-layer optimization needed."
        else:
            return "Performance acceptable. Monitor convergence of empirical probabilities."
