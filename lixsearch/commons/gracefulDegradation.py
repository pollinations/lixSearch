"""
Graceful Degradation Metrics Framework

Quantifies system behavior under component failures.

Failure scenarios:
- Disable web search
- Disable global store
- Disable session store
- Disable conversation cache
- Disable semantic cache

Measured impacts:
- ΔCompleteness: Change in answer coverage
- ΔFactuality: Change in citation quality
- ΔLatency: Change in response time
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from datetime import datetime
import numpy as np


class ComponentType(Enum):
    WEB_SEARCH = "web_search"
    GLOBAL_CACHE = "global_cache"
    SESSION_CACHE = "session_cache"
    CONVERSATION_CACHE = "conversation_cache"
    SEMANTIC_CACHE = "semantic_cache"
    EMBEDDING_SERVICE = "embedding_service"
    LLM_SERVICE = "llm_service"


@dataclass
class PerformanceDegradation:
    baseline: float
    degraded: float
    absolute_change: float = 0.0
    relative_change: float = 0.0
    
    def __post_init__(self):
        self.absolute_change = self.degraded - self.baseline
        if self.baseline != 0:
            self.relative_change = (self.degraded - self.baseline) / self.baseline
        else:
            self.relative_change = 0.0
    
    def is_acceptable(self, threshold: float = 0.2) -> bool:
        return abs(self.relative_change) <= threshold


@dataclass
class DegradationScenario:
    disabled_components: List[ComponentType] = field(default_factory=list)
    completeness_degradation: Optional[PerformanceDegradation] = None
    factuality_degradation: Optional[PerformanceDegradation] = None
    latency_degradation: Optional[PerformanceDegradation] = None
    
    @property
    def total_impact_score(self) -> float:
        scores = []
        
        if self.completeness_degradation:
            score = 1.0 - max(0.0, 1.0 - abs(self.completeness_degradation.relative_change))
            scores.append(score)
        
        if self.factuality_degradation:
            score = 1.0 - max(0.0, 1.0 - abs(self.factuality_degradation.relative_change))
            scores.append(score)
        
        if self.latency_degradation:
            score = 0.5 * min(1.0, abs(self.latency_degradation.relative_change))
            scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def get_summary(self) -> Dict:
        return {
            "disabled_components": [c.value for c in self.disabled_components],
            "total_impact_score": self.total_impact_score,
            "completeness": {
                "baseline": self.completeness_degradation.baseline,
                "degraded": self.completeness_degradation.degraded,
                "relative_change": self.completeness_degradation.relative_change
            } if self.completeness_degradation else None,
            "factuality": {
                "baseline": self.factuality_degradation.baseline,
                "degraded": self.factuality_degradation.degraded,
                "relative_change": self.factuality_degradation.relative_change
            } if self.factuality_degradation else None,
            "latency": {
                "baseline_ms": self.latency_degradation.baseline,
                "degraded_ms": self.latency_degradation.degraded,
                "relative_change": self.latency_degradation.relative_change
            } if self.latency_degradation else None
        }


class GracefulDegradationSimulator:
    
    def __init__(self):
        self.baseline_completeness = 0.85
        self.baseline_factuality = 0.80
        self.baseline_latency_ms = 500.0
        
        self.component_impact = {
            ComponentType.WEB_SEARCH: {
                "completeness": 0.20,
                "factuality": 0.15,
                "latency": 0.40
            },
            ComponentType.GLOBAL_CACHE: {
                "completeness": 0.05,
                "factuality": 0.03,
                "latency": 0.20
            },
            ComponentType.SESSION_CACHE: {
                "completeness": 0.03,
                "factuality": 0.02,
                "latency": 0.15
            },
            ComponentType.CONVERSATION_CACHE: {
                "completeness": 0.02,
                "factuality": 0.01,
                "latency": 0.10
            },
            ComponentType.SEMANTIC_CACHE: {
                "completeness": 0.03,
                "factuality": 0.02,
                "latency": 0.15
            },
            ComponentType.EMBEDDING_SERVICE: {
                "completeness": 0.10,
                "factuality": 0.08,
                "latency": 0.05
            },
            ComponentType.LLM_SERVICE: {
                "completeness": 0.50,
                "factuality": 0.50,
                "latency": 0.01
            }
        }
        
        logger.info(
            "[GracefulDegradationSimulator] Initialized with baseline: "
            f"completeness={self.baseline_completeness:.2f}, "
            f"factuality={self.baseline_factuality:.2f}, "
            f"latency={self.baseline_latency_ms:.0f}ms"
        )
    
    def simulate_single_component_failure(self,
                                        component: ComponentType) -> DegradationScenario:
        scenario = DegradationScenario(disabled_components=[component])
        
        impact = self.component_impact.get(component, {})
        
        completeness_loss = impact.get("completeness", 0.0)
        factuality_loss = impact.get("factuality", 0.0)
        latency_increase = impact.get("latency", 0.0)
        
        scenario.completeness_degradation = PerformanceDegradation(
            baseline=self.baseline_completeness,
            degraded=max(0.0, self.baseline_completeness * (1.0 - completeness_loss))
        )
        
        scenario.factuality_degradation = PerformanceDegradation(
            baseline=self.baseline_factuality,
            degraded=max(0.0, self.baseline_factuality * (1.0 - factuality_loss))
        )
        
        scenario.latency_degradation = PerformanceDegradation(
            baseline=self.baseline_latency_ms,
            degraded=self.baseline_latency_ms * (1.0 + latency_increase)
        )
        
        logger.debug(
            f"[Degradation] {component.value} failure simulated: "
            f"completeness Δ={scenario.completeness_degradation.relative_change:.1%}, "
            f"latency Δ={scenario.latency_degradation.relative_change:.1%}"
        )
        
        return scenario
    
    def simulate_multiple_component_failure(self,
                                           components: List[ComponentType]) -> DegradationScenario:
        scenario = DegradationScenario(disabled_components=components)
        
        total_completeness_loss = 0.0
        total_factuality_loss = 0.0
        total_latency_increase = 1.0
        
        for component in components:
            impact = self.component_impact.get(component, {})
            
            total_completeness_loss += impact.get("completeness", 0.0)
            total_factuality_loss += impact.get("factuality", 0.0)
            total_latency_increase *= (1.0 + impact.get("latency", 0.0))
        
        total_completeness_loss = min(1.0, total_completeness_loss)
        total_factuality_loss = min(1.0, total_factuality_loss)
        
        scenario.completeness_degradation = PerformanceDegradation(
            baseline=self.baseline_completeness,
            degraded=max(0.0, self.baseline_completeness * (1.0 - total_completeness_loss))
        )
        
        scenario.factuality_degradation = PerformanceDegradation(
            baseline=self.baseline_factuality,
            degraded=max(0.0, self.baseline_factuality * (1.0 - total_factuality_loss))
        )
        
        scenario.latency_degradation = PerformanceDegradation(
            baseline=self.baseline_latency_ms,
            degraded=self.baseline_latency_ms * total_latency_increase
        )
        
        component_names = [c.value for c in components]
        logger.info(
            f"[Degradation] Multi-component failure: {component_names} - "
            f"Impact score: {scenario.total_impact_score:.2f}"
        )
        
        return scenario
    
    def generate_all_failure_scenarios(self) -> List[DegradationScenario]:
        scenarios = []
        
        for component in ComponentType:
            scenario = self.simulate_single_component_failure(component)
            scenarios.append(scenario)
        
        critical_combinations = [
            [ComponentType.WEB_SEARCH, ComponentType.GLOBAL_CACHE],
            [ComponentType.SESSION_CACHE, ComponentType.CONVERSATION_CACHE],
            [ComponentType.EMBEDDING_SERVICE, ComponentType.LLM_SERVICE],
            [ComponentType.WEB_SEARCH, ComponentType.GLOBAL_CACHE, ComponentType.SESSION_CACHE]
        ]
        
        for components in critical_combinations:
            scenario = self.simulate_multiple_component_failure(components)
            scenarios.append(scenario)
        
        return scenarios
    
    def compute_system_resilience(self) -> Dict:
        scenarios = self.generate_all_failure_scenarios()
        
        if not scenarios:
            return {
                "resilience_score": 1.0,
                "scenarios_analyzed": 0,
                "average_impact": 0.0
            }
        
        impact_scores = [s.total_impact_score for s in scenarios]
        average_impact = np.mean(impact_scores)
        resilience_score = 1.0 - average_impact
        
        critical_scenarios = [s for s in scenarios if s.total_impact_score > 0.5]
        
        logger.info(
            f"[SystemResilience] Resilience score: {resilience_score:.3f} "
            f"(average impact: {average_impact:.3f}, "
            f"critical scenarios: {len(critical_scenarios)})"
        )
        
        return {
            "resilience_score": resilience_score,
            "scenarios_analyzed": len(scenarios),
            "average_impact_score": average_impact,
            "critical_scenarios": len(critical_scenarios),
            "critical_components": [c.value for c in self._identify_critical_components(scenarios)]
        }
    
    def _identify_critical_components(self, scenarios: List[DegradationScenario]) -> List[ComponentType]:
        impact_by_component = {}
        
        for scenario in scenarios:
            if len(scenario.disabled_components) == 1:
                component = scenario.disabled_components[0]
                impact_by_component[component] = scenario.total_impact_score
        
        sorted_components = sorted(impact_by_component.items(), key=lambda x: x[1], reverse=True)
        
        return [comp for comp, _ in sorted_components[:3]]
    
    def generate_mitigation_strategies(self, scenarios: List[DegradationScenario]) -> Dict:
        strategies = {
            "web_search": [
                "Implement fallback cache warmup",
                "Use more conservative cache thresholds",
                "Increase local indexing coverage"
            ],
            "global_cache": [
                "Increase session cache size",
                "Implement distributed caching",
                "Use semantic deduplication"
            ],
            "embedding_service": [
                "Use fallback embedding model",
                "Cache embeddings aggressively",
                "Implement embedding CDN"
            ],
            "llm_service": [
                "Queue requests and retry with exponential backoff",
                "Use cached summaries as fallback",
                "Implement response template system"
            ]
        }
        
        critical_components = [c.value for c in self._identify_critical_components(scenarios)]
        
        recommendations = []
        for component in critical_components:
            if component in strategies:
                recommendations.extend(strategies[component])
        
        return {
            "critical_components": critical_components,
            "mitigation_strategies": recommendations,
            "priority_order": critical_components
        }


class DegradationMetricsTracker:
    
    def __init__(self):
        self.scenarios_observed: Dict[str, DegradationScenario] = {}
        self.component_failure_history: List[Dict] = []
    
    def record_component_failure(self,
                                component: ComponentType,
                                duration_seconds: float,
                                observed_completeness_change: float,
                                observed_factuality_change: float,
                                observed_latency_change: float):
        record = {
            "component": component.value,
            "duration_seconds": duration_seconds,
            "timestamp": datetime.now().isoformat(),
            "observed_changes": {
                "completeness": observed_completeness_change,
                "factuality": observed_factuality_change,
                "latency": observed_latency_change
            }
        }
        
        self.component_failure_history.append(record)
        
        if len(self.component_failure_history) > 1000:
            self.component_failure_history = self.component_failure_history[-1000:]
        
        logger.warning(
            f"[DegradationProxy] Recorded {component.value} failure "
            f"(Δcompleteness={observed_completeness_change:+.1%}, "
            f"Δlatency={observed_latency_change:+.1%})"
        )
    
    def get_empirical_degradation_profile(self) -> Dict:
        profile = {}
        
        for component in ComponentType:
            relevant_records = [
                r for r in self.component_failure_history
                if r["component"] == component.value
            ]
            
            if relevant_records:
                completeness_changes = [
                    r["observed_changes"]["completeness"] for r in relevant_records
                ]
                factuality_changes = [
                    r["observed_changes"]["factuality"] for r in relevant_records
                ]
                latency_changes = [
                    r["observed_changes"]["latency"] for r in relevant_records
                ]
                
                profile[component.value] = {
                    "failure_count": len(relevant_records),
                    "avg_completeness_change": np.mean(completeness_changes),
                    "avg_factuality_change": np.mean(factuality_changes),
                    "avg_latency_change": np.mean(latency_changes),
                    "median_duration_seconds": np.median([
                        r["duration_seconds"] for r in relevant_records
                    ])
                }
        
        return profile
    
    def compare_simulated_vs_actual(self, simulator: GracefulDegradationSimulator) -> Dict:
        empirical = self.get_empirical_degradation_profile()
        simulated = simulator.component_impact
        
        comparison = {}
        
        for component_str, empirical_data in empirical.items():
            try:
                component = ComponentType[component_str.upper().replace(" ", "_")]
                simulated_impacts = simulated.get(component, {})
                
                comparison[component_str] = {
                    "simulated_completeness_impact": simulated_impacts.get("completeness", 0),
                    "actual_completeness_change": empirical_data["avg_completeness_change"],
                    "prediction_error_completeness": abs(
                        simulated_impacts.get("completeness", 0) -
                        empirical_data["avg_completeness_change"]
                    ),
                    "failure_count": empirical_data["failure_count"]
                }
            except:
                pass
        
        avg_error = np.mean([
            v["prediction_error_completeness"] for v in comparison.values()
            if "prediction_error_completeness" in v
        ]) if comparison else 0.0
        
        return {
            "model_accuracy": 1.0 - avg_error if avg_error < 1.0 else 0.0,
            "component_accuracy": comparison
        }
