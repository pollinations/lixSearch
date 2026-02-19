from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from loguru import logger
from collections import defaultdict, deque
import numpy as np
import json


class MetricType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    TOKEN_USAGE = "token_usage"
    COST = "cost"
    QUALITY = "quality"


@dataclass
class MetricPoint:
    timestamp: datetime
    value: float
    metric_type: MetricType
    tags: Dict = field(default_factory=dict)
    source: str = ""


class MetricsCollector:
    
    def __init__(self, window_size: int = 1000, sink_interval_seconds: int = 60):
        self.window_size = window_size
        self.sink_interval_seconds = sink_interval_seconds
        self.metrics_buffer: Dict[MetricType, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.last_sink_time = datetime.now()
    
    def record_metric(self,
                     metric_type: MetricType,
                     value: float,
                     tags: Optional[Dict] = None,
                     source: str = ""):
        
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            metric_type=metric_type,
            tags=tags or {},
            source=source
        )
        
        self.metrics_buffer[metric_type].append(point)
    
    def record_latency(self, latency_ms: float, component: str = ""):
        self.record_metric(MetricType.LATENCY, latency_ms, {"component": component})
    
    def record_cache_hit(self, is_hit: bool, layer: str = ""):
        self.record_metric(
            MetricType.CACHE_HIT_RATE,
            1.0 if is_hit else 0.0,
            {"layer": layer}
        )
    
    def record_tokens_used(self, tokens: int, model: str = ""):
        self.record_metric(MetricType.TOKEN_USAGE, float(tokens), {"model": model})
    
    def record_cost(self, cost_usd: float, operation: str = ""):
        self.record_metric(MetricType.COST, cost_usd, {"operation": operation})
    
    def record_error(self, error_type: str = ""):
        self.record_metric(MetricType.ERROR_RATE, 1.0, {"error_type": error_type})
    
    def get_metrics_for_window(self, metric_type: MetricType, minutes: int = 5) -> List[MetricPoint]:
        cutoff = datetime.now() - timedelta(minutes=minutes)
        
        return [
            point for point in self.metrics_buffer[metric_type]
            if point.timestamp >= cutoff
        ]
    
    def compute_percentile(self, metric_type: MetricType, percentile: float = 95.0) -> float:
        values = [p.value for p in self.metrics_buffer[metric_type]]
        
        if not values:
            return 0.0
        
        return float(np.percentile(values, percentile))
    
    def get_histogram(self, metric_type: MetricType, bins: int = 10) -> Dict:
        values = [p.value for p in self.metrics_buffer[metric_type]]
        
        if not values:
            return {"bins": [], "counts": []}
        
        hist, bin_edges = np.histogram(values, bins=bins)
        
        return {
            "bins": [float(e) for e in bin_edges],
            "counts": [int(c) for c in hist],
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values))
        }


class PerformanceMonitor:
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.collector = metrics_collector
        self.alerts: List[Dict] = []
        self.alert_thresholds = {
            "latency_p95_ms": 2000,
            "error_rate_percent": 5.0,
            "cache_hit_rate_percent": 30.0
        }
    
    def check_sla(self, sla_definition: Dict) -> Tuple[bool, List[str]]:
        violations = []
        
        if "p95_latency_ms" in sla_definition:
            p95 = self.collector.compute_percentile(MetricType.LATENCY, 95.0)
            if p95 > sla_definition["p95_latency_ms"]:
                violations.append(f"P95 latency {p95:.0f}ms exceeds SLA {sla_definition['p95_latency_ms']}ms")
        
        if "min_cache_hit_rate" in sla_definition:
            cache_points = self.collector.get_metrics_for_window(MetricType.CACHE_HIT_RATE, 5)
            if cache_points:
                hit_rate = np.mean([p.value for p in cache_points])
                if hit_rate < sla_definition["min_cache_hit_rate"]:
                    violations.append(
                        f"Cache hit rate {hit_rate:.1%} below SLA {sla_definition['min_cache_hit_rate']:.1%}"
                    )
        
        if "max_error_rate" in sla_definition:
            error_points = self.collector.get_metrics_for_window(MetricType.ERROR_RATE, 5)
            if error_points:
                error_rate = len(error_points) / max(1, self.collector.window_size)
                if error_rate > sla_definition["max_error_rate"]:
                    violations.append(
                        f"Error rate {error_rate:.1%} exceeds SLA {sla_definition['max_error_rate']:.1%}"
                    )
        
        return len(violations) == 0, violations
    
    def generate_alert(self, alert_type: str, severity: str, message: str):
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": alert_type,
            "severity": severity,
            "message": message
        }
        
        self.alerts.append(alert)
        
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        logger.warning(f"[Alert] {severity}: {message}")
    
    def check_anomalies(self) -> List[Dict]:
        anomalies = []
        
        latency_points = self.collector.get_metrics_for_window(MetricType.LATENCY, 5)
        if len(latency_points) > 10:
            latencies = [p.value for p in latency_points]
            mean_lat = np.mean(latencies)
            std_lat = np.std(latencies)
            
            for point in latency_points[-5:]:
                if abs(point.value - mean_lat) > 3 * std_lat:
                    anomalies.append({
                        "type": "latency_spike",
                        "timestamp": point.timestamp.isoformat(),
                        "value": point.value,
                        "expected_range": [mean_lat - 3*std_lat, mean_lat + 3*std_lat],
                        "component": point.tags.get("component", "unknown")
                    })
        
        error_points = self.collector.get_metrics_for_window(MetricType.ERROR_RATE, 5)
        if len(error_points) > 0:
            error_rate = len(error_points) / max(1, self.collector.window_size)
            if error_rate > self.alert_thresholds["error_rate_percent"] / 100:
                anomalies.append({
                    "type": "high_error_rate",
                    "timestamp": datetime.now().isoformat(),
                    "error_rate_percent": error_rate * 100,
                    "threshold_percent": self.alert_thresholds["error_rate_percent"]
                })
        
        return anomalies
    
    def get_dashboard_summary(self) -> Dict:
        latency_hist = self.collector.get_histogram(MetricType.LATENCY)
        cache_points = self.collector.get_metrics_for_window(MetricType.CACHE_HIT_RATE, 5)
        cache_hit_rate = np.mean([p.value for p in cache_points]) if cache_points else 0.0
        
        token_points = self.collector.get_metrics_for_window(MetricType.TOKEN_USAGE, 60)
        total_tokens = sum(p.value for p in token_points) if token_points else 0
        
        cost_points = self.collector.get_metrics_for_window(MetricType.COST, 60)
        total_cost = sum(p.value for p in cost_points) if cost_points else 0.0
        
        error_points = self.collector.get_metrics_for_window(MetricType.ERROR_RATE, 5)
        error_count = len(error_points)
        
        anomalies = self.check_anomalies()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "latency": {
                "p50_ms": latency_hist.get("percentile_50", 0),
                "p95_ms": self.collector.compute_percentile(MetricType.LATENCY, 95),
                "p99_ms": self.collector.compute_percentile(MetricType.LATENCY, 99),
                "mean_ms": latency_hist.get("mean", 0),
                "histogram": latency_hist
            },
            "cache": {
                "hit_rate_percent": cache_hit_rate * 100,
                "recent_hits": len([p for p in cache_points if p.value > 0.5]),
                "recent_misses": len([p for p in cache_points if p.value < 0.5])
            },
            "tokens": {
                "total_tokens_60min": int(total_tokens),
                "avg_tokens_per_min": int(total_tokens / 60) if token_points else 0
            },
            "cost": {
                "total_cost_usd_60min": round(total_cost, 4),
                "avg_cost_per_min": round(total_cost / 60, 6) if cost_points else 0
            },
            "errors": {
                "error_count_5min": error_count,
                "error_rate_percent": (error_count / max(1, self.collector.window_size)) * 100
            },
            "anomalies": anomalies,
            "alerts_count": len([a for a in self.alerts if a["severity"] == "WARNING"])
        }


class MetricsAggregator:
    
    def __init__(self):
        self.hourly_aggregates: List[Dict] = []
        self.daily_aggregates: List[Dict] = []
    
    def aggregate_by_hour(self, collector: MetricsCollector) -> Dict:
        now = datetime.now()
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        
        latency_points = [
            p for p in collector.metrics_buffer[MetricType.LATENCY]
            if p.timestamp >= hour_start
        ]
        
        cache_points = [
            p for p in collector.metrics_buffer[MetricType.CACHE_HIT_RATE]
            if p.timestamp >= hour_start
        ]
        
        token_points = [
            p for p in collector.metrics_buffer[MetricType.TOKEN_USAGE]
            if p.timestamp >= hour_start
        ]
        
        cost_points = [
            p for p in collector.metrics_buffer[MetricType.COST]
            if p.timestamp >= hour_start
        ]
        
        aggregate = {
            "hour": hour_start.isoformat(),
            "latency": {
                "mean_ms": float(np.mean([p.value for p in latency_points])) if latency_points else 0,
                "p95_ms": float(np.percentile([p.value for p in latency_points], 95)) if latency_points else 0,
                "count": len(latency_points)
            },
            "cache": {
                "hit_rate": float(np.mean([p.value for p in cache_points])) if cache_points else 0,
                "count": len(cache_points)
            },
            "tokens": {
                "total": int(sum(p.value for p in token_points)),
                "count": len(token_points)
            },
            "cost": {
                "total_usd": float(sum(p.value for p in cost_points)),
                "count": len(cost_points)
            }
        }
        
        self.hourly_aggregates.append(aggregate)
        if len(self.hourly_aggregates) > 168:
            self.hourly_aggregates = self.hourly_aggregates[-168:]
        
        return aggregate
    
    def aggregate_by_day(self) -> Optional[Dict]:
        if len(self.hourly_aggregates) < 24:
            return None
        
        recent_24h = self.hourly_aggregates[-24:]
        
        aggregate = {
            "day": datetime.now().date().isoformat(),
            "latency": {
                "mean_ms": float(np.mean([h["latency"]["mean_ms"] for h in recent_24h])),
                "p95_ms": float(np.mean([h["latency"]["p95_ms"] for h in recent_24h]))
            },
            "cache": {
                "hit_rate": float(np.mean([h["cache"]["hit_rate"] for h in recent_24h]))
            },
            "tokens": {
                "total": int(sum(h["tokens"]["total"] for h in recent_24h))
            },
            "cost": {
                "total_usd": float(sum(h["cost"]["total_usd"] for h in recent_24h))
            }
        }
        
        self.daily_aggregates.append(aggregate)
        if len(self.daily_aggregates) > 365:
            self.daily_aggregates = self.daily_aggregates[-365:]
        
        return aggregate
    
    def get_trend(self, metric_type: str, window_hours: int = 24) -> Dict:
        if metric_type == "latency":
            data = [(h["hour"], h["latency"]["mean_ms"]) for h in self.hourly_aggregates[-window_hours:]]
        elif metric_type == "cache_hit_rate":
            data = [(h["hour"], h["cache"]["hit_rate"] * 100) for h in self.hourly_aggregates[-window_hours:]]
        elif metric_type == "cost":
            data = [(h["hour"], h["cost"]["total_usd"]) for h in self.hourly_aggregates[-window_hours:]]
        else:
            data = []
        
        if not data:
            return {"trend": [], "direction": "flat"}
        
        values = [v for _, v in data]
        first_half_mean = np.mean(values[:len(values)//2]) if len(values) > 1 else values[0]
        second_half_mean = np.mean(values[len(values)//2:]) if len(values) > 1 else values[-1]
        
        if second_half_mean > first_half_mean * 1.1:
            direction = "increasing"
        elif second_half_mean < first_half_mean * 0.9:
            direction = "decreasing"
        else:
            direction = "stable"
        
        return {
            "metric": metric_type,
            "window_hours": window_hours,
            "data": data,
            "direction": direction,
            "trend_change_percent": ((second_half_mean - first_half_mean) / first_half_mean * 100) if first_half_mean > 0 else 0
        }


class DistributedTracer:
    
    def __init__(self):
        self.traces: Dict[str, Dict] = {}
        self.max_traces = 1000
    
    def start_trace(self, trace_id: str, operation: str) -> None:
        self.traces[trace_id] = {
            "trace_id": trace_id,
            "operation": operation,
            "start_time": datetime.now(),
            "spans": [],
            "status": "in_progress"
        }
    
    def add_span(self, trace_id: str, span_name: str, duration_ms: float) -> None:
        if trace_id in self.traces:
            self.traces[trace_id]["spans"].append({
                "name": span_name,
                "duration_ms": duration_ms,
                "timestamp": datetime.now().isoformat()
            })
    
    def end_trace(self, trace_id: str, status: str = "success") -> None:
        if trace_id in self.traces:
            self.traces[trace_id]["end_time"] = datetime.now()
            self.traces[trace_id]["status"] = status
            
            total_duration = (self.traces[trace_id]["end_time"] - 
                            self.traces[trace_id]["start_time"]).total_seconds() * 1000
            self.traces[trace_id]["total_duration_ms"] = total_duration
        
        if len(self.traces) > self.max_traces:
            oldest_trace_id = min(self.traces.keys(), 
                                 key=lambda k: self.traces[k].get("start_time", datetime.now()))
            del self.traces[oldest_trace_id]
    
    def get_trace(self, trace_id: str) -> Optional[Dict]:
        return self.traces.get(trace_id)
    
    def get_traces_by_operation(self, operation: str, limit: int = 10) -> List[Dict]:
        matching = [
            t for t in self.traces.values()
            if t.get("operation") == operation
        ]
        
        return sorted(matching, key=lambda x: x.get("start_time", datetime.now()), reverse=True)[:limit]
