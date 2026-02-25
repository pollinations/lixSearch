import logging
import time
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger("vectordb_monitor")


class VectorDBMonitor:
    
    def __init__(self):
        self.query_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        self.query_count = 0
        self.error_count = 0
        self.last_error = None
        self.last_heartbeat = datetime.now()
        self.connection_pool_stats = {}
        self.lock = asyncio.Lock()
    
    async def record_query(self, duration: float, is_cache_hit: bool):
        async with self.lock:
            self.query_count += 1
            self.query_times.append(duration)
            
            if len(self.query_times) > 1000:
                self.query_times = self.query_times[-1000:]
            
            if is_cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    async def record_error(self, error: Exception):
        async with self.lock:
            self.error_count += 1
            self.last_error = str(error)
    
    async def heartbeat(self):
        async with self.lock:
            self.last_heartbeat = datetime.now()
    
    async def get_stats(self) -> Dict[str, Any]:
        async with self.lock:
            if not self.query_times:
                avg_query_time = 0
                p50_time = 0
                p95_time = 0
                p99_time = 0
                min_time = 0
                max_time = 0
            else:
                sorted_times = sorted(self.query_times)
                avg_query_time = sum(self.query_times) / len(self.query_times)
                p50_time = sorted_times[len(sorted_times) // 2]
                p95_time = sorted_times[int(len(sorted_times) * 0.95)]
                p99_time = sorted_times[int(len(sorted_times) * 0.99)]
                min_time = min(sorted_times)
                max_time = max(sorted_times)
            
            total_requests = self.cache_hits + self.cache_misses
            cache_hit_rate = (
                self.cache_hits / total_requests if total_requests > 0 else 0
            )
            
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.last_heartbeat).total_seconds(),
                "queries": {
                    "total": self.query_count,
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "cache_hit_rate": cache_hit_rate,
                },
                "performance": {
                    "avg_query_time_ms": round(avg_query_time * 1000, 2),
                    "p50_query_time_ms": round(p50_time * 1000, 2),
                    "p95_query_time_ms": round(p95_time * 1000, 2),
                    "p99_query_time_ms": round(p99_time * 1000, 2),
                    "min_query_time_ms": round(min_time * 1000, 2),
                    "max_query_time_ms": round(max_time * 1000, 2),
                },
                "errors": {
                    "total_errors": self.error_count,
                    "last_error": self.last_error,
                },
                "connection_pool": self.connection_pool_stats,
            }
    
    async def update_pool_stats(self, pool_stats: Dict[str, Any]):
        async with self.lock:
            self.connection_pool_stats = pool_stats


_monitor_instance: Optional[VectorDBMonitor] = None


def get_vector_db_monitor() -> VectorDBMonitor:
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = VectorDBMonitor()
    return _monitor_instance


class QueryTimer:
    
    def __init__(self, monitor: VectorDBMonitor, is_cache_hit: bool = False):
        self.monitor = monitor
        self.is_cache_hit = is_cache_hit
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is not None:
            await self.monitor.record_error(exc_val)
        else:
            await self.monitor.record_query(duration, self.is_cache_hit)
        
        return False
