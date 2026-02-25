import asyncio
import threading
from typing import Optional, Any
from contextlib import asynccontextmanager, contextmanager
from loguru import logger
import time


class AsyncConnectionPool:
    
    def __init__(self, max_connections: int = 20, timeout: float = 10.0):
        self.max_connections = max_connections
        self.timeout = timeout
        self.available = asyncio.Queue(maxsize=max_connections)
        self.active_count = 0
        self.lock = asyncio.Lock()
        self.stats = {
            "total_requests": 0,
            "total_waited": 0,
            "max_concurrent": 0
        }
        
        for _ in range(max_connections):
            self.available.put_nowait(None)
        
        logger.info(f"[ConnectionPool] Initialized async pool with {max_connections} slots")
    
    @asynccontextmanager
    async def get_connection(self):
        start_time = time.time()
        
        async with self.lock:
            self.active_count += 1
            self.stats["total_requests"] += 1
            self.stats["max_concurrent"] = max(
                self.stats["max_concurrent"], 
                self.active_count
            )
        
        try:
            conn = await asyncio.wait_for(
                self.available.get(),
                timeout=self.timeout
            )
            
            waited_time = time.time() - start_time
            self.stats["total_waited"] += waited_time
            
            if waited_time > 1.0:
                logger.warning(f"[ConnectionPool] Long wait: {waited_time:.2f}s for connection")
            
            yield conn
            
        except asyncio.TimeoutError:
            logger.error(f"[ConnectionPool] Timeout waiting for connection after {self.timeout}s")
            raise
        finally:
            await self.available.put(conn)
            async with self.lock:
                self.active_count -= 1
    
    def get_utilization(self) -> float:
        return self.active_count / self.max_connections
    
    def get_stats(self) -> dict:
        avg_wait = (
            self.stats["total_waited"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0
            else 0
        )
        
        return {
            "max_connections": self.max_connections,
            "active": self.active_count,
            "utilization": self.get_utilization(),
            "total_requests": self.stats["total_requests"],
            "avg_wait_time": avg_wait,
            "max_concurrent": self.stats["max_concurrent"]
        }


class SyncConnectionPool:
    
    def __init__(self, max_connections: int = 20, timeout: float = 10.0):
        self.max_connections = max_connections
        self.timeout = timeout
        self.semaphore = threading.Semaphore(max_connections)
        self.lock = threading.RLock()
        self.active_count = 0
        self.stats = {
            "total_requests": 0,
            "total_waited": 0,
            "max_concurrent": 0
        }
        
        logger.info(f"[SyncConnectionPool] Initialized sync pool with {max_connections} slots")
    
    @contextmanager
    def get_connection(self):
        start_time = time.time()
        
        acquired = self.semaphore.acquire(timeout=self.timeout)
        
        if not acquired:
            raise TimeoutError(f"Could not acquire connection within {self.timeout}s")
        
        with self.lock:
            self.active_count += 1
            self.stats["total_requests"] += 1
            self.stats["max_concurrent"] = max(
                self.stats["max_concurrent"],
                self.active_count
            )
        
        try:
            waited_time = time.time() - start_time
            self.stats["total_waited"] += waited_time
            
            if waited_time > 1.0:
                logger.warning(f"[SyncConnectionPool] Long wait: {waited_time:.2f}s for connection")
            
            yield None
            
        finally:
            with self.lock:
                self.active_count -= 1
            self.semaphore.release()
    
    def get_utilization(self) -> float:
        return self.active_count / self.max_connections
    
    def get_stats(self) -> dict:
        avg_wait = (
            self.stats["total_waited"] / self.stats["total_requests"]
            if self.stats["total_requests"] > 0
            else 0
        )
        
        return {
            "max_connections": self.max_connections,
            "active": self.active_count,
            "utilization": self.get_utilization(),
            "total_requests": self.stats["total_requests"],
            "avg_wait_time": avg_wait,
            "max_concurrent": self.stats["max_concurrent"]
        }


_sync_pool = None
_async_pool = None


def get_sync_pool(max_connections: int = 20) -> SyncConnectionPool:
    global _sync_pool
    if _sync_pool is None:
        _sync_pool = SyncConnectionPool(max_connections=max_connections)
    return _sync_pool


def get_async_pool(max_connections: int = 20) -> AsyncConnectionPool:
    global _async_pool
    if _async_pool is None:
        _async_pool = AsyncConnectionPool(max_connections=max_connections)
    return _async_pool
