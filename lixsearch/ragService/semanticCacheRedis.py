import hashlib
import threading
import time
from typing import Dict, Optional, List, Tuple, Any
from loguru import logger
import numpy as np
import pickle
from datetime import datetime

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.error("[SemanticCacheRedis] Redis not available - Redis is REQUIRED for optimized caching")
    
    
    def stats(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "cached_queries": len(self.query_cache),
                "max_size": self.max_size,
                "utilization": len(self.query_cache) / self.max_size if self.max_size > 0 else 0,
                "ttl_seconds": self.ttl_seconds
            }


class URLEmbeddingCache:
    
    def __init__(
        self,
        ttl_seconds: int = 86400,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 1
    ):
        self.ttl_seconds = ttl_seconds
        self.redis_client = None
        self.lock = threading.RLock()
        
        if not REDIS_AVAILABLE:
            raise RuntimeError("[URLEmbeddingCache] Redis is required but not installed")
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            self.redis_client.ping()
            logger.info(f"[URLEmbeddingCache] Connected to Redis @ {redis_host}:{redis_port} (db={redis_db})")
        except Exception as e:
            logger.error(f"[URLEmbeddingCache] Failed to connect to Redis: {e}")
            raise
    
    def get(self, url: str) -> Optional[np.ndarray]:
        with self.lock:
            try:
                key = f"url_embedding:{url}"
                cached = self.redis_client.get(key)
                if cached:
                    embedding = np.frombuffer(cached, dtype=np.float32)
                    logger.debug(f"[URLEmbeddingCache] HIT: {url}")
                    return embedding
                return None
            except Exception as e:
                logger.warning(f"[URLEmbeddingCache] Get error for {url}: {e}")
                return None
    
    def set(self, url: str, embedding: np.ndarray) -> bool:
        with self.lock:
            try:
                key = f"url_embedding:{url}"
                if isinstance(embedding, np.ndarray):
                    embedding_bytes = embedding.astype(np.float32).tobytes()
                else:
                    embedding = np.array(embedding, dtype=np.float32)
                    embedding_bytes = embedding.tobytes()
                
                self.redis_client.setex(key, self.ttl_seconds, embedding_bytes)
                logger.debug(f"[URLEmbeddingCache] STORED: {url}")
                return True
            except Exception as e:
                logger.warning(f"[URLEmbeddingCache] Set error for {url}: {e}")
                return False
    
    def get_stats(self) -> Dict:
        try:
            info = self.redis_client.info("memory")
            return {
                "backend": "redis",
                "redis_memory_used": info.get("used_memory_human", "unknown"),
                "ttl_seconds": self.ttl_seconds
            }
        except Exception as e:
            logger.warning(f"[URLEmbeddingCache] Failed to get stats: {e}")
            return {"backend": "redis", "status": "error"}


class SessionContextWindow:
    
    def __init__(
        self,
        session_id: str,
        window_size: int = 20,
        ttl_seconds: int = 3600,
        max_tokens: Optional[int] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 2
    ):
        self.session_id = session_id
        self.window_size = window_size
        self.ttl_seconds = ttl_seconds
        self.max_tokens = max_tokens
        self.redis_client = None
        self.lock = threading.RLock()
        
        if not REDIS_AVAILABLE:
            raise RuntimeError("[SessionContextWindow] Redis is required but not installed")
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            self.redis_client.ping()
            logger.info(
                f"[SessionContextWindow] Initialized for session {session_id}: "
                f"window_size={window_size}, ttl={ttl_seconds}s"
            )
        except Exception as e:
            logger.error(f"[SessionContextWindow] Failed to connect to Redis: {e}")
            raise
    
    def _get_key(self) -> str:
        return f"session_context:{self.session_id}"
    
    def _get_order_key(self) -> str:
        return f"session_order:{self.session_id}"
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> int:
        with self.lock:
            try:
                key = self._get_key()
                order_key = self._get_order_key()
                timestamp = time.time()
                
                msg_id = int(timestamp * 1000) % (2**31)
                message = {
                    "role": role,
                    "content": content,
                    "timestamp": timestamp,
                    "metadata": metadata or {}
                }
                
                msg_key = f"{key}:{msg_id}"
                self.redis_client.setex(
                    msg_key,
                    self.ttl_seconds,
                    pickle.dumps(message)
                )
                
                self.redis_client.lpush(order_key, msg_id)
                self.redis_client.expire(order_key, self.ttl_seconds)
                
                window_len = self.redis_client.llen(order_key)
                if window_len > self.window_size:
                    excess = window_len - self.window_size
                    for _ in range(excess):
                        old_id = self.redis_client.rpop(order_key)
                        if old_id:
                            self.redis_client.delete(f"{key}:{old_id}")
                            logger.debug(f"[SessionContextWindow] LRU evicted message {old_id}")
                
                logger.debug(f"[SessionContextWindow] Added message {role} (window: {min(window_len, self.window_size)})")
                return min(window_len, self.window_size)
                
            except Exception as e:
                logger.warning(f"[SessionContextWindow] Failed to add message: {e}")
                return 0
    
    def get_context(self) -> List[Dict]:
        with self.lock:
            try:
                key = self._get_key()
                order_key = self._get_order_key()
                
                msg_ids = self.redis_client.lrange(order_key, 0, -1)
                
                messages = []
                for msg_id in reversed(msg_ids):
                    try:
                        msg_key = f"{key}:{msg_id}"
                        msg_data = self.redis_client.get(msg_key)
                        if msg_data:
                            message = pickle.loads(msg_data)
                            messages.append(message)
                    except Exception as e:
                        logger.warning(f"[SessionContextWindow] Failed to decode message {msg_id}: {e}")
                
                logger.debug(f"[SessionContextWindow] Retrieved {len(messages)} messages from window")
                return messages
            except Exception as e:
                logger.warning(f"[SessionContextWindow] Failed to get context: {e}")
                return []
    
    def get_formatted_context(self, max_lines: int = 50) -> str:
        messages = self.get_context()
        lines = []
        
        for msg in messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            if content:
                if len(content) > 200:
                    content = content[:200] + "..."
                lines.append(f"{role}: {content}")
        
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        
        return "\n".join(lines) if lines else ""
    
    def clear(self) -> bool:
        with self.lock:
            try:
                key = self._get_key()
                order_key = self._get_order_key()
                
                msg_ids = self.redis_client.lrange(order_key, 0, -1)
                
                for msg_id in msg_ids:
                    self.redis_client.delete(f"{key}:{msg_id}")
                
                self.redis_client.delete(order_key)
                
                logger.info(f"[SessionContextWindow] Cleared {len(msg_ids)} messages")
                return True
            except Exception as e:
                logger.warning(f"[SessionContextWindow] Failed to clear context: {e}")
                return False
    
    def get_stats(self) -> Dict:
        try:
            order_key = self._get_order_key()
            window_len = self.redis_client.llen(order_key)
            
            return {
                "session_id": self.session_id,
                "messages_in_window": window_len,
                "window_size": self.window_size,
                "utilization": window_len / self.window_size if self.window_size > 0 else 0,
                "ttl_seconds": self.ttl_seconds
            }
        except Exception as e:
            logger.warning(f"[SessionContextWindow] Failed to get stats: {e}")
            return {"session_id": self.session_id, "status": "error"}


class SemanticCacheRedis:
    
    def __init__(
        self, 
        ttl_seconds: int = 300, 
        similarity_threshold: float = 0.90,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0
    ):
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.redis_client = None
        self.lock = threading.RLock()
        
        if not REDIS_AVAILABLE:
            logger.error("[SemanticCacheRedis] Redis is REQUIRED but not installed")
            raise RuntimeError("Redis installation is required for semantic caching")
        
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            self.redis_client.ping()
            logger.info(f"[SemanticCacheRedis] Connected to Redis @ {redis_host}:{redis_port} (db={redis_db})")
        except Exception as e:
            logger.error(f"[SemanticCacheRedis] Failed to connect to Redis: {e}")
            raise
    
    def _get_redis_key(self, request_id: str, url: str) -> str:
        return f"semantic_cache:{request_id}:{url}"
    
    def get(self, url: str, query_embedding: np.ndarray, request_id: str = "default") -> Optional[Dict]:
        with self.lock:
            try:
                redis_key = self._get_redis_key(request_id, url)
                cached_data = self.redis_client.get(redis_key)
                
                if not cached_data:
                    return None
                
                cache_entry = pickle.loads(cached_data)
                best_match = None
                best_similarity = 0.0
                
                query_emb = np.array(query_embedding, dtype=np.float32)
                query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
                
                for cached_item in cache_entry.get("items", []):
                    cached_emb = np.array(cached_item["embedding"], dtype=np.float32)
                    cached_emb = cached_emb / (np.linalg.norm(cached_emb) + 1e-8)
                    similarity = float(np.dot(cached_emb, query_emb))
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = cached_item
                
                if best_similarity >= self.similarity_threshold and best_match:
                    logger.debug(f"[SemanticCacheRedis] HIT for {url} (similarity: {best_similarity:.3f})")
                    return best_match["response"]
                
                return None
            except Exception as e:
                logger.warning(f"[SemanticCacheRedis] Get error for {url}: {e}")
                return None
    
    def set(self, url: str, query_embedding: np.ndarray, response: Dict, request_id: str = "default") -> None:
        with self.lock:
            try:
                redis_key = self._get_redis_key(request_id, url)
                
                cached_data = self.redis_client.get(redis_key)
                if cached_data:
                    cache_entry = pickle.loads(cached_data)
                else:
                    cache_entry = {"items": []}
                
                cache_entry["items"].append({
                    "embedding": query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
                    "response": response,
                    "timestamp": time.time()
                })
                
                if len(cache_entry["items"]) > 50:
                    cache_entry["items"] = cache_entry["items"][-50:]
                
                self.redis_client.setex(
                    redis_key,
                    self.ttl_seconds,
                    pickle.dumps(cache_entry)
                )
                logger.debug(f"[SemanticCacheRedis] Stored: {redis_key}")
            except Exception as e:
                logger.warning(f"[SemanticCacheRedis] Set error: {e}")
    
    def get_stats(self) -> Dict:
        try:
            info = self.redis_client.info("memory")
            return {
                "backend": "redis",
                "redis_memory_used": info.get("used_memory_human", "unknown"),
                "ttl_seconds": self.ttl_seconds,
                "similarity_threshold": self.similarity_threshold
            }
        except Exception as e:
            logger.warning(f"[SemanticCacheRedis] Failed to get stats: {e}")
            return {"backend": "redis", "status": "error"}


SemanticCache = SemanticCacheRedis
