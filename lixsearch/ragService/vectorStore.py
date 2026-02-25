from loguru import logger 
import torch
import chromadb
from pathlib import Path
import threading
import numpy as np
from pipeline.config import EMBEDDING_DIMENSION
from typing import List, Dict
from datetime import datetime
import os
import time
from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent
from overrides import override


class NoOpProductTelemetry(ProductTelemetryClient):
    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return


class VectorStore:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, embedding_dim: int = None, embeddings_dir: str = "./embeddings"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, embedding_dim: int = None, embeddings_dir: str = "./embeddings"):
        if self._initialized:
            return
        
        if embedding_dim is None:
            embedding_dim = EMBEDDING_DIMENSION
        self.embedding_dim = embedding_dim
        self.embeddings_dir = embeddings_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine Chroma API implementation
        self.chroma_api_impl = os.getenv("CHROMA_API_IMPL", "embedded").lower()
        self.client = None
        self.collection = None
        self.chunk_count = 0
        self.lock = threading.RLock()
        self.metadata_path = os.path.join(embeddings_dir, "metadata.json")
        
        # Initialize Chroma client
        self._initialize_chroma_client()
        self._load_from_disk()
        
        self._initialized = True
        logger.info(f"[VectorStore] Initialized with {self.chunk_count} chunks on {self.device}")
    
    def _initialize_chroma_client(self) -> None:
        """Initialize Chroma client based on API implementation"""
        try:
            if self.chroma_api_impl == "http":
                # Use HTTP client for server mode
                host = os.getenv("CHROMA_SERVER_HOST", "localhost")
                port = int(os.getenv("CHROMA_SERVER_PORT", "8000"))
                
                logger.info(f"[VectorStore] Connecting to Chroma server at {host}:{port}")
                
                max_retries = 5
                retry_delay = 2
                
                for attempt in range(max_retries):
                    try:
                        self.client = chromadb.HttpClient(
                            host=host,
                            port=port,
                            settings=chromadb.config.Settings(
                                chroma_api_impl="rest",
                                allow_reset=True,
                                anonymized_telemetry=False,
                            )
                        )
                        
                        # Test connection
                        self.client.heartbeat()
                        logger.info(f"[VectorStore] Connected to Chroma server successfully")
                        break
                    except Exception as e:
                        if attempt < max_retries - 1:
                            logger.warning(
                                f"[VectorStore] Connection attempt {attempt + 1}/{max_retries} failed: {e}. "
                                f"Retrying in {retry_delay}s..."
                            )
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"[VectorStore] Failed to connect to Chroma server after {max_retries} attempts")
                            raise
            else:
                # Use embedded Persistent client
                Path(self.embeddings_dir).mkdir(parents=True, exist_ok=True)
                
                logger.info(f"[VectorStore] Using embedded Chroma with path {self.embeddings_dir}")
                
                try:
                    # Try with minimal settings for Chroma 1.5.1+
                    chroma_settings = chromadb.config.Settings(
                        anonymized_telemetry=False,
                        is_persistent=True
                    )
                    self.client = chromadb.PersistentClient(
                        path=self.embeddings_dir,
                        settings=chroma_settings
                    )
                except (ValueError, TypeError):
                    # Fallback: initialize without settings (Chroma 1.5.x doesn't require them)
                    logger.warning("[VectorStore] Initializing Chroma without custom settings")
                    self.client = chromadb.PersistentClient(path=self.embeddings_dir)
                
                logger.info(f"[VectorStore] Embedded Chroma initialized")
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="document_embeddings",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"[VectorStore] Collection 'document_embeddings' ready")
            
        except Exception as e:
            logger.error(f"[VectorStore] Failed to initialize Chroma: {e}")
            raise
    
    def add_chunks(self, chunks: List[Dict]) -> None:
        """Add chunks with automatic batching for efficient vector DB operations"""
        with self.lock:
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                emb = chunk["embedding"]
                if isinstance(emb, list):
                    emb = np.array(emb, dtype=np.float32)
                elif isinstance(emb, torch.Tensor):
                    emb = emb.cpu().numpy().astype(np.float32)
                else:
                    emb = np.array(emb, dtype=np.float32)
                
                # Normalize embedding
                emb = emb / (np.linalg.norm(emb) + 1e-8)
                
                chunk_id = str(self.chunk_count + i)
                ids.append(chunk_id)
                embeddings.append(emb.tolist())
                documents.append(chunk["text"])
                metadatas.append({
                    "url": chunk["url"],
                    "chunk_id": chunk.get("chunk_id", chunk_id),
                    "timestamp": chunk.get("timestamp", datetime.now().isoformat())
                })
                self.chunk_count += 1
            
            if ids:
                # Batch add with chunking to avoid timeouts
                batch_size = 100
                for i in range(0, len(ids), batch_size):
                    batch_end = min(i + batch_size, len(ids))
                    try:
                        self.collection.add(
                            ids=ids[i:batch_end],
                            embeddings=embeddings[i:batch_end],
                            documents=documents[i:batch_end],
                            metadatas=metadatas[i:batch_end]
                        )
                        logger.debug(f"[VectorStore] Added batch {i//batch_size + 1} ({batch_end - i} embeddings)")
                        time.sleep(0.05)  # Small delay between batches
                    except Exception as e:
                        logger.error(f"[VectorStore] Failed to add batch: {e}")
                        raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search vector store with fallback and retry logic"""
        with self.lock:
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.cpu().numpy().astype(np.float32)
            else:
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            # Normalize embedding
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    results = self.collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=min(top_k, self.chunk_count if self.chunk_count > 0 else 1)
                    )
                    
                    output = []
                    if results["ids"] and len(results["ids"]) > 0:
                        for i, (doc_id, distance, metadata, document) in enumerate(
                            zip(results["ids"][0], 
                                results["distances"][0], 
                                results["metadatas"][0], 
                                results["documents"][0])
                        ):
                            output.append({
                                "score": float(1 - distance),
                                "metadata": {
                                    **metadata,
                                    "text": document
                                }
                            })
                    
                    return output
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"[VectorStore] Query failed on attempt {attempt + 1}, retrying: {e}")
                        time.sleep(1)
                    else:
                        logger.error(f"[VectorStore] Query failed after {max_retries} attempts: {e}")
                        raise
    
    def persist_to_disk(self) -> None:
        """Persist data (automatic with Chroma)"""
        with self.lock:
            try:
                if self.chroma_api_impl == "embedded":
                    logger.info(f"[VectorStore] Data will be persisted by embedded Chroma to {self.embeddings_dir}")
                else:
                    logger.info(f"[VectorStore] Data is persisted by Chroma server")
            except Exception as e:
                logger.error(f"[VectorStore] Persist check failed: {e}")
    
    def _load_from_disk(self) -> None:
        try:
            count = self.collection.count()
            self.chunk_count = count
            if count > 0:
                logger.info(f"[VectorStore] Loaded {count} chunks from vector store")
        except Exception as e:
            logger.warning(f"[VectorStore] Could not load from vector store: {e}")
            self.chunk_count = 0
    
    def health_check(self) -> bool:
        """Check if vector store is accessible"""
        try:
            if self.chroma_api_impl == "http":
                # For HTTP client, test heartbeat
                self.client.heartbeat()
                return True
            else:
                # For embedded, check if client exists and collection is accessible
                if self.client and self.collection:
                    self.collection.count()
                    return True
        except Exception as e:
            logger.warning(f"[VectorStore] Health check failed: {e}")
            return False
        return False
    
    def reconnect(self) -> None:
        """Attempt to reconnect to vector store"""
        with self.lock:
            try:
                logger.info("[VectorStore] Attempting to reconnect to vector store...")
                self._initialize_chroma_client()
                logger.info("[VectorStore] Reconnected successfully")
            except Exception as e:
                logger.error(f"[VectorStore] Reconnection failed: {e}")
                raise
    
    def get_stats(self) -> Dict:
        with self.lock:
            stats = {
                "total_chunks": self.chunk_count,
                "device": self.device,
                "embedding_dim": self.embedding_dim,
                "chroma_api_impl": self.chroma_api_impl,
                "healthy": self.health_check()
            }
            
            if self.chroma_api_impl == "http":
                stats["server"] = f"{os.getenv('CHROMA_SERVER_HOST', 'localhost')}:{os.getenv('CHROMA_SERVER_PORT', '8000')}"
            
            return stats
    
    def search_with_cache_check(self, query_embedding: np.ndarray, top_k: int = 5, cache_similarity_threshold: float = 0.85) -> Dict:
        """Search with cache metadata (for compatibility)"""
        with self.lock:
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.cpu().numpy().astype(np.float32)
            else:
                query_embedding = np.array(query_embedding, dtype=np.float32)
            
            query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    results = self.collection.query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=min(top_k, self.chunk_count if self.chunk_count > 0 else 1)
                    )
                    
                    output = []
                    similarities = []
                    
                    if results["ids"] and len(results["ids"]) > 0:
                        for i, (doc_id, distance, metadata, document) in enumerate(
                            zip(results["ids"][0], 
                                results["distances"][0], 
                                results["metadatas"][0], 
                                results["documents"][0])
                        ):
                            similarity = float(1 - distance)
                            similarities.append(similarity)
                            output.append({
                                "score": similarity,
                                "metadata": {
                                    **metadata,
                                    "text": document
                                }
                            })
                    
                    best_match_similarity = similarities[0] if similarities else 0.0
                    avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
                    cache_hit = best_match_similarity >= cache_similarity_threshold
                    
                    return {
                        'results': output,
                        'cache_hit': cache_hit,
                        'avg_similarity': avg_similarity,
                        'best_match_similarity': best_match_similarity
                    }
                
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"[VectorStore] Query failed on attempt {attempt + 1}, retrying: {e}")
                        time.sleep(1)
                    else:
                        logger.error(f"[VectorStore] Query failed after {max_retries} attempts: {e}")
                        raise
