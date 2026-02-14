from typing import Dict, List, Optional
import threading
from loguru import logger
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime

from embedding_service import EmbeddingService, VectorStore
from semantic_cache import SemanticCache
from session_manager import SessionData
from utility import chunk_text, clean_text
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    EMBEDDINGS_DIR,
    SEMANTIC_CACHE_TTL_SECONDS,
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
    SESSION_SUMMARY_THRESHOLD,
)

class RAGEngine:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        semantic_cache: SemanticCache,
        session_data: SessionData
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.semantic_cache = semantic_cache
        self.session_data = session_data
        self.retrieval_pipeline = RetrievalPipeline(
            embedding_service,
            vector_store
        )
        logger.info("[RAG] Optimized RAG engine initialized")
    
    def retrieve_context(
        self,
        query: str,
        url: Optional[str] = None,
        top_k: int = 5
    ) -> Dict:
        try:
            query_embedding = self.embedding_service.embed_single(query)
            
            if url:
                cached_response = self.semantic_cache.get(url, query_embedding)
                if cached_response:
                    logger.info(f"[RAG] Semantic cache HIT for {url}")
                    return {
                        "source": "semantic_cache",
                        "url": url,
                        "response": cached_response,
                        "latency_ms": 1.0
                    }
            
            # CRITICAL FIX #8: Try session data first, then global vector store
            session_content = self._get_session_content_context(query_embedding, top_k)
            
            # Get global vector store results
            results = self.vector_store.search(query_embedding, top_k=top_k)
            context_texts = [r["metadata"]["text"] for r in results]
            sources = list(set([r["metadata"]["url"] for r in results]))
            
            context = "\n\n".join(context_texts)
            
            # Combine session content with global results
            if session_content["texts"]:
                logger.info(f"[RAG] Including {len(session_content['texts'])} session-specific content chunks")
                context = session_content["combined"] + "\n\n" + context if context else session_content["combined"]
                sources.extend(session_content["sources"])
                sources = list(set(sources))
            
            session_context = self._get_session_context()
            
            full_context = ""
            if session_context:
                full_context += f"Context from previous exchanges:\n{session_context}\n\n"
            
            full_context += f"Retrieved Information:\n{context}"
            
            retrieval_result = {
                "source": "vector_store",
                "query": query,
                "context": full_context,
                "sources": sources,
                "chunk_count": len(results),
                "scores": [r["score"] for r in results],
                "query_embedding": query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            }
            
            if url:
                self.semantic_cache.set(url, query_embedding, retrieval_result)
            
            return retrieval_result
        
        except Exception as e:
            logger.error(f"[RAG] Retrieval failed: {e}")
            return {
                "source": "error",
                "error": str(e),
                "context": "",
                "sources": []
            }
    
    def ingest_and_cache(self, url: str) -> Dict:
        try:
            chunk_count = self.retrieval_pipeline.ingest_url(url, max_words=3000)
            logger.info(f"[RAG] Ingested {chunk_count} chunks from {url}")
            return {
                "success": True,
                "url": url,
                "chunks": chunk_count
            }
        except Exception as e:
            logger.error(f"[RAG] Ingestion failed for {url}: {e}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }
    
    
    def get_full_context(self, query: str, top_k: int = 5) -> Dict:
        retrieval_result = self.retrieve_context(query, top_k=top_k)
        
        return {
            "query": query,
            "context": retrieval_result.get("context", ""),
            "sources": retrieval_result.get("sources", []),
            "cache_hit": retrieval_result.get("source") == "semantic_cache",
            "scores": retrieval_result.get("scores", [])
        }
    
    def get_stats(self) -> Dict:
        return {
            "vector_store": self.vector_store.get_stats(),
            "semantic_cache": self.semantic_cache.get_stats(),
            "session_memory": self.session_data.to_dict()
        }
    
    def build_rag_prompt_enhancement(self, session_id: str, top_k: int = 5) -> str:
        try:
            # Get session memory context
            context_parts = []
            
            if self.session_data:
                session_context = self._get_session_context()
                if session_context:
                    context_parts.append("=== Previous Context ===")
                    context_parts.append(session_context)
                    context_parts.append("")
            
            # Return formatted context for system prompt
            rag_prompt = "\n".join(context_parts) if context_parts else ""
            logger.info(f"[RAG] Built prompt enhancement: {len(rag_prompt)} chars")
            return rag_prompt
        
        except Exception as e:
            logger.error(f"[RAG] Failed to build prompt enhancement: {e}")
            return ""
    
    def _get_session_context(self) -> str:
        """Extract context from SessionData's conversation history."""
        if not self.session_data:
            return ""
        
        try:
            history = self.session_data.get_conversation_history()
            if not history:
                return ""
            
            # Build context from recent conversation turns
            context_parts = []
            # Keep last 2-3 turns for context
            for msg in history[-3:]:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if content:
                    context_parts.append(f"{role.capitalize()}: {content[:200]}")
            
            return "\n".join(context_parts) if context_parts else ""
        except Exception as e:
            logger.warning(f"[RAG] Failed to extract session context: {e}")
            return ""
    
    def _get_session_content_context(self, query_embedding: np.ndarray, top_k: int = 5) -> Dict:
        """CRITICAL FIX #8: Get relevant content from session's fetched URLs using ChromaDB."""
        if not self.session_data or not hasattr(self.session_data, 'processed_content'):
            return {"texts": [], "sources": [], "combined": ""}
        
        try:
            # If session has its own ChromaDB collection, use it for retrieval
            if hasattr(self.session_data, 'chroma_collection') and self.session_data.chroma_collection:
                results = self.session_data.chroma_collection.query(
                    query_embeddings=[query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding],
                    n_results=min(top_k, self.session_data.chroma_collection.count())
                )
                
                content_texts = []
                sources = []
                
                if results["documents"] and len(results["documents"]) > 0:
                    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
                        content_texts.append(document[:500])
                        if "url" in metadata:
                            sources.append(metadata["url"])
                
                combined = "\n\n[Session Content]\n".join(content_texts) if content_texts else ""
                logger.info(f"[RAG] Retrieved {len(content_texts)} session content chunks from ChromaDB")
                
                return {
                    "texts": content_texts,
                    "sources": sources,
                    "combined": combined
                }
            else:
                # Fallback: use simple similarity matching without ChromaDB
                content_texts = []
                sources = []
                
                if hasattr(self.session_data, 'content_order'):
                    for url in self.session_data.content_order[:top_k]:
                        content = self.session_data.processed_content.get(url, "")
                        if content:
                            content_texts.append(content[:500])
                            sources.append(url)
                
                combined = "\n\n[Session Content]\n".join(content_texts) if content_texts else ""
                logger.info(f"[RAG] Retrieved {len(content_texts)} session content chunks (fallback mode)")
                
                return {
                    "texts": content_texts,
                    "sources": sources,
                    "combined": combined
                }
        except Exception as e:
            logger.warning(f"[RAG] Failed to get session content context: {e}")
            return {"texts": [], "sources": [], "combined": ""}

class RetrievalPipeline:
    def __init__(self, embedding_service: EmbeddingService, vector_store: VectorStore):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
    
    def ingest_url(self, url: str, max_words: int = 3000) -> int:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, timeout=20, headers=headers)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                logger.warning(f"[Retrieval] Skipping non-HTML: {url}")
                return 0
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.extract()
            
            text = soup.get_text()
            text = clean_text(text)
            
            words = text.split()
            if len(words) > max_words:
                text = " ".join(words[:max_words])
            
            chunks = chunk_text(text, chunk_size=600, overlap=60)
            
            embeddings = self.embedding_service.embed(chunks, batch_size=32)
            
            chunk_dicts = [
                {
                    "url": url,
                    "chunk_id": i,
                    "text": chunk,
                    "embedding": embeddings[i],
                    "timestamp": datetime.now().isoformat()
                }
                for i, chunk in enumerate(chunks)
            ]
            
            self.vector_store.add_chunks(chunk_dicts)
            logger.info(f"[Retrieval] Ingested {len(chunks)} chunks from {url}")
            
            return len(chunks)
        
        except Exception as e:
            logger.error(f"[Retrieval] Failed to ingest {url}: {e}")
            return 0
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        try:
            query_embedding = self.embedding_service.embed_single(query)
            
            results = self.vector_store.search(query_embedding, top_k=top_k)
            
            return results
        
        except Exception as e:
            logger.error(f"[Retrieval] Retrieval failed: {e}")
            return []
    
    def build_context(self, query: str, top_k: int = 5, session_memory: str = "") -> Dict:
        results = self.retrieve(query, top_k=top_k)
        
        context_texts = [r["metadata"]["text"] for r in results]
        context = "\n\n".join(context_texts)
        
        sources = list(set([r["metadata"]["url"] for r in results]))
        
        prompt_context = ""
        if session_memory:
            prompt_context += f"Previous Context:\n{session_memory}\n\n"
        
        prompt_context += f"Retrieved Information:\n{context}"
        
        return {
            "context": prompt_context,
            "sources": sources,
            "chunk_count": len(results),
            "scores": [r["score"] for r in results]
        }


class RetrievalSystem:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = RetrievalSystem()
        return cls._instance
    
    def __init__(self):
        logger.info("[RetrievalSystem] Initializing...")
        self.embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL)
        logger.info(f"[RetrievalSystem] Embedding service device: {self.embedding_service.device}")
        
        # CRITICAL FIX #10: Use embedding dimension from config instead of hard-coded value
        self.vector_store = VectorStore(embedding_dim=EMBEDDING_DIMENSION, embeddings_dir=EMBEDDINGS_DIR)
        logger.info(f"[RetrievalSystem] Vector store device: {self.vector_store.device}")
        
        self.semantic_cache = SemanticCache(
            ttl_seconds=SEMANTIC_CACHE_TTL_SECONDS,
            similarity_threshold=SEMANTIC_CACHE_SIMILARITY_THRESHOLD
        )
        logger.info(f"[RetrievalSystem] Semantic cache: TTL={SEMANTIC_CACHE_TTL_SECONDS}s, threshold={SEMANTIC_CACHE_SIMILARITY_THRESHOLD}")
        
        # NOTE: SessionMemory removed in CRITICAL FIX #3 - using SessionData from SessionManager instead
        self.sessions_lock = threading.RLock()
        
        logger.info("[RetrievalSystem] ✅ Fully initialized with GPU acceleration")
    
    # CRITICAL FIX #3: Session management moved to SessionManager
    # These methods are deprecated and kept for backward compatibility only
    def create_session(self, session_id: str):
        """Deprecated: Use SessionManager.create_session() instead."""
        logger.warning(f"[RetrievalSystem] Deprecated create_session() called for {session_id}. Use SessionManager instead.")
        return None
    
    def get_session(self, session_id: str):
        """Deprecated: Use SessionManager.get_session() instead."""
        return None
    
    def get_rag_engine(self, session_id: str) -> RAGEngine:
        # CRITICAL FIX #3: Get existing SessionData from SessionManager instead of creating new SessionMemory
        from session_manager import get_session_manager
        session_manager = get_session_manager()
        session_data = session_manager.get_session(session_id)
        
        if not session_data:
            logger.warning(f"[RetrievalSystem] Session {session_id} not found in SessionManager")
            # Create a temporary session data if not found (edge case)
            session_data = session_manager.get_session(session_id) or type('SessionData', (), {'get_conversation_history': lambda: [], 'to_dict': lambda: {}})()
        
        return RAGEngine(
            self.embedding_service,
            self.vector_store,
            self.semantic_cache,
            session_data
        )
    
    def add_conversation_turn(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str,
        entities: List[str] = None
    ) -> None:
        # CRITICAL FIX #3: Use SessionManager to add conversation turns
        from session_manager import get_session_manager
        session_manager = get_session_manager()
        session_manager.add_message_to_history(
            session_id,
            "user",
            user_query
        )
        session_manager.add_message_to_history(
            session_id,
            "assistant",
            assistant_response,
            metadata={"entities": entities} if entities else None
        )
    
    def delete_session(self, session_id: str) -> None:
        # CRITICAL FIX #3: Use SessionManager to delete session
        from session_manager import get_session_manager
        session_manager = get_session_manager()
        session_manager.cleanup_session(session_id)
        logger.info(f"[RetrievalSystem] Deleted session {session_id}")
    
    def get_stats(self) -> Dict:
        # CRITICAL FIX #3: Get stats from SessionManager instead of internal sessions dict
        from session_manager import get_session_manager
        session_manager = get_session_manager()
        sessions_stats = session_manager.get_stats()
        
        return {
            "vector_store": self.vector_store.get_stats(),
            "semantic_cache": self.semantic_cache.get_stats(),
            "active_sessions": sessions_stats.get("total_sessions", 0)
        }
    
    def persist_vector_store(self) -> None:
        self.vector_store.persist_to_disk()


_retrieval_system = None


def initialize_retrieval_system() -> RetrievalSystem:
    global _retrieval_system
    _retrieval_system = RetrievalSystem.get_instance()
    return _retrieval_system


def get_retrieval_system() -> RetrievalSystem:
    if _retrieval_system is None:
        return initialize_retrieval_system()
    return _retrieval_system
