# 🚨 CRITICAL: RAG Pipeline Fix Required

## Problem Summary

The searchPipeline.py is calling a non-existent method on RAGEngine, which breaks the entire RAG context building.

### Current Broken Code
**File:** `api/searchPipeline.py:200`

```python
# BROKEN INSTANTIATION
rag_engine = RAGEngine(session_manager, top_k_entities=15)

# BROKEN METHOD CALL
rag_context = rag_engine.build_rag_prompt_enhancement(session_id)
```

### Why It's Broken

1. **Wrong Constructor:** RAGEngine expects:
   ```python
   RAGEngine(
       embedding_service: EmbeddingService,
       vector_store: VectorStore,
       semantic_cache: SemanticCache,
       session_memory: SessionMemory
   )
   ```
   But searchPipeline is calling it with:
   ```python
   RAGEngine(session_manager, top_k_entities=15)  # ❌ Wrong!
   ```

2. **Missing Method:** The method `build_rag_prompt_enhancement()` doesn't exist in RAGEngine class

---

## Solution: Use RetrievalSystem Singleton

Replace the broken RAG initialization with the proper RetrievalSystem pattern:

```python
from rag_engine import get_retrieval_system

# At request time:
retrieval_system = get_retrieval_system()
session_id = retrieval_system.get_instance().create_session(user_query)
rag_engine = retrieval_system.get_rag_engine(session_id)

# Build context properly:
rag_context = session_manager.get_rag_context(
    session_id, 
    refresh=False,
    query_embedding=None
)
```

---

## Implementation Steps

### Step 1: Fix searchPipeline.py initialization (line 199-219)

**Current (BROKEN):**
```python
session_manager = SessionManager(max_sessions=100, ttl_minutes=30)
rag_engine = RAGEngine(session_manager, top_k_entities=15)
session_id = session_manager.create_session(user_query)

rag_context = rag_engine.build_rag_prompt_enhancement(session_id)
```

**Replace with (FIXED):**
```python
from rag_engine import get_retrieval_system

# Initialize properly via RetrievalSystem
retrieval_system = get_retrieval_system()
session_manager = SessionManager(max_sessions=100, ttl_minutes=30)
session_id = session_manager.create_session(user_query)

# Get properly configured RAG engine for this session
rag_engine = retrieval_system.get_rag_engine(session_id)

# Build context from session data
rag_context = session_manager.get_rag_context(
    session_id,
    refresh=RAG_CONTEXT_REFRESH,
    query_embedding=None  # Will be set when query is received
)
```

---

## Vector Storage Locations (Per Session)

### 1. Global Vector Store
- **Location:** `./embeddings/faiss_index.bin`
- **Purpose:** Shared across all requests
- **GPU:** ✅ Yes (FAISS IndexFlatIP on GPU 0)
- **Persistence:** Every 300 seconds

### 2. Session-Local FAISS Index
- **Location:** `SessionData.faiss_index` (in-memory)
- **Purpose:** Per-request semantic search
- **GPU:** ❌ CPU only (needs GPU migration)
- **Lifetime:** Request duration only
- **Data:** Content from fetched URLs in this session

### 3. Semantic Cache
- **Location:** `SemanticCache.cache` (in-memory)
- **Structure:** `{url → {query_hash → {embedding, response, timestamp}}}`
- **TTL:** 3600 seconds (1 hour)
- **Similarity Threshold:** 0.90
- **Purpose:** Fast re-retrieval for similar queries on same URL

### 4. Conversation History
- **Location:** `SessionMemory.conversation_history` (in-memory)
- **Structure:** List of turn dicts with user/assistant messages
- **Compression:** Every 6 turns, keeps last 2 + summary
- **Lifetime:** Request duration

---

## GPU Configuration Status

### ✅ GPU ENABLED
- Embedding Service (SentenceTransformers)
- Global FAISS Index (IndexFlatIP on GPU 0)
- Whisper Audio Model

### ❌ GPU DISABLED (CPU ONLY)
- Session FAISS Index (IndexFlatL2 on CPU)
- Semantic Cache operations (NumPy on CPU)
- HTML parsing (BeautifulSoup CPU-only)

### Recommendation
Migrate session FAISS indices to GPU:
```python
import faiss

if torch.cuda.is_available():
    gpu_index = faiss.index_cpu_to_gpu(
        faiss.StandardGpuResources(), 0,
        faiss.IndexFlatIP(embedding_dim)
    )
    self.faiss_index = gpu_index
```

---

## TTL Configuration

| Setting | Value | Location |
|---------|-------|----------|
| Session TTL | 30 minutes | config.py:SESSION_TTL_MINUTES |
| Semantic Cache TTL | 3600 seconds (1h) | config.py:SEMANTIC_CACHE_TTL_SECONDS |
| Vector Persist | 300 seconds (5m) | config.py:PERSIST_VECTOR_STORE_INTERVAL |
| Request Timeout | 300 seconds (5m) | config.py:REQUEST_TIMEOUT |
| URL Fetch Timeout | 30 seconds | config.py:FETCH_TIMEOUT |

---

## Context Length Configuration

| Setting | Value | Purpose |
|---------|-------|---------|
| Words per URL | 3000 | MAX_TOTAL_SCRAPE_WORD_COUNT |
| YouTube transcript | 3000 | MAX_TRANSCRIPT_WORD_COUNT |
| Vector retrieval | 5 chunks | RETRIEVAL_TOP_K |
| LLM output | 3000 tokens | LLM_MAX_TOKENS |
| Embedding batch | 32 | EMBEDDING_BATCH_SIZE |

---

## Testing Checklist

- [ ] RAGEngine initialization with proper parameters
- [ ] RAG context generation from session data
- [ ] Semantic cache hit/miss working correctly
- [ ] Session FAISS index growing with content
- [ ] GPU memory usage within bounds
- [ ] Session cleanup after 30 minutes inactivity
- [ ] Conversation history compression after 6 turns
- [ ] Vector persistence every 5 minutes

---

## Files to Modify

1. **api/searchPipeline.py** - Fix RAGEngine initialization and RAG context building
2. **api/session_manager.py** - (Optional) Enable GPU FAISS indices
3. **api/config.py** - (If needed) Adjust context lengths based on testing

---

**Priority:** 🔴 CRITICAL - RAG context is completely broken and must be fixed before deployment.
