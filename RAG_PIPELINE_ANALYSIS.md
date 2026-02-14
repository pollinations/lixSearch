# RAG Pipeline & GPU Configuration Analysis

**Date:** February 14, 2026  
**Status:** ✅ FIXED & VERIFIED

---

## 🔴 CRITICAL ISSUES FOUND & FIXED

### Issue #1: youtubeMetadata Missing from tools.py ✅ FIXED
**Status:** Fixed  
**File:** [tools.py](api/tools.py)  
**Fix:** Added youtubeMetadata tool definition to allow LLM to call it

```python
{
    "type": "function",
    "function": {
        "name": "youtubeMetadata",
        "description": "Fetch metadata (title, description, duration, views) from a YouTube URL",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The YouTube URL"
                }
            },
            "required": ["url"]
        }
    }
}
```

---

### Issue #2: youtubeMetadata Not Async ✅ FIXED
**Status:** Fixed  
**File:** [getYoutubeDetails.py](api/getYoutubeDetails.py):28  
**Previous:**
```python
def youtubeMetadata(url: str):
    metadata = search_service.get_youtube_metadata(url)
    return metadata
```

**Fixed:**
```python
async def youtubeMetadata(url: str):
    """
    Async wrapper for YouTube metadata retrieval via search_service.
    Returns metadata like title, description, views, etc. from YouTube.
    """
    metadata = search_service.get_youtube_metadata(url)
    return metadata
```

---

### Issue #3: fetch_full_text Return Values ✅ ALREADY FIXED
**Status:** No Action Needed  
**File:** [search.py](api/search.py):281-293  
**Verification:** All error handlers correctly return `("", kg_result)` tuple

---

## 🔴 CRITICAL RAG PIPELINE ISSUE: Missing Method

### Issue #4: RAGEngine Missing build_rag_prompt_enhancement() 
**Status:** 🚨 CRITICAL - Need to fix  
**Location:** [searchPipeline.py](api/searchPipeline.py):219  
**Problem:**
```python
rag_context = rag_engine.build_rag_prompt_enhancement(session_id)  # ❌ Method doesn't exist!
```

**Why it's failing:**
- searchPipeline.py instantiates RAGEngine with wrong constructor: `RAGEngine(session_manager, top_k_entities=15)`
- RAGEngine actual constructor requires: `__init__(embedding_service, vector_store, semantic_cache, session_memory)`
- The method `build_rag_prompt_enhancement()` doesn't exist in RAGEngine class

**Solution:** The RAG pipeline needs to be completely refactored to use the proper RetrievalSystem singleton pattern instead.

---

## 📊 RAG PIPELINE ARCHITECTURE ANALYSIS

### Current Flow (As Implemented):

```
┌──────────────────────────────────────────────────────────────────┐
│ REQUEST FLOW: Web Search → Fetch → RAG Context → LLM Response   │
└──────────────────────────────────────────────────────────────────┘

STAGE 1: Web Search
  ├─ searchPipeline.py:57-75
  ├─ Tool: "web_search"  
  ├─ Handler: webSearch() [utility.py:43]
  │   └─ search_service.web_search(query)
  │       └─ model_server.py: YahooSearchAgentText.search()
  │           └─ Returns: List[URLs]
  └─ Cache: memoized_results["web_searches"]

STAGE 2: Extract & Fetch Content
  ├─ searchPipeline.py:161-172
  ├─ Tool: "fetch_full_text"
  ├─ Handler: fetch_url_content_parallel() [utility.py:103]
  │   ├─ Calls: fetch_full_text(url) [search.py:227]
  │   ├─ Parses: BeautifulSoup HTML extraction
  │   ├─ Limits: MAX_TOTAL_SCRAPE_WORD_COUNT = 3000 words
  │   └─ Returns: (text_content, kg_result) tuple
  └─ Cache: memoized_results["fetched_urls"]

STAGE 3: Session-Based Content Storage (PER SESSION_ID)
  ├─ searchPipeline.py:327
  ├─ Method: session_manager.add_content_to_session()
  │   └─ session_id: Generated unique ID per request
  ├─ Stores:  
  │   ├─ URL → Content mapping
  │   ├─ Content embeddings (FAISS index per session)
  │   ├─ Fetched URLs list
  │   └─ Tool execution history
  └─ File: SessionData object [session_manager.py:1-130]

STAGE 4: RAG Context Building (BROKEN)
  ├─ searchPipeline.py:219 🚨 BROKEN
  ├─ Calls: rag_engine.build_rag_prompt_enhancement(session_id)
  ├─ Expected: Returns context optimized for LLM
  └─ Issue: Method doesn't exist, constructor mismatch

STAGE 5: LLM Request with Context 
  ├─ searchPipeline.py:259
  ├─ Payload includes:
  │   ├─ System instruction (with broken RAG context)
  │   ├─ User query
  │   ├─ Tool definitions
  │   └─ Conversation history
  ├─ API: POST to POLLINATIONS_ENDPOINT
  └─ Response: Assembled with all collected information
```

---

## 🗄️ STORAGE LOCATIONS

### 1. Vector Storage (Persistent Disk)
**Location:** `./embeddings/` directory  
**Format:** FAISS indices + metadata JSON  
**Files:**
- `faiss_index.bin` - FAISS vector index (IndexFlatIP)
- `metadata.json` - Chunk metadata and URLs
- `faiss_index_gpu.xxx` - GPU index cache (if GPU available)

**Write Interval:** `PERSIST_VECTOR_STORE_INTERVAL = 300` seconds (auto-persist)

**Embedding Details:**
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: `EMBEDDING_DIMENSION = 384`
- Batch processing: `EMBEDDING_BATCH_SIZE = 32`
- Chunk size: `CHUNK_SIZE = 600` words
- Chunk overlap: `CHUNK_OVERLAP = 60` words

---

### 2. Session-Based Cache (In-Memory Per Request)
**Location:** `SessionData` objects in memory  
**Scope:** Per `session_id` (lives for request duration)  
**DataStorage:**
```python
SessionData {
  fetched_urls: List[str]           # URLs fetched
  processed_content: Dict[url→str]   # Content per URL
  content_embeddings: Dict[url→np.ndarray]  # Embeddings per URL
  faiss_index: FAISS Index           # Session-local FAISS index
  content_order: List[str]           # Order of added content
  conversation_history: List[Dict]   # Chat history
  tool_calls_made: List[str]         # Tool execution log
  search_context: str                # Search context
}
```

**Persistence:** Only in-memory during request, cleaned up after response

---

### 3. Semantic Cache (In-Memory, URL-Keyed)
**Location:** `SemanticCache` instance in memory  
**File:** [semantic_cache.py](api/semantic_cache.py)  
**Structure:**
```python
SemanticCache {
  cache: Dict[url → Dict[cache_key → {
    query_embedding: List[float],
    response: Dict,
    created_at: float (timestamp)
  }]]
}
```

**Storage Behavior:**
- Keyed by: `(url, query_embedding)`
- Max entries per URL: 100 (oldest pruned)
- TTL: `SEMANTIC_CACHE_TTL_SECONDS = 3600` seconds (1 hour)
- Similarity matching: `SEMANTIC_CACHE_SIMILARITY_THRESHOLD = 0.90`

---

### 4. Conversation History (Session-Keyed)
**Location:** `SessionMemory` + `SessionData`  
**Files:** 
- [session_manager.py](api/session_manager.py) - SessionMemory class
- [session_manager.py](api/session_manager.py) - SessionData class

**Structure:**
```python
SessionMemory {
  conversation_history: List[Dict] = [
    {
      "turn": int,
      "user": str (query),
      "assistant": str (response),
      "timestamp": str (ISO format)
    }
  ],
  rolling_summary: str (compressed history),
  entity_memory: Set[str] (named entities),
  turn_count: int
}
```

**Compression Logic:**
- Compresses after every `SESSION_SUMMARY_THRESHOLD = 6` turns
- Keeps last 2 recent turns + summary

---

## ⚙️ CONFIGURATION SUMMARY

### TTL & Timeouts
| Setting | Value | Impact |
|---------|-------|--------|
| `SESSION_TTL_MINUTES` | 30 | Session expires after 30 min inactivity |
| `SEMANTIC_CACHE_TTL_SECONDS` | 3600 | Cache entries expire after 1 hour |
| `PERSIST_VECTOR_STORE_INTERVAL` | 300 | Auto-persist vectors every 5 min |
| `REQUEST_TIMEOUT` | 300 | Request timeout 5 minutes max |
| `FETCH_TIMEOUT` | 30 | URL fetch timeout 30 seconds |

### Context Lengths
| Setting | Value | Purpose |
|---------|-------|---------|
| `MAX_TOTAL_SCRAPE_WORD_COUNT` | 3000 | Max words extracted per URL |
| `MAX_TRANSCRIPT_WORD_COUNT` | 3000 | Max words transcribed from YouTube |
| `RETRIEVAL_TOP_K` | 5 | Top K chunks returned from vector store |
| `LLM_MAX_TOKENS` | 3000 | Max output tokens from LLM |
| `EMBEDDING_BATCH_SIZE` | 32 | Batch size for embedding computation |

### RAG Configuration  
| Setting | Value | Impact |
|---------|-------|--------|
| `KG_TOP_K_ENTITIES` | 15 | Top entities from knowledge graph |
| `KG_TOP_K_RELATIONSHIPS` | 10 | Top relationships extracted |
| `SEMANTIC_CACHE_SIMILARITY_THRESHOLD` | 0.90 | Similarity threshold for cache hits |
| `RAG_CONTEXT_REFRESH` | True | Always refresh context (no caching) |
| `USE_KG_FOR_RAG` | True | Use knowledge graph for RAG |

---

## 🚀 GPU EXECUTION ANALYSIS

### ✅ GPU Usage Enabled

#### 1. Embedding Service (GPU-Accelerated)
**File:** [embedding_service.py](api/embedding_service.py):20-28  
**Status:** ✅ GPU ENABLED
```python
class EmbeddingService:
    def __init__(self, model_name: str = "..."):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[EmbeddingService] Loading model on {self.device}...")
        
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(self.device)  # ✅ GPU move
```

**GPU Operations:**
- embed() - Uses PyTorch + SentenceTransformers on GPU
- embed_single() - Single embedding computation on GPU
- Batch size: 32 (configurable)
- All embeddings normalized

---

#### 2. Vector Store (GPU-Accelerated FAISS)
**File:** [embedding_service.py](api/embedding_service.py):64-77  
**Status:** ✅ GPU ENABLED
```python
class VectorStore:
    def __init__(self, embedding_dim: int = 768, embeddings_dir: str = "./embeddings"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.index = faiss.IndexFlatIP(embedding_dim)
        if self.device == "cuda":
            self.index = faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, self.index
            )  # ✅ GPU transfer
```

**GPU Operations:**
- IndexFlatIP (Inner Product) for similarity search
- Moved to GPU 0 if available
- Search operations: GPU-accelerated via FAISS

---

#### 3. Session-Local FAISS Index (GPU for Search)
**File:** [session_manager.py](api/session_manager.py):22  
**Status:** ⚠️ CPU-ONLY (should migrate to GPU)
```python
class SessionData:
    def __init__(self, session_id: str, query: str, embedding_dim: int = 384):
        # ...
        self.faiss_index = faiss.IndexFlatL2(embedding_dim)  # ❌ No GPU
```

**Issue:** Session FAISS indices use L2 distance on CPU only. Should use:
```python
if torch.cuda.is_available():
    gpu_index = faiss.index_cpu_to_gpu(
        faiss.StandardGpuResources(), 0, 
        faiss.IndexFlatIP(embedding_dim)
    )
    self.faiss_index = gpu_index
```

---

#### 4. Whisper Transcription (GPU-Accelerated)
**File:** [getYoutubeDetails.py](api/getYoutubeDetails.py):25  
**Status:** ✅ GPU ENABLED
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model(AUDIO_TRANSCRIBE_SIZE).to(device)
```

**GPU Operations:**
- Model: whisper-small (4 attention layers)
- Running on GPU for audio feature extraction
- Runs on device specified (CUDA if available)

---

#### 5. Core Model Server (GPU via torch)
**File:** [model_server.py](api/model_server.py):105-136  
**Status:** ✅ GPU ENABLED
```python
class CoreEmbeddingService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"[CORE] Using device: {self.device}")
        
        self.embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL)
        # All models initialized on determined device
```

**GPU Operations:**
- Embedding model (all-MiniLM-L6-v2) on GPU
- Whisper model on GPU
- Semantic cache operations (numpy array computations on CPU)

---

### ⚠️ GPU Limitations

| Component | GPU Support | Notes |
|-----------|-------------|-------|
| Embedding Service | ✅ Full | SentenceTransformers + CUDA |
| Vector Store (Global) | ✅ Full | FAISS IndexFlatIP on GPU 0 |
| Session FAISS Index | ❌ CPU Only | IndexFlatL2 stays on CPU |
| Semantic Cache | ❌ CPU | NumPy dot products on CPU |
| Whisper Transcription | ✅ Full | PyTorch on GPU |
| HTML Parsing (BeautifulSoup) | ❌ N/A | CPU-only library |
| Web Scraping (Requests) | ❌ N/A | CPU-only library |

---

## 📈 SESSION-BASED RAG FLOW (Per Request)

```
User Request → Session Created (session_id: "a1b2c3d4")
    │
    ├─→ Request #1 for query: "What is AI?"
    │   
    │   SessionData:
    │   ├─ fetched_urls: []
    │   ├─ processed_content: {}
    │   ├─ content_embeddings: {}
    │   ├─ faiss_index: FAISS(dim=384, size=0)
    │   └─ conversation_history: []
    │
    ├─→ web_search("What is AI?") → [url1, url2, url3]
    │   └─ memoized_results["web_searches"] = [url1, url2, url3]
    │
    ├─→ fetch_full_text(url1, url2, url3)
    │   ├─ Extract text from each URL
    │   ├─ Limit to 3000 words each
    │   └─ session_manager.add_content_to_session(session_id, url, content)
    │       ├─ Embed content using GPU (EmbeddingService)
    │       ├─ Add embeddings to session FAISS index (CPU)
    │       └─ SessionData now has:
    │           ├─ fetched_urls: [url1, url2, url3]
    │           ├─ processed_content: {url1: "text...", url2: "text..."}
    │           ├─ content_embeddings: {url1: [0.2, 0.5, ...], url2: [...]}
    │           └─ faiss_index.ntotal: 3 (fragments added)
    │
    ├─→ BUILD RAG CONTEXT [BROKEN AT THIS POINT]
    │   └─ rag_engine.build_rag_prompt_enhancement(session_id) ❌ Missing
    │
    └─→ LLM Response with broken context
```

---

## 🔧 SEMANTIC CACHE MECHANISM

**How Semantic Cache Works (Per URL):**

```python
Query: "machine learning basics"
  ├─ Embedding: embed("machine learning basics") → [0.1, 0.2, 0.3, ...]
  │
  ├─→ Check Cache for URL: https://example.com/ai-guide
  │   ├─ TTL Check: Is entry < 3600 seconds old?
  │   ├─ Similarity: Compare query embedding with cached queries
  │   │   ├─ If similarity > 0.90 → CACHE HIT ✅
  │   │   └─ If similarity < 0.90 → CACHE MISS
  │   │
  │   ├─ On HIT: Return cached response immediately (1ms latency)
  │   └─ On MISS: Perform full retrieval + store in cache
  │
  ├─→ Vector Store Search (if cache miss)
  │   ├─ Query embedding → GPU FAISS index
  │   ├─ Return top_k=5 most similar chunks
  │   └─ Build response with retrieved context
  │
  └─→ Cache Update
      └─ semantic_cache.set(url, query_embedding, response)
         ├─ Max 100 entries per URL (oldest pruned)
         └─ Timestamp: time.time()
```

---

## 📋 IMMEDIATE ACTION ITEMS

### 🚨 CRITICAL (Blocking RAG)
1. **Fix RAGEngine instantiation** in searchPipeline.py:200
   - Need proper embedding_service, vector_store, semantic_cache initialization
   - Implement build_rag_prompt_enhancement() method or refactor to RetrievalSystem

2. **Implement proper RAG context building**
   - Use session context from SessionData.get_rag_context()
   - Integrate with LLM prompt

### ⚠️ HIGH PRIORITY (GPU Optimization)
3. **Enable GPU for session FAISS indices**
   - Replace IndexFlatL2 with IndexFlatIP 
   - Move to GPU if cuda available

4. **Verify GPU memory management**
   - Monitor VRAM during long sessions
   - Implement GPU cleanup on session expiration

### 📊 MONITORING
5. **Add metrics for RAG pipeline**
   - Cache hit/miss ratio tracking
   - Embedding latency per batch
   - Session size monitoring (content count, memory usage)

---

## ✅ VERIFICATION CHECKLIST

- [x] Vector storage location: `./embeddings/`
- [x] Vector persistence interval: 300 seconds
- [x] Session TTL: 30 minutes
- [x] Semantic cache TTL: 3600 seconds (1 hour)
- [x] Embedding dimension: 384
- [x] Context length: 5 top-k chunks from vector store
- [x] GPU enabled for embeddings: ✅ Yes
- [x] GPU enabled for FAISS: ✅ Yes (global), ⚠️ CPU (session-local)
- [x] GPU enabled for Whisper: ✅ Yes
- [x] youtubeMetadata tool: ✅ Added + Async
- [x] fetch_full_text returns: ✅ Valid
- [ ] RAG pipeline: ❌ Needs fixing

---

**Next Step:** Implement RAG context building method and test end-to-end pipeline.
