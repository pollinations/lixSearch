# Quick Fix Guide - Integration Issues

## 🔴 CRITICAL FIXES (Must Apply Immediately)

### Fix #1: Undefined `cwd` Variable 
**File**: `api/app.py` Line 84  
**Error**: `NameError: name 'cwd' is not defined`

**Current Code** (Lines 78-89):
```python
logger.info(f"[APP] Starting model server from {model_server_path}...")
model_server_process = subprocess.Popen(
    [sys.executable, model_server_path],
    cwd=cwd,  # ← UNDEFINED!
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)
```

**Fixed Code**:
```python
logger.info(f"[APP] Starting model server from {model_server_path}...")
# Add this line:
cwd = os.path.dirname(os.path.abspath(__file__))

model_server_process = subprocess.Popen(
    [sys.executable, model_server_path],
    cwd=cwd,  # ← Now defined!
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)
```

**Where to Add**: Insert after line 80, before subprocess.Popen call

---

### Fix #2: Missing `get_summary_stats()` Method
**File**: `api/rag_engine.py`  
**Error**: `AttributeError: 'RAGEngine' object has no attribute 'get_summary_stats'`

**Called From**: `api/chat_engine.py` Line 77

**Add This Method to RAGEngine class** (after line 133):
```python
def get_summary_stats(self, session_id: str) -> Dict:
    """Get summary statistics for a session"""
    try:
        stats = self.get_stats()
        return {
            "vector_store": {
                "chunks": stats.get("vector_store", {}).get("chunk_count", 0),
            },
            "semantic_cache": stats.get("semantic_cache", {}),
            "session_memory": {
                "turn_count": self.session_memory.turn_count if self.session_memory else 0
            }
        }
    except Exception as e:
        logger.warning(f"[RAG] Failed to get summary stats: {e}")
        return {
            "vector_store": {"chunks": 0},
            "semantic_cache": {},
            "session_memory": {"turn_count": 0}
        }
```

---

### Fix #3: Session Type Incompatibility
**Files**: `api/rag_engine.py` Lines 287-292, `api/searchPipeline.py` Lines 238-241  
**Problem**: SessionData vs SessionMemory mismatch

**Option 1 (Recommended)**: Create Unified Session Type
```python
# In session_manager.py, add:
class UnifiedSessionData:
    """Unified session handling for both data storage and RAG operations"""
    def __init__(self, session_id: str, query: str):
        self.session_id = session_id
        self.query = query
        # Content storage (from SessionData)
        self.fetched_urls: List[str] = []
        self.processed_content: Dict[str, str] = {}
        # Conversation management (from SessionMemory)
        self.conversation_history: List[Dict] = []
        self.rolling_summary: str = ""
        # ... other fields
```

**Option 2 (Quick Fix)**: Use SessionData in RetrievalSystem
```python
# In rag_engine.py, change get_rag_engine():
def get_rag_engine(self, session_id: str) -> RAGEngine:
    # Create a wrapper that adapts SessionData for RAGEngine needs
    session_memory = self.create_session(session_id)
    # ... continue with existing code
    # But ensure session_memory data syncs back to SessionData
```

See CRITICAL_INTEGRATION_ISSUES.md for detailed options.

---

### Fix #4: Wrong Parameter Type to `initialize_chat_engine()`
**File**: `api/app.py` Line 125  
**Error**: `AttributeError: 'RetrievalSystem' object has no attribute 'get_summary_stats'`

**Current Code** (Lines 118-125):
```python
try:
    start_model_server()
    
    session_manager = get_session_manager()
    retrieval_system = get_retrieval_system()
    initialize_chat_engine(session_manager, retrieval_system)  # ← Wrong type!
```

**Fixed Code**:
```python
try:
    start_model_server()
    
    session_manager = get_session_manager()
    retrieval_system = get_retrieval_system()
    
    # Create a default RAG engine for chat
    default_rag_engine = retrieval_system.get_rag_engine("default")
    initialize_chat_engine(session_manager, default_rag_engine)  # ← Correct type!
```

---

### Fix #5: IPC Service Registration - Remove `ipcService`
**Files**: `api/utility.py` Line 20, `api/getYoutubeDetails.py` Line 19

**In utility.py** (Line 20):
```python
# REMOVE this line:
modelManager.register("ipcService")
```

So utility.py lines 19-20 should be just:
```python
modelManager.register("accessSearchAgents")
# Line 20 deleted
```

**In getYoutubeDetails.py** (Line 19):
```python
# Current lines 18-20:
class modelManager(BaseManager): 
    pass

modelManager.register("accessSearchAgents")
# Don't add ipcService registration
```

**Verify model_server.py** (Line 865-867) registers correctly:
```python
ModelManager.register("CoreEmbeddingService", CoreEmbeddingService)
ModelManager.register("SessionManager", SessionManager)
ModelManager.register("accessSearchAgents", accessSearchAgents)
# ipcService is not registered and shouldn't be used
```

---

### Fix #6: BaseManager Registration Syntax in transcribe.py
**File**: `api/transcribe.py` Line 22  
**Error**: `AttributeError: manager doesn't have 'CoreEmbeddingService'`

**Current Code** (Lines 18-29):
```python
def get_core_service():
    """Connect to model_server's CoreEmbeddingService via IPC."""
    global _core_service
    if _core_service is None:
        try:
            class ModelManager(BaseManager):
                pass
            
            ModelManager.register("CoreEmbeddingService")  # ← Missing callable!
            manager = ModelManager(address=("localhost", 5010), authkey=b"ipcService")
            manager.connect()
            _core_service = manager.CoreEmbeddingService()
```

**Fixed Code**:
```python
def get_core_service():
    """Connect to model_server's CoreEmbeddingService via IPC."""
    global _core_service
    if _core_service is None:
        try:
            class ModelManager(BaseManager):
                pass
            
            # Import the callable from model_server
            from model_server import CoreEmbeddingService as CoreEmbeddingServiceClass
            ModelManager.register("CoreEmbeddingService", callable=CoreEmbeddingServiceClass)
            
            # OR use a string+callable approach:
            # ModelManager.register("CoreEmbeddingService", CoreEmbeddingService)
            
            manager = ModelManager(address=("localhost", 5010), authkey=b"ipcService")
            manager.connect()
            _core_service = manager.CoreEmbeddingService()
```

**Note**: The issue is that the client-side registration needs to match the server-side registration.

---

### Fix #7: Async/Sync Mismatch in YouTube Functions
**File**: `api/getYoutubeDetails.py` Lines 47-68

**Current Code**:
```python
async def youtubeMetadata(url: str):  # ← Async function
    if not _ipc_ready or search_service is None:
        logger.error("[YoutubeDetails] IPC service not available for YouTube metadata")
        return None
    try:
        metadata = search_service.get_youtube_metadata(url)  # ← Blocking sync call!
        return metadata
    except Exception as e:
        logger.error(f"[YoutubeDetails] Error fetching YouTube metadata: {e}")
        return None
```

**Option 1 - Keep Async, Use Executor** (Recommended):
```python
async def youtubeMetadata(url: str):
    if not _ipc_ready or search_service is None:
        logger.error("[YoutubeDetails] IPC service not available for YouTube metadata")
        return None
    try:
        # Run blocking call in thread pool
        loop = asyncio.get_event_loop()
        metadata = await loop.run_in_executor(
            None,
            search_service.get_youtube_metadata,
            url
        )
        return metadata
    except Exception as e:
        logger.error(f"[YoutubeDetails] Error fetching YouTube metadata: {e}")
        return None
```

**Option 2 - Make it Synchronous**:
```python
def youtubeMetadata(url: str):  # ← Remove async
    if not _ipc_ready or search_service is None:
        logger.error("[YoutubeDetails] IPC service not available for YouTube metadata")
        return None
    try:
        metadata = search_service.get_youtube_metadata(url)
        return metadata
    except Exception as e:
        logger.error(f"[YoutubeDetails] Error fetching YouTube metadata: {e}")
        return None

# In searchPipeline.py line 152, change:
# metadata = await youtubeMetadata(url)  ← Remove await
# To:
# metadata = youtubeMetadata(url)  ← No await
```

Also fix `transcribe_audio` function (lines 94-160) similarly if it has the same issue.

---

## ⚠️ HIGH PRIORITY FIXES

### Fix #8: Verify Ingest-and-Cache is Always Called
**File**: `api/searchPipeline.py` Lines 195-206

**Verify This Code Exists**:
```python
# ✅ Should have this ingest call after fetch_full_text:
elif function_name == "fetch_full_text":
    logger.info("Fetching webpage content")
    web_event = emit_event_func("INFO", "<TASK>Reading Webpage</TASK>")
    if web_event:
        yield web_event
    url = function_args.get("url")
    try:
        queries = memoized_results.get("search_query", "")
        if isinstance(queries, str):
            queries = [queries]
        parallel_results = await asyncio.wait_for(
            asyncio.to_thread(fetch_url_content_parallel, queries, [url]),
            timeout=15.0
        )
        
        # CRITICAL: Ingest fetched content into vector store for RAG
        try:
            rag_engine = retrieval_system.get_rag_engine(session_id)
            ingest_result = rag_engine.ingest_and_cache(url)  # ← Must be here!
            logger.info(f"[Pipeline] Ingested {ingest_result.get('chunks', 0)} chunks from {url}")
        except Exception as e:
            logger.warning(f"[Pipeline] Failed to ingest content: {e}")
        
        yield parallel_results if parallel_results else "[No content fetched]"
```

If this code is missing, add it.

---

### Fix #9: Make Embedding Dimensions Configurable
**Files**: `api/config.py`, `api/session_manager.py`, `api/rag_engine.py`

**In config.py**, add:
```python
# Extract embedding dimension from model name
def get_embedding_dim_from_model(model_name: str) -> int:
    """Get embedding dimension from model name"""
    dimension_map = {
        "paraphrase-multilingual-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "all-MiniLM-L6-v2": 384,
        # Add more models as needed
    }
    return dimension_map.get(model_name, 384)  # Default to 384

EMBEDDING_DIM = get_embedding_dim_from_model(EMBEDDING_MODEL)
```

**In session_manager.py**, change:
```python
# From:
def __init__(self, session_id: str, query: str, embedding_dim: int = 384):

# To:
from config import EMBEDDING_DIM

def __init__(self, session_id: str, query: str, embedding_dim: int = EMBEDDING_DIM):
```

**In rag_engine.py**, change:
```python
# From:
self.vector_store = VectorStore(embedding_dim=384, embeddings_dir=EMBEDDINGS_DIR)

# To:
from config import EMBEDDING_DIM
self.vector_store = VectorStore(embedding_dim=EMBEDDING_DIM, embeddings_dir=EMBEDDINGS_DIR)
```

**In embedding_service.py**, change:
```python
# From:
def __init__(self, embedding_dim: int = 384, embeddings_dir: str = "./embeddings"):

# To:
from config import EMBEDDING_DIM

def __init__(self, embedding_dim: int = EMBEDDING_DIM, embeddings_dir: str = "./embeddings"):
```

---

## Testing Checklist

After applying fixes, test:

- [ ] **Startup**: Application starts without NameError
  ```bash
  python api/app.py
  # Should see: "[APP] ElixpoSearch ready"
  ```

- [ ] **Chat Engine**: Chat endpoint works
  ```bash
  curl -X POST http://localhost:5000/api/chat \
    -H "Content-Type: application/json" \
    -d '{"session_id":"test","message":"hello"}'
  # Should get response, not AttributeError
  ```

- [ ] **Web Search**: Search works
  ```bash
  curl -X POST http://localhost:5000/api/search \
    -H "Content-Type: application/json" \
    -d '{"query":"python programming"}'
  # Should stream results
  ```

- [ ] **YouTube**: YouTube metadata works
  ```bash
  Test with a real YouTube URL
  # Should fetch metadata without hanging
  ```

- [ ] **Vector Store**: Content ingestion works
  ```bash
  Check logs for: "[Pipeline] Ingested X chunks from URL"
  # Should show successful ingest
  ```

---

## File Edit Commands

For quick reference, here are the exact locations to edit:

```
api/app.py:80-89      → Add cwd variable
api/rag_engine.py:133-150 → Add get_summary_stats() method  
api/app.py:125        → Change second parameter
api/utility.py:20     → Delete ipcService registration
api/transcribe.py:22  → Fix BaseManager.register
api/getYoutubeDetails.py:47-68 → Fix async function
api/searchPipeline.py:195-206 → Verify ingest call exists
```

---

## For More Details

See: [CRITICAL_INTEGRATION_ISSUES.md](CRITICAL_INTEGRATION_ISSUES.md)

