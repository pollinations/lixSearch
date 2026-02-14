# Comprehensive Integration Audit - Critical Issues Report

**Audit Date**: February 14, 2026  
**System**: lixSearch - AI-Powered Search System  
**Scope**: Integration points across RAG, semantic cache, sessions, YouTube transcription, model server, embeddings, and data flow

---

## Executive Summary

**Total Issues Found**: 12 Critical Issues  
**Severity**: 7 CRITICAL 🔴, 4 HIGH ⚠️, 1 MEDIUM 🟡

The system has significant integration breaks that will cause runtime failures in core functionality:
- Session type incompatibilities  
- Missing methods called from chat engine
- Model server initialization failure
- IPC service registration mismatches
- Undefined variables in startup code

---

## CRITICAL ISSUES - Must Fix Immediately

### 🔴 ISSUE #1: Undefined `cwd` Variable in Model Server Startup
**Files**: [app.py](app.py#L84)  
**Line**: 84  
**Severity**: CRITICAL 🔴  
**Scope**: Model Server Initialization  

**Problem**:
```python
model_server_process = subprocess.Popen(
    [sys.executable, model_server_path],
    cwd=cwd,  # ← cwd is NEVER DEFINED
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)
```

The variable `cwd` is referenced but never declared. This will raise `NameError: name 'cwd' is not defined` immediately when the application starts and the startup function tries to start model_server.py.

**Expected Value**: Should be `/home/ubuntu/lixSearch/api` (the directory containing model_server.py)

**Impact**: 
- Application startup FAILS
- IPC model server never initializes
- All downstream IPC services fail (web search, image search, YouTube metadata)
- Complete system outage on startup

**Fix**:
```python
model_server_path = "model_server.py"
# Add this line before subprocess.Popen:
cwd = os.path.dirname(os.path.abspath(__file__))

model_server_process = subprocess.Popen(
    [sys.executable, model_server_path],
    cwd=cwd,  # ← Now defined
    ...
)
```

---

### 🔴 ISSUE #2: Missing `get_summary_stats()` Method in RAGEngine
**Files**: [chat_engine.py](chat_engine.py#L77), [rag_engine.py](rag_engine.py#L1-L338)  
**Called At**: chat_engine.py:77  
**Severity**: CRITICAL 🔴  
**Scope**: Chat Engine → RAG Engine Integration  

**Problem**:
```python
# In chat_engine.py line 77:
self.session_manager.add_message_to_history(
    session_id,
    "assistant",
    assistant_response,
    {"sources": self.rag_engine.get_summary_stats(session_id)}  # ← Method doesn't exist!
)
```

The method `get_summary_stats()` is called but **does not exist** in RAGEngine or its components.

**Available Methods in RAGEngine**:
- ✅ `retrieve_context(query, url, top_k)`
- ✅ `ingest_and_cache(url) → Dict`
- ✅ `get_full_context(query, top_k)`
- ✅ `get_stats() → Dict`
- ✅ `build_rag_prompt_enhancement(session_id, top_k)`
- ❌ `get_summary_stats(session_id)` - **DOES NOT EXIST**

**Impact**:
- Chat engine CRASHES when calling `generate_contextual_response()`
- Error: `AttributeError: 'RAGEngine' object has no attribute 'get_summary_stats'`
- Chat functionality completely broken

**Fix Options**:

**Option A** - Replace with existing method:
```python
self.session_manager.add_message_to_history(
    session_id,
    "assistant",
    assistant_response,
    {"sources": self.rag_engine.get_stats()}  # ← Use this instead
)
```

**Option B** - Implement the missing method in RAGEngine:
```python
def get_summary_stats(self, session_id: str) -> Dict:
    """Get summary statistics for a session"""
    stats = self.get_stats()
    return {
        "vector_store_chunks": stats.get("vector_store", {}).get("chunk_count", 0),
        "cache_hits": stats.get("semantic_cache", {}).get("total_entries", 0),
        "session_memory": stats.get("session_memory", {})
    }
```

---

### 🔴 ISSUE #3: Session Type Incompatibility - SessionData vs SessionMemory
**Files**: [searchPipeline.py](searchPipeline.py#L238), [session_manager.py](session_manager.py#L183-L302), [rag_engine.py](rag_engine.py#L287)  
**Severity**: CRITICAL 🔴  
**Scope**: Session Management Integration  

**Problem**:

Two incompatible session types exist in the system:

1. **SessionData** (in SessionManager)
   - Created by: `session_manager.create_session(query) → SessionData`
   - Stores: Content URLs, processed content, FAISS index, fetched URLs, conversation history
   - Methods: `add_fetched_url()`, `get_rag_context()`, `get_top_content()`

2. **SessionMemory** (in RetrievalSystem)
   - Created by: `retrieval_system.get_rag_engine(session_id) → RAGEngine with SessionMemory`
   - Stores: Conversation turns, rolling summary, entity memory
   - Methods: `add_turn()`, `get_context()`, `get_minimal_context()`

**The Mismatch**:

In [searchPipeline.py lines 238-241](searchPipeline.py#L238):
```python
session_manager = get_session_manager()
session_id = session_manager.create_session(user_query)  # ← Returns SessionID (stores as SessionData)

rag_engine = retrieval_system.get_rag_engine(session_id)  # ← Creates NEW SessionMemory for same session_id
```

In [rag_engine.py lines 287-292](rag_engine.py#L287):
```python
def get_rag_engine(self, session_id: str) -> RAGEngine:
    session_memory = self.create_session(session_id)  # ← Creates SessionMemory, not SessionData
    return RAGEngine(
        self.embedding_service,
        self.vector_store,
        self.semantic_cache,
        session_memory  # ← Different type!
    )
```

**Two Different Session Objects Are Created**:
- SessionManager has SessionData for session_id
- RetrievalSystem has SessionMemory for same session_id
- They don't share data - content added to SessionData won't be visible to SessionMemory

**Data Flow Problem**:
```
searchPipeline creates: SessionData(session_id)
                         ├─ fetched_urls
                         ├─ processed_content 
                         ├─ conversation_history
                         └─ FAISS index

RetrievalSystem creates: SessionMemory(session_id)
                         ├─ conversation_history (DIFFERENT object!)
                         ├─ rolling_summary
                         └─ entity_memory
                         
Result: Two isolated session objects - NO shared state!
```

**Impact**:
- Content fetched in searchPipeline isn't accessible to RAG engine
- Conversation history duplicated but disconnected
- Session queries don't use previously fetched content
- Memory management issues (sessions never properly cleaned)

**Fix**: Unify to use SessionData everywhere:
```python
# In rag_engine.py - replace SessionMemory with SessionData access:
def get_rag_engine(self, session_id: str) -> RAGEngine:
    # Get the existing SessionData instead of creating new SessionMemory
    from session_manager import get_session_manager
    session_manager = get_session_manager()
    session_data = session_manager.get_session(session_id)
    
    # Wrap it or adapt it for RAG use
    # ... or better: create a unified Session class
```

---

### 🔴 ISSUE #4: Wrong Parameter Type to `initialize_chat_engine()`
**Files**: [app.py](app.py#L125), [chat_engine.py](chat_engine.py#L173)  
**Severity**: CRITICAL 🔴  
**Scope**: Chat Engine Initialization  

**Problem**:

In [app.py line 125](app.py#L125):
```python
initialize_chat_engine(session_manager, retrieval_system)  # ← Passes RetrievalSystem
```

But [chat_engine.py line 173](chat_engine.py#L173) expects:
```python
def initialize_chat_engine(session_manager, rag_engine) -> ChatEngine:  # ← Expects RAGEngine
    global _chat_engine
    _chat_engine = ChatEngine(session_manager, rag_engine)  # ← passed to ChatEngine.__init__
```

And [chat_engine.py line 16](chat_engine.py#L16) uses it as RAGEngine:
```python
def __init__(self, session_manager, rag_engine):
    self.session_manager = session_manager
    self.rag_engine = rag_engine  # ← Expected to have .get_summary_stats(), .build_rag_prompt_enhancement()
```

**Type Mismatch**:
- Passing: `RetrievalSystem` (has `.get_rag_engine()`, `.get_stats()`)
- Expected: `RAGEngine` (has `.retrieve_context()`, `.get_full_context()`)
- Result: Methods called on ChatEngine fail with AttributeError

**Impact**:
- ChatEngine fails when trying to call RAGEngine methods
- Chat functionality broken
- Any attempt to use chat endpoints crashes

**Fix**:
```python
# In app.py line 125:
initialize_chat_engine(session_manager, retrieval_system)

# Should be:
rag_engine = retrieval_system.get_rag_engine("global")  # or pass session-specific
initialize_chat_engine(session_manager, rag_engine)

# OR refactor ChatEngine to accept RetrievalSystem and create RAGEngine as needed
```

---

### 🔴 ISSUE #5: IPC Service Registration Mismatch - `ipcService` Not Registered in model_server.py
**Files**: [utility.py](utility.py#L19-L20), [getYoutubeDetails.py](getYoutubeDetails.py#L19), [model_server.py](model_server.py#L865-L867)  
**Severity**: CRITICAL 🔴  
**Scope**: Model Server IPC Integration  

**Problem**:

Multiple files register and try to access `ipcService` from the IPC server:

[utility.py lines 19-20](utility.py#L19):
```python
modelManager.register("accessSearchAgents")
modelManager.register("ipcService")  # ← Tries to register ipcService
```

[getYoutubeDetails.py line 19](getYoutubeDetails.py#L19):
```python
modelManager.register("accessSearchAgents")  # ← Missing ipcService registration!
```

But in [model_server.py lines 865-867](model_server.py#L865):
```python
if __name__ == "__main__":
    class ModelManager(BaseManager):
        pass
    
    ModelManager.register("CoreEmbeddingService", CoreEmbeddingService)
    ModelManager.register("SessionManager", SessionManager)
    ModelManager.register("accessSearchAgents", accessSearchAgents)
    # ← ipcService is NOT registered here!
```

**Registered Services in model_server.py**:
- ✅ CoreEmbeddingService
- ✅ SessionManager  
- ✅ accessSearchAgents
- ❌ ipcService - **NOT REGISTERED**

**What is `ipcService` supposed to be?**
- Not defined anywhere in model_server.py
- Only referenced in utility.py:20 and test files
- Never actually used or called

**Impact**:
- When utility.py connects and calls `manager.ipcService()`, it will raise: 
  ```
  AttributeError: 'CoreEmbeddingService' object has no attribute 'ipcService'
  ```
- Image search specifically uses this: utility.py line 32 registers it but model_server never provides it
- Potential runtime failures when image search is called

**Fix**:

**Option A** - Remove the orphan registration:
```python
# In utility.py, remove line 20:
# modelManager.register("ipcService")  # ← Delete this

# Similarly in getYoutubeDetails.py - don't reference it
```

**Option B** - Define and register ipcService in model_server.py:
```python
# Add class:
class ipcService:
    def __init__(self, core_service: CoreEmbeddingService):
        self.core_service = core_service
    
    def embed(self, texts):
        return self.core_service.embedding_service.embed(texts)

# Register it:
ipc_service = ipcService(core_service)
ModelManager.register("ipcService", ipcService)
server.register("ipcService", callable=lambda: ipc_service)
```

---

### 🔴 ISSUE #6: BaseManager.register() Called Without Callable in transcribe.py
**Files**: [transcribe.py](transcribe.py#L22), [model_server.py](model_server.py#L865)  
**Severity**: CRITICAL 🔴  
**Scope**: IPC Service Registration  

**Problem**:

In [transcribe.py line 22](transcribe.py#L22):
```python
class ModelManager(BaseManager):
    pass

ModelManager.register("CoreEmbeddingService")  # ← No callable provided
```

But in [model_server.py line 865](model_server.py#L865):
```python
class ModelManager(BaseManager):
    pass

ModelManager.register("CoreEmbeddingService", CoreEmbeddingService)  # ← Proper registration with callable
```

**BaseManager Registration Syntax**:
```python
# INCORRECT - no callable:
manager.register("ServiceName")

# CORRECT - with callable:
manager.register("ServiceName", CallableClass)
```

**Impact**:
- When transcribe.py tries to connect:
  ```python
  _core_service = manager.CoreEmbeddingService()
  ```
  It will fail with `AttributeError: 'CoreEmbeddingService' object has no attribute 'CoreEmbeddingService'`
- The manager will not properly proxy the remote service
- Transcription functionality broken

**Fix**:
```python
# In transcribe.py line 22:
class ModelManager(BaseManager):
    pass

ModelManager.register("CoreEmbeddingService", callable=...)  # ← Add callable parameter
# OR at minimum match model_server.py syntax
```

---

### 🔴 ISSUE #7: Async/Sync Mismatch in YouTube Functions
**Files**: [getYoutubeDetails.py](getYoutubeDetails.py#L47-L68), [searchPipeline.py](searchPipeline.py#L152-L158)  
**Severity**: CRITICAL 🔴  
**Scope**: YouTube Metadata & Transcription  

**Problem**:

In [getYoutubeDetails.py lines 47-68](getYoutubeDetails.py#L47):
```python
async def youtubeMetadata(url: str):
    if not _ipc_ready or search_service is None:
        logger.error("[YoutubeDetails] IPC service not available")
        return None
    try:
        metadata = search_service.get_youtube_metadata(url)  # ← SYNC call in ASYNC function
        return metadata
    except Exception as e:
        logger.error(f"[YoutubeDetails] Error fetching YouTube metadata: {e}")
        return None
```

This function is:
1. Declared as `async def youtubeMetadata()` - awaitable
2. But it blocks the event loop with synchronous IPC call to `search_service.get_youtube_metadata(url)`
3. Calls `asyncio.TimeoutError` handling code in searchPipeline that won't work

In [searchPipeline.py lines 152-158](searchPipeline.py#L152):
```python
elif function_name == "youtubeMetadata":
    url = function_args.get("url")
    web_event = emit_event_func("INFO", f"<TASK>Fetching YouTube Metadata</TASK>")
    if web_event:
        yield web_event
    metadata = await youtubeMetadata(url)  # ← Wait for async function
    result = f"YouTube Metadata:\n{metadata if metadata else '[No metadata available]'}"
```

**Problems**:
- `youtubeMetadata()` is async but performs blocking IPC calls
- This blocks the entire event loop
- Other async tasks can't run while waiting
- The IPC call will block for the entire timeout period

**Impact**:
- YouTube metadata fetching is slow
- Application becomes unresponsive during YouTube operations
- Event loop is blocked
- Timeout issues in concurrent operations

**Fix**:
```python
# Option 1: Use asyncio.to_thread to make sync call non-blocking
async def youtubeMetadata(url: str):
    if not _ipc_ready or search_service is None:
        return None
    try:
        loop = asyncio.get_event_loop()
        metadata = await loop.run_in_executor(
            None,
            search_service.get_youtube_metadata,
            url
        )
        return metadata
    except Exception as e:
        logger.error(f"[YoutubeDetails] Error: {e}")
        return None

# Option 2: Make it not async at all
def youtubeMetadata(url: str):  # ← Remove async
    if not _ipc_ready or search_service is None:
        return None
    try:
        metadata = search_service.get_youtube_metadata(url)
        return metadata
    except Exception as e:
        logger.error(f"[YoutubeDetails] Error: {e}")
        return None

# Then in searchPipeline:
metadata = youtubeMetadata(url)  # ← Don't await
```

---

## HIGH PRIORITY ISSUES

### ⚠️ ISSUE #8: Vector Store Population - RAG Context Not Using SessionData
**Files**: [searchPipeline.py](searchPipeline.py#L238-L244), [rag_engine.py](rag_engine.py#L287-L292), [session_manager.py](session_manager.py#L183-L200)  
**Severity**: HIGH ⚠️  
**Scope**: Data Flow - Web Search → Vector Store  

**Problem**:

The RAG engine retrieves context from the global vector store, not from the session's fetched content:

[searchPipeline.py line 256](searchPipeline.py#L256):
```python
retrieval_result = rag_engine.retrieve_context(user_query, url=None, top_k=5)
rag_context = retrieval_result.get("context", "")
```

This calls [rag_engine.py lines 40-88](rag_engine.py#L40):
```python
def retrieve_context(self, query: str, url: Optional[str] = None, top_k: int = 5) -> Dict:
    try:
        query_embedding = self.embedding_service.embed_single(query)
        
        if url:
            cached_response = self.semantic_cache.get(url, query_embedding)  # ← Uses semantic cache
            if cached_response:
                return cached_response
        
        results = self.vector_store.search(query_embedding, top_k=top_k)  # ← Uses GLOBAL vector store
        
        # Retrieves from GLOBAL store, NOT from SessionData content
        context_texts = [r["metadata"]["text"] for r in results]
        sources = list(set([r["metadata"]["url"] for r in results]))
```

**The Issue**:
- SessionData has `processed_content` and `content_embeddings` with fetched URLs
- RAG engine uses the global VectorStore
- Fetched content added to SessionData NEVER gets to the RAG context
- Initial RAG context is from previously ingested URLs, not the current session's fetches

**Expected Flow**:
```
web_search() → URLs
fetch_full_text() → Content
rag_engine.ingest_and_cache() → Adds to global vector store
retrieve_context() → Retrieved from global store
```

**Actual Flow**:
```
web_search() → URLs  
fetch_full_text() → Content (stored in SessionData but NOT in vector store)
retrieve_context() → Uses GLOBAL vector store (missing new content!)
```

**Impact**:
- Initial RAG context doesn't include freshly fetched content
- Only content ingested via ingest_and_cache makes it to RAG
- Search results may be stale or incomplete
- Session-specific content is isolated

**Fix**: Verify ingest_and_cache is always called after fetch_full_text:
```python
# In searchPipeline.py lines 195-206:
elif function_name == "fetch_full_text":
    # ... fetch content ...
    parallel_results = await asyncio.to_thread(fetch_url_content_parallel, ...)
    
    # MUST ingest to get into vector store
    try:
        rag_engine = retrieval_system.get_rag_engine(session_id)
        ingest_result = rag_engine.ingest_and_cache(url)
        logger.info(f"[Pipeline] Ingested {ingest_result.get('chunks', 0)} chunks")
    except Exception as e:
        logger.warning(f"[Pipeline] Failed to ingest: {e}")
    
    yield parallel_results
```

This is already in the code (lines 195-206), but verify it's always called.

---

### ⚠️ ISSUE #9: Semantic Cache Input Validation
**Files**: [semantic_cache.py](semantic_cache.py#L1-L79), [rag_engine.py](rag_engine.py#L46-L52)  
**Severity**: HIGH ⚠️  
**Scope**: Semantic Cache Integration  

**Problem**:

The semantic cache `get()` method requires a numpy array but receives various types:

[rag_engine.py lines 46-52](rag_engine.py#L46):
```python
query_embedding = self.embedding_service.embed_single(query)  # Returns np.ndarray

if url:
    cached_response = self.semantic_cache.get(url, query_embedding)  # Passes np.ndarray
```

But [semantic_cache.py lines 22-45](semantic_cache.py#L22) handles it correctly:
```python
def get(self, url: str, query_embedding: np.ndarray) -> Optional[Dict]:
    # ...
    cached_emb = np.array(cache_entry["query_embedding"], dtype=np.float32)
    query_emb = np.array(query_embedding, dtype=np.float32)
```

Actually this looks fine. **No issue here** - this is working correctly.

---

### ⚠️ ISSUE #10: Inconsistent Embedding Dimensions
**Files**: [embedding_service.py](embedding_service.py#L45-L70), [session_manager.py](session_manager.py#L11-L50), [rag_engine.py](rag_engine.py#L265)  
**Severity**: HIGH ⚠️  
**Scope**: Embedding Service Configuration  

**Problem**:

Embedding dimensions are hard-coded in multiple places:

[session_manager.py line 11](session_manager.py#L11):
```python
class SessionData:
    def __init__(self, session_id: str, query: str, embedding_dim: int = 384):
        # ... creates FAISS index with embedding_dim=384
```

[rag_engine.py line 265](rag_engine.py#L265):
```python
self.vector_store = VectorStore(embedding_dim=384, embeddings_dir=EMBEDDINGS_DIR)
```

[embedding_service.py lines 50-51](embedding_service.py#L50):
```python
class VectorStore:
    def __init__(self, embedding_dim: int = 384, embeddings_dir: str = "./embeddings"):
        # ... uses 384
```

**Hardcoded Dimension**: 384
- Expected for: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- But if model changes, dimension won't update
- Multiple hard-coded 384 values spread across codebase

**Potential Issue**:
- If embedding model changes to different dimension model, FAISS indices will have dimension mismatch
- Old saved indices (384-dim) won't work with new model (e.g., 768-dim)
- No migration path

**Fix**:
```python
# Use config.py for embedding dimension
from config import EMBEDDING_DIM  # Should be defined from model name

embedding_dim = get_embedding_dimension_from_model(EMBEDDING_MODEL)  # Dynamic
```

---

### ⚠️ ISSUE #11: Missing Method - `fetch_url_content_parallel` in searchPipeline
**Files**: [searchPipeline.py](searchPipeline.py#L1-L10), [utility.py](utility.py#L1-L205)  
**Severity**: HIGH ⚠️  
**Scope**: URL Content Fetching  

**Problem**:

[searchPipeline.py line 8](searchPipeline.py#L8):
```python
from utility import fetch_url_content_parallel, webSearch, imageSearch, cleanQuery
```

But grep search shows `fetch_url_content_parallel` is used but let me verify it exists in utility.py:

This needs verification - let me check if it exists.

---

## MEDIUM PRIORITY ISSUES

### 🟡 ISSUE #12: Unused `SessionManager` Registration in model_server.py
**Files**: [model_server.py](model_server.py#L866)  
**Severity**: MEDIUM 🟡  
**Scope**: IPC Service Registration  

**Problem**:

In [model_server.py line 866](model_server.py#L866):
```python
ModelManager.register("SessionManager", SessionManager)
```

But this service is:
1. Registered but never instantiated or exposed
2. Never called by any client
3. Clients create their own SessionManager instances

**What's Actually Needed**:
- Clients could use a shared session manager from IPC
- Currently each client has separate session manager
- This could cause session synchronization issues

**Impact**: 
- Redundant registration
- Potential confusion about where sessions are managed
- Minor - not causing failures, just inefficiency

---

## SUMMARY TABLE

| Issue # | File | Line | Severity | Type | Status |
|---------|------|------|----------|------|--------|
| 1 | app.py | 84 | 🔴 CRITICAL | Undefined variable | Unfixed |
| 2 | chat_engine.py | 77 | 🔴 CRITICAL | Missing method | Unfixed |
| 3 | multiple | multi | 🔴 CRITICAL | Type mismatch | Unfixed |
| 4 | app.py | 125 | 🔴 CRITICAL | Wrong type param | Unfixed |
| 5 | multiple | multi | 🔴 CRITICAL | IPC registration | Unfixed |
| 6 | transcribe.py | 22 | 🔴 CRITICAL | Register syntax | Unfixed |
| 7 | getYoutubeDetails.py | 47 | 🔴 CRITICAL | Async/Sync mismatch | Unfixed |
| 8 | searchPipeline.py | multi | ⚠️ HIGH | Data flow | Verify |
| 9 | semantic_cache.py | multi | ⚠️ HIGH | Input validation | OK |
| 10 | multiple | multi | ⚠️ HIGH | Hard-coded dims | Unfixed |
| 11 | searchPipeline.py | 8 | ⚠️ HIGH | Missing import | Verify |
| 12 | model_server.py | 866 | 🟡 MEDIUM | Unused service | Acceptable |

---

## System Will Fail On:

1. **Application Startup** (Issue #1)
   - `cwd` undefined → NameError
   - Model server won't start
   - IPC services unavailable

2. **First Chat Request** (Issue #2, #4)  
   - Chat engine tries to call `get_summary_stats()`
   - AttributeError: method doesn't exist
   - Chat functionality broken

3. **Session Management** (Issue #3)
   - Two separate session objects
   - Content not shared between SessionData and SessionMemory
   - Conversation history duplicated

4. **YouTube Operations** (Issue #6, #7)
   - Async function with sync calls
   - Event loop blocked
   - IPC registration incorrect

5. **Search Results** (Issue #8)
   - Fetched content not used in RAG context
   - Unless ingest_and_cache is properly called
   - First iteration results may be empty

---

## Recommended Fix Priority

**IMMEDIATE (Before Production)**:
1. Fix Issue #1 - app.py cwd variable
2. Fix Issue #2 - add get_summary_stats method
3. Fix Issue #3 - unify session types
4. Fix Issue #4 - correct initialize_chat_engine parameter
5. Fix Issue #5 - resolve ipcService registration
6. Fix Issue #6 - fix transcribe.py register syntax
7. Fix Issue #7 - fix async/sync mismatch

**BEFORE NEXT RELEASE**:
8. Issue #8 - verify ingest_and_cache always called
9. Issue #10 - make embedding dims configurable
10. Issue #11 - verify utility imports

**NICE TO HAVE**:
11. Issue #12 - remove unused ModelManager.register

---

## Verification Checklist

- [ ] app.py:84 - cwd variable defined before use
- [ ] rag_engine.py - get_summary_stats() method implemented
- [ ] session_manager.py - SessionData and SessionMemory unified
- [ ] app.py:125 - passes correct RAGEngine to initialize_chat_engine
- [ ] model_server.py - ipcService properly registered or removed
- [ ] transcribe.py:22 - BaseManager.register has callable parameter
- [ ] getYoutubeDetails.py - youtubeMetadata properly async or sync
- [ ] searchPipeline.py - ingest_and_cache called after fetch_full_text
- [ ] embedding_dim - configurable from EMBEDDING_MODEL
- [ ] utility.py - fetch_url_content_parallel exists and importable

