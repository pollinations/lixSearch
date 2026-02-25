# lixSearch: Full System Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Architectural Layers](#architectural-layers)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Request Lifecycle](#request-lifecycle)
6. [Integration Architecture](#integration-architecture)
7. [Deployment Model](#deployment-model)

---

## System Overview

**lixSearch** is a multi-layered intelligent search system that combines:
- Real-time web search with streaming results
- Semantic RAG (Retrieval-Augmented Generation)
- Session-based context management
- LLM-powered synthesis
- Advanced caching strategies

### Key Goals
 Sub-second semantic matching through adaptive caching  
 Parallel query execution across multiple evidence sources  
 Cost optimization via token estimation and compression  
 Context-aware response synthesis using session memory  
 Graceful degradation when components fail  

---

## Architectural Layers

### Layer 1: API Gateway Layer (Quart/Hypercorn)

```mermaid
graph TD
    Client["👤 Client<br/>HTTP/WebSocket"]
    Gateway["Quart Server<br/>0.0.0.0:8000"]
    RequestID["RequestID Middleware<br/>X-Request-ID Header"]
    CORS["CORS Handler"]
    Routes["Route Dispatcher"]
    
    Search["/api/search<br/>POST/GET"]
    Chat["/api/chat<br/>POST"]
    Session["/api/session<br/>Crud Ops"]
    Health["/api/health<br/>GET"]
    WebSocket["/ws/search<br/>WebSocket"]
    
    Processing["Response Processing"]
    SSE["SSE Streaming<br/>Server-Sent Events"]
    JSON["OpenAI-Compatible<br/>JSON Format"]
    Error["Error Handlers"]
    Logging["Request Logging"]
    
    Client -->|HTTP/WS| Gateway
    Gateway --> RequestID
    RequestID --> CORS
    CORS --> Routes
    Routes --> Search
    Routes --> Chat
    Routes --> Session
    Routes --> Health
    Routes --> WebSocket
    
    Search --> Processing
    Chat --> Processing
    Session --> Processing
    
    Processing --> SSE
    Processing --> JSON
    Processing --> Logging
    Processing --> Error
    
    style Gateway fill:#E3F2FD
    style RequestID fill:#BBDEFB
    style Processing fill:#C8E6C9
    style Error fill:#FFCDD2
```

**Gateways:**
- `health.py` - Health checks
- `search.py` - Search endpoint (streaming SSE)
- `chat.py` - Chat with multi-turn context
- `session.py` - Session CRUD + KB operations
- `stats.py` - System statistics
- `websocket.py` - WebSocket streaming

**Key Features:**
- Streaming responses via Server-Sent Events (SSE)
- OpenAI-compatible response format
- Request ID tracking for tracing
- Async/await throughout with Quart

---

### Layer 2: Pipeline & Orchestration Layer

```mermaid
graph TD
    Input["User Query +<br/>Image URL"]
    SearchPipeline["searchPipeline.py<br/>Entry Point"]
    
    Validate["1. Validate Query<br/>& Image URL"]
    CreateSession["2. Create Session<br/>& Track Request"]
    Decompose["3. Query Decomposition<br/>Aspect Detection"]
    
    LixSearch["lixsearch.py<br/>Main Orchestrator"]
    
    ToolExec["optimized_tool_execution.py<br/>Parallel Execution"]
    
    WebSearch["Web Search<br/>Playwright"]
    FetchText["Fetch Full Text<br/>BeautifulSoup"]
    YouTubeAPI["YouTube Metadata<br/>API Call"]
    ImageAnalysis["Image Analysis<br/>Vision API"]
    
    Aggregate["Aggregate Results"]
    RAGContext["Retrieve RAG Context<br/>Semantic Cache + Vector Search"]
    LLMSynthesize["LLM Synthesis<br/>ChatEngine"]
    StreamResponse["Stream Response<br/>SSE Events"]
    
    OptModules["Optimization Modules"]
    TokenCost["tokenCostOptimization"]
    FormalOpt["formalOptimization"]
    AdaptiveThresh["adaptiveThresholding"]
    
    Input --> SearchPipeline
    SearchPipeline --> Validate
    Validate --> CreateSession
    CreateSession --> Decompose
    Decompose --> LixSearch
    
    LixSearch --> ToolExec
    ToolExec -->|parallel| WebSearch
    ToolExec -->|parallel| FetchText
    ToolExec -->|parallel| YouTubeAPI
    ToolExec -->|parallel| ImageAnalysis
    
    WebSearch --> Aggregate
    FetchText --> Aggregate
    YouTubeAPI --> Aggregate
    ImageAnalysis --> Aggregate
    
    Aggregate --> RAGContext
    RAGContext --> LLMSynthesize
    LLMSynthesize --> StreamResponse
    
    LixSearch -.->|uses| OptModules
    OptModules --> TokenCost
    OptModules --> FormalOpt
    OptModules --> AdaptiveThresh
    
    style LixSearch fill:#FFF3E0
    style ToolExec fill:#F3E5F5
    style RAGContext fill:#E8F5E9
    style LLMSynthesize fill:#FCE4EC
    style OptModules fill:#FBE9E7
```

**Key Modules:**

#### lixsearch.py (Main Orchestrator)

```mermaid
graph TD
    Start[\"run_elixposearch_pipeline<br/>query, image, event_id\"]
    Decompose[\"_decompose_query<br/>Break into sub-queries\"]
    ToolExec[\"optimized_tool_execution<br/>Parallel execution\"]
    RAGContext[\"_get_rag_context<br/>Retrieve cached evidence\"]
    Synthesis[\"LLM synthesis<br/>Generate response\"]
    SSEStream[\"SSE streaming<br/>Yield formatted events\"]
    End[\"Return AsyncGenerator<br/>Event chunks\"]
    
    Start --> Decompose
    Decompose --> ToolExec
    ToolExec --> RAGContext
    RAGContext --> Synthesis
    Synthesis --> SSEStream
    SSEStream --> End
    
    style Start fill:#FFF3E0
    style End fill:#FFF3E0
    style SSEStream fill:#FFE0B2
```

#### searchPipeline.py (Flow Controller)

```mermaid
graph TD
    Start[\"run_elixposearch_pipeline<br/>entry point\"]
    Validate[\"1. Validate query<br/>& image_url\"]
    CreateSess[\"2. Create session<br/>Track request_id\"]
    ToolExec[\"3. Execute tools<br/>in parallel\"]
    Aggregate[\"4. Aggregate results<br/>Deduplicate URLs\"]
    RAGRetrieve[\"5. Retrieve RAG<br/>context\"]
    LLMCall[\"6. Call LLM<br/>with context\"]
    Stream[\"7. Stream response<br/>chunks as SSE\"]
    End[\"Return event stream<br/>to gateway\"]
    
    Start --> Validate
    Validate --> CreateSess
    CreateSess --> ToolExec
    ToolExec --> Aggregate
    Aggregate --> RAGRetrieve
    RAGRetrieve --> LLMCall
    LLMCall --> Stream
    Stream --> End
    
    style Start fill:#F3E5F5
    style End fill:#F3E5F5
    style ToolExec fill:#E1BEE7
    style RAGRetrieve fill:#CE93D8
```

#### optimized_tool_execution.py (Tool Runner)

```mermaid
graph TD
    Start[\"optimized_tool_execution<br/>search_tools list\"]
    
    WebSearch[\"Web Search<br/>Playwright\"]
    YouTubeFetch[\"YouTube Metadata<br/>API Call\"]
    ImageAnalysis[\"Image Analysis<br/>Vision Model\"]
    Functions[\"Function Calls<br/>getTimeZone, generateImage\"]
    
    Async1[\"Async<br/>Task 1\"]
    Async2[\"Async<br/>Task 2\"]
    Async3[\"Async<br/>Task 3\"]
    Async4[\"Async<br/>Task 4\"]
    
    Gather[\"Gather all results<br/>asyncio.gather\"]
    Aggregate[\"Aggregate results<br/>De-duplicate\"]
    Format[\"Format output<br/>Structured data\"]
    End[\"Return aggregated<br/>results to pipeline\"]
    
    Start --> WebSearch
    Start --> YouTubeFetch
    Start --> ImageAnalysis
    Start --> Functions
    
    WebSearch --> Async1
    YouTubeFetch --> Async2
    ImageAnalysis --> Async3
    Functions --> Async4
    
    Async1 --> Gather
    Async2 --> Gather
    Async3 --> Gather
    Async4 --> Gather
    
    Gather --> Aggregate
    Aggregate --> Format
    Format --> End
    
    style Gather fill:#B2DFDB
    style Async1 fill:#4DB6AC
    style Async2 fill:#4DB6AC
    style Async3 fill:#4DB6AC
    style Async4 fill:#4DB6AC
```

---

### Layer 3: RAG Service Layer

```mermaid
graph TD
    Query["Query Input"]
    RAGEngine["RAG Engine<br/>ragEngine.py"]
    RetrieveContext["retrieve_context<br/>query, url -> RAG"]
    IngestCache["ingest_and_cache<br/>url -> embeddings"]
    BuildPrompt["build_rag_prompt_enhancement<br/>-> combine"]
    GetStats["get_stats<br/>-> metrics"]
    
    SemanticCache["Semantic Cache<br/>semanticCache.py"]
    CacheHit["✓ Cache Hit<br/>1-10ms"]
    CacheMiss["✗ Cache Miss<br/>Continue"]
    
    EmbedService["Embedding Service<br/>embeddingService.py"]
    EmbedModel["SentenceTransformer<br/>all-MiniLM-L6-v2<br/>384 dimensions"]
    EmbedSingle["embed_single<br/>text->vector"]
    EmbedBatch["embed<br/>texts[]->batch"]
    
    VectorStore["Vector Store<br/>vectorStore.py"]
    ChromaDB["ChromaDB<br/>HNSW Index"]
    AddChunks["add_chunks<br/>batch insert"]
    SearchVec["search<br/>cosine similarity"]
    PersistDisk["persist_to_disk<br/>./embeddings/"]
    
    RetPipeline["Retrieval Pipeline<br/>retrievalPipeline.py"]
    IngestURL["ingest_url"]
    FetchHTML["Fetch HTML<br/>3000 words max"]
    CleanText["Clean Text<br/>remove scripts"]
    ChunkText["Chunk Text<br/>600 words, 60 overlap"]
    EmbedChunks["Embed Chunks<br/>batch mode"]
    StoreVector["Store in Vector<br/>Store"]
    
    RetrieveQuery["retrieve"]
    EmbedQueryVec["Embed Query"]
    SearchSim["Search Similarity<br/>top-K"]
    ReturnResults["Return Results<br/>+ metadata"]
    
    BuildContext["build_context"]
    RelevantChunks["Retrieve Chunks"]
    CombineSession["Combine with<br/>Session Memory"]
    FormatPrompt["Format for LLM"]
    
    Query --> RAGEngine
    RAGEngine --> RetrieveContext
    RAGEngine --> IngestCache
    RAGEngine --> BuildPrompt
    RAGEngine --> GetStats
    
    RetrieveContext --> SemanticCache
    SemanticCache -->|hit| CacheHit
    SemanticCache -->|miss| CacheMiss
    
    CacheMiss --> EmbedService
    IngestCache --> EmbedService
    
    EmbedService --> EmbedModel
    EmbedService --> EmbedSingle
    EmbedService --> EmbedBatch
    
    EmbedSingle --> VectorStore
    EmbedBatch --> VectorStore
    
    VectorStore --> ChromaDB
    VectorStore --> AddChunks
    VectorStore --> SearchVec
    VectorStore --> PersistDisk
    
    IngestCache --> RetPipeline
    RetPipeline --> IngestURL
    IngestURL --> FetchHTML
    FetchHTML --> CleanText
    CleanText --> ChunkText
    ChunkText --> EmbedChunks
    EmbedChunks --> StoreVector
    StoreVector --> ChromaDB
    
    Query --> RetPipeline
    RetPipeline --> RetrieveQuery
    RetrieveQuery --> EmbedQueryVec
    EmbedQueryVec --> SearchSim
    SearchSim --> ReturnResults
    
    BuildPrompt --> BuildContext
    ReturnResults --> BuildContext
    BuildContext --> RelevantChunks
    RelevantChunks --> CombineSession
    CombineSession --> FormatPrompt
    
    style RAGEngine fill:#E8F5E9
    style SemanticCache fill:#C8E6C9
    style EmbedService fill:#A5D6A7
    style VectorStore fill:#81C784
    style RetPipeline fill:#66BB6A
    style ChromaDB fill:#4CAF50
```

**Retrieval Flow:**

```mermaid
graph TD
    Query["New Query<br/>User Input"]
    EmbedQuery["embed_single query<br/>-> 384-dim vector"]
    CheckCache{"semanticCache.get<br/>url + embedding?"}
    
    CacheHit["✓ Cache HIT<br/>Return cached_response<br/>⚡ 1-10ms"]
    
    CacheMiss["✗ Cache MISS"]
    VecSearch["vectorStore.search<br/>embedding, top_k=5"]
    HNSWIndex["HNSW Index<br/>Find top-5 chunks"]
    ReturnResults["Return<br/>metadata, text, score"]
    SetCache["semanticCache.set<br/>Cache for future hits"]
    FinalReturn["Return Results<br/>To Pipeline"]
    
    Query --> EmbedQuery
    EmbedQuery --> CheckCache
    CheckCache -->|HIT| CacheHit
    CheckCache -->|MISS| CacheMiss
    CacheHit --> FinalReturn
    CacheMiss --> VecSearch
    VecSearch --> HNSWIndex
    HNSWIndex --> ReturnResults
    ReturnResults --> SetCache
    SetCache --> FinalReturn
    
    style CacheHit fill:#C8E6C9
    style CacheMiss fill:#FFCDD2
    style Query fill:#E3F2FD
    style FinalReturn fill:#F3E5F5
```

---

### Layer 4: Search Service Layer

```mermaid
graph TD
    Pipeline["Tool Execution<br/>Request"]
    
    SearchFacade["searching/main.py<br/>Service Facade"]
    IPCCheck{"IPC Connection<br/>Available?"}
    IPCClient["IPC Client<br/>localhost:5010"]
    LocalFallback["Local Services<br/>Fallback"]
    
    WebSearch["playwright_web_search.py<br/>Web Search"]
    BrowserAuto["Async Browser<br/>Automation"]
    SearchEngine["Search Engines<br/>Google/Bing/DDG"]
    ParseResults["Parse Title +<br/>Snippets"]
    UserAgent["User-Agent<br/>Rotation"]
    Timeout["Timeout: 30s"]
    WebSearchOut["Output: URL,<br/>Title, Snippet"]
    
    FetchText["fetch_full_text.py<br/>Content Extraction"]
    HTTPGet["HTTP GET<br/>Spoofed Headers"]
    BeautifulSoup["BeautifulSoup<br/>Parsing"]
    RemoveJunk["Remove Scripts/<br/>Styles/Nav"]
    ExtractContent["Extract Main<br/>Content"]
    WordLimit["Limit: 3000<br/>words max"]
    FetchOut["Output: Cleaned<br/>Text"]
    
    Tools["tools.py<br/>Function Calls"]
    YouTube["getYoutubeDetails<br/>-> Video Metadata"]
    ImagePrompt["getImagePrompt<br/>-> Image Analysis"]
    TimeZone["getTimeZone<br/>-> Location Data"]
    GenerateImage["generateImage<br/>-> Pollinations API"]
    
    Results["Aggregated Results<br/>To Pipeline"]
    
    Pipeline --> SearchFacade
    SearchFacade --> IPCCheck
    IPCCheck -->|YES| IPCClient
    IPCCheck -->|NO| LocalFallback
    
    SearchFacade -->|web search| WebSearch
    WebSearch --> BrowserAuto
    BrowserAuto --> SearchEngine
    SearchEngine --> ParseResults
    ParseResults --> UserAgent
    UserAgent --> Timeout
    Timeout --> WebSearchOut
    
    SearchFacade -->|fetch content| FetchText
    FetchText --> HTTPGet
    HTTPGet --> BeautifulSoup
    BeautifulSoup --> RemoveJunk
    RemoveJunk --> ExtractContent
    ExtractContent --> WordLimit
    WordLimit --> FetchOut
    
    SearchFacade -->|function calls| Tools
    Tools --> YouTube
    Tools --> ImagePrompt
    Tools --> TimeZone
    Tools --> GenerateImage
    
    WebSearchOut --> Results
    FetchOut --> Results
    YouTube --> Results
    ImagePrompt --> Results
    TimeZone --> Results
    GenerateImage --> Results
    
    style SearchFacade fill:#F8BBD0
    style WebSearch fill:#F48FB1
    style FetchText fill:#EC407A
    style Tools fill:#E91E63
    style Results fill:#AD1457
```

---

### Layer 5: Chat Engine & Session Layer

```mermaid
graph TD
    UserMessage["User Message<br/>Multi-turn Chat"]
    
    ChatEngine["ChatEngine<br/>chatEngine.py"]
    GenContextual["generate_contextual_response"]
    ChatSearch["chat_with_search"]
    
    BuildHistory["Build Message<br/>History"]
    RAGRetrieval["Retrieve RAG<br/>Context"]
    LLMCall["Call LLM<br/>Pollinations API"]
    StreamAsync["Stream AsyncGenerator<br/>Response Chunks"]
    
    SearchFirst["Execute Search<br/>First"]
    IncludeResults["Include Search<br/>Results"]
    EnhancedPrompt["Enhanced Prompt<br/>Synthesis"]
    
    SessionMgr["SessionManager<br/>sessionManager.py"]
    Storage["Storage:<br/>Dict<br/>session_id →<br/>SessionData"]
    MaxSessions["Max Sessions: 1000<br/>TTL: 30 min<br/>Thread-safe: RLock"]
    
    CreateSession["create_session<br/>query -> id"]
    GetSession["get_session<br/>id -> Data"]
    AddMessage["add_message_to_history"]
    GetHistory["get_conversation_history"]
    AddContent["add_content_to_session<br/>url + embedding"]
    GetRAGContext["get_rag_context<br/>-> combined"]
    
    SessionData["SessionData<br/>sessionData.py"]
    SessionID["session_id<br/>unique"]
    History["conversation<br/>history[]"]
    FetchedURLs["fetched_urls<br/>url -> content"]
    SearchURLs["web_search_urls<br/>results[]"]
    YouTubeURLs["youtube_urls<br/>metadata[]"]
    ToolCalls["tool_calls<br/>exec log"]
    Embeddings["embeddings<br/>session_emb[]"]
    LastActivity["last_activity<br/>timestamp"]
    
    GetRAGCtx["get_rag_context<br/>summary"]
    GetTopContent["get_top_content<br/>k most relevant"]
    Memory["session_memory<br/>compressed"]
    
    UserMessage --> ChatEngine
    ChatEngine --> GenContextual
    ChatEngine --> ChatSearch
    
    GenContextual --> BuildHistory
    GenContextual --> RAGRetrieval
    GenContextual --> LLMCall
    GenContextual --> StreamAsync
    
    ChatSearch --> SearchFirst
    ChatSearch --> IncludeResults
    ChatSearch --> EnhancedPrompt
    
    ChatEngine -.->|depends on| SessionMgr
    SessionMgr --> Storage
    SessionMgr --> MaxSessions
    
    SessionMgr --> CreateSession
    SessionMgr --> GetSession
    SessionMgr --> AddMessage
    SessionMgr --> GetHistory
    SessionMgr --> AddContent
    SessionMgr --> GetRAGContext
    
    SessionMgr -.->|manages| SessionData
    SessionData --> SessionID
    SessionData --> History
    SessionData --> FetchedURLs
    SessionData --> SearchURLs
    SessionData --> YouTubeURLs
    SessionData --> ToolCalls
    SessionData --> Embeddings
    SessionData --> LastActivity
    
    SessionData --> GetRAGCtx
    SessionData --> GetTopContent
    SessionData --> Memory
    
    style ChatEngine fill:#FCE4EC
    style SessionMgr fill:#F3E5F5
    style SessionData fill:#E1BEE7
    style StreamAsync fill:#C2185B
```

---

### Layer 6: IPC Service Layer (Optional Distributed)

```mermaid
graph LR
    Main["Main API Server<br/>:8000"]
    SearchingService["searching/main.py<br/>Service Facade"]
    
    IPCClient["IPC Client<br/>LocalHost:5010"]
    IPCConnection{"IPC Connection<br/>Active?"}
    
    CoreService["CoreEmbeddingService<br/>ipcService/"]
    InstanceID["_instance_id<br/>unique service ID"]
    
    EmbedServiceDeployed["EmbeddingService<br/>Deployed"]
    VectorStoreDeployed["VectorStore<br/>Deployed"]
    SemanticCacheDeployed["SemanticCache<br/>Deployed"]
    RetPipelineDeployed["RetrievalPipeline<br/>Deployed"]
    
    IngestURL["ingest_url<br/>url -> chunks"]
    RetrieveQuery["retrieve<br/>query, top_k"]
    BuildContext["build_retrieval_context"]
    GetStats["get_stats<br/>-> metrics"]
    
    ThreadPool["ThreadPoolExecutor<br/>max_workers=2"]
    GPULock["GPU Lock<br/>Safe Access"]
    PersistWorker["Persistence Thread<br/>Background"]
    
    LocalFallback["Local Services<br/>Fallback"]
    LocalEmbed["Local Embedding<br/>Service"]
    LocalVector["Local Vector<br/>Store"]
    
    Main --> SearchingService
    SearchingService --> IPCClient
    IPCClient --> IPCConnection
    
    IPCConnection -->|YES| CoreService
    IPCConnection -->|NO| LocalFallback
    
    CoreService --> InstanceID
    CoreService --> EmbedServiceDeployed
    CoreService --> VectorStoreDeployed
    CoreService --> SemanticCacheDeployed
    CoreService --> RetPipelineDeployed
    
    CoreService --> IngestURL
    CoreService --> RetrieveQuery
    CoreService --> BuildContext
    CoreService --> GetStats
    
    CoreService --> ThreadPool
    CoreService --> GPULock
    CoreService --> PersistWorker
    
    LocalFallback --> LocalEmbed
    LocalFallback --> LocalVector
    
    style Main fill:#E3F2FD
    style CoreService fill:#E8EAF6
    style IPCConnection fill:#FFF9C4
    style LocalFallback fill:#C8E6C9
```

---

## Core Components

### 1. Request ID & Tracing
- **requestID.py**: Middleware injects X-Request-ID header
- **Lifetime**: Passed through all layers for observability
- **Format**: UUID truncated to N characters

### 2. Instruction Set
- **system_instruction**: System behavior & constraints
- **user_instruction**: User input formatting
- **synthesis_instruction**: LLM response synthesis rules

### 3. Tools & Function Calls
```
tools.py:
├─ Web Search Tools
│  └─ playwright_web_search(query) → results
├─ Content Retrieval
│  └─ fetch_full_text(url) → cleaned text
├─ External APIs
│  ├─ getYoutubeDetails(url) → metadata
│  ├─ getImagePrompt(image_url) → analysis
│  ├─ generateImage(prompt) → image URL
│  └─ getTimeZone(location) → timezone
└─ RAG Tools
   ├─ retrieve_from_vector_store(query, k)
   └─ ingest_url_to_vector_store(url)
```

### 4. Observability & Monitoring
- **commons/observabilityMonitoring.py**: Metrics collection
- **commons/robustnessFramework.py**: Failure tracking
- **commons/gracefulDegradation.py**: Degradation analysis

---

## Data Flow

### Complete Request Flow: "/api/search"
```mermaid
sequenceDiagram
  actor User
  participant Gateway as API Gateway<br/>gateways/search.py
  participant Pipeline as SearchPipeline<br/>searchPipeline.py
  participant Tools as Tool Execution<br/>optimized_tool_execution
  participant RAG as RAG Engine<br/>ragEngine.py
  participant LLM as ChatEngine +<br/>Pollinations API
  participant Session as SessionManager<br/>sessionManager.py
  participant Client as Client<br/>SSE Stream

  User->>Gateway: 1. POST /api/search<br/>{query, image_url, stream=true}
  Gateway->>Gateway: Validate query & image_url<br/>Extract X-Request-ID header
  Gateway->>Pipeline: Route to pipeline

  Pipeline->>Pipeline: 2a. Clean query & extract URLs
  Pipeline->>Session: 2b. Create session
  Session-->>Pipeline: session_id
  Pipeline->>Pipeline: 2c. Decompose query if complex

  Pipeline->>Tools: 2d. Parallel tool execution
  par Web Search
    Tools->>Tools: Playwright web search
  and Fetch Content
    Tools->>Tools: Fetch full text (BeautifulSoup)
  and YouTube Metadata
    Tools->>Tools: YouTube API call
  and Image Analysis
    Tools->>Tools: Image analysis (if provided)
  end
  Tools-->>Pipeline: Search results aggregated

  Pipeline->>RAG: 3. retrieve_context(query)
  RAG->>RAG: 3a. Embed query (embeddingService)
  RAG->>RAG: 3b. Check semantic cache per URL
  alt Cache Hit
    RAG-->>RAG: Return cached_response
  else Cache Miss
    RAG->>RAG: 3c. Search vector store (ChromaDB)
    RAG->>RAG: 3d. Combine with session memory
    RAG->>RAG: 3e. Cache result (semanticCache)
  end
  RAG-->>Pipeline: RAG context retrieved

  Pipeline->>LLM: 4. generate_contextual_response()
  LLM->>LLM: 4a. Build message history
  LLM->>LLM: 4b. Format system prompt
  LLM->>LLM: 4c. Include RAG context
  LLM->>LLM: 4d. POST to Pollinations API
  LLM->>LLM: 4e. Parse response

  LLM-->>Client: 5. Stream SSE events<br/>(info, final-part, final, error)
  Client-->>User: Response chunks in real-time

  Pipeline->>Session: 6. Update session<br/>Store response in history
  Session->>Session: Log metrics & TTL tracking

  User->>User: 7. USER RECEIVES<br/>STREAMED RESPONSE
```

## Request Lifecycle

### Example: Multi-turn Chat Session

```mermaid
graph TD
    Step1["Step 1: Create Session<br/>POST /api/session/create"]
    Step1Out["Response:<br/>session_id: 'abc123'"]
    
    Step2["Step 2: First Chat Turn<br/>POST /api/session/abc123/chat<br/>message: 'What are latest AI news?'"]
    Step2Proc["Process:<br/>• Tool execution<br/>• RAG context retrieval<br/>• LLM synthesis<br/>• Stream response"]
    Step2Out["Response:<br/>SSE event stream"]
    Step2Update["Update:<br/>add_message_to_history"]
    
    Step3["Step 3: Follow-up Turn<br/>POST /api/session/abc123/chat<br/>message: 'Can you summarize that?'"]
    Step3Proc["Process:<br/>• References previous conversation<br/>• RAG includes prior context<br/>• Continuity-aware LLM<br/>• Memory embeddings"]
    Step3Out["Response:<br/>SSE event stream"]
    
    Step4["Step 4: Get Session Info<br/>GET /api/session/abc123"]
    Step4Out["Response:<br/>metadata, history, tool calls"]
    
    Step5["Step 5: Clean Up<br/>DELETE /api/session/abc123"]
    Step5Out["Release memory<br/>cleanup_session"]
    
    Step1 --> Step1Out
    Step2 --> Step2Proc
    Step2Proc --> Step2Out
    Step2Out --> Step2Update
    Step2Update --> Step3
    Step3 --> Step3Proc
    Step3Proc --> Step3Out
    Step3Out --> Step4
    Step4 --> Step4Out
    Step4Out --> Step5
    Step5 --> Step5Out
    
    style Step1 fill:#E3F2FD
    style Step2 fill:#E8EAF6
    style Step3 fill:#F3E5F5
    style Step4Out fill:#E8F5E9
    style Step5Out fill:#FFCCBC
```

---

## Integration Architecture

### Component Dependency Graph

```mermaid
graph TB
    API[API Gateway<br/>Quart/Hypercorn]
    Pipeline[SearchPipeline]
    LixSearch[lixsearch.py<br/>Main Orchestrator]
    
    API -->|routes to| Pipeline
    Pipeline -->|executes| LixSearch
    
    LixSearch -->|parallel execution| ToolExec[optimized_tool_execution]
    ToolExec -->|uses| WebSearch[playwright_web_search]
    ToolExec -->|uses| FetchFull[fetch_full_text]
    ToolExec -->|uses| Tools[tools.py<br/>function calls]
    
    LixSearch -->|retrieves context| RAGEngine[RAG Engine]
    RAGEngine -->|checks| SemanticCache
    RAGEngine -->|searches| VectorStore[VectorStore<br/>ChromaDB]
    RAGEngine -->|retrieves| RetrievalPipeline
    
    RetrievalPipeline -->|embeds| EmbeddingService[EmbeddingService<br/>SentenceTransformer]
    RetrievalPipeline -->|chunks text| ChunkUtil[commons/minimal.py]
    
    LixSearch -->|synthesizes with| ChatEngine[ChatEngine]
    ChatEngine -->|accesses| SessionMgr[SessionManager]
    SessionMgr -->|stores| SessionData[SessionData]
    
    ChatEngine -->|calls LLM| Pollinations[Pollinations API<br/>LLM Backend]
    
    ToolExec -->|IPC connection| IPC[CoreEmbeddingService<br/>IPC on :5010]
    IPC -->|fallback to local| RetrievalPipeline
    
    style API fill:#E3F2FD
    style Pipeline fill:#F3E5F5
    style LixSearch fill:#FFF3E0
    style RAGEngine fill:#E8F5E9
    style ChatEngine fill:#FCE4EC
```

---

## Deployment Model

### Single-Process Deployment (Default)

```mermaid
graph TB
    Client["👤 Client<br/>HTTP/WebSocket"]
    
    Process["Single Python Process<br/>lixSearch API"]
    
    QuartApp["Quart App<br/>Async Server<br/>0.0.0.0:8000"]
    SearchPipe["SearchPipeline"]
    ChatEng["ChatEngine"]
    SessionMgr["SessionManager"]
    ErrorHandler["Error Handlers"]
    
    RAGServices["RAG Services<br/>Same Process"]
    RAGEngine["RAGEngine"]
    EmbedService["EmbeddingService"]
    VectorStore["VectorStore<br/>ChromaDB"]
    SemanticCache["SemanticCache"]
    
    SearchServices["Search Services<br/>Same Process"]
    Playwright["Playwright<br/>Browser"]
    HTTPClients["HTTP Clients"]
    ToolExec["Tool Executors"]
    
    ExternalAPIs["External APIs<br/>HTTP"]
    Pollinations["Pollinations<br/>LLM"]
    YouTubeAPI["YouTube API"]
    ImageAPIs["Image APIs"]
    
    Client --> QuartApp
    
    QuartApp --> SearchPipe
    QuartApp --> ChatEng
    QuartApp --> SessionMgr
    QuartApp --> ErrorHandler
    
    SearchPipe --> RAGServices
    ChatEng --> RAGServices
    
    RAGServices --> RAGEngine
    RAGServices --> EmbedService
    RAGServices --> VectorStore
    RAGServices --> SemanticCache
    
    SearchPipe --> SearchServices
    SearchServices --> Playwright
    SearchServices --> HTTPClients
    SearchServices --> ToolExec
    
    EmbedService -.-> ExternalAPIs
    ToolExec -.-> ExternalAPIs
    ChatEng -.-> Pollinations
    ToolExec -.-> YouTubeAPI
    ToolExec -.-> ImageAPIs
    
    style Process fill:#E3F2FD
    style QuartApp fill:#BBDEFB
    style RAGServices fill:#C8E6C9
    style SearchServices fill:#FFE0B2
    style ExternalAPIs fill:#F5F5F5
```

### Distributed Deployment (Optional IPC)

```mermaid
graph TB
    Client["👤 Client"]
    
    MainServer["Main API Server<br/>:8000<br/>Process 1"]
    SearchPipe["SearchPipeline"]
    ChatEng["ChatEngine"]
    SessionMgr["SessionManager"]
    
    IPCNetwork["IPC Network<br/>localhost:5010<br/>RPC Call"]
    
    EmbedProcess["Embedding Service<br/>:5010<br/>Process 2<br/>Separate Process"]
    CoreService["CoreEmbeddingService"]
    EmbedService2["EmbeddingService"]
    VectorStore2["VectorStore"]
    SemanticCache2["SemanticCache"]
    RetPipeline2["RetrievalPipeline"]
    
    Client --> MainServer
    MainServer --> SearchPipe
    MainServer --> ChatEng
    MainServer --> SessionMgr
    
    SearchPipe -->|retrieval| IPCNetwork
    ChatEng -->|context| IPCNetwork
    
    IPCNetwork --> EmbedProcess
    EmbedProcess --> CoreService
    CoreService --> EmbedService2
    CoreService --> VectorStore2
    CoreService --> SemanticCache2
    CoreService --> RetPipeline2
    
    Benefits["Benefits:<br/>✓ GPU isolation<br/>✓ Independent scaling<br />✓ Memory separation<br/>✓ Fallback on failure"]
    
    EmbedProcess -.-> Benefits
    
    style MainServer fill:#E3F2FD
    style EmbedProcess fill:#E8F5E9
    style IPCNetwork fill:#FFF9C4
    style Benefits fill:#C8E6C9
```

---


## Key Features & Guarantees

### Performance
- **Cache Hit Latency**: 5-15ms (conversation/semantic)
- **Web Search Latency**: 500-2000ms
- **Vector Search**: 10-50ms (ChromaDB HNSW)
- **Streaming**: Real-time SSE chunks

### Reliability
- Graceful degradation if components fail
- Fallback: IPC → local services
- Request ID tracing across all layers
- Comprehensive error handling

### Scalability
- Session expiry (30m TTL) prevents memory leak
- Cache cleanup on startup and runtime
- Batch embeddings (configurable)
- Parallel tool execution

### Privacy & Safety
- Internal reasoning filtering
- User-friendly task messages
- No leaking of system prompts
- Per-request isolation

---

## System Architecture Diagram

```mermaid
graph TB
    User["👤 User<br/>HTTP/WebSocket"]
    
    subgraph API["API Layer"]
        Gateway["Quart Gateway"]
        Middleware["RequestID Middleware"]
        Routes["Routes<br/>search, chat, session, stats"]
    end
    
    subgraph Pipeline["Pipeline & Orchestration"]
        SearchPipeline["SearchPipeline"]
        LixSearch["lixsearch.py<br/>Main Orchestrator"]
        ToolExec["optimized_tool_execution"]
        Decompose["queryDecomposition"]
    end
    
    subgraph Search["Search & Fetch"]
        WebSearch["playwright_web_search"]
        FetchText["fetch_full_text"]
        Tools["function_calls<br/>YouTube, Image, etc"]
    end
    
    subgraph RAG["RAG Service"]
        RAGEngine["RAGEngine"]
        SemanticCache["SemanticCache<br/>URL-bucketed"]
        EmbedService["EmbeddingService<br/>SentenceTransformer"]
        VecStore["VectorStore<br/>ChromaDB HNSW"]
        RetPipeline["RetrievalPipeline"]
    end
    
    subgraph Chat["Chat & Session"]
        ChatEngine["ChatEngine"]
        SessionMgr["SessionManager"]
        SessionData["SessionData"]
    end
    
    subgraph LLM["External"]
        Pollinations["Pollinations API<br/>LLM Inference"]
    end
    
    User -->|HTTP POST| Gateway
    Gateway --> Middleware
    Middleware --> Routes
    Routes -->|/search| SearchPipeline
    Routes -->|/chat| ChatEngine
    Routes -->|/session| SessionMgr
    
    SearchPipeline --> LixSearch
    LixSearch --> Decompose
    LixSearch --> ToolExec
    
    ToolExec -->|web search| WebSearch
    ToolExec -->|fetch| FetchText
    ToolExec -->|calls| Tools
    
    LixSearch --> RAGEngine
    RAGEngine -->|check| SemanticCache
    RAGEngine -->|miss| VecStore
    VecStore -->|depends on| EmbedService
    VecStore -->|depends on| RetPipeline
    
    RAGEngine -->|context| LixSearch
    LixSearch -->|synthesize| ChatEngine
    ChatEngine -->|context| SessionMgr
    SessionMgr -->|store| SessionData
    
    ChatEngine -->|prompt + context| Pollinations
    Pollinations -->|response| ChatEngine
    
    style User fill:#B3E5FC
    style API fill:#C8E6C9
    style Pipeline fill:#FFE0B2
    style Search fill:#F8BBD0
    style RAG fill:#E1BEE7
    style Chat fill:#FFCCBC
    style LLM fill:#FFF9C4
```

---

## Summary

**lixSearch** is a modern, production-ready search system with:

 **Layered Architecture**: API → Pipeline → RAG → Search → Chat → Session
 **Streaming Responses**: Real-time SSE for user feedback
 **Semantic Caching**: 0.90+ similarity detection with adaptive thresholds
 **Parallel Execution**: Tools run concurrently for speed
 **Context Awareness**: Full conversation history + session memory
 **Cost Optimization**: Token counting, context compression, cache savings
 **Graceful Degradation**: Works even if components fail
 **Scalable Design**: Session TTL prevents memory bloat
 **Observable**: Request tracing via X-Request-ID throughout

The system achieves **sub-100ms cache hits**, **500-2000ms web search**, and **20-30% cost savings** through intelligent resource allocation.
