# ElixpoSearch API

A production-grade RAG (Retrieval-Augmented Generation) search API powered by LLMs, vector embeddings, and intelligent web search.

## Quick Start

```bash
# Activate virtual environment
source searchenv/bin/activate

# Start the API
cd api
python app.py

# API available at: http://localhost:8000
# Health check: http://localhost:8000/api/health
```

## Features

- **Intelligent Search**: Web search + RAG-based context retrieval
- **Session Management**: Persistent conversations with semantic caching
- **Chat Interface**: Streaming responses with source attribution
- **Multi-modal Support**: Image analysis and audio transcription
- **Knowledge Graphs**: Dynamic entity extraction and relationship mapping
- **Async Processing**: Fast, non-blocking request handling

## Architecture Overview

```
                        ┌─────────────┐
                        │   Client    │
                        │ (REST/WS)   │
                        └──────┬──────┘
                               │ HTTP/WS
                        ┌──────▼──────┐
                        │  Quart API  │◄─── Handles requests
                        │  (app.py)   │
                        └──────┬──────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
    ┌────────┐         ┌─────────────┐        ┌───────────┐
    │Pipeline│         │  RAGService │        │  Sessions │
    │        │         │             │        │ & Chat    │
    └────┬───┘         └────┬────────┘        └─────┬─────┘
         │                  │                       │
         │ (searches)       │ (embeds & retrieves)  │
         ▼                  ▼                       ▼
     ┌───────────────────────────┬───────────────────────┐
     │       IPC Service         │   SessionManager      │
     │ (subprocess)              │                       │
     │                           │                       │
     │ • CoreEmbeddingService    │ • Session CRUD        │
     │ • SearchAgents            │ • History management  │
     │ • Chroma DB (vectors)     │ • TTL cleanup         │
     └───────────────────────────┴───────────────────────┘
```

## Core Modules

| Module | Purpose |
|--------|---------|
| **app.py** | Main entry point, route handlers, server startup |
| **pipeline/** | Search pipeline orchestration & LLM tools |
| **ragService/** | Embedding, vector store, retrieval |
| **ipcService/** | IPC subprocess for heavy operations |
| **sessions/** | Session & conversation management |
| **chatEngine/** | Chat response generation |
| **commons/** | Shared utilities (search, IPC) |
| **searching/** | Web scraping and content extraction |
| **functionCalls/** | Image analysis, YouTube, timezone |

## API Endpoints

### Search
```bash
POST /api/search
Content-Type: application/json

{
  "query": "latest AI trends",
  "image_url": "https://example.com/image.jpg" (optional),
  "session_id": "abc123" (optional)
}

# Response: Server-Sent Events (SSE) stream
```

### Sessions
```bash
# Create session
POST /api/session/create
{"query": "my search query"}

# Get session info
GET /api/session/{session_id}

# Delete session
DELETE /api/session/{session_id}

# Get chat history
GET /api/session/{session_id}/history
```

### Chat
```bash
# Chat with search
POST /api/chat
{
  "message": "What does this image show?",
  "image_url": "https://example.com/img.jpg",
  "session_id": "abc123"
}

# Session-specific chat (uses session context)
POST /api/session/{session_id}/chat
{"message": "Tell me more"}

# OpenAI-compatible completions
POST /api/session/{session_id}/chat/completions
{
  "messages": [{"role": "user", "content": "..."}],
  "stream": true
}
```

### Context Management
```bash
# Get knowledge graph
GET /api/session/{session_id}/kg

# Query knowledge graph
POST /api/session/{session_id}/query
{"query": "entity search", "top_k": 5}

# Get entity evidence
GET /api/session/{session_id}/entity/{entity_name}

# Get session summary/stats
GET /api/session/{session_id}/summary
```

### WebSocket
```bash
ws://localhost:8000/ws/search

# Send: {"query": "...", "image_url": "..."}
# Receive: {event, data, request_id}
```

## Configuration

Edit `pipeline/config.py`:

```python
# Model & API
POLLINATIONS_ENDPOINT = "https://gen.pollinations.ai/v1/chat/completions"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_DIR = "./embeddings"

# Session management
MAX_SESSIONS = 1000
SESSION_TTL_MINUTES = 30

# Search & retrieval
SEARCH_MAX_RESULTS = 8
RETRIEVAL_TOP_K = 5
RAG_CONTEXT_REFRESH = True

# Caching
SEMANTIC_CACHE_TTL_SECONDS = 3600
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = 0.95
```

## Key Design Decisions

1. **IPC Service**: Heavy operations (embeddings, web search) run in a separate Python subprocess for isolation
2. **Streaming Responses**: SSE (Server-Sent Events) for real-time result streaming
3. **Semantic Caching**: Similarity-based caching reduces redundant computations
4. **Session-Based Context**: Each conversation maintains its own vector store and RAG context
5. **Tool-Based LLM**: Structured tool calling for reliable function execution

## Performance

- **Startup**: ~5 seconds (IPC service + model initialization)
- **First Search**: ~10-15 seconds (cold start, model loading)
- **Subsequent Searches**: ~5-8 seconds (with IPC server warm)
- **Cache Hits**: ~1-2 seconds (semantic cache matches)

## Troubleshooting

### IPC Service won't start
```bash
# Check if port 5010 is in use
lsof -i :5010

# Manually start for debugging
cd api/ipcService && python main.py
```

### Model inference timeout
```python
# Increase timeout in pipeline/config.py
REQUEST_TIMEOUT = 300  # seconds
```

### Vector store issues
```bash
# Rebuild vector store
rm -rf embeddings/
# Restart app - it will rebuild on first run
```

## Development

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run type checking
mypy api/

# Check imports
python -m compileall api/

# Monitor logs
tail -f output.log
```

## Files & Imports Reference

After refactoring, all imports follow this pattern:
- **From root api modules**: `from pipeline.config import ...`
- **From subfolders**: `from moduleName.submodule import ...`
- **No circular imports**: Tools avoid importing from Pipeline/RAG back to themselves

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed module dependencies.

## License

Part of ElixpoSearch - See repo LICENSE file

