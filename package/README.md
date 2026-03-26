# lix-open-search

Python client SDK for [lixSearch](https://github.com/elixpo/lixSearch) — a multi-tool AI search engine with web search, video, images, deep research, and RAG-augmented synthesis.

```bash
pip install lix-open-search
```

## Quick Start

### One-shot search

```python
from lix_open_search import LixSearch

lix = LixSearch("http://localhost:9002")

result = lix.search("quantum computing breakthroughs 2026")
print(result.content)
```

### Streaming

```python
for chunk in lix.search_stream("latest advances in fusion energy"):
    print(chunk.content, end="", flush=True)
```

### Multi-turn conversation

```python
lix = LixSearch("http://localhost:9002")

# First turn
result = lix.chat([
    {"role": "user", "content": "Compare Tesla and BYD sales in 2025"}
], session_id="my-session")
print(result.content)

# Follow-up (session remembers context)
result = lix.chat([
    {"role": "user", "content": "Compare Tesla and BYD sales in 2025"},
    {"role": "assistant", "content": result.content},
    {"role": "user", "content": "What about their market cap?"}
], session_id="my-session")
print(result.content)
```

### Image + text (multimodal)

```python
result = lix.search(
    "What building is this and when was it built?",
    images=["https://example.com/photo.jpg"]
)
```

### Surf (raw URLs, no LLM)

```python
result = lix.surf("best Python testing frameworks", limit=10, images=True)
print(result.urls)    # ['https://...', ...]
print(result.images)  # ['https://...', ...]
```

### Async

```python
import asyncio
from lix_open_search import AsyncLixSearch

async def main():
    async with AsyncLixSearch("http://localhost:9002") as lix:
        result = await lix.search("SpaceX Starship updates")
        print(result.content)

        async for chunk in lix.search_stream("latest AI papers"):
            print(chunk.content, end="", flush=True)

asyncio.run(main())
```

## API Reference

### `LixSearch(base_url, api_key=None, timeout=120)`

| Method | Returns | Description |
|--------|---------|-------------|
| `search(query, session_id=, images=)` | `SearchResult` | Search with full LLM synthesis |
| `search_stream(query, session_id=, images=)` | `Iterator[StreamChunk]` | Streaming search |
| `chat(messages, session_id=, stream=)` | `SearchResult` or `Iterator` | Multi-turn conversation |
| `surf(query, limit=5, images=False)` | `SurfResult` | Raw URL + image search (no LLM) |
| `create_session(query)` | `Session` | Create a persistent session |
| `get_session(session_id)` | `Session` | Get session info |
| `get_history(session_id)` | `list[Message]` | Get conversation history |
| `delete_session(session_id)` | `None` | Delete a session |
| `health()` | `dict` | Health check |

### `AsyncLixSearch(base_url, api_key=None, timeout=120)`

Same methods as `LixSearch`, but `async`. Use `async with` as context manager.

### Response Models

**`SearchResult`** — LLM-synthesized response
- `.content` — response text (markdown)
- `.model` — model name
- `.session_id` — session ID if provided
- `.usage` — token usage stats
- `.raw` — full OpenAI-format response dict

**`StreamChunk`** — single streaming token
- `.content` — text fragment
- `.finish_reason` — `None` or `"stop"`
- `.raw` — raw SSE chunk

**`SurfResult`** — raw search results
- `.urls` — list of URLs
- `.images` — list of image URLs
- `.query` — the query

**`Session`** / **`Message`** — session and message objects

## OpenAI Compatibility

lixSearch exposes an OpenAI-compatible API. You can also use the standard OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9002/v1", api_key="unused")

response = client.chat.completions.create(
    model="lixsearch",
    messages=[{"role": "user", "content": "latest news on AI regulation"}],
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Running the Server

The SDK connects to a running lixSearch instance. Deploy it with Docker:

```bash
git clone https://github.com/elixpo/lixSearch.git
cd lixSearch
cp .env.example .env  # fill in TOKEN, MODEL, HF_TOKEN
./deploy.sh build
./deploy.sh start 3
```

Or point the client at the hosted instance:

```python
lix = LixSearch("https://search.elixpo.com", api_key="your-key")
```

## License

MIT
