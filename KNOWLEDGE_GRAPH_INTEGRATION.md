# Knowledge Graph Integration with Request ID Tracking

## Overview

This implementation integrates the knowledge graph system with request ID tracking, allowing the search pipeline to build, store, and retrieve knowledge graphs for specific queries. Each search request now automatically builds knowledge graphs from fetched URLs and stores them indexed by request_id for future reference and query optimization.

## Architecture

### Components

1. **kg_manager.py** - New module for managing knowledge graphs by request_id
   - Thread-safe KG Manager with LRU cache
   - Aggregates KGs across multiple URLs in a single request
   - Provides query context building
   - Auto-cleanup of expired entries

2. **search.py** - Updated fetch_full_text function
   - Now accepts optional `request_id` parameter
   - Builds knowledge graphs and stores them in KG Manager
   - Maintains backward compatibility

3. **utility.py** - Updated fetch_url_content_parallel
   - Passes request_id through the parallel fetch pipeline
   - Collects KG data from all fetched URLs

4. **searchPipeline.py** - Updated run_elixposearch_pipeline
   - Accepts request_id parameter
   - Stores request_id in memoized_results
   - Passes request_id through tool execution

5. **app.py** - Updated main application
   - Passes request_id from RequestTask to search pipeline
   - Provides REST API endpoints for KG retrieval and management

## Data Flow

```
Request → app.py (RequestTask with request_id)
    ↓
run_elixposearch_pipeline(request_id=task.request_id)
    ↓
fetch_url_content_parallel(request_id=request_id)
    ↓
fetch_full_text(url, request_id=request_id)
    ↓
build_knowledge_graph() → kg_manager.add_kg(request_id, url, text, kg)
    ↓
Knowledge Graph stored indexed by request_id
```

## KG Manager API

### Core Methods

#### `add_kg(request_id, url, text, kg)`
Stores a knowledge graph for a specific request and URL.

```python
kg_manager.add_kg(request_id="abc123", url="https://example.com", text=content, kg=kg_object)
```

#### `get_request_kg(request_id, url=None)`
Retrieves knowledge graph(s) for a request.
- If url provided: returns KG for that specific URL
- If url is None: returns aggregated KG across all URLs

```python
# Get aggregated KG
kg_data = kg_manager.get_request_kg("abc123")

# Get KG for specific URL
kg_data = kg_manager.get_request_kg("abc123", url="https://example.com")
```

#### `get_top_entities(request_id, top_k=15)`
Returns top-k most important entities across all KGs in a request.

```python
entities = kg_manager.get_top_entities("abc123", top_k=10)
# Returns: [("Apple", 0.95), ("CEO", 0.87), ...]
```

#### `get_entity_relationships(request_id, entity)`
Returns all relationships involving a specific entity.

```python
rels = kg_manager.get_entity_relationships("abc123", "Apple")
# Returns: [("apple", "founded_by", "steve jobs"), ...]
```

#### `build_query_context(request_id)`
Builds rich context string from all KGs for query understanding.

```python
context = kg_manager.build_query_context("abc123")
# Returns formatted context with entities and relationships
```

#### `export_request_kg(request_id)`
Exports complete KG data with metadata for a request.

```python
export = kg_manager.export_request_kg("abc123")
# Contains: request_id, metadata, graphs, top_entities, export_time
```

#### `get_stats()`
Returns manager statistics.

```python
stats = kg_manager.get_stats()
# Returns: {total_requests, max_cache_size, storage_size}
```

## REST API Endpoints

### 1. GET `/kg/request/<request_id>`
Get aggregated knowledge graph for a specific request.

**Response:**
```json
{
  "request_id": "abc123",
  "metadata": {
    "created_at": "2024-01-25T10:30:00",
    "urls": ["https://example.com", "https://example2.com"],
    "total_entities": 45,
    "total_relationships": 120
  },
  "knowledge_graph": {
    "entities": {...},
    "relationships": [...],
    "importance_scores": {...},
    "entity_graph": {...}
  }
}
```

### 2. GET `/kg/request/<request_id>/entities?top_k=15`
Get top-k entities for a request.

**Query Parameters:**
- `top_k` (optional, default=15): Number of top entities to return

**Response:**
```json
{
  "request_id": "abc123",
  "top_entities": [
    ["apple", 0.95],
    ["steve jobs", 0.87],
    ["technology", 0.82]
  ]
}
```

### 3. GET `/kg/request/<request_id>/context`
Get query context built from knowledge graphs.

**Response:**
```json
{
  "request_id": "abc123",
  "context": "Key entities identified:\n- apple (relevance: 0.95)\n  Relationships: 12 connections\n- steve jobs (relevance: 0.87)\n  Relationships: 8 connections\n..."
}
```

### 4. GET `/kg/request/<request_id>/export`
Export complete knowledge graph data with metadata.

**Response:**
```json
{
  "request_id": "abc123",
  "metadata": {...},
  "graphs": {
    "https://example.com": {...},
    "https://example2.com": {...}
  },
  "top_entities": [...],
  "export_time": "2024-01-25T10:31:00"
}
```

### 5. GET `/kg/manager/stats`
Get KG manager statistics.

**Response:**
```json
{
  "total_requests": 42,
  "max_cache_size": 100,
  "storage_size": 85
}
```

### 6. DELETE `/kg/request/<request_id>`
Clear knowledge graph data for a specific request.

**Response:**
```json
{
  "message": "Cleared knowledge graph for request abc123"
}
```

## Usage Examples

### Python Integration

```python
# In your search pipeline code
from kg_manager import kg_manager

# After search completes, retrieve the KG
request_id = "abc123"
top_entities = kg_manager.get_top_entities(request_id, top_k=10)

# Use entities for query refinement
context = kg_manager.build_query_context(request_id)
print(f"Query context:\n{context}")

# Export for storage/analysis
export = kg_manager.export_request_kg(request_id)
```

### Using REST API

```bash
# Get top entities for a request
curl http://localhost:5000/kg/request/abc123/entities?top_k=10

# Get query context
curl http://localhost:5000/kg/request/abc123/context

# Export complete KG
curl http://localhost:5000/kg/request/abc123/export

# Get manager stats
curl http://localhost:5000/kg/manager/stats

# Cleanup a request's KG
curl -X DELETE http://localhost:5000/kg/request/abc123
```

## Key Features

1. **Request-scoped Storage**: Each search request has its own knowledge graph namespace
2. **Parallel URL Processing**: KGs built in parallel during URL fetching
3. **Automatic Aggregation**: Multiple KGs automatically merged for cross-URL entity relationships
4. **Auto-cleanup**: Expired KGs removed automatically (configurable TTL)
5. **Thread-safe**: Uses locks for concurrent access
6. **Query Context Building**: Automatically generates rich context from entities and relationships
7. **REST API**: Full HTTP interface for programmatic access

## Configuration

The KG Manager can be configured when instantiated:

```python
from kg_manager import KGManager

# Create custom manager instance
kg_mgr = KGManager(
    max_cache_size=100,      # Max requests to store
    ttl_hours=2               # Time-to-live for cached KGs
)
```

## Performance Considerations

- **Memory**: Each cached KG consumes ~10-50KB depending on entity/relationship count
- **TTL**: Default 2 hours; older KGs automatically removed
- **Max Cache**: Default 100 requests; oldest entries removed when exceeded
- **Concurrent Access**: Thread-safe for up to 15 concurrent requests (app.py default)

## Future Enhancements

1. **Persistent Storage**: Save KGs to database for long-term retention
2. **KG Merging**: Merge KGs across multiple requests for cross-request learning
3. **Entity Linking**: Link entities across requests to build global knowledge base
4. **Query Rewriting**: Use KG to automatically refine and expand queries
5. **Relationship Extraction**: Enhanced relationship type detection
6. **Custom Importance Scoring**: Configurable scoring algorithms

## Backward Compatibility

All changes are backward compatible:
- `request_id` parameters are optional
- Existing code without request IDs continues to work
- Knowledge graphs still returned in search results
- No breaking changes to existing APIs

## Testing

The implementation includes:
- Thread-safe operations tested with concurrent access
- Memory-efficient caching with TTL-based cleanup
- Error handling for all edge cases
- Graceful fallbacks when KG building fails

## Files Modified/Created

### Created:
- `api/kg_manager.py` - New KG Manager module

### Modified:
- `api/search.py` - Added request_id parameter to fetch_full_text
- `api/utility.py` - Added request_id parameter to fetch_url_content_parallel
- `api/searchPipeline.py` - Added request_id to pipeline and tool execution
- `api/app.py` - Updated to pass request_id and added REST API endpoints

### Backward Compatible:
- `api/web_scraper.py` - No changes needed (re-exports from search.py)
- `api/knowledge_graph.py` - No changes needed (core KG building logic)
