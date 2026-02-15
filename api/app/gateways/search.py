import logging
import uuid
import json
from datetime import datetime, timezone
from quart import request, jsonify, Response
from pipeline.searchPipeline import run_elixposearch_pipeline
from app.utils import validate_query, validate_url, format_openai_response
import os

logger = logging.getLogger("lixsearch-api")


async def search(pipeline_initialized: bool):
    """Search endpoint with optional streaming.
    
    Parameters:
    - query: Search query (required)
    - image_url: Optional image URL for image search
    - stream: Whether to stream SSE events (default: true)
      - stream=true: Returns Server-Sent Events with real-time updates
      - stream=false: Returns single OpenAI-format JSON response
    """
    if not pipeline_initialized:
        return jsonify({"error": "Server not initialized"}), 503

    try:
        if request.method == 'GET':
            query = request.args.get("query", "").strip()
            image_url = request.args.get("image_url")
            stream_param = request.args.get("stream", "true").lower()
        else:
            data = await request.get_json()
            query = data.get("query", "").strip()
            image_url = data.get("image_url")
            stream_param = str(data.get("stream", "true")).lower()

        # Parse stream parameter (default True)
        stream_mode = stream_param not in ("false", "0", "no")

        if not validate_query(query):
            return jsonify({"error": "Invalid or missing query"}), 400

        if image_url and not validate_url(image_url):
            return jsonify({"error": "Invalid image_url"}), 400

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])

        logger.info(f"[{request_id}] Search: {query[:50]}... [stream={stream_mode}]")

        # Streaming mode: SSE with real-time events
        if stream_mode:
            async def event_stream_generator():
                async for chunk in run_elixposearch_pipeline(
                    user_query=query,
                    user_image=image_url,
                    event_id=request_id,  # Enable SSE format
                    request_id=request_id
                ):
                    yield chunk.encode('utf-8') if isinstance(chunk, str) else chunk

            return Response(
                event_stream_generator(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Content-Type': 'text/event-stream',
                    'Access-Control-Allow-Origin': '*'
                }
            )
        
        # Non-streaming mode: Single JSON response
        else:
            response_content = None
            async for chunk in run_elixposearch_pipeline(
                user_query=query,
                user_image=image_url,
                event_id=None,  # Disable SSE format, yield raw content
                request_id=request_id
            ):
                response_content = chunk

            if not response_content:
                return jsonify({"error": "No response generated"}), 500

            openai_response = format_openai_response(response_content, request_id)

            return Response(
                openai_response,
                mimetype='application/json',
                headers={
                    'Cache-Control': 'no-cache',
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                }
            )

    except Exception as e:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])
        logger.error(f"[{request_id}] Search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
