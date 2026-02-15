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
    """Search endpoint - supports both POST and GET requests, returns OpenAI-format JSON."""
    if not pipeline_initialized:
        return jsonify({"error": "Server not initialized"}), 503

    try:
        # Handle both POST and GET requests
        if request.method == 'GET':
            query = request.args.get("query", "").strip()
            image_url = request.args.get("image_url")
        else:  # POST
            data = await request.get_json()
            query = data.get("query", "").strip()
            image_url = data.get("image_url")

        if not validate_query(query):
            return jsonify({"error": "Invalid or missing query"}), 400

        if image_url and not validate_url(image_url):
            return jsonify({"error": "Invalid image_url"}), 400

        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])

        logger.info(f"[{request_id}] Search: {query[:50]}...")

        # Collect response from pipeline
        response_content = None
        async for chunk in run_elixposearch_pipeline(
            user_query=query,
            user_image=image_url,
            event_id=None,  # Don't use SSE format at pipeline level
            request_id=request_id
        ):
            # Pipeline yields raw content when event_id is None
            response_content = chunk

        if not response_content:
            return jsonify({"error": "No response generated"}), 500

        # Format as OpenAI-compatible JSON
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
