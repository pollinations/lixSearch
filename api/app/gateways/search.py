import logging
import uuid
from quart import request, jsonify, Response
from pipeline.searchPipeline import run_elixposearch_pipeline
from app.utils import validate_query, validate_url

logger = logging.getLogger("lixsearch-api")


async def search(pipeline_initialized: bool):
    """Search endpoint - supports both POST and GET requests."""
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

        async def event_generator():
            async for chunk in run_elixposearch_pipeline(
                user_query=query,
                user_image=image_url,
                event_id=request_id
            ):
                yield chunk.encode('utf-8')

        return Response(
            event_generator(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Content-Type': 'text/event-stream',
                'Access-Control-Allow-Origin': '*'
            }
        )

    except Exception as e:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])
        logger.error(f"[{request_id}] Search error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
