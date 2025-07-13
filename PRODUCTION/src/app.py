import os
import sys
import time
import json
import threading
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from collections import defaultdict
from datetime import datetime
from searchPipeline import run_elixposearch_pipeline, track_event, mark_event_done, event_sources
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("elixpo")
app = Flask(__name__)
CORS(app)

@app.route("/search", methods=["GET", "POST"])
@app.route("/search/", methods=["GET", "POST"])
@app.route("/search/<path:anything>", methods=["GET", "POST"])
def search(anything=None):
    """
    Normal GET/POST endpoint for ElixpoSearch.
    """
    if request.method == "POST":
        data = request.get_json(force=True)
        user_query = data.get("query", "")
    else:
        user_query = request.args.get("query", "")

    if not user_query:
        return jsonify({"error": "Missing 'query' parameter."}), 400

    output = []
    def capture_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        output.append(msg)

    orig_print = __builtins__.print
    __builtins__.print = capture_print
    try:
        run_elixposearch_pipeline(user_query)
    finally:
        __builtins__.print = orig_print

    return jsonify({"output": "\n".join(output)})

@app.route("/search/sse", methods=["GET", "POST"])
@app.route("/search/sse/", methods=["GET", "POST"])
@app.route("/search/sse/<path:anything>", methods=["GET", "POST"])
def search_sse(anything=None):
    """
    SSE endpoint for live event streaming.
    """
    if request.method == "POST":
        data = request.get_json(force=True)
        user_query = data.get("query", "")
        logger.info(f"Received POST query: {user_query}")
    else:
        user_query = request.args.get("query", "")

    if not user_query:
        return jsonify({"error": "Missing 'query' parameter."}), 400

    event_id = f"{int(time.time() * 1000)}-{os.urandom(4).hex()}"
    event_sources[event_id] = []

    def pipeline_thread():
        try:
            run_elixposearch_pipeline(user_query, event_id=event_id)
        except Exception as e:
            # Always mark done on error
            mark_event_done(event_id)
            track_event(event_id, {"timestamp": time.time(), "message": f"[ERROR] {e}"})
            logger.info(f"[ERROR] {e}")

    threading.Thread(target=pipeline_thread, daemon=True).start()

    def robust_event_stream(event_id):
        last_sent = 0
        while True:
            events = event_sources.get(event_id, [])
            if events == "done":
                break
            if last_sent < len(events):
                for i in range(last_sent, len(events)):
                    event = events[i]
                    # Always send every event as a generic message event
                    yield f"data: {json.dumps(event)}\n\n"
                    # If this is the final response, also send as a special event
                    if event.get("final"):
                        yield f"event: final\ndata: {json.dumps(event)}\n\n"
                last_sent = len(events)
            time.sleep(0.02)
        yield "event: close\ndata: {}\n\n"

    return Response(
        stream_with_context(robust_event_stream(event_id)),
        mimetype="text/event-stream"
    )

@app.route("/v1/chat/completions", methods=["GET", "POST"])
def openai_compatible():
    """
    OpenAI-compatible endpoint.
    """
    if request.method == "POST":
        data = request.get_json(force=True)
        user_query = ""
        if "messages" in data and isinstance(data["messages"], list):
            for msg in reversed(data["messages"]):
                if msg.get("role") == "user":
                    user_query = msg.get("content", "")
                    break
        elif "query" in data:
            user_query = data["query"]
    else:
        user_query = request.args.get("query", "")

    if not user_query:
        return jsonify({"error": "Missing user query."}), 400

    output = []
    def capture_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        output.append(msg)

    orig_print = __builtins__.print
    __builtins__.print = capture_print
    try:
        run_elixposearch_pipeline(user_query)
    finally:
        __builtins__.print = orig_print

    return jsonify({
        "id": f"elixpo-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "elixposearch",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "\n".join(output)
            },
            "finish_reason": "stop"
        }]
    })

@app.route("/")
def index():
    return jsonify({
        "service": "ElixpoSearch API",
        "endpoints": ["/search", "/search/sse", "/v1/chat/completions"],
        "rate_limit": "None (disabled)",
        "model": "elixposearch"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
