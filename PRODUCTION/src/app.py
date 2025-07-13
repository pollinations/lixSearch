import os
import sys
import time
import json
import threading
from functools import wraps
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from collections import defaultdict
from datetime import datetime, timedelta
from waitress import serve
from searchPipeline import run_elixposearch_pipeline



# Import the ElixpoSearch pipeline
sys.path.append(os.path.dirname(__file__))

# --- Rate Limiting ---
RATE_LIMIT = 10  # requests
RATE_PERIOD = 60  # seconds

client_requests = defaultdict(list)

def rate_limiter(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        client_ip = request.remote_addr or "global"
        now = datetime.utcnow()
        window = [t for t in client_requests[client_ip] if now - t < timedelta(seconds=RATE_PERIOD)]
        if len(window) >= RATE_LIMIT:
            return jsonify({"error": "Rate limit exceeded. Try again later."}), 429
        window.append(now)
        client_requests[client_ip] = window
        return func(*args, **kwargs)
    return wrapper

# --- Event Sourcing ---
event_sources = {}

def event_stream_generator(event_id):
    """Yield events for a given event_id."""
    last_sent = 0
    while True:
        events = event_sources.get(event_id, [])
        if last_sent < len(events):
            for i in range(last_sent, len(events)):
                yield f"data: {json.dumps(events[i])}\n\n"
            last_sent = len(events)
        time.sleep(0.5)
        if event_sources.get(event_id, None) == "done":
            break

def track_event(event_id, event):
    if event_id not in event_sources or event_sources[event_id] == "done":
        event_sources[event_id] = []
    event_sources[event_id].append(event)

def mark_event_done(event_id):
    event_sources[event_id] = "done"

# --- Flask App ---
app = Flask(__name__)
CORS(app)

@app.route("/search", methods=["GET", "POST"])
@rate_limiter
def search():
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

    # Run the pipeline and capture stdout for progress
    output = []
    def capture_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        output.append(msg)

    # Patch print temporarily
    orig_print = __builtins__.print
    __builtins__.print = capture_print
    try:
        run_elixposearch_pipeline(user_query)
    finally:
        __builtins__.print = orig_print

    return jsonify({"output": "\n".join(output)})

@app.route("/search/sse", methods=["POST"])
@rate_limiter
def search_sse():
    """
    SSE endpoint for live event streaming.
    """
    data = request.get_json(force=True)
    user_query = data.get("query", "")
    if not user_query:
        return jsonify({"error": "Missing 'query' parameter."}), 400

    event_id = f"{int(time.time() * 1000)}-{os.urandom(4).hex()}"
    event_sources[event_id] = []

    def pipeline_thread():
        def event_print(*args, **kwargs):
            msg = " ".join(str(a) for a in args)
            track_event(event_id, {"timestamp": time.time(), "message": msg})
        orig_print = __builtins__.print
        __builtins__.print = event_print
        try:
            run_elixposearch_pipeline(user_query)
        finally:
            __builtins__.print = orig_print
            mark_event_done(event_id)

    threading.Thread(target=pipeline_thread, daemon=True).start()

    return Response(
        stream_with_context(event_stream_generator(event_id)),
        mimetype="text/event-stream"
    )

@app.route("/v1/chat/completions", methods=["POST"])
@rate_limiter
def openai_compatible():
    """
    OpenAI-compatible endpoint.
    """
    data = request.get_json(force=True)
    user_query = ""
    if "messages" in data and isinstance(data["messages"], list):
        # Use the last user message
        for msg in reversed(data["messages"]):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break
    elif "query" in data:
        user_query = data["query"]

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

    # OpenAI compatible response
    return jsonify({
        "id": f"elixpo-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "elixpo-search",
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
        "rate_limit": f"{RATE_LIMIT} requests per {RATE_PERIOD} seconds"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting ElixpoSearch API on port {port} ...")
    serve(app, host="0.0.0.0", port=port)