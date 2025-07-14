from flask_cors import CORS
from flask import Flask, request, Response, stream_with_context
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import time
from searchPipeline import run_elixposearch_pipeline
import random
import logging
import sys 


app = Flask(__name__)
CORS(app)

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://",
    strategy="moving-window"
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', stream=sys.stdout)
app.logger.setLevel(logging.DEBUG)
logging.getLogger('werkzeug').setLevel(logging.DEBUG)

@limiter.limit("10 per minute")
@app.route("/search/sse", methods=["POST"])
def search_sse():
    user_query = request.json.get("query", "")
    event_id = f"elixpo-{int(time.time() * 1000)}-{random.randint(1000,9999)}"
    app.logger.info(f"Received SSE search request: event_id={event_id}, query={user_query}")
    try:
        response = Response(
            stream_with_context(run_elixposearch_pipeline(user_query, event_id)),
            mimetype="text/event-stream"
        )
        app.logger.debug(f"SSE response prepared for event_id={event_id}")
        return response
    except Exception as e:
        app.logger.error(f"Error in search_sse: {e}", exc_info=True)
        return {"error": "Internal server error"}, 500


@app.route('/search', methods=['GET', 'POST'])
@app.route('/search/', methods=['GET', 'POST'])
@app.route('/search/<path:anything>', methods=['GET', 'POST'])
@app.route("/search", methods=["GET", "POST"])
def search_json():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        user_query = data.get("query") or data.get("message") or data.get("prompt") or ""
        app.logger.info(f"POST /search with body: {data}")
    else:
        user_query = request.args.get("query")
        app.logger.info(f"GET /search with query: {user_query}")
    if not user_query:
        app.logger.warning("Missing query in /search request")
        return {"error": "Missing query"}, 400

    try:
        final_response = run_elixposearch_pipeline(user_query)
        # Handle generator return value properly
        if hasattr(final_response, '__iter__') and not isinstance(final_response, str):
            try:
                last_value = None
                while True:
                    last_value = next(final_response)
            except StopIteration as e:
                final_response = e.value
            except Exception as e:
                app.logger.error(f"Error iterating generator in /search: {e}", exc_info=True)
                final_response = None

        app.logger.debug(f"Returning response for /search: {final_response}")
        return {"result": final_response or "Didn't Wait"}
    except Exception as e:
        app.logger.error(f"Error in /search: {e}", exc_info=True)
        return {"error": "Internal server error"}, 500

@app.route("/search/v1/chat/completions", methods=["GET", "POST"])
def openai_chat_completions():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        messages = data.get("messages", [])
        app.logger.info(f"POST /search/v1/chat/completions with body: {data}")
    else:
        # For GET, allow ?query=... or ?message=...
        query = request.args.get("query") or request.args.get("message") or ""
        messages = [{"role": "user", "content": query}] if query else []
        app.logger.info(f"GET /search/v1/chat/completions with query: {query}")

    if not messages or not isinstance(messages, list):
        app.logger.warning("Missing or invalid messages in /search/v1/chat/completions request")
        return {"error": "Missing or invalid messages"}, 400

    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    if not user_query:
        app.logger.warning("No user message found in /search/v1/chat/completions request")
        return {"error": "No user message found"}, 400

    try:
        final_response = run_elixposearch_pipeline(user_query)
        # Handle generator return value properly
        if hasattr(final_response, '__iter__') and not isinstance(final_response, str):
            try:
                last_value = None
                while True:
                    last_value = next(final_response)
            except StopIteration as e:
                final_response = e.value
            except Exception as e:
                app.logger.error(f"Error iterating generator in /search/v1/chat/completions: {e}", exc_info=True)
                final_response = None

        response = {
            "id": "chatcmpl-elixpo",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "elixpo-search",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": final_response or ""
                    },
                    "finish_reason": "stop"
                }
            ]
        }
        app.logger.debug(f"Returning response for /search/v1/chat/completions: {response}")
        return response
    except Exception as e:
        app.logger.error(f"Error in /search/v1/chat/completions: {e}", exc_info=True)
        return {"error": "Internal server error"}, 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)