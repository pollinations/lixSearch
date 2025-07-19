from quart import Quart, request, jsonify, Response
from quart_cors import cors
import asyncio
import time
import random
import logging
import sys
from searchPipeline import run_elixposearch_pipeline

app = Quart(__name__)
app = cors(app, allow_origin="*")

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', stream=sys.stdout)
app.logger.setLevel(logging.DEBUG)


@app.route("/search/sse", methods=["POST"])
async def search_sse():
    data = await request.get_json()
    user_query = data.get("query", "")
    event_id = f"elixpo-{int(time.time() * 1000)}-{random.randint(1000, 9999)}"
    app.logger.info(f"Received SSE search request: event_id={event_id}, query={user_query}")

    async def event_stream():
        async for chunk in run_elixposearch_pipeline(user_query, event_id):  # Use async for
            yield chunk
        while True:
            yield ":\n\n"  # Ping
            await asyncio.sleep(30)

    return Response(event_stream(), content_type="text/event-stream")


@app.route('/search', methods=['GET', 'POST'])
@app.route('/search/<path:anything>', methods=['GET', 'POST'])
async def search_json(anything=None):
    is_openai_chat = False
    if request.method == "POST":
        data = await request.get_json(force=True, silent=True) or {}
        user_query = ""
        messages = data.get("messages", [])
        if messages and isinstance(messages, list):
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_query = msg.get("content", "").strip()
                    is_openai_chat = True
                    break

        if not user_query:
            user_query = data.get("query") or data.get("message") or data.get("prompt") or ""
    else:
        user_query = request.args.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Missing query"}), 400

    # Call the async generator and collect final result
    final_result_content = []
    try:
        async for chunk in run_elixposearch_pipeline(user_query, event_id=None):
            app.logger.debug(f"[DEBUG] Chunk: {chunk}")
            
            # Parse SSE format
            lines = chunk.splitlines()
            event_type = None
            data_lines = []

            for line in lines:
                if line.startswith("event:"):
                    event_type = line.replace("event:", "").strip()
                elif line.startswith("data:"):
                    data_lines.append(line.replace("data:", "").strip())

            data_text = "\n".join(data_lines)

            # Collect final content
            if data_text and event_type in ["final", "final-part"]:
                final_result_content.append(data_text)

    except Exception as e:
        app.logger.error(f"Error running pipeline: {e}", exc_info=True)
        final_response = f"Error: {e}"
        if is_openai_chat:
            return jsonify([{"message": {"role": "assistant", "content": final_response}}])
        else:
            return jsonify({"result": final_response})

    # Join all final content
    final_response = "\n".join(final_result_content).strip()
    
    if not final_response:
        final_response = "No results found"

    app.logger.debug(f"Returning response for /search: {final_response[:200]}...")

    if is_openai_chat:
        return jsonify([
            {
                "message": {
                    "role": "assistant",
                    "content": final_response
                }
            }
        ])
    else:
        return jsonify({"result": final_response})



@app.route("/test", methods=["GET"])
async def test():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    import hypercorn.asyncio
    from hypercorn.config import Config
    config = Config()
    config.bind = ["0.0.0.0:5000"]
    config.use_reloader = False
    config.workers = 1
    asyncio.run(hypercorn.asyncio.serve(app, config))
