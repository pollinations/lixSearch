import requests
import json
from clean_query import cleanQuery
from search import web_search 
from getYoutubeDetails import get_youtube_metadata, get_youtube_transcript
from scrape import fetch_full_text 
from tools import tools 
from datetime import datetime, timezone
from getTimeZone import get_timezone_and_offset, convert_utc_to_local 
import time
import random
import concurrent.futures





def run_elixposearch_pipeline(user_query: str):
    current_utc_datetime = datetime.now(timezone.utc)
    current_utc_time = current_utc_datetime.strftime("%H:%M UTC")
    current_utc_date = current_utc_datetime.strftime("%Y-%m-%d")
    POLLINATIONS_ENDPOINT = "https://text.pollinations.ai/openai"
    headers = {"Content-Type": "application/json"}

    memoized_results = {
        "timezone_info": {},
        "web_searches": {},
        "fetched_urls": {},
        "youtube_metadata": {},
        "youtube_transcripts": {}
    }

    messages = [
        {
            "role": "system",
            "content": f"""
        # ElixpoSearch - Your Advanced AI Web Research Assistant

        🎯 **Mission:** Fully answer the user's query with reliable, well-researched, and thoroughly explained information.  

        Your responses should always prioritize being:
        - **Detailed and Comprehensive:** Aim for depth, not just surface-level answers.
        - **Clear and Understandable:** Write explanations that any educated reader can follow.
        - **Well-Structured:** Organize your answers logically, with clarity.
        - **Long where necessary:** Never rush to brief or overly short answers when more detail would help understanding.
        - **Polished:** Avoid incomplete thoughts, vague summaries, or fragmented outputs.

        ---
        🛠️ **Available Tools You Can Use Freely:**
        - `cleanQuery(query: str)`: Extracts URLs (websites, YouTube) and a cleaned text query from a raw input string.
        - `web_search(query: str)`: Performs a web search and returns a list of top URLs with short snippets (dictionaries with 'url' and 'snippet'). **Crucially, this tool does NOT fetch full text.** You must decide to call `fetch_full_text` on specific URLs from its results if you need the content.
        - `get_youtube_metadata(url: str)`: Fetches metadata (title, description, duration, etc.) for a given YouTube video URL.
        - `get_youtube_transcript(url: str)`: Fetches the full transcript of a given YouTube video URL. **Always present transcripts as clean, clear, fully readable text. Never output messy JSON or fragmented phrases.**
        - `fetch_full_text(url: str)`: Scrapes and extracts the main readable text content and all image URLs from a given webpage. **Use this AFTER `web_search` if you need detailed information from a promising URL.**
        - `get_timezone_and_offset(location_name: str)`: Gets the IANA timezone string and the current UTC offset in hours for a specified location (e.g., "Kolkata"). This tool also directly returns the current local date and time for that location based on the current UTC time.
        - `convert_utc_to_local(utc_datetime: str, utc_offset_hours: float)`: Converts a given UTC datetime string to a local datetime using a provided UTC offset. Use this if you need to convert a *specific historical or future* UTC time. For current local time, `get_timezone_and_offset` is usually sufficient.

        ---
        📅 **Current UTC Date:** {current_utc_date}
        ⏰ **Current UTC Time:** {current_utc_time}

        ---
        🧠 **Reasoning & Workflow - This is an ITERATIVE Process:**

        1️. **Understand & Deconstruct:** Carefully analyze the user's query.
            - If complex or ambiguous, break it into smaller, clear sub-questions.
            - Identify implicit needs. Example: "umbrella today in Kolkata" implies "What is Kolkata's current weather?" and "What is the local time?"

        2️. **Iterative Tool Calling Cycle:**
            - You will respond with tool calls; I will execute them and return results.
            - After each result, reason step-by-step:
                - **A) Call more tools if needed.**
                - **B) Provide a Final Answer when fully ready.**
            - Do NOT rush to answer until sufficient information is gathered.

        3️. **Strategic Tool Chaining:**
            - **Time/Location Sensitive Queries (weather, local events):**
                - FIRST: Use `get_timezone_and_offset(location_name)` to determine local time.
                - THEN: Use `web_search` with specific queries (e.g., "Kolkata weather forecast today").
                - FINALLY: Use `fetch_full_text` for details from authoritative URLs (e.g., weather.com).
            - **General Info Queries:**
                - Generate 1-5 well-refined `web_search` queries.
                - Evaluate snippets, then fetch details with `fetch_full_text` from reliable URLs.
            - **URL-Based Queries:**
                - Always start with `cleanQuery`.
                - Websites: `fetch_full_text`
                - YouTube: `get_youtube_metadata` + `get_youtube_transcript` (present cleanly and fully readable)

        4️. **Evidence Gathering:**
            - Collect from multiple trusted sources (aim for 2-5 quality sources).
            - Prioritize official, authoritative sites.
            - Confirm consistency of information across sources.

        ---
        📄 **Final Response Guidelines:**
        - Be **detailed and as long as needed** to ensure full understanding.
        - Summarize clearly, cite sources (URLs) explicitly.
        - Always mention if information is time-sensitive relative to **{current_utc_date} at {current_utc_time} UTC** or the *determined local time*.
        - **For YouTube transcripts:** Provide a **clear, clean, well-formatted transcript** that is easy for the user to read and understand. Give detailed long response, No technical or raw JSON formatting.
        - If reliable answers cannot be found, state this honestly and suggest practical next steps for the user.

        ---
        🎙️ **Tone & Style:**
        - Professional, clear, confident.
        - Friendly and engaging with occasional light wit where appropriate.
        - Never expose your internal tools or reasoning explicitly to the user.

        ---
        Begin solving the user's query thoughtfully.  
        First, determine whether you need tools. If so, call them. If not, answer in detail.
        """
        },
        {
            "role": "user",
            "content": f"This is the raw query from the user: {user_query}"
        }
    ]

    max_iterations = 7
    current_iteration = 0
    collected_sources = []
    collected_images = []

    final_message_content = None

    while current_iteration < max_iterations:
        current_iteration += 1
        print(f"[DEBUG] Pollinations API call, Iteration {current_iteration}...")

        payload = {
            "model": "openai",
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "token": "fEWo70t94146ZYgk",
            "temperature": 0.7,
            "frequency_penalty": 0.0,
            "private": True,
            "seed": random.randint(1000, 9999)
        }

        try:
            response = requests.post(POLLINATIONS_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Pollinations API call failed at iteration {current_iteration}: {e}")
            break

        assistant_message = response_data["choices"][0]["message"]
        messages.append(assistant_message)

        tool_calls = assistant_message.get("tool_calls")

        if not tool_calls:
            final_message_content = assistant_message.get("content")
            break

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            function_args = json.loads(tool_call["function"]["arguments"])
            print(f"[DEBUG] Executing tool: {function_name} with args: {function_args}")

            tool_result = "[Tool execution failed or returned no data.]"
            try:
                if function_name == "cleanQuery":
                    websites, youtube, cleaned_query = cleanQuery(function_args.get("query"))
                    tool_result = f"Cleaned Query: {cleaned_query}\nWebsites: {websites}\nYouTube URLs: {youtube}"

                elif function_name == "get_timezone_and_offset":
                    location_name = function_args.get("location_name")
                    if location_name in memoized_results["timezone_info"]:
                        tool_result = memoized_results["timezone_info"][location_name]
                        print(f"[DEBUG] Using cached timezone info for {location_name}")
                    else:
                        offset_str = get_timezone_and_offset(location_name)
                        local_datetime_str = convert_utc_to_local(current_utc_datetime, offset_str)
                        tool_result = (
                            f"Location: {location_name}\n"
                            f"UTC Offset: {offset_str}\n"
                            f"Local Date & Time in {location_name}: {local_datetime_str}"
                        )
                        memoized_results["timezone_info"][location_name] = tool_result
                elif function_name == "convert_utc_to_local":
                    utc_dt_str = function_args.get("utc_datetime")
                    utc_offset = function_args.get("utc_offset_hours")
                    utc_dt_obj = datetime.fromisoformat(utc_dt_str.replace('Z', '+00:00'))
                    tool_result = convert_utc_to_local(utc_dt_obj, utc_offset).strftime('%Y-%m-%d %H:%M:%S')

                elif function_name == "web_search":
                    search_query = function_args.get("query")
                    search_results_raw = web_search(search_query)
                    summaries = ""
                    for url in search_results_raw:
                        try:
                            text_content, image_urls = fetch_full_text(url)
                            text_content = text_content[:500]
                            summaries += f"\nURL: {url}\nSummary: {text_content if text_content else '[Could not fetch or summarize this page.]'}\nImages: {image_urls}\n"
                        except Exception as e:
                            print(f"[WARN] Failed to fetch or summarize {url}: {e}")
                            continue
                    tool_result = summaries

                elif function_name == "get_youtube_metadata":
                    url = function_args.get("url")
                    if url in memoized_results["youtube_metadata"]:
                        tool_result = memoized_results["youtube_metadata"][url]
                        print(f"[DEBUG] Using cached YouTube metadata for {url}")
                    else:
                        metadata = get_youtube_metadata(url)
                        tool_result = json.dumps(metadata) # Return as JSON string
                        memoized_results["youtube_metadata"][url] = tool_result
                    # YouTube URLs are also sources
                    if url and url not in collected_sources:
                        collected_sources.append(url)

                elif function_name == "get_youtube_transcript":
                    url = function_args.get("url")
                    if url in memoized_results["youtube_transcripts"]:
                        tool_result = memoized_results["youtube_transcripts"][url]
                        print(f"[DEBUG] Using cached YouTube transcript for {url}")
                    else:
                        transcript = get_youtube_transcript(url)
                        # Truncate transcript for context window efficiency
                        tool_result = f"YouTube Transcript for {url}:\n{transcript[:2000] if transcript else '[No transcript available]'}..."
                        memoized_results["youtube_transcripts"][url] = tool_result
                    # Transcript URL is implicitly part of the video source
                    if url and url not in collected_sources: # Redundant if metadata already added, but safe
                        collected_sources.append(url)

                elif function_name == "fetch_full_text":
                    url_to_fetch = function_args.get("url")
                    if url_to_fetch in memoized_results["fetched_urls"]:
                        tool_result = memoized_results["fetched_urls"][url_to_fetch]
                        print(f"[DEBUG] Using cached fetch_full_text result for {url_to_fetch}")
                    else:
                        print(f"[INFO] Attempting to fetch full text from: {url_to_fetch}")
                        text_content, image_urls = fetch_full_text(url_to_fetch)
                        tool_result_content = text_content[:2000] if text_content else '[No text content]'
                        tool_result = f"URL: {url_to_fetch}\nText Preview: {tool_result_content}...\nImages Found: {len(image_urls)}"
                        if url_to_fetch and url_to_fetch not in collected_sources:
                            collected_sources.append(url_to_fetch)
                        for img in image_urls:
                            if img not in collected_images:
                                collected_images.append(img)
                        memoized_results["fetched_urls"][url_to_fetch] = tool_result

                else:
                    tool_result = f"Unknown tool: {function_name}"
            except Exception as e:
                tool_result = f"[ERROR] Tool {function_name} failed: {type(e).__name__}: {e}"
                print(f"[DEBUG] Tool {function_name} failed: {type(e).__name__}: {e}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": function_name,
                "content": tool_result
            })

    if final_message_content:
        response_with_sources = final_message_content

        sources_md = ""
        if collected_sources:
            sources_md += "\n\n---\n**Sources:**\n"
            unique_sources = sorted(list(set(collected_sources)))
            for i, src in enumerate(unique_sources):
                sources_md += f"{i+1}. [{src}]({src})\n"
        if collected_images:
            sources_md += "\n**Images Found:**\n"
            unique_images = sorted(list(set(collected_images)))
            for img in unique_images:
                sources_md += f"![image]({img})\n"

        response_with_sources += sources_md
        print("\n--- ElixpoSearch Final Answer ---\n")
        print(response_with_sources)
    else:
        fallback_msg = f"[ERROR] ElixpoSearch could not generate a comprehensive answer after {max_iterations} iterations. " \
                       "This might be due to an ambiguous query, lack of information, or an internal error."
        print(fallback_msg)

if __name__ == "__main__":
    run_elixposearch_pipeline("What's the weather condition of paris now?")
    