import requests
import json
from clean_query import cleanQuery
from search import web_search, GoogleSearchAgentText, GoogleSearchAgentImage, image_search
from getYoutubeDetails import get_youtube_metadata, get_youtube_transcript
from scrape import fetch_full_text
from getImagePrompt import generate_prompt_from_image, image_url_to_base64, replyFromImage
from tools import tools
from datetime import datetime, timezone
from getTimeZone import get_timezone_and_offset, convert_utc_to_local
import random
import concurrent.futures
import logging
import dotenv
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("elixpo")
dotenv.load_dotenv()

POLLINATIONS_TOKEN=os.getenv("TOKEN")
MODEL=os.getenv("MODEL")
REFRRER=os.getenv("REFERRER")
POLLINATIONS_ENDPOINT = "https://text.pollinations.ai/openai"
print(MODEL, POLLINATIONS_TOKEN)

def fetch_url_content_parallel(urls, max_workers=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_full_text, url): url for url in urls}
        results = {}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                text_content, image_urls = future.result()
                results[url] = (text_content, image_urls)
            except Exception as e:
                print(f"[ERROR] Failed fetching {url}: {e}")
                results[url] = ('[Failed]', [])
        print(f"[INFO] Fetched all URL information in parallel.")
        return results


def fetch_youtube_parallel(urls, mode='metadata', max_workers=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        if mode == 'metadata':
            futures = {executor.submit(get_youtube_metadata, url): url for url in urls}
        else:
            futures = {executor.submit(get_youtube_transcript, url): url for url in urls}

        results = {}
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                results[url] = future.result()
            except Exception as e:
                print(f"[ERROR] YouTube {mode} failed for {url}: {e}")
                results[url] = '[Failed]'
        return results

def format_sse(event: str, data: str) -> str:
    lines = data.splitlines()
    data_str = ''.join(f"data: {line}\n" for line in lines)
    return f"event: {event}\n{data_str}\n\n"

async def run_elixposearch_pipeline(user_query: str, user_image: str, event_id: str = None):
    print(f"[INFO] Starting ElixpoSearch Pipeline for query: {user_query}")
    
    def emit_event(event_type, message):
        if event_id:
            return format_sse(event_type, message)
        return None
    
    initial_event = emit_event("INFO", " Initiating Pipeline ")
    if initial_event:
        yield initial_event
    
    google_agent_text = GoogleSearchAgentText()
    google_agent_image = GoogleSearchAgentImage()

    try:
        await google_agent_text.start()
        await google_agent_image.start()
        google_req_count = 0  

        current_utc_datetime = datetime.now(timezone.utc)
        current_utc_time = current_utc_datetime.strftime("%H:%M UTC")
        current_utc_date = current_utc_datetime.strftime("%Y-%m-%d")
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
    Mission: Answer the user's query with reliable, well-researched, and well-explained information.
    
    **CRITICAL: Answer directly if you know the answer to a question (basic facts, math, general knowledge) without using tools.**
    
    Use tools only when:
    - You need current/recent information (news, stock prices, weather, etc.)
    - Current political positions or office holders (presidents, prime ministers, etc.)
    - The query explicitly asks for web research or sources
    - You are genuinely uncertain about factual information
    - The query contains specific URLs to analyze
    - User uploads an image
    - Time-sensitive information is requested
    - Queries asking "now", "current", "latest", "today", "recent"

    Your answers must prioritize:
    - Clarity and correctness
    - Concise explanations
    - Markdown formatting
    - Relevant citations if sources are used

    ---

    Available Tools:
    - cleanQuery(query: str) → Extracts URLs and cleaned user query.
    - web_search(query: str) → Returns websites and summaries (no full text).
    - fetch_full_text(url: str) → Extracts full main content and images.
    - get_youtube_metadata(url: str) → Metadata from a YouTube link.
    - get_youtube_transcript(url: str) → Transcript from a YouTube link.
    - get_timezone_and_offset(location: str) → Timezone + UTC offset.
    - convert_utc_to_local(utc_time, offset) → Converts to local time.
    - generate_prompt_from_image(image_url: str) → Suggests a search query based on image content.
    - replyFromImage(image_url: str, query: str) → Answers a question using both query and image context.
    - image_url_to_base64(image_url: str) → Converts image URL to base64 for prompt generation.
    - image_search(image_query: str, max_images=10) → Performs reverse or similar image search.

    ---

    Context:
    Current UTC Date: {current_utc_date}  
    Current UTC Time: {current_utc_time}

    ---

    IMAGE-RELATED BEHAVIOR:

    1. User Provides Image ONLY (No Text Query):
    - **Step 1:** Convert the `user_image` URL to base64 using `image_url_to_base64()`.
    - **Step 2:** Generate a search prompt from the image using `generate_prompt_from_image()` with the base64 image.
    - **Step 3:** Perform an image search using the generated prompt with `image_search()` to find **10 relevant images**.
    - **Step 4:** Present these 10 images directly to the user. No detailed text answer is needed, just a concise intro.

    2. User Provides Image AND Text Query:
    - **Step 1:** Convert the `user_image` URL to base64 using `image_url_to_base64()`.
    - **Step 2:** Use `replyFromImage()` with the base64 image and the `user_query` to get an initial understanding and response based on the image context. This response will form the basis of your answer.
    - **Step 3:** Based on the `user_query` and the initial `replyFromImage()` response, determine if additional web search (`web_search()`) is necessary for broader context, current information, or detailed explanation beyond what the image alone provides.
    - **Step 4:** If a web search is needed, generate 1-3 highly focused search queries combining elements from the image and the user's text query. Execute `web_search()` and `fetch_full_text()` on relevant results.
    - **Step 5:** Concurrently, perform an `image_search()` using a relevant query derived from the image content and the user's text query to find **10 relevant images** that visually relate to the user's request.
    - **Step 6:** Synthesize information from `replyFromImage()`, web search results (if any), and image search results into a comprehensive, detailed answer.

    3. User Provides Text Query ONLY (No Image):
    - **Step 1:** Follow the standard text-based query decision framework.
    - **Step 2:** If `web_search()` is performed, also generate a concise image search query based on the topic of the text query and perform `image_search()` to find **10 relevant images**.
    - **Step 3:** Include these 10 relevant images in the final response.

    ---

    General Decision Framework:
    1. **Basic Knowledge/Math/Facts**: Answer directly (e.g., "What is 1+1?", "What is the capital of France?", "How does photosynthesis work?")
    2. **Current Events/News/Politics**: Use `web_search` tool (e.g., "Who is the current president?", "Latest news", "Current prime minister")
    3. **Specific URLs provided**: Use appropriate tools (`fetch_full_text`, `get_youtube_metadata`, `get_youtube_transcript`) to analyze them.
    4. **Explicit research requests**: Use tools as needed.
    5. **Time-sensitive data**: Use tools for current information.
    6. **Keywords like "now", "current", "latest", "today"**: Use `web_search` tool.
    7. **Image Tools**: Always use relevant image tools (`image_url_to_base64`, `generate_prompt_from_image`, `replyFromImage`, `image_search`) when an image is involved, following the specific behaviors outlined above.

    **CRITICAL: For any query asking about current political positions, office holders, or using words like "now", "current", "latest" - ALWAYS use `web_search` first.**

    Final Response Structure:
    1. **Answer** (detailed explanation of the query)
    2. **Visually Similar Images** (If an image was uploaded by the user, this section will contain 10 images directly related to the user's uploaded image.)
    3. **Images from Related Web Results** (If a web search was performed, this section will contain up to 10 images found during the web search related to the content.)
    4. **Sources & References** (List all URLs from which information was gathered.)
    5. **Summary** (A concise summary of the answer.)

    Tone:
    Professional, clear, confident, and detailed. Ensure all relevant information is covered.

    **Answer in English, unless explicitly asked to use another language.**
    """
},
{
    "role": "user", 
    "content": f"""Query: {user_query} -- Image: {user_image}"""
}
]

        max_iterations = 7
        current_iteration = 0
        collected_sources = []
        collected_images_from_web = []
        collected_similar_images = []
        final_message_content = None
        
        # Initial check for image-only query
        if user_image and not user_query:
            base64_img = image_url_to_base64(user_image)
            generated_prompt = await generate_prompt_from_image(base64_img)
            
            search_results_raw = await image_search(generated_prompt, google_agent_image, max_images=10)
            collected_similar_images.extend(search_results_raw)
            
            final_message_content = "Here are some images visually similar to the one you provided."
            # Break early if only image is provided and handled
            # The remaining logic in 'finally' block will format and send the response.
            await google_agent_image.close()
            await google_agent_text.close()
            yield format_sse("final", final_message_content + "\n\n---")
            return


        while current_iteration < max_iterations:
            current_iteration += 1
            
            iteration_event = emit_event("INFO", f" Research Iteration {current_iteration} \n")
            if iteration_event:
                yield iteration_event
                
            payload = {
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "token": POLLINATIONS_TOKEN,
                "referrer": REFRRER,
                "temperature": 0.2,
                "private": True,
                "seed": random.randint(1000, 9999)
            }

            try:
                response = requests.post(POLLINATIONS_ENDPOINT, headers=headers, json=payload)
                response.raise_for_status()
                response_data = response.json()
            except requests.exceptions.RequestException as e:
                if event_id:
                    yield format_sse("error", f"[ERROR] Pollinations API call failed at iteration {current_iteration}: {e}")
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
                if event_id:
                    print(f"[INFO] Tool call detected: {function_name} with args: {function_args}")
                    yield format_sse("INFO", f" Execution In Progress ({function_name}) \n")
                
                tool_result = "[Tool execution failed or returned no data.]"

                try:
                    if function_name == "cleanQuery":
                        websites, youtube, cleaned_query = cleanQuery(function_args.get("query"))
                        tool_result = f"Cleaned Query: {cleaned_query}\nWebsites: {websites}\nYouTube URLs: {youtube}"

                    elif function_name == "get_timezone_and_offset":
                        location_name = function_args.get("location_name")
                        if location_name in memoized_results["timezone_info"]:
                            tool_result = memoized_results["timezone_info"][location_name]
                        else:
                            offset_str = get_timezone_and_offset(location_name)
                            local_datetime_str = convert_utc_to_local(current_utc_datetime, offset_str)
                            tool_result = f"Location: {location_name}\nUTC Offset: {offset_str}\nLocal Date & Time: {local_datetime_str}"
                            memoized_results["timezone_info"][location_name] = tool_result

                    elif function_name == "convert_utc_to_local":
                        utc_dt_str = function_args.get("utc_datetime")
                        utc_offset = function_args.get("utc_offset_hours")
                        utc_dt_obj = datetime.fromisoformat(utc_dt_str.replace('Z', '+00:00'))
                        tool_result = convert_utc_to_local(utc_dt_obj, utc_offset).strftime('%Y-%m-%d %H:%M:%S')

                    elif function_name == "web_search":
                        web_event = emit_event("INFO", f" Surfing Internet \n")
                        if web_event:
                            yield web_event
                        print(f"[INFO] Performing web search")
                        search_query = function_args.get("query")

                        google_req_count += 1
                        if google_req_count > 50:
                            print("[INFO] Restarting GoogleSearchAgent after 50 requests.")
                            await google_agent_text.close()
                            google_agent_text = GoogleSearchAgentText()
                            await google_agent_text.start()
                            google_req_count = 1 

                        search_results_raw = await web_search(search_query, google_agent_text)
                        print(f"[INFO] Web search returned {len(search_results_raw)} results")
                        summaries = ""
                        parallel_results = fetch_url_content_parallel(search_results_raw)
                        for url, (text_content, image_urls) in parallel_results.items():
                            summaries += f"\nURL: {url}\nSummary: {text_content}\nImages: {image_urls}\n"
                            collected_sources.append(url)
                            collected_images_from_web.extend(image_urls)
                        tool_result = summaries

                    elif function_name == "generate_prompt_from_image":
                        base64_img_result = image_url_to_base64(function_args.get("image_url"))
                        get_prompt = await generate_prompt_from_image(base64_img_result)
                        tool_result = f"Generated Search Query: {get_prompt}"

                    elif function_name == "replyFromImage":
                        base64_img_result = image_url_to_base64(function_args.get("image_url"))
                        query = function_args.get("query")
                        reply = await replyFromImage(base64_img_result, query)
                        tool_result = f"Reply from Image: {reply}"
                    
                    elif function_name == "image_search":
                        image_query = function_args.get("image_query")
                        max_images = function_args.get("max_images", 10)
                        google_req_count += 1
                        if google_req_count > 50:
                            print("[INFO] Restarting GoogleSearchAgent after 50 requests.")
                            await google_agent_image.close()
                            google_agent_image = GoogleSearchAgentImage()
                            await google_agent_image.start()
                            google_req_count = 1  

                        search_results_raw = await image_search(image_query, google_agent_image, max_images)
                        print(f"[INFO] Image search returned {len(search_results_raw)} results")
                        collected_similar_images.extend(search_results_raw)
                        tool_result = f"Image Search Results: {search_results_raw}"

                    
                    elif function_name == "get_youtube_metadata":
                        print(f"[INFO] Getting YouTube metadata for URLs")
                        urls = [function_args.get("url")]
                        results = fetch_youtube_parallel(urls, mode='metadata')
                        for url, metadata in results.items():
                            tool_result = json.dumps(metadata)
                            memoized_results["youtube_metadata"][url] = tool_result
                            collected_sources.append(url)

                    elif function_name == "get_youtube_transcript":
                        print(f"[INFO] Getting YouTube transcripts for URLs")
                        if event_id:
                            yield format_sse("INFO", f" Watching Youtube \n")
                        urls = [function_args.get("url")]
                        results = fetch_youtube_parallel(urls, mode='transcript')
                        for url, transcript in results.items():
                            tool_result = f"YouTube Transcript for {url}:\n{transcript if transcript else '[No transcript available]'}..."
                            memoized_results["youtube_transcripts"][url] = tool_result
                            collected_sources.append(url)

                    elif function_name == "fetch_full_text":
                        print(f"[INFO] Fetching full text for URLs")
                        if event_id:
                            yield format_sse("INFO", f" Writing Script \n")
                        urls = [function_args.get("url")]
                        parallel_results = fetch_url_content_parallel(urls)
                        for url, (text_content, image_urls) in parallel_results.items():
                            tool_result = f"URL: {url}\nText Preview: {text_content}...\nImages Found: {len(image_urls)}"
                            collected_sources.append(url)
                            collected_images_from_web.extend(image_urls)
                            memoized_results["fetched_urls"][url] = tool_result

                    else:
                        tool_result = f"Unknown tool: {function_name}"

                except Exception as e:
                    tool_result = f"[ERROR] Tool {function_name} failed: {type(e).__name__}: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": tool_result
                })
                if event_id:
                    print(f"[INFO] Tool {function_name} executed successfully.")
                    yield format_sse("INFO", f" Execution Completed! \n")

        if final_message_content:
            print(f"[INFO] Preparing final response")
            response_with_sources = final_message_content
            sources_md = ""

            if collected_similar_images:
                images_to_show = [img for img in collected_similar_images if img and img.startswith("http")][:10]
                if images_to_show:
                    sources_md += "\n\n---\n**Visually Similar Images:**\n"
                    for img in images_to_show:
                        sources_md += f"![Similar Image]({img})\n"

            if collected_images_from_web:
                web_images = [img for img in collected_images_from_web if img and img.startswith("http")]
                if web_images:
                    sources_md += "\n\n---\n**Images from Related Web Results:**\n"
                    for img in web_images[:10]:
                        sources_md += f"![Web Image]({img})\n"

            if collected_sources:
                sources_md += "\n\n---\n**Sources & References:**\n"
                unique_sources = sorted(set(collected_sources))
                for i, src in enumerate(unique_sources):
                    sources_md += f"{i+1}. [{src}]({src})\n"

            response_with_sources += "\n\n---\n**Summary:**\n"
            response_with_sources += "This answer includes the latest information, relevant images, and sources for further reading. If you need more details or updates, please specify your interest.\n"

            response_with_sources += sources_md

            print(f"[INFO] Final response prepared with sources and images")
            
            if event_id:
                yield format_sse("INFO", " SUCCESS")
                chunk_size = 5000 
                for i in range(0, len(response_with_sources), chunk_size):
                    chunk = response_with_sources[i:i+chunk_size]
                    event_name = "final" if i + chunk_size >= len(response_with_sources) else "final-part"
                    yield format_sse(event_name, chunk)
            else:
                yield format_sse("final", response_with_sources)
            return  
        else:
            error_msg = f"[ERROR] ElixpoSearch failed after {max_iterations} iterations."
            if event_id:
                yield format_sse("error", error_msg)
                return  
            else:
                print(error_msg)
    finally:
        try:
            await google_agent_image.close()
            await google_agent_text.close()
        except Exception as e:
            print(f"[ERROR] Failed to close GoogleSearchAgent: {e}")


if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Test Cases
        # 1. Image only
        # user_query = ""
        # user_image = "https://media.istockphoto.com/id/1421310827/photo/young-graceful-ballerina-is-performing-classic-dance-beauty-and-elegance-of-classic-ballet.jpg?s=612x612&w=0&k=20&c=GQ1DVEarW4Y-lGD6y8jCb3YPIgap7gj-6ReS3C7Qi3Y=" 
        
        # 2. Image + Text Query
        user_query = "What can you tell me about this type of dance and its current popularity?"
        user_image = "https://media.istockphoto.com/id/1421310827/photo/young-graceful-ballerina-is-performing-classic-dance-beauty-and-elegance-of-classic-ballet.jpg?s=612x612&w=0&k=20&c=GQ1DVEarW4Y-lGD6y8jCb3YPIgap7gj-6ReS3C7Qi3Y=" 

        # 3. Text only
        # user_query = "What is the capital of France and show me some images of it?"
        # user_image = ""
        
        event_id = None # Set to a string like "test_session_123" if you want SSE output

        async_generator = run_elixposearch_pipeline(user_query, user_image, event_id=event_id)
        answer = None
        
        try:
            async for event_chunk in async_generator:
                print(event_chunk) # Print each SSE chunk as it comes
                if event_chunk and "event: final" in event_chunk:
                    lines = event_chunk.split('\n')
                    for line in lines:
                        if line.startswith('data: '):
                            if answer is None:
                                answer = line[6:]
                            else:
                                answer += line[6:]
                elif event_chunk and "event: final-part" in event_chunk:
                    lines = event_chunk.split('\n')
                    for line in lines:
                        if line.startswith('data: '):
                            if answer is None:
                                answer = line[6:]
                            else:
                                answer += line[6:]
        except Exception as e:
            print(f"Error during async generator iteration: {e}")
            answer = "Failed to get answer due to an error."
        
        if answer:
            print(f"\n--- Final Answer Received ---\n{answer}")
        else:
            print("\n--- No answer received ---")
    
    asyncio.run(main())