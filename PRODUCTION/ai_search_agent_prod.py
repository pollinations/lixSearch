import requests
import json
import datetime
import re
import time
import sys
from urllib.parse import urljoin, urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from pytube import YouTube, exceptions
# from pytube.request import default_range_size # Not needed for this fix
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
import math
import mimetypes
from tqdm import tqdm
import random
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from dotenv import load_dotenv
import html
from concurrent.futures import ThreadPoolExecutor
import threading
import logging
import concurrent.futures

MAX_SEARCH_RESULTS_PER_QUERY = 8
MAX_SCRAPE_WORD_COUNT = 2000
MAX_TOTAL_SCRAPE_WORD_COUNT = 8000
MIN_PAGES_TO_SCRAPE = 3
MAX_PAGES_TO_SCRAPE = 10
MAX_IMAGES_TO_INCLUDE = 3
MAX_TRANSCRIPT_WORD_COUNT = 6000
DUCKDUCKGO_REQUEST_DELAY = 3
REQUEST_RETRY_DELAY = 5
MAX_REQUEST_RETRIES = 3
MAX_DUCKDUCKGO_RETRIES = 5
CLASSIFICATION_MODEL = "OpenAI GPT-4.1-nano"
SYNTHESIS_MODEL = "openai-large"
MAX_CONCURRENT_REQUESTS = 5

query_pollinations_ai_show_log = False
get_youtube_transcript_show_log = False
get_youtube_video_metadata_show_log = False
scrape_website_show_log = False
plan_execution_llm_show_log = False
perform_duckduckgo_text_search_show_log = False
load_dotenv()

class DummyContextManager:
    """A dummy context manager to replace tqdm when logs are off."""
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        pass
    def set_postfix_str(self, *args, **kwargs): pass
    def update(self, *args, **kwargs): pass
    def close(self): pass

def conditional_tqdm(iterable, show_logs, *args, **kwargs):
    """Use tqdm only if show_logs is True."""
    if show_logs:
        return tqdm(iterable, *args, **kwargs)
    else:
        return iterable

def conditional_print(message, show_logs):
    """Print message only if show_logs is True."""
    if show_logs:
        print(message)

def exponential_backoff(attempt, base_delay=REQUEST_RETRY_DELAY, max_delay=60):
    """Calculate exponential backoff delay with jitter."""
    delay = min(max_delay, base_delay * (2 ** attempt))
    return delay + random.uniform(0, base_delay * 0.5)

def retry_operation(func, *args, retries=MAX_REQUEST_RETRIES, show_logs=True, **kwargs):
    """Retry an operation with exponential backoff."""
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < retries - 1:
                conditional_print(f"Attempt {attempt + 1}/{retries} failed for {func.__name__}: {type(e).__name__}: {e}. Retrying in {exponential_backoff(attempt):.2f}s.", show_logs)
                time.sleep(exponential_backoff(attempt))
            else:
                conditional_print(f"Attempt {attempt + 1}/{retries} failed for {func.__name__}: {type(e).__name__}: {e}. Max retries reached.", show_logs)
                raise # Re-raise the last exception

    # Should theoretically not be reached if retries > 0 and exception is always raised on last attempt
    raise RuntimeError(f"Operation {func.__name__} failed after {retries} attempts.")


def query_pollinations_ai(messages, model=SYNTHESIS_MODEL, retries=MAX_REQUEST_RETRIES, show_logs=query_pollinations_ai_show_log):
    """Query Pollinations AI API with retry logic."""
    def _query():
        payload = {
            "model": model,
            "messages": messages,
            "seed": random.randint(0, 2**31 - 1),
            "token" : os.getenv("POLLINATIONS_TOKEN"),
            "referrer" : os.getenv("POLLINATIONS_REFERRER")
        }
        url = "https://text.pollinations.ai/openai"
        headers = {
            "Content-Type": "application/json"
        }
        # Use a higher timeout for the AI query itself
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120) # Increased timeout
        response.raise_for_status()
        return response.json()

    try:
        return retry_operation(_query, retries=retries, show_logs=show_logs)
    except requests.exceptions.HTTPError as e:
        conditional_print(f"HTTP error {e.response.status_code} querying Pollinations AI ({model}): {e}. Failed after retries.", show_logs)
        # For specific client errors like 401, 403, 404, we might not want to retry in `retry_operation`,
        # but handling it here after retries are exhausted is fine.
        return None
    except Exception as e:
        conditional_print(f"Error querying Pollinations AI ({model}): {type(e).__name__}: {e}. Failed after retries.", show_logs)
        return None


def extract_urls_from_query(query):
    """Extract URLs from the query string and return the cleaned query and URL lists."""
    # Improved regex to be a bit more robust and non-greedy
    urls = re.findall(r'(https?://[^\s]+)', query) # Match http or https followed by non-whitespace chars
    cleaned_query = query
    website_urls = []
    youtube_urls = []

    for url in urls:
        # Remove the found URL from the query string
        cleaned_query = cleaned_query.replace(url, '').strip()
        # Simple check to avoid adding trailing punctuation attached to URLs
        url = url.rstrip('.,;!?"\'')

        parsed_url = urlparse(url)
        if "youtube.com" in parsed_url.netloc or "youtu.be" in parsed_url.netloc:
            youtube_urls.append(url)
        elif parsed_url.scheme in ['http', 'https']: # Ensure it's a valid web URL
            website_urls.append(url)
        # Remove consecutive spaces left by removal
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()


    return website_urls, youtube_urls, cleaned_query

def get_youtube_video_id(url):
    """Extract YouTube video ID from various YouTube URL formats."""
    parsed_url = urlparse(url)
    if "youtube.com" in parsed_url.netloc:
        # Handle watch, embed, v, etc.
        video_id = parse_qs(parsed_url.query).get('v')
        if video_id:
            return video_id[0]
        # Handle /v/ and /embed/ URLs
        if parsed_url.path:
            match = re.search(r'/(?:embed|v)/([^/?#&]+)', parsed_url.path)
            if match:
                return match.group(1)
    elif "youtu.be" in parsed_url.netloc:
        # Handle youtu.be short URLs
        if parsed_url.path and len(parsed_url.path) > 1:
            return parsed_url.path[1:].split('/')[0] # Take the part after /, before any other slashes
    return None

def get_youtube_transcript(video_id, show_logs=get_youtube_transcript_show_log):
    """Fetch and return the English transcript for a YouTube video ID."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try English first, then auto-generated English
        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
             conditional_print(f"No 'en' transcript found, trying 'a.en' for video ID: {video_id}", show_logs)
             try:
                transcript = transcript_list.find_transcript(['a.en'])
             except NoTranscriptFound:
                conditional_print(f"No 'a.en' transcript found for video ID: {video_id}", show_logs)
                return None


        entries = transcript.fetch()
        full_text = " ".join(entry['text'] for entry in entries)

        words = full_text.split()
        if len(words) > MAX_TRANSCRIPT_WORD_COUNT:
            # Trim to word count and ensure it ends cleanly
            return " ".join(words[:MAX_TRANSCRIPT_WORD_COUNT]) + "..."
        
        return full_text

    except NoTranscriptFound:
        conditional_print(f"No transcript available at all for video ID: {video_id}", show_logs)
    except TranscriptsDisabled:
        conditional_print(f"Transcripts are disabled for video ID: {video_id}", show_logs)
    except Exception as e:
        # Catch other potential errors during fetching or processing
        conditional_print(f"Unexpected error while fetching transcript for {video_id}: {type(e).__name__} - {e}", show_logs)

    return None


def get_youtube_video_metadata(url, show_logs=get_youtube_video_metadata_show_log):
    """Fetch YouTube video metadata using pytube."""
    video_id = get_youtube_video_id(url)
    if not video_id:
        conditional_print(f"[Pytube] Invalid URL provided for metadata: {url}", show_logs)
        return None

    try:
        yt = YouTube(url)
        # Accessing attributes forces fetching metadata
        _ = yt.title
        _ = yt.author
        _ = yt.publish_date
        _ = yt.length
        _ = yt.views
        _ = yt.description
        _ = yt.thumbnail_url

        metadata = {
            "title": yt.title,
            "author": yt.author,
            "publish_date": yt.publish_date.strftime("%Y-%m-%d %H:%M:%S") if yt.publish_date else "Unknown",
            "length": f"{yt.length // 60}m {yt.length % 60}s" if yt.length is not None else "Unknown",
            "views": f"{yt.views:,}" if yt.views is not None else "Unknown",
            "description": (
                yt.description[:500] + "..." if yt.description and len(yt.description) > 500 # Trim description
                else yt.description or "No description available"
            ),
            "thumbnail_url": yt.thumbnail_url,
            "url": url # Include the original URL
        }
        return metadata

    except exceptions.VideoUnavailable:
        conditional_print(f"[Pytube] VideoUnavailable: {url}", show_logs)
    except exceptions.LiveStreamError:
        conditional_print(f"[Pytube] LiveStreamError (likely a live stream or not fully processed yet): {url}", show_logs)
    except exceptions.RegexMatchError:
        conditional_print(f"[Pytube] RegexMatchError: Invalid or malformed URL - {url}", show_logs)
    except exceptions.ExtractError:
        conditional_print(f"[Pytube] ExtractError: Could not extract data for - {url}", show_logs)
    except Exception as e:
        # Catch any other unexpected exceptions during pytube operations
        conditional_print(f"[Pytube] Unexpected error for {url}: {type(e).__name__} - {e}", show_logs)

    return None


def is_likely_search_result_url(url):
    """Heuristic to check if a URL is likely from a search engine result page itself."""
    if not url:
        return False
    parsed_url = urlparse(url)
    # Check common search engine domains
    if any(domain in parsed_url.netloc for domain in ['google.com', 'duckduckgo.com', 'bing.com', 'yahoo.com', 'baidu.com', 'yandex.ru']):
        # Check common paths and query parameters for search results pages
        if parsed_url.path.startswith(('/search', '/html', '/res', '/web', '/url', '/clank')) or parsed_url.path == '/':
             return True
        query_params = parse_qs(parsed_url.query)
        if any(param in query_params for param in ['q', 'query', 'p', 'wd', 'text', 'url']):
             return True
    return False

def is_likely_image_url_heuristic(url):
    """Heuristic to filter out unlikely image URLs based on common patterns."""
    if not url:
        return False

    lower_url = url.lower()

    # Basic file extension check (handle query params)
    path_part = urlparse(url).path
    if not any(path_part.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg']):
        return False

    # Keywords often found in non-content images (icons, ads, etc.)
    if any(keyword in lower_url for keyword in ['/icon/', '/logo/', '/loader/', '/sprite/', '/thumbnail/', '/small/', '/avatar/', '/advert', '/ad_/', 'pixel', '1x1']):
        return False

    # Size patterns often found in thumbnails or tiny images
    if re.search(r'/\d+x\d+/', lower_url) or re.search(r'-\d+x\d+\.', lower_url):
        # Allow larger sizes, but filter out typical thumbnail sizes
        # This is a bit rough, could be improved
        size_matches = re.findall(r'(\d+)x(\d+)', lower_url)
        for w, h in size_matches:
             if int(w) <= 150 and int(h) <= 150: # Filter out small sizes
                 return False

    return True


def scrape_website(url, scrape_images=True, total_word_count_limit=MAX_TOTAL_SCRAPE_WORD_COUNT, show_logs=scrape_website_show_log):
    """Scrape text content and image URLs from a single website."""
    def _scrape():
        # This inner function uses variables from the outer scope (closure)
        # It does NOT need total_word_count_limit passed as an argument
        text_content = ""
        image_urls = []
        # Use a more common user agent
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

        try:
            response = requests.get(url, timeout=20, headers=headers) # Increased timeout
            response.raise_for_status() # Raise an exception for bad status codes

            # Check content type before proceeding with BeautifulSoup
            content_type = response.headers.get('Content-Type', '').lower()
            if not 'text/html' in content_type:
                 conditional_print(f"Skipping non-HTML content from {url} (Content-Type: {content_type})", show_logs)
                 return "", []

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove elements that are unlikely to contain main content or are navigational/boilerplate
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'noscript', 'iframe', 'svg']):
                element.extract()

            # Find main content area - check common tags
            main_content_elements = soup.find_all(['main', 'article', 'div'], class_=['main', 'content', 'article', 'post', 'body'])
            # If specific content areas aren't found, fall back to body or the whole soup
            if not main_content_elements:
                main_content_elements = [soup.find('body')] if soup.find('body') else [soup]

            temp_text = []
            word_count = 0
            # Process elements within main content areas
            for main_elem in main_content_elements:
                 if word_count >= total_word_count_limit: # Accessing total_word_count_limit from outer scope
                      break
                 for tag in main_elem.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote']): # Focus on paragraph-like and heading tags
                      if word_count >= total_word_count_limit: # Accessing total_word_count_limit from outer scope
                           break
                      text = tag.get_text()
                      # Clean up whitespace aggressively
                      text = re.sub(r'\s+', ' ', text).strip()
                      if text:
                          # Append text and update word count, checking limit
                          words = text.split()
                          words_to_add = words[:total_word_count_limit - word_count] # Accessing total_word_count_limit from outer scope
                          temp_text.append(" ".join(words_to_add))
                          word_count += len(words_to_add)

            text_content = '\n\n'.join(temp_text) # Use double newline for paragraphs

            # Append ellipsis if truncated
            if word_count >= total_word_count_limit and len(text_content.split()) > total_word_count_limit: # Accessing total_word_count_limit from outer scope
                 text_content = ' '.join(text_content.split()[:total_word_count_limit]) + '...' # Accessing total_word_count_limit from outer scope


            if scrape_images and len(image_urls) < MAX_IMAGES_TO_INCLUDE:
                # Find images in the main content areas or the whole page if main not found
                img_tags = []
                for main_elem in main_content_elements:
                     img_tags.extend(main_elem.find_all('img'))
                if not main_content_elements: # If main elements weren't found, search everywhere
                     img_tags.extend(soup.find_all('img'))


                processed_img_srcs = set() # To avoid duplicates from src/data-src on same img
                for img in img_tags:
                    if len(image_urls) >= MAX_IMAGES_TO_INCLUDE:
                        break
                    img_src = img.get('src')
                    img_data_src = img.get('data-src')

                    # Prioritize data-src if available
                    img_url = img_data_src if img_data_src else img_src

                    if img_url and img_url not in processed_img_srcs:
                        processed_img_srcs.add(img_url) # Mark as processed for this img tag

                        img_url_full = urljoin(url, img_url)
                        parsed_img_url = urlparse(img_url_full)

                        if parsed_img_url.scheme in ['http', 'https']:
                             # Apply heuristic filter
                             if is_likely_image_url_heuristic(img_url_full):
                                  image_urls.append(img_url_full)


            return text_content, image_urls

        except requests.exceptions.Timeout:
            conditional_print(f"Timeout scraping URL: {url}", show_logs)
            return "", []
        except requests.exceptions.RequestException as e:
             # Catch HTTP errors, connection errors, etc.
            conditional_print(f"Request error scraping URL: {url}: {type(e).__name__}: {e}", show_logs)
            return "", []
        except Exception as e:
             # Catch any other unexpected errors during scraping (e.g., BS4 parsing issues)
            conditional_print(f"Error processing URL: {url}: {type(e).__name__}: {e}", show_logs)
            return "", []


    # Retry the entire scrape operation
    try:
        # FIX: Removed the unexpected keyword argument `total_word_count_limit`
        return retry_operation(_scrape, retries=MAX_REQUEST_RETRIES, show_logs=show_logs)
    except Exception as e:
        # retry_operation already printed the last error, just return empty
        return "", []

def plan_execution_llm(user_query, website_urls, youtube_urls, cleaned_query, current_time_utc, location, show_logs=plan_execution_llm_show_log):
    """Use LLM to plan the information gathering strategy."""
    effective_cleaned_query = cleaned_query if cleaned_query else user_query

    # The prompt structure and desired JSON format remain the same for the planning model
    messages = [
        {"role": "system", "content": """You are an AI assistant designed to plan how to gather information to answer a user query. You have access to your native knowledge, the ability to process provided website and YouTube URLs, and the ability to perform web searches.

        Analyze the user's **original query**, the **provided URLs**, and the **cleaned query text** (without URLs). Determine the best strategy to find the answer.

        Consider these points:
        1.  **Primary Focus:** Is the query mainly about the content of the provided URLs, or is it a general question where URLs are supplementary, or unrelated?
        2.  **Native Knowledge:** What parts of the query can you answer using your own knowledge? Be specific about the *topic* or *aspect* you think you can answer natively. If none, state 'None'.
        3.  **Provided URLs:** Are the provided URLs relevant? Should they be processed (scraped for websites, transcript/metadata for YouTube)?
        4.  **Web Search:** Is a web search needed to find information *not* likely present in the provided URLs or your native knowledge? If so, formulate specific search queries. Avoid searching for information you expect to find in the provided URLs or your native knowledge.
        5.  **Search Results:** If web searches are needed, how many unique pages from the search results should be considered for scraping? Estimate between 3 and 10 (inclusive).

        Strictly respond in a structured, parseable JSON format:
        ```json
        {
          "native_parts": "String describing the topic/aspect answerable natively, or 'None'",
          "search_queries": ["query 1", "query 2", ...], // List of string queries, can be empty
          "scrape_provided_websites": true/false,
          "process_provided_youtube": true/false,
          "estimated_pages_to_scrape_from_search": 5, // Number between 3 and 10, only relevant if search_queries is not empty
          "query_focus": "URL Focused (Websites)" | "URL Focused (YouTube)" | "URL Focused (Both)" | "Mixed" | "Purely Native" | "Other Web Focused" | "Unclear" // Categorize the primary nature of the query
        }
        ```
        Ensure the JSON is valid and nothing outside the JSON block.
        Be concise in your descriptions and queries.

        Context:
        Current Time UTC: """ + current_time_utc + """
        Location (approximated): """ + location + """

        Original User Query: """ + user_query + """
        Provided Website URLs: """ + ", ".join(website_urls) if website_urls else "None" + """
        Provided YouTube URLs: """ + ", ".join(youtube_urls) if youtube_urls else "None" + """
        Cleaned Query Text (no URLs): """ + effective_cleaned_query
        },
        {"role": "user", "content": "Plan the execution strategy in the specified JSON format."}
    ]

    # Default plan in case AI fails or returns invalid format
    default_plan = {
        "native_parts": "None",
        "search_queries": [effective_cleaned_query] if effective_cleaned_query else [], # Search only if there's text
        "scrape_provided_websites": len(website_urls) > 0,
        "process_provided_youtube": len(youtube_urls) > 0,
        # If there's a query or provided URLs, assume we need to scrape some search results
        "estimated_pages_to_scrape_from_search": MAX_PAGES_TO_SCRAPE if effective_cleaned_query else 0,
        "query_focus": "Mixed" if (website_urls or youtube_urls) and effective_cleaned_query else ("URL Focused (Both)" if website_urls and youtube_urls else ("URL Focused (Websites)" if website_urls else ("URL Focused (YouTube)" if youtube_urls else "Purely Native" if not effective_cleaned_query and not (website_urls or youtube_urls) else "Other Web Focused")))
    }


    response = query_pollinations_ai(messages, model=CLASSIFICATION_MODEL, show_logs=show_logs)

    plan = default_plan # Start with default

    if response and 'choices' in response and len(response['choices']) > 0:
        ai_output = response['choices'][0]['message']['content'].strip()
        json_match = re.search(r"```json\n(.*)\n```", ai_output, re.DOTALL)
        if json_match:
            try:
                parsed_plan = json.loads(json_match.group(1))
                # Validate and sanitize parsed_plan, merging with defaults
                plan["native_parts"] = str(parsed_plan.get("native_parts", default_plan["native_parts"]))
                # Ensure search_queries is a list of non-empty strings
                plan["search_queries"] = [str(q).strip() for q in parsed_plan.get("search_queries", []) if isinstance(q, str) and q.strip()]
                plan["scrape_provided_websites"] = bool(parsed_plan.get("scrape_provided_websites", default_plan["scrape_provided_websites"]))
                plan["process_provided_youtube"] = bool(parsed_plan.get("process_provided_youtube", default_plan["process_provided_youtube"]))
                estimated = int(parsed_plan.get("estimated_pages_to_scrape_from_search", default_plan["estimated_pages_to_scrape_from_search"]))
                # Clamp estimated pages within bounds, but only if searching
                plan["estimated_pages_to_scrape_from_search"] = max(MIN_PAGES_TO_SCRAPE, min(MAX_PAGES_TO_SCRAPE, estimated)) if plan["search_queries"] else 0
                plan["query_focus"] = str(parsed_plan.get("query_focus", default_plan["query_focus"]))

                conditional_print("\n--- AI Execution Plan ---", show_logs)
                conditional_print(json.dumps(plan, indent=2), show_logs)
                conditional_print("-------------------------", show_logs)
                return plan
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                conditional_print(f"Error parsing AI plan JSON: {e}. Using default plan.", show_logs)
                conditional_print(f"AI output: {ai_output}", show_logs)
                # Fallback to default_plan already initialized
                return plan
        else:
            conditional_print("AI response did not contain valid JSON plan. Using default plan.", show_logs)
            conditional_print(f"AI output: {ai_output}", show_logs)
            # Fallback to default_plan already initialized
            return plan

    conditional_print("Could not get execution plan from AI. Using default plan.", show_logs)
    # Fallback to default_plan already initialized
    return plan


def perform_duckduckgo_text_search(query, max_results, retries=MAX_DUCKDUCKGO_RETRIES, show_logs=perform_duckduckgo_text_search_show_log):
    """Perform a DuckDuckGo text search with retry logic and delay."""
    def _search():
        # Add a delay before each search request to be polite and avoid rate limits
        time.sleep(DUCKDUCKGO_REQUEST_DELAY)
        with DDGS() as ddgs:
            # Using result.get() is safer in case a key is missing
            search_results = [{k: result.get(k) for k in ['title', 'href', 'body']} for result in ddgs.text(query, max_results=max_results)]
            # Consider adding a check if search_results is unexpectedly empty
            if not search_results:
                 conditional_print(f"DDGS returned empty results for query '{query}'.", show_logs)
                 return [] # Return empty list instead of raising error for empty results

            return search_results

    try:
        return retry_operation(_search, retries=retries, show_logs=show_logs)
    except requests.exceptions.HTTPError as e:
        # Specific handling for errors where retrying won't help immediately
        if e.response.status_code in [403, 429]:
             conditional_print(f"DDGS Received {e.response.status_code} for query '{query}'. Skipping after retries.", show_logs)
        else:
            conditional_print(f"DDGS HTTP error {e.response.status_code} for query '{query}'. Failed after retries.", show_logs)
        return []
    except Exception as e:
        conditional_print(f"DDGS Error performing text search for query '{query}' after retries: {type(e).__name__}: {e}.", show_logs)
        return []

def search_and_synthesize(user_input_query, show_sources=True, scrape_images=True, show_logs=True, output_format='markdown'):
    """
    Orchestrates the information gathering and synthesis process.

    Args:
        user_input_query (str): The original query from the user.
        show_sources (bool): Whether to include source information in the output.
                             For markdown output, this is appended text.
                             For json output, this adds a 'sources' key.
        scrape_images (bool): Whether to attempt scraping and reporting image URLs.
                              For markdown, appended text. For json, adds an 'images' key.
        show_logs (bool): Whether to print detailed progress logs.
        output_format (str): 'markdown' or 'json'. Determines the format of the
                             primary content returned and whether sources/images
                             are appended to content (markdown) or returned in
                             separate keys (json).

    Returns:
        tuple: (final_output, status_code, collected_data)
               final_output (str): The synthesized answer (markdown string).
               status_code (int): The HTTP status code reflecting success or failure.
               collected_data (dict): Dictionary containing gathered information (urls, images, native part).
    """
    collected_data = {
        "native_knowledge_part": None,
        "scraped_website_urls": [],
        "processed_youtube_urls": [],
        "failed_youtube_urls": [],
        "found_image_urls": []
    }
    final_markdown_output = ""
    status_code = 200

    if not user_input_query:
        collected_data["error"] = "No query provided."
        return "Error: No query provided.", 400, collected_data

    current_time_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    location = ""
    try:
        # Fetch location for planning context - handle potential errors gracefully
        location_response = retry_operation(requests.get, "https://ipinfo.io/json", timeout=5, retries=1, show_logs=False) # Don't log ipinfo errors
        if location_response and location_response.status_code == 200:
            location_data = location_response.json()
            location = location_data.get("city", "")
    except Exception:
        location = "" # Silently fail if location fetching fails

    # --- Special Handling for SUMMARIZE/SUMMARIZEW ---
    # Note: This special handling overrides the general planning and execution flow
    # and focuses specifically on the provided YouTube URL and optional web search.
    summarize_match = re.match(r'^(SUMMARIZE|SUMMARIZEW)\s+(https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(?:[-\w.!~*\'()@;:$+,?&/=#%]*))(?:\s+(.*))?$', user_input_query.strip(), re.IGNORECASE)

    if summarize_match:
        keyword = summarize_match.group(1).upper()
        youtube_url_for_summary = summarize_match.group(2)
        extra_query_text = summarize_match.group(3) if summarize_match.group(3) else ""

        video_id = get_youtube_video_id(youtube_url_for_summary)

        if not video_id:
            error_msg = f"Error: The provided URL '{youtube_url_for_summary}' is not a valid YouTube video URL for {keyword}."
            collected_data["error"] = error_msg
            return error_msg, 400, collected_data

        conditional_print(f"\n--- Special Handling: {keyword} ---", show_logs)
        conditional_print(f"Target YouTube URL: {youtube_url_for_summary}", show_logs)
        if extra_query_text:
            conditional_print(f"Additional Query: '{extra_query_text}'", show_logs)
        else:
            conditional_print("No additional query text.", show_logs)

        youtube_transcripts_content = None
        youtube_metadata = None

        conditional_print("Fetching YouTube data...", show_logs)

        # Fetch transcript and metadata. Retries are handled inside the functions.
        transcript = get_youtube_transcript(video_id, show_logs=show_logs)
        metadata = get_youtube_video_metadata(youtube_url_for_summary, show_logs=show_logs)

        if transcript:
            youtube_transcripts_content = f"--- YouTube Transcript Content from {youtube_url_for_summary} ---\n{transcript}\n\n"
        if metadata:
             # Include metadata in content if transcript exists, or as the main content if not
             metadata_string = f"Title: {metadata.get('title', 'N/A')}\n"
             metadata_string += f"Author: {metadata.get('author', 'N/A')}\n"
             metadata_string += f"Published: {metadata.get('publish_date', 'N/A')}\n"
             metadata_string += f"Length: {metadata.get('length', 'N/A')}\n"
             metadata_string += f"Views: {metadata.get('views', 'N/A')}\n"
             metadata_string += f"Description: {metadata.get('description', 'N/A')}\n\n"

             if youtube_transcripts_content:
                  # Prepend metadata to transcript content
                  youtube_transcripts_content = f"--- Metadata from YouTube: {youtube_url_for_summary} ---\n{metadata_string}" + youtube_transcripts_content
             else:
                  # If no transcript, metadata becomes the primary content for this URL
                  youtube_transcripts_content = f"--- Metadata from YouTube: {youtube_url_for_summary} ---\n{metadata_string}No transcript available.\n\n"


        if youtube_transcripts_content:
             collected_data["processed_youtube_urls"].append(youtube_url_for_summary)
        else:
             collected_data["failed_youtube_urls"].append(youtube_url_for_summary)
             conditional_print(f"Could not get transcript or metadata for {youtube_url_for_summary} after attempts.", show_logs)

        scraped_text_content = ""
        total_scraped_words = 0
        search_urls = []

        # SUMMARIZEW means perform a web search for the extra text
        if keyword == "SUMMARIZEW" and extra_query_text:
             conditional_print("\nPerforming Web Search for additional query...", show_logs)
             # Perform search, retries handled internally
             search_results = perform_duckduckgo_text_search(extra_query_text, max_results=MAX_SEARCH_RESULTS_PER_QUERY, show_logs=show_logs)
             search_urls = [result.get('href') for result in search_results if result and result.get('href')]

             unique_search_urls = []
             # Filter out search engine result page URLs and duplicates
             for url in search_urls:
                 if url and urlparse(url).scheme in ['http', 'https'] and not is_likely_search_result_url(url) and url not in unique_search_urls:
                     unique_search_urls.append(url)

             # For SUMMARIZEW + extra query, we'll scrape a fixed small number of pages, maybe MAX_PAGES_TO_SCRAPE/2
             pages_to_scrape_count = min(MAX_PAGES_TO_SCRAPE // 2, len(unique_search_urls))
             pages_to_scrape_count = max(pages_to_scrape_count, 1) if unique_search_urls else 0 # Scrape at least 1 if URLs exist


             if unique_search_urls and pages_to_scrape_count > 0 and total_scraped_words < MAX_TOTAL_SCRAPE_WORD_COUNT:
                 conditional_print(f"Scraping up to {pages_to_scrape_count} pages from search results for additional query...", show_logs)
                 if show_logs:
                    pbar = tqdm(total=pages_to_scrape_count, desc="Scraping Search Results (SUMMARIZEW)", unit="page")
                 else:
                    pbar = DummyContextManager()

                 urls_for_scraping_search = unique_search_urls[:pages_to_scrape_count]

                 # Use ThreadPoolExecutor for concurrent scraping
                 with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, pages_to_scrape_count)) as executor, pbar as pb:
                     future_to_url = {executor.submit(scrape_website, url, scrape_images, MAX_TOTAL_SCRAPE_WORD_COUNT - total_scraped_words, show_logs): url for url in urls_for_scraping_search}

                     for future in concurrent.futures.as_completed(future_to_url):
                          if total_scraped_words >= MAX_TOTAL_SCRAPE_WORD_COUNT:
                             # Cancel remaining futures if word limit reached
                             for pending_future in future_to_url.keys():
                                pending_future.cancel()
                             pb.set_postfix_str("Total word limit reached")
                             break # Exit loop

                          url = future_to_url[future]
                          try:
                              # Pass total_word_count_limit correctly to individual scrapes
                              content, images = future.result() # Result includes content and images
                              if content:
                                  # Need to manage total_scraped_words atomically if multithreading,
                                  # but since MAX_TOTAL_SCRAPE_WORD_COUNT is checked *before* submitting
                                  # new tasks and *before* processing results, a simple update here
                                  # is *mostly* safe, though might slightly exceed the limit.
                                  # For strict adherence, a lock would be needed. Let's assume slight overshoot is acceptable.
                                  content_words = len(content.split())
                                  if total_scraped_words + content_words > MAX_TOTAL_SCRAPE_WORD_COUNT:
                                       remaining_words = MAX_TOTAL_SCRAPE_WORD_COUNT - total_scraped_words
                                       content = ' '.join(content.split()[:remaining_words]) + '...' if remaining_words > 0 else ""
                                       content_words = len(content.split()) if remaining_words > 0 else 0 # Update actual added word count

                                  if content_words > 0:
                                       scraped_text_content += f"\n\n--- Content from {url} ---\n{content}"
                                       total_scraped_words += content_words
                                       collected_data["scraped_website_urls"].append(url)

                                       if scrape_images:
                                            for img_url in images:
                                                if len(collected_data["found_image_urls"]) < MAX_IMAGES_TO_INCLUDE and img_url not in collected_data["found_image_urls"]:
                                                    collected_data["found_image_urls"].append(img_url)
                                                elif len(collected_data["found_image_urls"]) >= MAX_IMAGES_TO_INCLUDE:
                                                     break # Stop adding images

                                       pb.set_postfix_str(f"Scraped {content_words} words from {urlparse(url).hostname}")
                                  else:
                                      pb.set_postfix_str(f"Scraped 0 words from {urlparse(url).hostname}")

                              else:
                                  pb.set_postfix_str(f"Failed to scrape {urlparse(url).hostname}")

                          except Exception as exc:
                              conditional_print(f"Scraping {url} generated an exception: {exc}", show_logs)
                              pb.set_postfix_str(f"Error scraping {urlparse(url).hostname}")

                          pb.update(1) # Update progress bar regardless of success

                 if total_scraped_words >= MAX_TOTAL_SCRAPE_WORD_COUNT:
                      conditional_print("Reached total scraped word limit.", show_logs)
             elif not unique_search_urls:
                 conditional_print("No unique search result URLs found for scraping.", show_logs)
             else:
                 conditional_print("Scraping skipped (0 pages estimated or total word limit already met).", show_logs)


        # Build the synthesis prompt for special cases
        synthesis_prompt_special = """You are a helpful assistant. Synthesize an answer based *only* on the provided information from the YouTube transcript and potentially scraped web pages.

        Present the answer in Markdown format.

        Incorporate the information logically and provide the required answer contextually.
        Include dates, times, and specific details from the provided content where relevant. Pay close attention to time zones when dealing with time-related queries and convert times to be relative to the provided Current Time UTC if necessary based on the scraped data.

        **Important:** When citing information derived directly from a source, include a brief inline citation or reference to the source URL where appropriate within the synthesized answer. For example: "According to [Source URL], ..." or "The video at [YouTube URL] explains that...". Aim for a natural flow, not excessive citations.

        If the provided information from all sources is insufficient to answer *any* part of the query, state that you could not find a definitive answer based on the available data. If a *specific* provided URL could not be processed or yielded no information, mention that you were unable to get information from that source.

        Avoid mentioning the web search, scraping, or transcript fetching process explicitly in the final answer (except for the inline citations and mentioning inability to process specific sources if applicable).

        User Query: """ + user_input_query + """

        Current Time UTC: """ + current_time_utc + """

        Provided Information:
        """
        if youtube_transcripts_content:
             synthesis_prompt_special += youtube_transcripts_content # Already formatted with header inside the variable
        elif youtube_url_for_summary in collected_data["failed_youtube_urls"]:
             synthesis_prompt_special += f"--- YouTube Content from {youtube_url_for_summary} ---\nCould not retrieve transcript or metadata for this video after multiple attempts.\n\n"
        else:
             synthesis_prompt_special += f"--- YouTube Content from {youtube_url_for_summary} ---\nNo content available (unexpected error).\n\n"


        if scraped_text_content:
             synthesis_prompt_special += f"--- Scraped Web Page Content (for additional query) ---\n{scraped_text_content}\n"
        elif keyword == "SUMMARIZEW" and extra_query_text:
             # Only mention failure if a search/scrape was intended for extra text
             synthesis_prompt_special += f"--- Scraped Web Page Content (for additional query) ---\nAttempted to scrape web content for '{extra_query_text}' but could not retrieve information.\n\n"
        else:
             synthesis_prompt_special += "--- Scraped Web Page Content (for additional query) ---\nNone available or processed.\n\n"

        # Add specific instruction based on the keyword and available data
        if keyword == "SUMMARIZE":
            if youtube_transcripts_content:
                synthesis_prompt_special += "Please summarize the provided YouTube transcript in detail."
            else:
                 synthesis_prompt_special += "No YouTube transcript was provided or could be retrieved after multiple attempts. Therefore, I cannot summarize it."
        elif keyword == "SUMMARIZEW":
            if youtube_transcripts_content and scraped_text_content:
                 synthesis_prompt_special += f"Synthesize information from the YouTube transcript and the scraped web content to answer '{extra_query_text}'. Focus primarily on the video content but include relevant details from the web results."
            elif youtube_transcripts_content:
                 synthesis_prompt_special += f"Synthesize information from the YouTube transcript to answer '{extra_query_text}'. No relevant web content was found or retrieved after multiple attempts."
            elif scraped_text_content:
                 synthesis_prompt_special += f"No YouTube transcript was provided or could be retrieved after multiple attempts. Synthesize information from the scraped web content to answer '{extra_query_text}'."
            else:
                 synthesis_prompt_special += f"No YouTube transcript or web content was provided or could be retrieved after multiple attempts. I cannot answer '{extra_query_text}'."


        final_answer_messages_special = [
            {"role": "system", "content": synthesis_prompt_special},
            {"role": "user", "content": "Synthesize the final answer in Markdown based on the provided information."}
        ]

        # Determine status based on whether *any* content was gathered
        if not youtube_transcripts_content and not scraped_text_content:
             status_code = 404 # Not Found if no content at all
             final_markdown_output = "--- No Information Found ---\nCould not retrieve transcript or metadata for the YouTube video after multiple attempts, and no other relevant information was found or requested.\n---------------------------"
        else:
            if show_logs:
                pbar = tqdm(total=1, desc="Synthesizing Answer (Special)", unit="step")
            else:
                pbar = DummyContextManager()

            with pbar as pb:
                final_answer_response = query_pollinations_ai(final_answer_messages_special, model=SYNTHESIS_MODEL, show_logs=show_logs)
                if final_answer_response and 'choices' in final_answer_response and len(final_answer_response['choices']) > 0:
                    final_markdown_output = final_answer_response['choices'][0]['message']['content']
                    pb.set_postfix_str("Success")
                else:
                    pb.set_postfix_str("Failed")
                    status_code = 500 # Synthesis failed
                    final_markdown_output = "--- Synthesis Failed ---\nAn error occurred during the synthesis process, but some information was gathered:\n\n"
                    if youtube_transcripts_content: final_markdown_output += "**YouTube Content:** Available\n"
                    if scraped_text_content: final_markdown_output += "**Scraped Web Content:** Available\n"
                    if collected_data["failed_youtube_urls"]: final_markdown_output += f"**Failed to process YouTube:** {', '.join(collected_data['failed_youtube_urls'])}\n"
                    final_markdown_output += "\nConsider retrying the query."

                pb.update(1)

        # Append sources/images for markdown output format if requested
        if output_format == 'markdown' and show_sources and (collected_data["processed_youtube_urls"] or collected_data["scraped_website_urls"] or collected_data["failed_youtube_urls"] or collected_data["found_image_urls"]):
             final_markdown_output += "\n\n## Sources\n"
             if collected_data["processed_youtube_urls"]:
                  final_markdown_output += "### Transcript Source (Processed YouTube Video)\n"
                  for url in collected_data["processed_youtube_urls"]:
                      final_markdown_output += f"- {url}\n"
             if collected_data["failed_youtube_urls"]:
                  final_markdown_output += "### Failed YouTube Source\n"
                  final_markdown_output += f"Could not retrieve content for: {', '.join(collected_data['failed_youtube_urls'])}\n"
             if collected_data["scraped_website_urls"]:
                 final_markdown_output += "### Text Sources (Scraped Websites)\n"
                 for url in collected_data["scraped_website_urls"]:
                     final_markdown_output += f"- {url}\n"
             if scrape_images and collected_data["found_image_urls"]:
                 final_markdown_output += "### Images Found on Scraped Pages\n"
                 # Basic check to ensure images aren't ridiculously long URLs
                 for url in collected_data["found_image_urls"]:
                      final_markdown_output += f"- {url[:200]}{'...' if len(url) > 200 else ''}\n"
             elif scrape_images and (collected_data["scraped_website_urls"] or collected_data["processed_youtube_urls"]):
                  final_markdown_output += "### Images Found on Scraped Pages\n"
                  final_markdown_output += "No relevant images found on processed sources within limits.\n"

             final_markdown_output += "---\n"

        return final_markdown_output, status_code, collected_data

    # --- End Special Handling ---


    # --- Standard Handling ---
    website_urls, youtube_urls, cleaned_query = extract_urls_from_query(user_input_query)

    plan = plan_execution_llm(user_input_query, website_urls, youtube_urls, cleaned_query, current_time_utc, location, show_logs=show_logs)

    native_answer_content = ""
    if plan.get("native_parts") and plan["native_parts"] != "None":
        collected_data["native_knowledge_part"] = plan["native_parts"]
        if show_logs:
             pbar = tqdm(total=1, desc="Getting Native Answer", unit="step")
        else:
             pbar = DummyContextManager()

        with pbar as pb:
            # Ask the AI a specific question based on the identified native part
            # Including the cleaned query provides context
            native_answer_messages = [
                {"role": "system", "content": f"Answer the following question in detail with proper description based on your knowledge. Focus on the topic: {plan['native_parts']}"},
                {"role": "user", "content": cleaned_query} # Use cleaned query for context
            ]
            native_answer_response = query_pollinations_ai(native_answer_messages, model=SYNTHESIS_MODEL, show_logs=show_logs)
            if native_answer_response and 'choices' in native_answer_response and len(native_answer_response['choices']) > 0:
                native_answer_content = native_answer_response['choices'][0]['message']['content']
                pb.set_postfix_str("Success")
            else:
                pb.set_postfix_str("Failed")
                conditional_print("Warning: Could not get native answer.", show_logs)
            pb.update(1)


    # Initialize the list that will hold strings and the main content variable
    youtube_content_strings = []
    processed_youtube_urls = []
    failed_youtube_urls = []
    youtube_transcripts_content = "" # Initialize here to avoid UnboundLocalError


    if plan.get("process_provided_youtube") and youtube_urls:
        conditional_print("\nProcessing YouTube URLs...", show_logs)
        if show_logs:
            pbar = tqdm(total=len(youtube_urls), desc="Processing YouTube URLs", unit="video")
        else:
            pbar = DummyContextManager()

        # We need to fetch both transcript and metadata, potentially concurrently per URL,
        # then combine them per URL before joining into youtube_transcripts_content.
        # Let's fetch transcripts and metadata separately first, then combine.
        # This approach ensures we have all results before formatting the strings.

        # Store futures for transcripts and metadata fetches
        transcript_futures = {}
        metadata_futures = {}

        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(youtube_urls) * 2)) as executor: # Allow slightly more workers for paired tasks
            for url in youtube_urls:
                video_id = get_youtube_video_id(url)
                if video_id:
                    # Submit transcript fetch
                    transcript_futures[executor.submit(get_youtube_transcript, video_id, show_logs)] = url
                    # Submit metadata fetch
                    metadata_futures[executor.submit(get_youtube_video_metadata, url, show_logs)] = url
                else:
                    # Add invalid URL to failed list immediately
                    collected_data["failed_youtube_urls"].append(url)
                    conditional_print(f"Skipped invalid YouTube URL: {url}", show_logs)


            # Collect results as they complete
            completed_transcripts = {}
            completed_metadata = {}
            # Use a combined list of futures to track overall progress
            all_futures = list(transcript_futures.keys()) + list(metadata_futures.keys())

            if show_logs:
                 processing_pbar = tqdm(total=len(all_futures), desc="Fetching YouTube Data", unit="item")
            else:
                 processing_pbar = DummyContextManager()

            with processing_pbar as pb:
                 for future in concurrent.futures.as_completed(all_futures):
                     url = transcript_futures.get(future) or metadata_futures.get(future)
                     video_id = get_youtube_video_id(url)
                     task_type = "Transcript" if future in transcript_futures else "Metadata"

                     try:
                         result = future.result()
                         if task_type == "Transcript":
                             completed_transcripts[url] = result
                             pb.set_postfix_str(f"Fetched Transcript for {video_id or url}")
                         else: # Metadata
                             completed_metadata[url] = result
                             pb.set_postfix_str(f"Fetched Metadata for {video_id or url}")

                     except Exception as exc:
                         conditional_print(f"{task_type} fetch for {url} generated exception: {exc}", show_logs)
                         pb.set_postfix_str(f"Error fetching {task_type} for {video_id or url}")
                         # Note: We don't add to failed_youtube_urls here, we'll do that after checking if *either* transcript or metadata succeeded.
                     finally:
                        pb.update(1) # Update progress bar for each completed task


        # Now combine fetched transcripts and metadata per URL and format the string
        youtube_content_strings = []
        processed_youtube_urls = []

        for url in youtube_urls:
             video_id = get_youtube_video_id(url)
             if not video_id: # Already handled invalid URLs before submitting futures
                 continue

             transcript = completed_transcripts.get(url)
             metadata = completed_metadata.get(url)

             if transcript or metadata:
                 content_string = f"\n\n--- Content from YouTube: {url} ---\n"
                 if metadata:
                     content_string += f"Title: {metadata.get('title', 'N/A')}\n"
                     content_string += f"Author: {metadata.get('author', 'N/A')}\n"
                     content_string += f"Published: {metadata.get('publish_date', 'N/A')}\n"
                     content_string += f"Length: {metadata.get('length', 'N/A')}\n"
                     content_string += f"Views: {metadata.get('views', 'N/A')}\n"
                     content_string += f"Description: {metadata.get('description', 'N/A')}\n\n"
                 if transcript:
                     content_string += transcript
                 else:
                     content_string += "Transcript not available.\n" # Indicate if only metadata was found

                 youtube_content_strings.append(content_string)
                 processed_youtube_urls.append(url)
             else:
                 # If neither transcript nor metadata was successfully fetched for this URL
                 collected_data["failed_youtube_urls"].append(url)
                 conditional_print(f"Could not get transcript or metadata for {url} after attempts.", show_logs)


        # Assign the combined string regardless of whether any content was found
        youtube_transcripts_content = "".join(youtube_content_strings)
        collected_data["processed_youtube_urls"] = processed_youtube_urls # Update collected_data with successful ones


    scraped_text_content = ""
    total_scraped_words = 0
    urls_to_scrape = []

    if plan.get("scrape_provided_websites") and website_urls:
         urls_to_scrape.extend(website_urls)

    search_urls = []
    if plan.get("search_queries"):
         conditional_print("\nPerforming Web Search...", show_logs)
         all_search_results = []
         # Use ThreadPoolExecutor for concurrent search queries
         with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(plan["search_queries"]))) as executor:
             search_futures = {executor.submit(perform_duckduckgo_text_search, query, MAX_SEARCH_RESULTS_PER_QUERY, MAX_DUCKDUCKGO_RETRIES, show_logs): query for query in plan["search_queries"]}

             for future in concurrent.futures.as_completed(search_futures):
                 query = search_futures[future]
                 try:
                     results = future.result()
                     all_search_results.extend(results)
                 except Exception as exc:
                     conditional_print(f"Search for '{query}' generated an exception: {exc}", show_logs)

         # Collect hrefs from all search results
         search_urls = [result.get('href') for result in all_search_results if result and result.get('href')]


    # Add search result URLs to the scraping list, ensuring uniqueness and filtering out search result pages
    unique_search_urls = []
    for url in search_urls:
        if url and urlparse(url).scheme in ['http', 'https'] and not is_likely_search_result_url(url) and url not in unique_search_urls and url not in urls_to_scrape: # Also check against provided URLs
            unique_search_urls.append(url)

    estimated_pages_from_search = plan.get("estimated_pages_to_scrape_from_search", 0)

    # Combine provided URLs marked for scraping with search results (up to the estimated limit)
    urls_for_scraping = []
    # Add provided websites first
    if plan.get("scrape_provided_websites"):
        urls_for_scraping.extend([url for url in website_urls if urlparse(url).scheme in ['http', 'https']]) # Only add valid URLs

    # Add unique search result URLs up to the estimated limit
    search_result_urls_to_add = [url for url in unique_search_urls if url not in urls_for_scraping]
    urls_for_scraping.extend(search_result_urls_to_add[:estimated_pages_from_search])

    # Ensure final list is unique (e.g., if a provided URL also appeared in search results)
    urls_for_scraping = list(dict.fromkeys(urls_for_scraping))

    if urls_for_scraping and total_scraped_words < MAX_TOTAL_SCRAPE_WORD_COUNT:
        conditional_print(f"\nScraping {len(urls_for_scraping)} Web Pages...", show_logs)
        if show_logs:
            pbar = tqdm(total=len(urls_for_scraping), desc="Scraping Websites", unit="page")
        else:
            pbar = DummyContextManager()

        with ThreadPoolExecutor(max_workers=min(MAX_CONCURRENT_REQUESTS, len(urls_for_scraping))) as executor, pbar as pb:
            # Use a dictionary to map futures to URLs to process results as they complete
            # Pass remaining word count limit to each scrape job
            future_to_url = {executor.submit(scrape_website, url, scrape_images, MAX_TOTAL_SCRAPE_WORD_COUNT - total_scraped_words, show_logs): url for url in urls_for_scraping}

            for future in concurrent.futures.as_completed(future_to_url):
                # Re-check word limit before processing results from completed futures
                if total_scraped_words >= MAX_TOTAL_SCRAPE_WORD_COUNT:
                    # Cancel remaining futures if word limit reached
                    # This is a best effort, some futures might already be running/completed
                    for pending_future in future_to_url.keys():
                         if not pending_future.done():
                             pending_future.cancel()
                    # Ensure pbar reflects that scraping stopped early
                    if pb.n < pb.total: # Only update if not already finished
                         pb.update(pb.total - pb.n) # Fill progress bar
                         pb.set_postfix_str("Total word limit reached, scraping stopped")

                    break # Exit loop

                url = future_to_url[future]
                try:
                    # Pass total_word_count_limit correctly to individual scrapes
                    content, images = future.result() # Result includes content and images

                    if content:
                        # Need to manage total_scraped_words atomically if multithreading,
                        # but since MAX_TOTAL_SCRAPE_WORD_COUNT is checked *before* submitting
                        # new tasks and *before* processing results, a simple update here
                        # is *mostly* safe, though might slightly exceed the limit.
                        # For strict adherence, a lock would be needed. Let's assume slight overshoot is acceptable.
                        content_words = len(content.split())
                        # Trim content if adding it would exceed the total limit
                        if total_scraped_words + content_words > MAX_TOTAL_SCRAPE_WORD_COUNT:
                             remaining_words = MAX_TOTAL_SCRAPE_WORD_COUNT - total_scraped_words
                             # Ensure remaining_words is not negative
                             if remaining_words <= 0:
                                  content = ""
                                  content_words = 0
                             else:
                                 content = ' '.join(content.split()[:remaining_words]) + '...'
                                 content_words = len(content.split()) # Update actual added word count

                        if content_words > 0:
                             scraped_text_content += f"\n\n--- Content from {url} ---\n{content}"
                             total_scraped_words += content_words
                             collected_data["scraped_website_urls"].append(url)

                             if scrape_images:
                                  # Add images if within the MAX_IMAGES_TO_INCLUDE limit
                                  for img_url in images:
                                     if len(collected_data["found_image_urls"]) < MAX_IMAGES_TO_INCLUDE and img_url not in collected_data["found_image_urls"]:
                                         collected_data["found_image_urls"].append(img_url)
                                     elif len(collected_data["found_image_urls"]) >= MAX_IMAGES_TO_INCLUDE:
                                          break # Stop adding images

                             pb.set_postfix_str(f"Scraped {content_words} words from {urlparse(url).hostname}")
                        else:
                             pb.set_postfix_str(f"Scraped 0 words from {urlparse(url).hostname}")

                    else:
                         # If content is empty, it failed or was blocked/non-html
                         pb.set_postfix_str(f"Failed to scrape {urlparse(url).hostname}")

                except Exception as exc:
                    conditional_print(f"Scraping {url} generated an exception: {exc}", show_logs)
                    pb.set_postfix_str(f"Error scraping {urlparse(url).hostname}")

                pb.update(1) # Update progress bar regardless of success, as task is complete

            if total_scraped_words >= MAX_TOTAL_SCRAPE_WORD_COUNT:
                 conditional_print("Reached total scraped word limit.", show_logs)
    elif not urls_for_scraping:
         conditional_print("\nNo unique URLs found for scraping or plan specified no scraping.", show_logs)
    else:
         conditional_print("\nScraping skipped (total word limit already met before starting).", show_logs)


    conditional_print("\nSending Information to AI for Synthesis...", show_logs)

    # Check if there's any relevant content to synthesize from
    # This check is now safe because youtube_transcripts_content is initialized to ""
    if not native_answer_content and not scraped_text_content and not youtube_transcripts_content:
         status_code = 404 # Not Found
         final_markdown_output = "--- No Information Found ---\nCould not find enough information (either natively, from web searches/provided URLs, or YouTube transcripts) to answer the query after multiple attempts.\n---------------------------"
    else:
        synthesis_prompt = """You are a helpful assistant. Synthesize a comprehensive, detailed, and confident answer to the user's original query based *only* on the provided information from native knowledge, scraped web pages, and YouTube transcripts.

        Present the answer in Markdown format.

        Incorporate the information logically and provide the required answer contextually.
        Include dates, times, and specific details from the provided content where relevant to make the information more lively and grounded in the sources. Pay close attention to time zones when dealing with time-related queries and convert times to be relative to the provided Current Time UTC if necessary based on the scraped data.

        **Important:** When citing information derived directly from a source (scraped website or YouTube content), include a brief inline citation or reference to the source URL where appropriate within the synthesized answer. For example: "According to [Source URL], ..." or "The video at [YouTube URL] explains that...". Aim for a natural flow, not excessive citations.

        If the provided information from all sources is insufficient to answer *any* part of the query, state that you could not find a definitive answer based on the available data. If a *specific* provided URL could not be processed or yielded no information, mention that you were unable to get information from that source.

        Avoid mentioning the web search, scraping, or transcript fetching process explicitly in the final answer (except for the inline citations and mentioning inability to process specific sources if applicable).

        User Query: """ + user_input_query + """

        Current Time UTC: """ + current_time_utc + """

        Provided Information:
        """

        if native_answer_content:
            synthesis_prompt += f"--- Native Knowledge ({collected_data.get('native_knowledge_part', 'Topic')}) ---\n{native_answer_content}\n\n"
        else:
            synthesis_prompt += "--- Native Knowledge ---\nNone available or requested.\n\n"

        if youtube_transcripts_content:
             synthesis_prompt += f"--- YouTube Transcript and Metadata Content ---\n{youtube_transcripts_content}\n\n" # Already formatted with headers inside variable
        elif plan.get("process_provided_youtube") and youtube_urls:
            # Only mention failure if processing YouTube was part of the plan
            synthesis_prompt += "--- YouTube Transcript and Metadata Content ---\nAttempted to process provided YouTube URLs but could not retrieve content or metadata after multiple attempts.\n\n"
        else:
             synthesis_prompt += "--- YouTube Transcript and Metadata Content ---\nNone available or processed.\n\n"

        if scraped_text_content:
            synthesis_prompt += f"--- Scraped Web Page Content ---\n{scraped_text_content}\n"
        elif (plan.get("scrape_provided_websites") and website_urls) or (plan.get("search_queries") and search_urls):
            # Only mention failure if scraping websites was part of the plan
             synthesis_prompt += "--- Scraped Web Page Content ---\nAttempted to scrape web content but could not retrieve information.\n\n"
        else:
             synthesis_prompt += "--- Scraped Web Page Content ---\nNone available or processed.\n\n"


        final_answer_messages = [
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": "Synthesize the final answer in Markdown based on the provided information, including inline citations."}
        ]

        if show_logs:
            pbar = tqdm(total=1, desc="Synthesizing Answer", unit="step")
        else:
            pbar = DummyContextManager()

        with pbar as pb:
            final_answer_response = query_pollinations_ai(final_answer_messages, model=SYNTHESIS_MODEL, show_logs=show_logs)
            if final_answer_response and 'choices' in final_answer_response and len(final_answer_response['choices']) > 0:
                final_markdown_output = final_answer_response['choices'][0]['message']['content']
                pb.set_postfix_str("Success")
            else:
                pb.set_postfix_str("Failed")
                status_code = 500 # Synthesis failed
                final_markdown_output = "--- Synthesis Failed ---\nAn error occurred during the synthesis process, but some information was gathered:\n\n"
                if native_answer_content: final_markdown_output += "**Native Knowledge:** Available\n"
                if youtube_transcripts_content: final_markdown_output += "**YouTube Content:** Available\n"
                if scraped_text_content: final_markdown_output += "**Scraped Web Content:** Available\n"
                if collected_data["failed_youtube_urls"]: final_markdown_output += f"**Failed to process YouTube:** {', '.join(collected_data['failed_youtube_urls'])}\n"
                final_markdown_output += "\nConsider retrying the query."

            pb.update(1)

    # Append sources/images for markdown output format if requested
    if output_format == 'markdown' and show_sources and (collected_data["native_knowledge_part"] != "None" or collected_data["scraped_website_urls"] or collected_data["processed_youtube_urls"] or collected_data["failed_youtube_urls"] or collected_data["found_image_urls"]):
         final_markdown_output += "\n\n## Sources\n"
         if collected_data["native_knowledge_part"] != "None":
              final_markdown_output += "### Answered Natively\n"
              final_markdown_output += f"Parts of the query related to: {collected_data['native_knowledge_part']}\n"
         if collected_data["scraped_website_urls"]:
             final_markdown_output += "### Text Sources (Scraped Websites)\n"
             for url in collected_data["scraped_website_urls"]:
                 final_markdown_output += f"- {url}\n"
         if collected_data["processed_youtube_urls"]:
              final_markdown_output += "### Transcript Sources (Processed YouTube Videos)\n"
              for url in collected_data["processed_youtube_urls"]:
                  final_markdown_output += f"- {url}\n"
         if collected_data["failed_youtube_urls"]:
              final_markdown_output += "### Failed YouTube Sources\n"
              final_markdown_output += f"Could not retrieve content for: {', '.join(collected_data['failed_youtube_urls'])}\n"

         if scrape_images and collected_data["found_image_urls"]:
             final_markdown_output += "### Images Found on Scraped Pages\n"
             # Basic check to ensure images aren't ridiculously long URLs
             for url in collected_data["found_image_urls"]:
                 final_markdown_output += f"- {url[:200]}{'...' if len(url) > 200 else ''}\n"
         elif scrape_images and (collected_data["scraped_website_urls"] or collected_data["processed_youtube_urls"]):
              # Only mention "no images found" if scraping/processing happened
              if collected_data["scraped_website_urls"] or collected_data["processed_youtube_urls"]:
                   final_markdown_output += "### Images Found on Scraped Pages\n"
                   final_markdown_output += "No relevant images found on processed sources within limits.\n"


         final_markdown_output += "---\n"

    return final_markdown_output, status_code, collected_data

app = Flask(__name__)
CORS(app)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri="memory://", # Use a persistent storage for production if needed
    strategy="moving-window"
)

# Use a separate executor for the potentially long search/scrape/synth task
process_executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
# semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS) # Semaphore is handled by the executor's max_workers

# Add basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@limiter.limit("10 per minute")
@app.route('/search', methods=['GET', 'POST'])
def handle_search():
    user_input_query = None
    show_logs = True # Default to True unless explicitly set
    scrape_images = True # Default to True unless explicitly set
    show_sources = True # Default to True unless explicitly set
    output_format = 'markdown' # Default output format

    collected_data = {} # Dictionary to store source/image data for JSON output

    if request.method == 'POST':
        try:
            data = request.get_json()
            if not isinstance(data, dict):
                 return jsonify({"error": "Invalid JSON payload."}), 400

            # Check if it's the OpenAI-style messages format
            if 'messages' in data and isinstance(data['messages'], list):
                output_format = 'json'
                messages = data['messages']
                # Find the latest user message for the query
                user_message_content = None
                for message in reversed(messages):
                    if isinstance(message, dict) and message.get('role') == 'user':
                        user_message_content = message.get('content')
                        if isinstance(user_message_content, str):
                             user_input_query = user_message_content.strip()
                             break # Found the latest user query

                if user_input_query is None:
                    return jsonify({"error": "No user message found in the 'messages' array."}), 400

                # Get optional parameters from the top level for JSON format
                show_logs = bool(data.get('show_logs', True)) # Default True
                scrape_images = bool(data.get('show_images', True)) # Default True
                show_sources = bool(data.get('show_sources', True)) # Default True


            # Check if it's the old {"query": "..."} format
            elif 'query' in data and isinstance(data['query'], str):
                output_format = 'markdown' # Old format implies markdown response
                user_input_query = data.get('query').strip()
                # Get optional parameters from the JSON body
                show_logs_param = data.get('show_logs', None)
                show_image_param = data.get('show_images', None)
                show_sources_param = data.get('show_sources', None)

                show_logs = str(show_logs_param).lower() == 'true' if show_logs_param is not None else True
                scrape_images = str(show_image_param).lower() == 'true' if show_image_param is not None else True
                show_sources = str(show_sources_param).lower() == 'true' if show_sources_param is not None else True

            else:
                return jsonify({"error": "Invalid POST body. Expected {'query': '...'} or {'messages': [...]}."}), 400

        except Exception as e:
             # Catch JSONDecodeError and other parsing issues
             return jsonify({"error": f"Failed to parse POST body: {type(e).__name__}: {e}"}), 400

    elif request.method == 'GET':
        # Existing GET handling (implies markdown output)
        output_format = 'markdown'
        user_input_query = request.args.get('query')
        show_logs_param = request.args.get('show_logs', None)
        show_image_param = request.args.get('show_images', None)
        show_sources_param = request.args.get('show_sources', None)

        show_logs = str(show_logs_param).lower() == 'true' if show_logs_param is not None else True
        scrape_images = str(show_image_param).lower() == 'true' if show_image_param is not None else True
        show_sources = str(show_sources_param).lower() == 'true' if show_sources_param is not None else True

    # Validate query string common to all methods
    if not user_input_query or not isinstance(user_input_query, str) or len(user_input_query) == 0:
        return jsonify({"error": "Query parameter 'query' is required and cannot be empty."}), 400
    if len(user_input_query) > 1000:
        return jsonify({"error": "Query parameter 'query' is too long (max 1000 characters)."}), 400

    # --- Handle special test query ---
    if user_input_query.strip().lower() == "pollinations_test":
        try:
             test_messages = [{"role": "user", "content": "Respond with 'Service is reachable and responding (test mode).'"}]
             # Use a minimal model and low retries for a quick test
             response = query_pollinations_ai(test_messages, model="OpenAI GPT-3.5 Turbo", retries=1, show_logs=show_logs)
             if response and 'choices' in response and len(response['choices']) > 0:
                  test_response_content = response['choices'][0]['message']['content']
                  if "service is reachable" in test_response_content.lower():
                       # If output_format is json, return JSON test success
                       if output_format == 'json':
                            return jsonify({
                                "choices": [
                                     {"message": {"role": "assistant", "content": test_response_content}}
                                ],
                                "test_status": "success"
                            }), 200
                       else:
                            return Response(test_response_content, mimetype='text/markdown', status=200)
                  else:
                       error_msg = f"Pollinations AI test: AI responded unexpectedly. Output:\n\n{test_response_content}"
                       if output_format == 'json':
                            return jsonify({"error": error_msg, "test_status": "failed"}), 500
                       else:
                            return Response(error_msg, mimetype='text/markdown', status=500)
             else:
                  error_msg = "Pollinations AI test: AI call failed or returned empty response."
                  if output_format == 'json':
                       return jsonify({"error": error_msg, "test_status": "failed"}), 500
                  else:
                       return Response(error_msg, mimetype='text/markdown', status=500)
        except Exception as e:
             error_msg = f"Pollinations AI test: An error occurred during the test AI call: {type(e).__name__}: {e}"
             if output_format == 'json':
                  return jsonify({"error": error_msg, "test_status": "error"}), 500
             else:
                  return Response(error_msg, mimetype='text/markdown', status=500)

    
    future = process_executor.submit(
        search_and_synthesize,
        user_input_query,
        show_sources=show_sources,
        scrape_images=scrape_images,
        show_logs=show_logs,
        output_format=output_format 
    )

    try:
        markdown_output, status_code, collected_data = future.result(timeout=300) 

        if output_format == 'json':
            response_body = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": markdown_output
                        }
                    }
                ]
            }
            if show_sources:
                 source_info = {}
                 if collected_data.get("native_knowledge_part") and collected_data["native_knowledge_part"] != "None":
                      source_info["native_knowledge"] = collected_data["native_knowledge_part"]
                 if collected_data.get("scraped_website_urls"):
                      source_info["scraped_websites"] = collected_data["scraped_website_urls"]
                 if collected_data.get("processed_youtube_urls"):
                      source_info["processed_youtube"] = collected_data["processed_youtube_urls"]
                 if collected_data.get("failed_youtube_urls"):
                      source_info["failed_youtube"] = collected_data["failed_youtube_urls"]
                 if source_info: 
                     response_body["sources"] = source_info

            if scrape_images and collected_data.get("found_image_urls"):
                 response_body["images"] = collected_data["found_image_urls"]
            elif scrape_images and (collected_data.get("scraped_website_urls") or collected_data.get("processed_youtube_urls")):
                 response_body["images"] = [] 
                 pass

            # Include error message in JSON if present in collected_data (e.g., for 400/404)
            if collected_data.get("error"):
                 response_body["error"] = collected_data["error"]

            return jsonify(response_body), status_code
        else: # markdown output
            # For markdown, search_and_synthesize already appended sources/images if show_sources is True
            return Response(markdown_output, mimetype='text/markdown', status=status_code)

    except concurrent.futures.TimeoutError:
        logging.error("Search and synthesis process timed out.")
        error_msg = "The request timed out."
        status_code = 504 # Gateway Timeout
        if output_format == 'json':
             return jsonify({"error": error_msg}), status_code
        else:
             return Response(f"Error: {error_msg}\n---\nA timeout occurred while processing your request.", mimetype='text/markdown', status=status_code)

    except Exception as e:
        logging.exception("An unhandled error occurred during search and synthesis.")
        error_msg = f"An unexpected error occurred: {type(e).__name__}: {e}"
        status_code = 500 # Internal Server Error
        if output_format == 'json':
             return jsonify({"error": error_msg}), status_code
        else:
             return Response(f"Error: {error_msg}\n---\nAn internal error prevented the request from completing.", mimetype='text/markdown', status=status_code)

@app.route('/', methods=['GET'])
def index():
    """Simple index page for testing API reachability."""
    return "Pollinations Search API is running.", 200

# if __name__ == "__main__":
    # Running in __main__ with debug=False is standard for production-like deployment
    # Using 127.0.0.1 binds to localhost only. Use 0.0.0.0 for external access (use with caution).
    # app.run(host="127.0.0.1", port=5000, debug=False)