import requests
import json
from clean_query import cleanQuery
from yahooSearch import YahooSearchAgentText, YahooSearchAgentImage, ddgs_search, mojeek_form_search, ddgs_search_module_search
from getYoutubeDetails import get_youtube_metadata, get_youtube_transcript
from scrape import fetch_full_text
from getImagePrompt import generate_prompt_from_image, replyFromImage
from tools import tools
from datetime import datetime, timezone
from getTimeZone import get_local_time
import random
import concurrent.futures
import logging
import dotenv
import os
import asyncio
import threading
from collections import deque
from textEmbedModel import retrieve_top_k
import numpy as np
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("elixpo")
dotenv.load_dotenv()
POLLINATIONS_TOKEN=os.getenv("TOKEN")
MODEL=os.getenv("MODEL")
REFRRER=os.getenv("REFERRER")
POLLINATIONS_ENDPOINT = "https://text.pollinations.ai/openai"
print(MODEL, POLLINATIONS_TOKEN)

# Load embedding model once (global)



# Agent Management with Queue System
class SearchAgentManager:
    def __init__(self, max_concurrent_agents=10):
        self.max_concurrent_agents = max_concurrent_agents
        self.active_text_agents = 0
        self.active_image_agents = 0
        self.text_queue = deque()
        self.image_queue = deque()
        self.text_lock = asyncio.Lock()
        self.image_lock = asyncio.Lock()
        self.port_range_start = 9000
        self.port_range_end = 9999
        self.used_ports = set()
        
    def _get_random_port(self):
        while True:
            port = random.randint(self.port_range_start, self.port_range_end)
            if port not in self.used_ports:
                self.used_ports.add(port)
                return port
    
    def _release_port(self, port):
        self.used_ports.discard(port)
    
    async def get_text_search_result(self, query, max_links=10):
        async with self.text_lock:
            # Try Yahoo first
            if self.active_text_agents < self.max_concurrent_agents:
                self.active_text_agents += 1
                port = self._get_random_port()
                try:
                    logger.info(f"Creating new Yahoo text agent on port {port} for query: {query}")
                    agent = YahooSearchAgentText(custom_port=port)
                    await agent.start()
                    await asyncio.sleep(random.uniform(1, 3))
                    results = await agent.search(query, max_links)
                    if results:
                        logger.info(f"Yahoo search completed for '{query}' with {len(results)} results")
                        return results
                    else:
                        logger.warning(f"Yahoo search returned no results for '{query}'. Trying DDGS fallback.")
                except Exception as e:
                    logger.error(f"Yahoo search failed for '{query}': {e}. Trying DDGS fallback.")
                finally:
                    try:
                        await agent.close()
                    except:
                        pass
                    self.active_text_agents -= 1
                    self._release_port(port)
                    logger.info(f"Text agent on port {port} closed. Active agents: {self.active_text_agents}")
            else:
                logger.warning(f"Max concurrent text agents reached ({self.max_concurrent_agents}). Queueing request for: {query}")
                await asyncio.sleep(random.uniform(2, 5))
                return await self.get_text_search_result(query, max_links)

        # Fallback 1: DDGS
        try:
            logger.info(f"Trying DDGS fallback for: {query}")
            ddgs_results = ddgs_search_module_search(query)
            if ddgs_results:
                logger.info(f"DDGS fallback returned {len(ddgs_results)} results for '{query}'")
                return ddgs_results[:max_links]
            else:
                logger.warning(f"DDGS fallback returned no results for '{query}'. Trying Mojeek fallback.")
        except Exception as e:
            logger.error(f"DDGS fallback failed for '{query}': {e}. Trying Mojeek fallback.")

        # Fallback 2: Mojeek
        try:
            logger.info(f"Trying Mojeek fallback for: {query}")
            mojeek_results = mojeek_form_search(query)
            if mojeek_results:
                logger.info(f"Mojeek fallback returned {len(mojeek_results)} results for '{query}'")
                return mojeek_results[:max_links]
            else:
                logger.warning(f"Mojeek fallback returned no results for '{query}'.")
        except Exception as e:
            logger.error(f"Mojeek fallback failed for '{query}': {e}")

        logger.error(f"All search engines failed for query: '{query}'")
        return []
                    
    
    async def get_image_search_result(self, query, max_images=10):
        """Get image search results with agent management and queueing"""
        async with self.image_lock:
            if self.active_image_agents < self.max_concurrent_agents:
                self.active_image_agents += 1
                port = self._get_random_port()
                
                try:
                    logger.info(f"Creating new image agent on port {port} for query: {query}")
                    agent = YahooSearchAgentImage(custom_port=port)
                    await agent.start()
                    await asyncio.sleep(random.uniform(1, 3))
                    
                    results = await agent.search_images(query, max_images)
                    logger.info(f"Image search completed for '{query}' with {len(results)} results")
                    if results:
                        result_dict = {f"yahoo_source_{i}": [url] for i, url in enumerate(results)}
                        return json.dumps(result_dict)
                    else:
                        return json.dumps({})
                    
                except Exception as e:
                    logger.error(f"Image search failed for '{query}': {e}")
                    return json.dumps({})
                    
                finally:
                    try:
                        await agent.close()
                    except:
                        pass
                    self.active_image_agents -= 1
                    self._release_port(port)
                    logger.info(f"Image agent on port {port} closed. Active agents: {self.active_image_agents}")
            
            else:
                logger.warning(f"Max concurrent image agents reached ({self.max_concurrent_agents}). Queueing request for: {query}")
                await asyncio.sleep(random.uniform(2, 5))
                return await self.get_image_search_result(query, max_images)
    
    async def get_agent_status(self):
        """Get current status of agents"""
        async with self.text_lock:
            text_count = self.active_text_agents
        async with self.image_lock:
            image_count = self.active_image_agents
        
        return {
            "active_text_agents": text_count,
            "active_image_agents": image_count,
            "total_active": text_count + image_count,
            "max_concurrent": self.max_concurrent_agents,
            "used_ports": len(self.used_ports)
        }

# Global instance
agent_manager = SearchAgentManager(max_concurrent_agents=10)

# Standalone search functions for backward compatibility
async def web_search(query, agent=None):
    return await agent_manager.get_text_search_result(query, max_links=5)

async def image_search(query, agent=None, max_images=10):
    return await agent_manager.get_image_search_result(query, max_images)

def fetch_url_content_parallel(urls, max_workers=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_full_text, url): url for url in urls}
        results = ""
        for future in concurrent.futures.as_completed(futures):
            url = futures[future]
            try:
                text_content = future.result()
                clean_text = str(text_content).encode('unicode_escape').decode('utf-8')
                clean_text = clean_text.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
                clean_text = ''.join(c for c in clean_text if c.isprintable())
                results += f"\nURL: {url}\nText Preview: {clean_text.strip()}"
            except Exception as e:
                logger.error(f"Failed fetching {url}: {e}")
                results += f"\nURL: {url}\n Failed to fetch content of this URL"
        logger.info(f"Fetched all URL information in parallel.")
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
                logger.error(f"YouTube {mode} failed for {url}: {e}")
                results[url] = '[Failed]'
        return results

def format_sse(event: str, data: str, task_num: int = 0, finished: str = "no", progress: int = 0, stage: str = "info") -> str:
    lines = data.splitlines()
    data_str = ''.join(f"data: {line}\n" for line in lines)
    return (
        f"event: {event}\n"
        f"task: {task_num}\n"
        f"finished: {finished}\n"
        f"progress: {progress}\n"
        f"stage: {stage}\n"
        f"{data_str}\n\n"
    )

async def run_deep_research_pipeline(user_query: str, user_image: str, event_id: str = None):
    logger.info(f"Starting DeepResearch Pipeline for query: '{user_query}' with image: '{user_image[:50] + '...' if user_image else 'None'}'")

    def emit_event(event_type, message, task_num=0, finished="no", progress=0, stage="info"):
        if event_id:
            return format_sse(event_type, message, task_num, finished, progress, stage)
        return None

    initial_event = emit_event("INFO", "Initiating Deep Research Pipeline", 0, "no", 0, "info")
    if initial_event:
        yield initial_event

    try:
        current_utc_time = datetime.now(timezone.utc)
        headers = {"Content-Type": "application/json"}

        system_prompt = f"""
Mission: Perform deep research on the user's query by decomposing it into 3-5 sub-tasks.
CRITICAL: If you know the answer directly (basic facts, math, general knowledge), answer without research.
Otherwise, for each sub-task:
- Clearly state the sub-task.
- Find 2-3 highly relevant URLs for that sub-task.
- Extract and summarize the most important information from those URLs.
- Use text embeddings to select the most relevant content for the sub-task.
- After each sub-task, provide a markdown summary with sources.
- After all sub-tasks, synthesize a final, comprehensive answer.
- Use markdown formatting, clear structure, and cite all sources.
- All progress and results must be streamed as SSE events with task numbers, finished status, and progress percentage.
- Only the last task's SSE should have finished: yes.
- Be concise, insightful, and friendly.

Guidelines:
- Use web search only when the query needs recent info, sources, or is time-sensitive.
- If an image is provided, analyze it and relate findings to the query.
- For multi-part queries, break down and address each part.
- Always infer user intent—don't wait for explicit trigger words.
- Never expose UTC or internal data.
- Use clear, professional, and friendly tone.
- Always respond in English unless asked otherwise.
- Add a clever, light signoff at the end.

Final Response Format:
1. Answer — detailed and insightful
2. Related Images — when applicable
3. Sources — when tools used
4. Signoff — clever, light, and relevant

All output must be useful, rich in information, and easy to read.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {user_query} {'Image: ' + user_image if user_image else ''}"}
        ]

        # Step 1: Ask LLM to break down the query into 3-5 sub-tasks
        planning_payload = {
            "model": MODEL,
            "messages": messages + [
                {"role": "user", "content": "Break down the query into 3-5 research tasks. Respond as numbered list, each with a short description."}
            ],
            "token": POLLINATIONS_TOKEN,
            "referrer": REFRRER,
            "private": True,
            "seed": random.randint(1000, 9999)
        }
        planning_resp = requests.post(POLLINATIONS_ENDPOINT, headers=headers, json=planning_payload)
        planning_resp.raise_for_status()
        planning_tasks = planning_resp.json()["choices"][0]["message"]["content"]
        # Parse tasks as list
        tasks = [line.strip(" .") for line in planning_tasks.split("\n") if line.strip() and line[0].isdigit()]
        if not tasks:
            tasks = [planning_tasks.strip()]
        logger.info(f"Decomposed into {len(tasks)} tasks: {tasks}")

        for task_num, task in enumerate(tasks, 1):
            progress = int((task_num - 1) / len(tasks) * 100)
            try:
                task_start_event = emit_event("INFO", f"Starting Task {task_num}: {task}", task_num, "no", progress, "info")
                if task_start_event:
                    yield task_start_event

                # Step 2: Search for 2-5 URLs for this sub-task
                try:
                    search_results = await agent_manager.get_text_search_result(task, max_links=5)
                    urls = [url for url in search_results if url and url.startswith("http")]
                except Exception as e:
                    logger.error(f"Search failed for task {task_num}: {e}", exc_info=True)
                    error_event = emit_event("ERROR", f"Search failed for task {task_num}: {e}", task_num, "no", progress, "error")
                    if error_event:
                        yield error_event
                    urls = []
                surfing_event = emit_event("SURFING_WEB", f"Task {task_num}: Surfing URLs: {urls}", task_num, "no", progress, "info")
                if surfing_event:
                    yield surfing_event

                # Step 3: Fetch and embed content from URLs
                url_texts = []
                for url in urls:
                    try:
                        loop = asyncio.get_event_loop()
                        text = await loop.run_in_executor(None, fetch_full_text, url)
                        url_texts.append((url, text[:3000]))
                    except Exception as e:
                        logger.error(f"Failed to fetch {url}: {e}")
                        error_event = emit_event("ERROR", f"Failed to fetch {url}: {e}", task_num, "no", progress, "error")
                        if error_event:
                            yield error_event

                # Always apply text embedding to select the most relevant content, even if only one or two URLs
                if url_texts:
                    doc_texts = [t[1] for t in url_texts]
                    try:
                        # Always use embedding to select top chunks, even if doc_texts is small
                        top_chunks = retrieve_top_k(doc_texts, task, k=min(3, len(doc_texts)))
                        context = "\n\n".join(top_chunks)
                    except Exception as e:
                        logger.error(f"Embedding failed for task {task_num}: {e}")
                        error_event = emit_event("ERROR", f"Embedding failed for task {task_num}: {e}", task_num, "no", progress, "error")
                        if error_event:
                            yield error_event
                        context = ""
                else:
                    context = ""

                # Step 4: Ask LLM to summarize this sub-task with context
                try:
                    summary_payload = {
                        "model": MODEL,
                        "messages": [
                            {"role": "system", "content": "You are a research assistant. Summarize the following context for the given sub-task. Use markdown, cite sources."},
                            {"role": "user", "content": f"Sub-task: {task}\n\nContext:\n{context[:4000]}\n\nSources:\n" + "\n".join([u[0] for u in url_texts])}
                        ],
                        "token": POLLINATIONS_TOKEN,
                        "referrer": REFRRER,
                        "private": True,
                        "seed": random.randint(1000, 9999)
                    }
                    loop = asyncio.get_event_loop()
                    summary_resp = await loop.run_in_executor(None, lambda: requests.post(POLLINATIONS_ENDPOINT, headers=headers, json=summary_payload))
                    summary_resp.raise_for_status()
                    summary = summary_resp.json()["choices"][0]["message"]["content"]

                    progress = int(task_num / len(tasks) * 100)
                    summary_event = emit_event(
                        "TASK_SUMMARY",
                        f"### Task {task_num}: {task}\n\n{summary}\n\n---\nSources:\n" + "\n".join([f"- {u[0]}" for u in url_texts]),
                        task_num,
                        "intermediate" if task_num < len(tasks) else "yes",
                        progress,
                        "summary"
                    )
                    if summary_event:
                        yield summary_event
                except Exception as e:
                    logger.error(f"Error summarizing task {task_num}: {e}", exc_info=True)
                    error_event = emit_event("ERROR", f"Task {task_num} summary failed: {e}", task_num, "no", progress, "error")
                    if error_event:
                        yield error_event

            except Exception as e:
                logger.error(f"Error in task {task_num}: {e}", exc_info=True)
                error_event = emit_event("ERROR", f"Task {task_num} failed: {e}", task_num, "no", progress, "error")
                if error_event:
                    yield error_event
            # Continue to next task even if error

        # Step 5: Final synthesis (always attempt, even if some tasks failed)
        try:
            final_payload = {
                "model": MODEL,
                "messages": messages + [
                    {"role": "user", "content": f"Based on the research summaries above, write a comprehensive answer to the original query. Use markdown, cite sources, and add a clever signoff."}
                ],
                "token": POLLINATIONS_TOKEN,
                "referrer": REFRRER,
                "private": True,
                "seed": random.randint(1000, 9999)
            }
            loop = asyncio.get_event_loop()
            final_resp = await loop.run_in_executor(None, lambda: requests.post(POLLINATIONS_ENDPOINT, headers=headers, json=final_payload))
            final_resp.raise_for_status()
            final_answer = final_resp.json()["choices"][0]["message"]["content"]

            final_event = emit_event("FINAL_ANSWER", final_answer, len(tasks), "yes", 100, "summary")
            if final_event:
                yield final_event
        except Exception as e:
            logger.error(f"Final synthesis failed: {e}", exc_info=True)
            error_event = emit_event("ERROR", f"Final synthesis failed: {e}", len(tasks), "yes", 100, "error")
            if error_event:
                yield error_event

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        if event_id:
            yield format_sse("error", f"[ERROR] DeepResearch failed: {e}", 0, "yes", 0, "error")
    finally:
        logger.info("Deep Research Completed")


if __name__ == "__main__":

    async def main():
        user_query = "The eiffel tower"
        user_image = None
        event_id = "cli"
        async_generator = run_deep_research_pipeline(user_query, user_image, event_id=event_id)
        try:
            async for event_chunk in async_generator:
                print(event_chunk)
        except Exception as e:
            print(f"Error: {e}")
    asyncio.run(main())