import re
from loguru import logger
from .main import _init_ipc_manager
import asyncio
search_service = None
from commons.searching_based import fetch_full_text

def webSearch(query: str):
    if not _init_ipc_manager() or search_service is None:
        logger.warning("[Utility] IPC service not available for web search")
        return []
    try:
        urls = search_service.web_search(query)
        return urls
    except Exception as e:
        logger.error(f"[Utility] Web search failed: {e}")
        return []


async def imageSearch(query: str, max_images: int = 10) -> list:
    if not _init_ipc_manager() or search_service is None:
        logger.warning("[Utility] IPC service not available for image search")
        return []
    try:
        loop = asyncio.get_event_loop()
        urls = await loop.run_in_executor(
            None,
            lambda: search_service.image_search(query, max_images=max_images)
        )
        logger.debug(f"[Utility] Image search returned {len(urls)} results for: {query[:50]}")
        return urls
    except Exception as e:
        logger.error(f"[Utility] Image search failed: {e}")
        return []

def preprocess_text(text):
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    text = re.sub(r'[^\w\s.,!?;:]', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    meaningful_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20 and len(sentence.split()) > 3:
            if not any(word in sentence.lower() for word in ['feedback', 'menu', 'navigation', 'click', 'download']):
                meaningful_sentences.append(sentence)
    
    return meaningful_sentences


def fetch_url_content_parallel(queries, urls, max_workers=10, request_id: str = None) -> str:
    results = []
    for url in urls:
        try:
            text_content = fetch_full_text(url, request_id=request_id)
            
            clean_text = str(text_content).encode('unicode_escape').decode('utf-8')
            clean_text = clean_text.replace('\\n', ' ').replace('\\r', ' ').replace('\\t', ' ')
            clean_text = ''.join(c for c in clean_text if c.isprintable())
            results.append(f"URL: {url}\n{clean_text.strip()}")
            logger.debug(f"[Utility] Fetched {len(clean_text)} chars from {url}")
        except Exception as e:
            logger.error(f"[Utility] Failed fetching {url}: {e}")

    combined_text = "\n".join(results)
    logger.info(f"[Utility] Fetched all URLs in parallel, total: {len(combined_text)} chars")
    
    return combined_text
