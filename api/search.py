import time
import random
import asyncio
from typing import List, Tuple, Dict, Optional
from urllib.parse import quote, urlparse
import requests
from bs4 import BeautifulSoup
import re
import logging
import ipaddress
from config import MAX_TOTAL_SCRAPE_WORD_COUNT, RETRIEVAL_TOP_K
from loguru import logger
from multiprocessing.managers import BaseManager

__all__ = ['fetch_full_text', 'playwright_web_search', 'warmup_playwright', 'ingest_url_to_vector_store', 'retrieve_from_vector_store']


def fetch_full_text(
    url,
    total_word_count_limit=MAX_TOTAL_SCRAPE_WORD_COUNT,
    request_id: Optional[str] = None,
) -> str:
    if not _validate_url_for_fetch(url):
        logger.error(f"[Fetch] URL validation failed: {url}")
        return ""
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, timeout=20, headers=headers)
        if response.status_code != 200:
            logger.error(f"[FETCH] Error fetching {url}: Status {response.status_code}")
            return ""
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            logger.warning(f"[FETCH] Skipping non-HTML content from {url} (Content-Type: {content_type})")
            return ""

        soup = BeautifulSoup(response.content, 'html.parser')

        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'noscript', 'iframe', 'svg']):
            element.extract()

        main_content_elements = soup.find_all(['main', 'article', 'div', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'span', 'p', 'article'], class_=[
            'main', 'content', 'article', 'post', 'body', 'main-content', 'entry-content', 'blog-post'
        ])
        if not main_content_elements:
            main_content_elements = [soup.find('body')] if soup.find('body') else [soup]

        temp_text = []
        word_count = 0
        for main_elem in main_content_elements:
            if word_count >= total_word_count_limit:
                break
            for tag in main_elem.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'blockquote', 'div']):
                text = re.sub(r'\s+', ' ', tag.get_text()).strip()
                if text:
                    words = text.split()
                    words_to_add = words[:total_word_count_limit - word_count]
                    if words_to_add:
                        temp_text.append(" ".join(words_to_add))
                        word_count += len(words_to_add)

        text_content = '\n\n'.join(temp_text)
        if word_count >= total_word_count_limit:
            text_content = ' '.join(text_content.split()[:total_word_count_limit]) + '...'

        cleaned_text = text_content.strip()
        return cleaned_text

    except requests.exceptions.Timeout:
        logger.error(f"[Fetch] Timeout scraping URL: {url}")
        return ""
    except requests.exceptions.RequestException as e:
        logger.error(f"[Fetch] Request error scraping URL: {url}: {type(e).__name__}: {e}")
        return ""
    except Exception as e:
        logger.error(f"[Fetch] Error processing URL: {url}: {type(e).__name__}: {e}")
        return ""




if __name__ == "__main__":
    async def main():
        query = "an evening in paris"
        urls, search_time = await playwright_web_search(query, max_links=4, images=True)
        print(f"Search completed in {search_time:.3f} seconds")
        print("URLs found:")
        for url in urls:
            print(f" - {url}")
    
    asyncio.run(main())


_global_embedding_service = None
_global_vector_store = None
_global_retrieval_pipeline = None


# IPC Client setup for connecting to model_server
class ModelServerClient(BaseManager):
    pass

ModelServerClient.register('CoreEmbeddingService')

_model_server = None

def get_model_server():
    """Lazy connection to the model_server via IPC"""
    global _model_server
    if _model_server is None:
        try:
            _model_server = ModelServerClient(address=("localhost", 5010), authkey=b"ipcService")
            _model_server.connect()
            logger.info("[SEARCH] Connected to model_server via IPC")
        except Exception as e:
            logger.error(f"[SEARCH] Failed to connect to model_server: {e}")
            raise
    return _model_server


def _ensure_retrieval_services():
    global _global_embedding_service, _global_vector_store, _global_retrieval_pipeline
    
    if _global_embedding_service is None:
        try:
            from embedding_service import EmbeddingService
            from embedding_service import VectorStore
            from rag_engine import RetrievalPipeline
            from config import EMBEDDING_MODEL, EMBEDDINGS_DIR
            
            logger.info("[SEARCH] Initializing retrieval services...")
            _global_embedding_service = EmbeddingService(model_name=EMBEDDING_MODEL)
            _global_vector_store = VectorStore(embeddings_dir=EMBEDDINGS_DIR)
            _global_retrieval_pipeline = RetrievalPipeline(
                _global_embedding_service,
                _global_vector_store
            )
            logger.info("[SEARCH] Retrieval services initialized")
        except Exception as e:
            logger.error(f"[SEARCH] Failed to initialize retrieval services: {e}")
            raise


def ingest_url_to_vector_store(url: str) -> Dict:
    try:
        model_server = get_model_server()
        core_service = model_server.CoreEmbeddingService()
        ingest_result = core_service.ingest_url(url)
        logger.info(f"[SEARCH] Ingested URL {url} via IPC: {ingest_result}")
        return ingest_result
    except Exception as e:
        logger.error(f"[SEARCH] Failed to ingest URL {url} via IPC: {e}")
        # Fallback to local services if IPC fails
        try:
            _ensure_retrieval_services()
            chunk_count = _global_retrieval_pipeline.ingest_url(url, max_words=3000)
            return {
                "success": True,
                "url": url,
                "chunks_ingested": chunk_count
            }
        except Exception as fallback_e:
            logger.error(f"[SEARCH] Fallback ingest also failed: {fallback_e}")
            return {
                "success": False,
                "url": url,
                "error": str(e)
            }


def retrieve_from_vector_store(query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
    try:
        model_server = get_model_server()
        core_service = model_server.CoreEmbeddingService()
        results = core_service.retrieve(query, top_k=top_k)
        logger.info(f"[SEARCH] Retrieved {len(results)} results via IPC")
        return results
    except Exception as e:
        logger.error(f"[SEARCH] Failed to retrieve via IPC: {e}")
        # Fallback to local services if IPC fails
        try:
            _ensure_retrieval_services()
            return _global_retrieval_pipeline.retrieve(query, top_k=top_k)
        except Exception as fallback_e:
            logger.error(f"[SEARCH] Fallback retrieve also failed: {fallback_e}")
            return []


def get_vector_store_stats() -> Dict:
    _ensure_retrieval_services()
    return _global_vector_store.get_stats()


def persist_vector_store() -> None:
    _ensure_retrieval_services()
    _global_vector_store.persist_to_disk()
