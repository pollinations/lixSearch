from collections import deque
from loguru import logger
from multiprocessing.managers import BaseManager
from search import fetch_full_text
import concurrent.futures
import asyncio
import re
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Tuple, Optional
import numpy as np
import time


_deepsearch_store = {}

class modelManager(BaseManager): 
    pass

modelManager.register("accessSearchAgents")

search_service = None
_ipc_ready = False
_ipc_initialized = False


def cleanQuery(query):
    logger.debug("[Utility] Cleaning user query")
    urls = re.findall(r'(https?://[^\s]+)', query)
    cleaned_query = query
    website_urls = []
    youtube_urls = []

    for url in urls:
        cleaned_query = cleaned_query.replace(url, '').strip()
        url_cleaned = url.rstrip('.,;!?"\'')

        parsed_url = urlparse(url_cleaned)
        if "youtube.com" in parsed_url.netloc or "youtu.be" in parsed_url.netloc:
            youtube_urls.append(url_cleaned)
        elif parsed_url.scheme in ['http', 'https']:
            website_urls.append(url_cleaned)
        cleaned_query = re.sub(r'\s+', ' ', cleaned_query).strip()

    return website_urls, youtube_urls, cleaned_query


def testSearching():
    test_queries = ["Latest news from Nepal", "Political updates in Nepal"]
    test_urls = [
        "https://english.nepalnews.com/",
        "https://apnews.com/article/nepal-gen-z-protests-army-kathmandu-2e4d9e835216b11fa238d7bcf8915cbf",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    ]
    contents = fetch_url_content_parallel(test_queries, test_urls)
    for idx, content in enumerate(contents):
        print(f"Content snippet {idx+1}:", content[:200])
   


if __name__ == "__main__":
    pass