import time
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from typing import List, Tuple
import nltk
import os
import requests
import random
from dotenv import load_dotenv
from web_scraper import fetch_full_text

load_dotenv()

NLTK_DIR = "searchenv/nltk_data"
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)

if not os.path.exists(os.path.join(NLTK_DIR, "tokenizers/punkt")):
    nltk.download("punkt", download_dir=NLTK_DIR)
    nltk.download("punkt_tab", download_dir=NLTK_DIR)


t0 = time.perf_counter()

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda",
    cache_folder="model_cache"
)

_ = model.encode("warmup", show_progress_bar=False)

t1 = time.perf_counter()
MODEL_LOAD_TIME = t1 - t0


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b)


def normalize(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def chunk_text(text: str, max_sentences: int = 5) -> List[str]:
    sentences = sent_tokenize(text)
    chunks = []

    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk.strip():
            chunks.append(chunk)

    return chunks



def generate_intermediate_response(query: str, embed_result: str, max_tokens: int = 500) -> str:
    system_prompt = f"""You are an expert search response formatter. Your task is to take a user query and raw search results, and frame them into a natural, smooth, and engaging response that reads like a well-crafted search summary.
    
Guidelines:
- Format the response to flow naturally from the query
- Highlight the most relevant information
- Make it conversational yet informative
- Use clear structure and formatting when appropriate
- Ensure the response sounds human and polished
- If there are multiple pieces of information, organize them logically
- Avoid overwhelming the user with raw data but pack as much semantic information as you can.
- Keep the response concise but comprehensive
- Fit the response within the {max_tokens} token limit but in detail."""
    
    payload = {
        "model": "gemini-fast",
        "messages": [
            {
                "role": "system",
                "content": system_prompt.replace("\n", " ").strip()
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nRaw Search Result:\n{embed_result}"
            }
        ],
        "temperature": 0.7,
        "stream": False,
        "private": True,
        "max_tokens": max_tokens,
        "seed": random.randint(1000, 1000000)
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('TOKEN')}"
    }
    
    try:
        response = requests.post(
            "https://gen.pollinations.ai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.status_code}, {response.text}")
        
        data = response.json()
        try:
            reply = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected response format: {data}") from e
        
        return reply.strip()
    
    except requests.exceptions.Timeout:
        print(f"Timeout occurred formatting response for query: {query}")
        return f"Based on your search for '{query}': {embed_result}"
    except Exception as e:
        print(f"Error in generate_intermediate_response: {e}")
        return f"Based on your search for '{query}': {embed_result}"
    




def select_top_sentences(
    query: str,
    docs: List[str],
    top_k_chunks: int = 4,
    top_k_sentences: int = 8,
) -> Tuple[List[Tuple[str, float]], float]:
    """
    Returns top relevant sentences + inference time
    """

    start = time.perf_counter()

    chunks = []
    chunk_to_sentences = []
    for doc in docs:
        doc_chunks = chunk_text(doc)
        for ch in doc_chunks:
            chunks.append(ch)
            chunk_to_sentences.append(sent_tokenize(ch))

    if not chunks:
        return [], 0.0

    embeddings = model.encode(
        [query] + chunks,
        batch_size=16,
        show_progress_bar=False
    )

    query_emb = embeddings[0:1]
    chunk_embs = embeddings[1:]

    query_emb = normalize(query_emb)
    chunk_embs = normalize(chunk_embs)

    scores = (chunk_embs @ query_emb.T).squeeze()

    top_chunk_idxs = np.argsort(scores)[-top_k_chunks:][::-1]

    candidate_sentences = []

    for idx in top_chunk_idxs:
        chunk_score = scores[idx]
        for s in chunk_to_sentences[idx]:
            score = chunk_score
            if query.lower() in s.lower():
                score += 0.06
            candidate_sentences.append((s, float(score)))

    candidate_sentences.sort(key=lambda x: x[1], reverse=True)

    end = time.perf_counter()

    return candidate_sentences[:top_k_sentences], end - start




query = "turing machine definition"
t2 = time.perf_counter()
text = fetch_full_text("https://www.geeksforgeeks.org/theory-of-computation/turing-machine-in-toc/")
# print(text)
t3 = time.perf_counter()
print(f"Web scrape time  : {t3 - t2:.3f} seconds")
docs = [text] if text else []
results, inference_time = select_top_sentences(query, docs)

sent = ""
for sent, score in results:
    if score > 0.6:
        print(sent)
t4 = time.perf_counter()
final_resp = generate_intermediate_response(query, sent)
print("\nFinal formatted response:\n")
print(final_resp)
t5 = time.perf_counter()
print("\n\n")
print(f"\nResponse formatting time: {t5 - t4:.3f} seconds")
print(f"\nModel load time   : {MODEL_LOAD_TIME:.3f} seconds")
print(f"Inference time    : {inference_time:.3f} seconds\n")