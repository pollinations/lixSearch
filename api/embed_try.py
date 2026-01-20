import time
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from typing import List, Tuple, Dict
import nltk
import os
import asyncio
from search import playwright_web_search
from knowledge_graph import build_knowledge_graph


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




def select_top_sentences(
    query: str,
    docs: List[str],
    top_k_chunks: int = 4,
    top_k_sentences: int = 8,
    kg_data: List[Dict] = None,
) -> Tuple[List[Tuple[str, float]], float]:
    """
    Select top sentences using both embedding similarity and knowledge graph importance
    
    Args:
        query: Search query
        docs: List of documents
        top_k_chunks: Top K chunks to consider
        top_k_sentences: Top K sentences to return
        kg_data: Knowledge graph data from scraping (optional)
    
    Returns:
        Tuple of (top_sentences, processing_time)
    """
    start = time.perf_counter()

    chunks = []
    chunk_to_sentences = []
    chunk_kg_scores = []  # Importance scores from KG for each chunk
    
    for doc in docs:
        doc_chunks = chunk_text(doc)
        for ch in doc_chunks:
            chunks.append(ch)
            chunk_to_sentences.append(sent_tokenize(ch))
            
            # Calculate KG importance score if available
            kg_score = 0.0
            if kg_data:
                # Check if chunk contains entities from KG
                for kg_dict in kg_data:
                    if isinstance(kg_dict, dict) and "top_entities" in kg_dict:
                        for entity, score in kg_dict["top_entities"][:5]:
                            if entity.lower() in ch.lower():
                                kg_score = max(kg_score, float(score))
            
            chunk_kg_scores.append(kg_score)

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
    
    # Combine embedding scores with KG importance scores (70% embedding, 30% KG)
    combined_scores = []
    for i, score in enumerate(scores):
        kg_weight = chunk_kg_scores[i] if i < len(chunk_kg_scores) else 0.0
        combined_score = (0.7 * float(score)) + (0.3 * kg_weight)
        combined_scores.append(combined_score)
    
    combined_scores = np.array(combined_scores)
    top_chunk_idxs = np.argsort(combined_scores)[-top_k_chunks:][::-1]

    candidate_sentences = []

    for idx in top_chunk_idxs:
        chunk_score = combined_scores[idx]
        for s in chunk_to_sentences[idx]:
            score = chunk_score
            if query.lower() in s.lower():
                score += 0.06
            candidate_sentences.append((s, float(score)))

    candidate_sentences.sort(key=lambda x: x[1], reverse=True)

    end = time.perf_counter()

    return candidate_sentences[:top_k_sentences], end - start
# Main execution
if __name__ == "__main__":
    query = "latest news between america and venezuela?"
    pass