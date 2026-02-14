from typing import List
import re

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 60) -> List[str]:
    words = text.split()
    chunks = []
    stride = chunk_size - overlap
    
    for i in range(0, len(words), stride):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) > 10:
            chunks.append(" ".join(chunk_words))
    
    return chunks


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip()
    return text


def normalize_url(url: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return f"{parsed.netloc}{parsed.path}"

