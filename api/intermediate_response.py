import requests
import random
from urllib.parse import quote
import os
from dotenv import load_dotenv
from typing import Dict, List, Optional
load_dotenv()

def generate_intermediate_response(
    query: str,
    embed_result: str,
    max_tokens: int = 500,
    kg_context: Optional[Dict] = None,
) -> str:
    """
    Generate intermediate response using embeddings and knowledge graph context
    
    Args:
        query: User search query
        embed_result: Raw embedding search result
        max_tokens: Max tokens in response
        kg_context: Knowledge graph context from scraping
    
    Returns:
        Formatted response string
    """
    # Build enhanced context from knowledge graph
    kg_enhancement = ""
    if kg_context and isinstance(kg_context, dict):
        # Extract key entities
        top_entities = kg_context.get("top_entities", [])
        if top_entities:
            entity_list = ", ".join([f"{ent[0]} ({ent[1]:.2f})" for ent in top_entities[:5]])
            kg_enhancement = f"\n\nKey Entities: {entity_list}"
        
        # Extract relationships
        relationships = kg_context.get("relationships", [])
        if relationships:
            rel_list = "; ".join([f"{s} → {r} → {o}" for s, r, o in relationships[:3]])
            kg_enhancement += f"\n\nRelationships: {rel_list}"
    
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
- Fit the response within the {max_tokens} token limit but in detail
- When key entities and relationships are provided, weave them naturally into the response
- Prioritize information marked as high-importance by the knowledge graph"""
    
    payload = {
        "model": "gemini-fast",
        "messages": [
            {
                "role": "system",
                "content": system_prompt.replace("\n", " ").strip()
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nRaw Search Result:\n{embed_result}{kg_enhancement}"
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
    