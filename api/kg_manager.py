"""
Knowledge Graph Manager - Manages knowledge graphs indexed by request_id
Allows building, storing, and retrieving knowledge graphs for specific search requests
"""

from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime, timedelta
import threading
from knowledge_graph import KnowledgeGraph, build_knowledge_graph
import logging

logger = logging.getLogger(__name__)

class KGManager:
    """Thread-safe manager for storing and retrieving knowledge graphs by request_id"""

    def __init__(self, max_cache_size: int = 100, ttl_hours: int = 2):
        """
        Initialize KG Manager

        Args:
            max_cache_size: Maximum number of request KGs to store
            ttl_hours: Time-to-live for cached knowledge graphs in hours
        """
        self.kg_storage: Dict[str, Dict] = {}
        self.metadata: Dict[str, Dict] = {}
        self.max_cache_size = max_cache_size
        self.ttl = timedelta(hours=ttl_hours)
        self.lock = threading.RLock()

    def add_kg(self, request_id: str, url: str, text: str, kg: KnowledgeGraph) -> Dict:
        """
        Add a knowledge graph for a specific request

        Args:
            request_id: Unique request identifier
            url: Source URL
            text: Original text used to build KG
            kg: KnowledgeGraph object

        Returns:
            Dictionary with stored KG data
        """
        with self.lock:
            # Clean expired entries if cache is full
            if len(self.kg_storage) >= self.max_cache_size:
                self._cleanup_expired()

            if request_id not in self.kg_storage:
                self.kg_storage[request_id] = {}
                self.metadata[request_id] = {
                    "created_at": datetime.now().isoformat(),
                    "urls": [],
                    "total_entities": 0,
                    "total_relationships": 0
                }

            kg_dict = kg.to_dict()
            self.kg_storage[request_id][url] = {
                "kg": kg_dict,
                "text_length": len(text),
                "timestamp": datetime.now().isoformat()
            }

            # Update metadata
            self.metadata[request_id]["urls"].append(url)
            self.metadata[request_id]["total_entities"] = sum(
                len(kg_storage["kg"]["entities"])
                for kg_storage in self.kg_storage[request_id].values()
            )
            self.metadata[request_id]["total_relationships"] = sum(
                len(kg_storage["kg"]["relationships"])
                for kg_storage in self.kg_storage[request_id].values()
            )

            logger.info(f"[KG] Added KG for request {request_id} from {url}")
            return self.kg_storage[request_id][url]

    def get_request_kg(self, request_id: str, url: Optional[str] = None) -> Optional[Dict]:
        """
        Get knowledge graph for a specific request

        Args:
            request_id: Request identifier
            url: Specific URL (optional). If None, returns aggregated KG

        Returns:
            Knowledge graph dictionary or None if not found
        """
        with self.lock:
            if request_id not in self.kg_storage:
                return None

            if url:
                return self.kg_storage[request_id].get(url)

            # Return aggregated KG for all URLs in this request
            return self._aggregate_kgs(request_id)

    def get_request_metadata(self, request_id: str) -> Optional[Dict]:
        """Get metadata for a specific request"""
        with self.lock:
            return self.metadata.get(request_id)

    def get_top_entities(self, request_id: str, top_k: int = 15) -> List[Tuple[str, float]]:
        """
        Get top entities across all KGs in a request

        Args:
            request_id: Request identifier
            top_k: Number of top entities to return

        Returns:
            List of (entity, importance_score) tuples
        """
        with self.lock:
            if request_id not in self.kg_storage:
                return []

            entity_scores = {}
            for kg_data in self.kg_storage[request_id].values():
                for entity, score in kg_data["kg"]["importance_scores"].items():
                    if entity in entity_scores:
                        entity_scores[entity] = max(entity_scores[entity], score)
                    else:
                        entity_scores[entity] = score

            sorted_entities = sorted(entity_scores.items(), key=lambda x: x[1], reverse=True)
            return sorted_entities[:top_k]

    def get_entity_relationships(self, request_id: str, entity: str) -> List[Tuple[str, str, str]]:
        """
        Get all relationships involving a specific entity across all KGs in a request

        Args:
            request_id: Request identifier
            entity: Entity to search for

        Returns:
            List of (subject, relation, object) tuples
        """
        with self.lock:
            if request_id not in self.kg_storage:
                return []

            entity_lower = entity.lower().strip()
            relationships = []

            for kg_data in self.kg_storage[request_id].values():
                for subject, relation, obj in kg_data["kg"]["relationships"]:
                    if subject == entity_lower or obj == entity_lower:
                        relationships.append((subject, relation, obj))

            return relationships

    def build_query_context(self, request_id: str) -> str:
        """
        Build rich context from all KGs in a request for better query understanding

        Args:
            request_id: Request identifier

        Returns:
            Formatted context string
        """
        with self.lock:
            if request_id not in self.kg_storage:
                return ""

            top_entities = self.get_top_entities(request_id, top_k=10)
            if not top_entities:
                return ""

            context_parts = []
            context_parts.append("Key entities identified:")

            for entity, score in top_entities:
                context_parts.append(f"- {entity} (relevance: {score:.2f})")

                relationships = self.get_entity_relationships(request_id, entity)
                if relationships:
                    context_parts.append(f"  Relationships: {len(relationships)} connections")

            return "\n".join(context_parts)

    def export_request_kg(self, request_id: str) -> Dict:
        """
        Export complete knowledge graph data for a request

        Args:
            request_id: Request identifier

        Returns:
            Complete exportable KG dictionary
        """
        with self.lock:
            if request_id not in self.kg_storage:
                return {}

            return {
                "request_id": request_id,
                "metadata": self.metadata.get(request_id),
                "graphs": self.kg_storage[request_id],
                "top_entities": self.get_top_entities(request_id, top_k=15),
                "export_time": datetime.now().isoformat()
            }

    def _aggregate_kgs(self, request_id: str) -> Dict:
        """Aggregate all KGs in a request into a single KG"""
        aggregated = {
            "entities": {},
            "relationships": [],
            "importance_scores": {},
            "entity_graph": {}
        }

        for kg_data in self.kg_storage[request_id].values():
            kg = kg_data["kg"]

            # Merge entities
            aggregated["entities"].update(kg["entities"])

            # Merge relationships
            aggregated["relationships"].extend(kg["relationships"])

            # Merge importance scores (take max)
            for entity, score in kg["importance_scores"].items():
                if entity in aggregated["importance_scores"]:
                    aggregated["importance_scores"][entity] = max(
                        aggregated["importance_scores"][entity],
                        score
                    )
                else:
                    aggregated["importance_scores"][entity] = score

            # Merge entity graph
            for entity, connected in kg["entity_graph"].items():
                if entity not in aggregated["entity_graph"]:
                    aggregated["entity_graph"][entity] = []
                aggregated["entity_graph"][entity].extend(connected)

        return aggregated

    def _cleanup_expired(self):
        """Remove expired entries from storage"""
        now = datetime.now()
        expired_requests = []

        for request_id, metadata in self.metadata.items():
            created_at = datetime.fromisoformat(metadata["created_at"])
            if now - created_at > self.ttl:
                expired_requests.append(request_id)

        for request_id in expired_requests:
            del self.kg_storage[request_id]
            del self.metadata[request_id]
            logger.info(f"[KG] Cleaned up expired KG for request {request_id}")

    def clear_request(self, request_id: str):
        """Clear KG data for a specific request"""
        with self.lock:
            if request_id in self.kg_storage:
                del self.kg_storage[request_id]
                del self.metadata[request_id]
                logger.info(f"[KG] Cleared KG for request {request_id}")

    def get_stats(self) -> Dict:
        """Get manager statistics"""
        with self.lock:
            return {
                "total_requests": len(self.kg_storage),
                "max_cache_size": self.max_cache_size,
                "storage_size": sum(
                    len(request_kgs)
                    for request_kgs in self.kg_storage.values()
                )
            }


# Global KG Manager instance
kg_manager = KGManager()
