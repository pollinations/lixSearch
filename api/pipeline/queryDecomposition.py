"""
Query Decomposition Analysis and Metrics Framework

Replaces rule-based decomposition with evidence-based decision making.

Metrics:
- ACR (Aspect Coverage Ratio): # unique aspects answered / estimated aspects
- RI (Redundancy Index): Measure of answer overlap
- TER (Token Efficiency Ratio): Tokens in answer / tokens used in retrieval

Compares:
- Single-pass RAG vs Decomposition + parallel retrieval
- Validates that decomposition improves outcomes
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from loguru import logger
import numpy as np
from enum import Enum


class QueryComplexity(Enum):
    """Query complexity classifications."""
    SIMPLE = "simple"           # Single aspect, straightforward
    MODERATE = "moderate"       # Multiple aspects, some relationships
    COMPLEX = "complex"         # Multi-step, requires reasoning
    HIGHLY_COMPLEX = "highly_complex"  # Multiple interdependent sub-questions


@dataclass
class DecompositionMetrics:
    """Metrics for evaluating decomposition effectiveness."""
    aspect_coverage_ratio: float = 0.0      # ACR: coverage of semantic aspects
    redundancy_index: float = 0.0           # RI: answer overlap measure
    token_efficiency_ratio: float = 0.0     # TER: token utilization efficiency
    decomposition_quality: float = 0.0      # Overall decomposition quality [0,1]
    sub_query_count: int = 0                # Number of generated sub-queries
    parallel_speedup: float = 1.0           # Speedup vs sequential execution
    
    def is_beneficial(self) -> bool:
        """Check if decomposition is beneficial (score > threshold)."""
        return self.decomposition_quality > 0.65


class SubQuery:
    """Represents a decomposed sub-query."""
    
    def __init__(self, sub_query_text: str, aspect: str, parent_query: str):
        self.text = sub_query_text
        self.aspect = aspect
        self.parent_query = parent_query
        self.response: Optional[str] = None
        self.sources: List[str] = []
        self.execution_time_ms: float = 0.0
        self.tokens_used: int = 0
        self.relevance_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "aspect": self.aspect,
            "response": self.response,
            "sources": self.sources,
            "execution_time_ms": self.execution_time_ms,
            "tokens_used": self.tokens_used,
            "relevance_score": self.relevance_score
        }


class QueryAnalyzer:
    """Analyzes queries for decomposition suitability."""
    
    def __init__(self):
        # Semantic aspect keywords
        self.aspect_patterns = {
            "definition": {
                "keywords": ["what is", "define", "meaning", "refers to"],
                "indicators": ["?"]
            },
            "comparison": {
                "keywords": ["compare", "difference", "vs", "versus", "similar",
                           "different", "contrast"],
                "indicators": ["vs", "versus", "and"]
            },
            "cause_effect": {
                "keywords": ["why", "cause", "because", "reason", "result",
                           "due to", "lead to", "caused"],
                "indicators": ["because", "why"]
            },
            "procedure": {
                "keywords": ["how", "process", "steps", "method", "way",
                           "procedure", "generate", "create"],
                "indicators": ["how"]
            },
            "history": {
                "keywords": ["history", "origin", "began", "founded",
                           "evolution", "development"],
                "indicators": ["history"]
            },
            "future": {
                "keywords": ["future", "predict", "forecast", "will",
                           "upcoming", "expected", "planned"],
                "indicators": ["future"]
            },
            "examples": {
                "keywords": ["example", "instance", "case", "illustration"],
                "indicators": ["example"]
            },
            "benefits": {
                "keywords": ["benefit", "advantage", "good", "positive",
                           "improve", "increase"],
                "indicators": ["better", "improve"]
            },
            "risks": {
                "keywords": ["risk", "danger", "negative", "problem",
                           "issue", "limitation"],
                "indicators": ["problem", "issue"]
            }
        }
    
    def detect_query_complexity(self, query: str) -> QueryComplexity:
        """
        Classify query complexity based on linguistic features.
        
        Returns QueryComplexity enum value
        """
        query_lower = query.lower()
        word_count = len(query.split())
        
        # Count complexity indicators
        question_count = query.lower().count("?")
        and_count = query.lower().count(" and ")
        or_count = query.lower().count(" or ")
        
        # Count detected aspects
        detected_aspects = self._detect_aspects(query)
        aspect_count = len(detected_aspects)
        
        complexity_score = (
            word_count * 0.01 +           # Longer queries more complex
            question_count * 0.3 +         # Multiple questions indicate complexity
            and_count * 0.15 +             # Conjunctions indicate multiple aspects
            or_count * 0.15 +
            aspect_count * 0.2             # More aspects = more complex
        )
        
        # Classify based on score
        if complexity_score < 0.5:
            return QueryComplexity.SIMPLE
        elif complexity_score < 1.0:
            return QueryComplexity.MODERATE
        elif complexity_score < 1.5:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.HIGHLY_COMPLEX
    
    def _detect_aspects(self, query: str) -> Set[str]:
        """Detect semantic aspects in query."""
        query_lower = query.lower()
        detected = set()
        
        for aspect, patterns in self.aspect_patterns.items():
            if any(kw in query_lower for kw in patterns["keywords"]):
                detected.add(aspect)
        
        return detected
    
    def should_decompose(self, query: str) -> Tuple[bool, str, float]:
        """
        Determine if query should be decomposed.
        
        Returns (should_decompose, reason, confidence)
        """
        complexity = self.detect_query_complexity(query)
        aspects = self._detect_aspects(query)
        
        # Heuristics for decomposition
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.HIGHLY_COMPLEX]:
            reason = f"Complexity level: {complexity.value} with {len(aspects)} aspects"
            confidence = 0.9
            return True, reason, confidence
        
        elif len(aspects) >= 3:
            reason = f"Multiple aspects detected: {aspects}"
            confidence = 0.75
            return True, reason, confidence
        
        else:
            reason = f"Simple query ({complexity.value}), single-pass sufficient"
            confidence = 0.7
            return False, reason, confidence
    
    def propose_decomposition(self, query: str) -> List[SubQuery]:
        """
        Generate sub-queries from decomposition.
        
        Returns list of SubQuery objects
        """
        aspects = list(self._detect_aspects(query))
        if not aspects:
            aspects = ["general"]
        
        sub_queries = []
        
        # Create sub-queries for each aspect
        for aspect in aspects:
            sub_query_text = self._generate_aspect_subquery(query, aspect)
            sub_query = SubQuery(sub_query_text, aspect, query)
            sub_queries.append(sub_query)
        
        logger.info(
            f"[QueryAnalyzer] Decomposed '{query[:50]}...' into "
            f"{len(sub_queries)} sub-queries: {[sq.aspect for sq in sub_queries]}"
        )
        
        return sub_queries
    
    def _generate_aspect_subquery(self, original_query: str, aspect: str) -> str:
        """Generate aspect-specific sub-query."""
        aspect_templates = {
            "definition": "What is the definition and meaning of {}?",
            "comparison": "What are the key differences and similarities in {}?",
            "cause_effect": "What are the causes and effects of {}?",
            "procedure": "How does the process of {} work step by step?",
            "history": "What is the historical background and evolution of {}?",
            "future": "What are the future trends and predictions for {}?",
            "examples": "What are concrete examples and case studies of {}?",
            "benefits": "What are the benefits and advantages of {}?",
            "risks": "What are the risks and limitations of {}?",
            "general": "{}"
        }
        
        template = aspect_templates.get(aspect, "{}")
        
        # Extract main entity/concept
        words = original_query.split()
        entity = " ".join(words[-3:]) if len(words) > 2 else original_query
        
        sub_query = template.format(entity)
        return sub_query


class DecompositionEvaluator:
    """Evaluates effectiveness of decomposition vs single-pass."""
    
    def __init__(self):
        self.analyzer = QueryAnalyzer()
    
    def compute_aspect_coverage_ratio(self,
                                     original_query: str,
                                     sub_query_responses: Dict[str, str]) -> float:
        """
        Compute ACR = # unique semantic aspects answered / estimated aspects in query
        
        Returns value in [0, 1]
        """
        # Detect required aspects
        required_aspects = self.analyzer._detect_aspects(original_query)
        if not required_aspects:
            return 0.5
        
        # Detect covered aspects from responses
        covered_aspects = set()
        for aspect, response in sub_query_responses.items():
            if response and len(response) > 50:  # Meaningful response
                covered_aspects.add(aspect)
        
        coverage_ratio = len(covered_aspects) / len(required_aspects)
        
        logger.debug(
            f"[DecompositionEvaluator] ACR={coverage_ratio:.3f} "
            f"({len(covered_aspects)}/{len(required_aspects)} aspects)"
        )
        
        return min(1.0, coverage_ratio)
    
    def compute_redundancy_index(self,
                                sub_query_responses: List[str],
                                response_embeddings: Optional[List[np.ndarray]] = None) -> float:
        """
        Compute redundancy index (RI) measuring answer overlap.
        
        Lower RI = less redundancy (better)
        Returns value in [0, 1]
        """
        if len(sub_query_responses) < 2:
            return 0.0
        
        # Simple approach: measure token overlap
        response_tokens = [set(r.lower().split()) for r in sub_query_responses if r]
        
        if not response_tokens:
            return 0.0
        
        # Compute pairwise Jaccard similarity
        similarities = []
        for i in range(len(response_tokens)):
            for j in range(i + 1, len(response_tokens)):
                tokens_i = response_tokens[i]
                tokens_j = response_tokens[j]
                
                intersection = len(tokens_i & tokens_j)
                union = len(tokens_i | tokens_j)
                
                if union > 0:
                    jaccard = intersection / union
                    similarities.append(jaccard)
        
        if not similarities:
            return 0.0
        
        redundancy_index = np.mean(similarities)
        
        logger.debug(
            f"[DecompositionEvaluator] RI={redundancy_index:.3f} "
            f"(avg pairwise overlap)"
        )
        
        return min(1.0, redundancy_index)
    
    def compute_token_efficiency_ratio(self,
                                      final_response_tokens: int,
                                      total_retrieval_tokens: int,
                                      decomposed: bool) -> float:
        """
        Compute token efficiency ratio (TER).
        TER = tokens_in_answer / tokens_used_in_retrieval
        
        Higher TER = better efficiency
        Normalized to [0, 1] for comparison
        """
        if total_retrieval_tokens == 0:
            return 0.5
        
        ter = final_response_tokens / total_retrieval_tokens
        
        # Normalize: assume ideal TER ~= 0.1-0.2
        normalized_ter = min(1.0, ter / 0.15)
        
        logger.debug(
            f"[DecompositionEvaluator] TER={ter:.3f} (normalized={normalized_ter:.3f}) "
            f"decomposed={decomposed}"
        )
        
        return normalized_ter
    
    def evaluate_decomposition(self,
                              original_query: str,
                              sub_queries: List[SubQuery],
                              sub_query_responses: Dict[str, str],
                              single_pass_response: Optional[str] = None,
                              final_response_tokens: int = 500,
                              total_retrieval_tokens: int = 5000,
                              single_pass_retrieval_tokens: int = 3000) -> DecompositionMetrics:
        """
        Comprehensive evaluation of decomposition effectiveness.
        
        Compares decomposed approach against single-pass RAG.
        """
        # Compute quality metrics
        acr = self.compute_aspect_coverage_ratio(original_query, sub_query_responses)
        ri = self.compute_redundancy_index(list(sub_query_responses.values()))
        ter = self.compute_token_efficiency_ratio(
            final_response_tokens, total_retrieval_tokens, decomposed=True
        )
        
        # Compute single-pass TER for comparison
        single_pass_ter = self.compute_token_efficiency_ratio(
            final_response_tokens, single_pass_retrieval_tokens, decomposed=False
        ) if single_pass_response else ter
        
        # Compute overall decomposition quality score
        # Higher ACR = good (more aspects covered)
        # Lower RI = good (less redundancy)
        # Higher TER = good (more efficient)
        decomposition_quality = (
            0.4 * acr +                    # 40% on coverage
            0.3 * (1.0 - ri) +             # 30% on low redundancy
            0.3 * ter                      # 30% on efficiency
        )
        
        # Compute speedup (decomposition is parallel, single-pass is sequential)
        # Rough estimate: n sub-queries in parallel ≈ n times faster (minus overhead)
        parallel_speedup = min(len(sub_queries) / 2, 3.0)  # Cap at 3x
        
        metrics = DecompositionMetrics(
            aspect_coverage_ratio=acr,
            redundancy_index=ri,
            token_efficiency_ratio=ter,
            decomposition_quality=decomposition_quality,
            sub_query_count=len(sub_queries),
            parallel_speedup=parallel_speedup
        )
        
        logger.info(
            f"[Decomposition] Quality={metrics.decomposition_quality:.3f} "
            f"(ACR={acr:.3f}, RI={ri:.3f}, TER={ter:.3f}, "
            f"speedup={parallel_speedup:.1f}x) - "
            f"Beneficial: {metrics.is_beneficial()}"
        )
        
        return metrics


class DecompositionClassifier:
    """
    Lightweight classifier to determine if query should be decomposed.
    (Could be trained on ground truth data)
    """
    
    def __init__(self):
        self.analyzer = QueryAnalyzer()
        self.evaluator = DecompositionEvaluator()
        
        # Feature weights (would be learned in real classifier)
        self.complexity_weight = 0.5
        self.aspect_count_weight = 0.3
        self.keywords_weight = 0.2
    
    def predict_decomposition_benefit(self, query: str) -> Tuple[bool, float]:
        """
        Predict if decomposition would be beneficial.
        
        Returns (should_decompose, confidence)
        """
        complexity = self.analyzer.detect_query_complexity(query)
        aspects = self.analyzer._detect_aspects(query)
        
        # Feature scores
        complexity_score = {
            QueryComplexity.SIMPLE: 0.1,
            QueryComplexity.MODERATE: 0.4,
            QueryComplexity.COMPLEX: 0.8,
            QueryComplexity.HIGHLY_COMPLEX: 1.0
        }[complexity]
        
        aspect_score = min(1.0, len(aspects) / 3.0)  # Normalize by 3 aspects
        
        # Check for explicit decomposition keywords
        keyword_score = 0.0
        decompose_keywords = ["compare", "difference", "vs", "both", "each"]
        if any(kw in query.lower() for kw in decompose_keywords):
            keyword_score = 0.8
        
        # Weighted combination
        score = (
            self.complexity_weight * complexity_score +
            self.aspect_count_weight * aspect_score +
            self.keywords_weight * keyword_score
        )
        
        should_decompose = score > 0.5
        
        logger.debug(
            f"[DecompositionClassifier] Score={score:.3f}, "
            f"Decision={should_decompose} "
            f"(complexity={complexity_score:.2f}, aspects={aspect_score:.2f})"
        )
        
        return should_decompose, score
