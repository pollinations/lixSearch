from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
from datetime import datetime
import numpy as np


class TokenCountMethod(Enum):
    TIKTOKEN = "tiktoken"
    ESTIMATED = "estimated"
    COUNTED = "counted"


@dataclass
class TokenCost:
    input_tokens: int = 0
    output_tokens: int = 0
    input_cost_usd: float = 0.0
    output_cost_usd: float = 0.0
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def total_cost_usd(self) -> float:
        return self.input_cost_usd + self.output_cost_usd


@dataclass
class PricingModel:
    model_name: str
    input_cost_per_1k_tokens: float = 0.001
    output_cost_per_1k_tokens: float = 0.002
    max_context_tokens: int = 128000
    
    def compute_cost(self, input_tokens: int, output_tokens: int) -> TokenCost:
        input_cost = (input_tokens / 1000.0) * self.input_cost_per_1k_tokens
        output_cost = (output_tokens / 1000.0) * self.output_cost_per_1k_tokens
        
        return TokenCost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost
        )


class TokenEstimator:
    
    def __init__(self):
        self.avg_token_per_word = 1.3
        self.avg_token_per_char = 0.25
        self.pricing_models = {
            "gpt-4": PricingModel("gpt-4", 0.03, 0.06),
            "gpt-3.5": PricingModel("gpt-3.5", 0.0005, 0.0015),
            "gemini-pro": PricingModel("gemini-pro", 0.0005, 0.0015),
            "claude-3": PricingModel("claude-3", 0.003, 0.015),
            "default": PricingModel("default", 0.001, 0.002)
        }
    
    def estimate_tokens_from_text(self, text: str, method: TokenCountMethod = TokenCountMethod.ESTIMATED) -> int:
        if method == TokenCountMethod.ESTIMATED:
            word_count = len(text.split())
            char_count = len(text)
            est_word = int(word_count * self.avg_token_per_word)
            est_char = int(char_count * self.avg_token_per_char)
            return max(est_word, est_char)
        return len(text.split())
    
    def estimate_tokens_from_messages(self, messages: List[Dict]) -> int:
        total_tokens = 0
        for msg in messages:
            content = msg.get("content", "")
            total_tokens += self.estimate_tokens_from_text(content)
        return total_tokens + (len(messages) * 4)
    
    def estimate_response_tokens(self, query_tokens: int, model: str = "default") -> int:
        if model.startswith("gpt-4"):
            return min(int(query_tokens * 0.5), 2000)
        elif model.startswith("gpt-3.5"):
            return min(int(query_tokens * 0.6), 1500)
        elif model.startswith("claude"):
            return min(int(query_tokens * 0.4), 1000)
        return min(int(query_tokens * 0.5), 1500)
    
    def get_pricing_model(self, model_name: str) -> PricingModel:
        for key in self.pricing_models:
            if key in model_name.lower():
                return self.pricing_models[key]
        return self.pricing_models["default"]


class TokenCompressor:
    
    def __init__(self, target_reduction: float = 0.3):
        self.target_reduction = target_reduction
    
    def compress_context(self, context: str, max_tokens: int) -> Tuple[str, Dict]:
        estimated = len(context.split()) * 1.3
        
        if estimated <= max_tokens:
            return context, {
                "original_estimated_tokens": int(estimated),
                "compressed_estimated_tokens": int(estimated),
                "reduction_ratio": 0.0,
                "method": "none"
            }
        
        reduction_ratio = 1.0 - (max_tokens / estimated)
        sentences = context.split('. ')
        target_sentences = max(1, int(len(sentences) * (1 - reduction_ratio)))
        
        compressed = '. '.join(sentences[:target_sentences])
        if not compressed.endswith('.'):
            compressed += '.'
        
        compressed_estimated = len(compressed.split()) * 1.3
        
        return compressed, {
            "original_estimated_tokens": int(estimated),
            "compressed_estimated_tokens": int(compressed_estimated),
            "reduction_ratio": reduction_ratio,
            "method": "sentence_truncation"
        }
    
    def summarize_context(self, text: str, compression_ratio: float = 0.5) -> str:
        words = text.split()
        target_words = max(50, int(len(words) * (1 - compression_ratio)))
        
        important_words = set(w.lower() for w in words if len(w) > 5)
        key_sentences = []
        
        for sentence in text.split('. '):
            if any(iw in sentence.lower() for iw in important_words):
                key_sentences.append(sentence)
        
        result = '. '.join(key_sentences[:max(3, len(key_sentences) // 3)])
        return result if result else text[:target_words]
    
    def deduplicate_context(self, texts: List[str]) -> Tuple[List[str], Dict]:
        unique_texts = []
        seen_normalized = set()
        removed_count = 0
        
        for text in texts:
            normalized = ' '.join(text.lower().split())
            
            is_duplicate = False
            for seen in seen_normalized:
                similarity = self._string_similarity(normalized, seen)
                if similarity > 0.85:
                    is_duplicate = True
                    removed_count += 1
                    break
            
            if not is_duplicate:
                unique_texts.append(text)
                seen_normalized.add(normalized)
        
        return unique_texts, {
            "original_count": len(texts),
            "deduplicated_count": len(unique_texts),
            "removed_count": removed_count,
            "reduction_ratio": removed_count / len(texts) if texts else 0.0
        }
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        longer = s1 if len(s1) > len(s2) else s2
        shorter = s2 if len(s1) > len(s2) else s1
        
        if len(longer) == 0:
            return 1.0
        
        common = 0
        for c in shorter:
            if c in longer:
                common += 1
        
        return common / len(longer)


class CostOptimizer:
    
    def __init__(self, token_estimator: TokenEstimator):
        self.token_estimator = token_estimator
        self.compressor = TokenCompressor()
        self.cost_history: List[Dict] = []
        self.optimization_history: List[Dict] = []
    
    def compute_retrieval_cost(self,
                              query: str,
                              retrieved_docs: List[str],
                              model: str = "default") -> Dict:
        pricing = self.token_estimator.get_pricing_model(model)
        
        query_tokens = self.token_estimator.estimate_tokens_from_text(query)
        
        doc_tokens = sum(self.token_estimator.estimate_tokens_from_text(doc) for doc in retrieved_docs)
        
        response_tokens = self.token_estimator.estimate_response_tokens(query_tokens + doc_tokens, model)
        
        input_cost = pricing.compute_cost(query_tokens + doc_tokens, 0)
        output_cost = pricing.compute_cost(0, response_tokens)
        
        total_cost = TokenCost(
            input_tokens=query_tokens + doc_tokens,
            output_tokens=response_tokens,
            input_cost_usd=input_cost.input_cost_usd,
            output_cost_usd=output_cost.output_cost_usd
        )
        
        return {
            "query_tokens": query_tokens,
            "retrieval_tokens": doc_tokens,
            "response_tokens": response_tokens,
            "total_cost": total_cost.total_cost_usd,
            "input_cost": total_cost.input_cost_usd,
            "output_cost": total_cost.output_cost_usd,
            "model": model,
            "pricing": {
                "input_per_1k": pricing.input_cost_per_1k_tokens,
                "output_per_1k": pricing.output_cost_per_1k_tokens
            }
        }
    
    def optimize_retrieval_cost(self,
                               query: str,
                               retrieved_docs: List[str],
                               max_budget_usd: float = 0.05,
                               model: str = "default") -> Dict:
        
        baseline_cost = self.compute_retrieval_cost(query, retrieved_docs, model)
        
        if baseline_cost["total_cost"] <= max_budget_usd:
            return {
                "baseline_cost": baseline_cost["total_cost"],
                "optimized_cost": baseline_cost["total_cost"],
                "savings": 0.0,
                "savings_ratio": 0.0,
                "optimizations_applied": [],
                "optimized_docs": retrieved_docs,
                "status": "within_budget"
            }
        
        optimized_docs = retrieved_docs.copy()
        optimizations_applied = []
        
        unique_docs, dedup_info = self.compressor.deduplicate_context(optimized_docs)
        if dedup_info["removed_count"] > 0:
            optimized_docs = unique_docs
            optimizations_applied.append(f"deduplication: removed {dedup_info['removed_count']} docs")
        
        if len(optimized_docs) > 5:
            optimized_docs = optimized_docs[:5]
            optimizations_applied.append("doc_limiting: kept top 5")
        
        compressed_docs = []
        for doc in optimized_docs:
            compressed, comp_info = self.compressor.compress_context(doc, max_tokens=300)
            compressed_docs.append(compressed)
        
        optimized_docs = compressed_docs
        optimizations_applied.append("context_compression")
        
        optimized_cost_info = self.compute_retrieval_cost(query, optimized_docs, model)
        optimized_cost = optimized_cost_info["total_cost"]
        
        savings = baseline_cost["total_cost"] - optimized_cost
        savings_ratio = savings / baseline_cost["total_cost"] if baseline_cost["total_cost"] > 0 else 0.0
        
        return {
            "baseline_cost": baseline_cost["total_cost"],
            "optimized_cost": optimized_cost,
            "savings": savings,
            "savings_ratio": savings_ratio,
            "optimizations_applied": optimizations_applied,
            "optimized_docs": optimized_docs,
            "status": "optimized" if optimized_cost <= max_budget_usd else "partially_optimized",
            "baseline_tokens": baseline_cost["query_tokens"] + baseline_cost["retrieval_tokens"],
            "optimized_tokens": self.token_estimator.estimate_tokens_from_text(query) + sum(
                self.token_estimator.estimate_tokens_from_text(doc) for doc in optimized_docs
            )
        }
    
    def compute_session_cost(self, session_messages: List[Dict], model: str = "default") -> Dict:
        pricing = self.token_estimator.get_pricing_model(model)
        
        total_input_tokens = self.token_estimator.estimate_tokens_from_messages(session_messages)
        estimated_output_tokens = self.token_estimator.estimate_response_tokens(total_input_tokens, model)
        
        cost = pricing.compute_cost(total_input_tokens, estimated_output_tokens)
        
        return {
            "session_input_tokens": total_input_tokens,
            "session_output_tokens": estimated_output_tokens,
            "session_total_cost": cost.total_cost_usd,
            "message_count": len(session_messages),
            "avg_cost_per_message": cost.total_cost_usd / len(session_messages) if session_messages else 0.0,
            "model": model
        }
    
    def predict_cost_trajectory(self,
                               session_messages: List[Dict],
                               future_messages: int = 10,
                               model: str = "default") -> Dict:
        
        current_cost_info = self.compute_session_cost(session_messages, model)
        current_total_cost = current_cost_info["session_total_cost"]
        avg_cost_per_msg = current_cost_info["avg_cost_per_message"]
        
        projected_total = current_total_cost + (avg_cost_per_msg * future_messages)
        
        pricing = self.token_estimator.get_pricing_model(model)
        token_budget_ratio = (current_cost_info["session_input_tokens"] + 
                            current_cost_info["session_output_tokens"]) / pricing.max_context_tokens
        
        estimated_remaining_messages = max(1, int((1 - token_budget_ratio) * 100))
        
        trajectory = []
        for i in range(future_messages):
            trajectory.append({
                "message_num": len(session_messages) + i,
                "estimated_cumulative_cost": current_total_cost + (avg_cost_per_msg * (i + 1)),
                "estimated_tokens": int((current_cost_info["session_input_tokens"] + 
                                       current_cost_info["session_output_tokens"]) * (1 + (i + 1) * 0.15))
            })
        
        return {
            "current_session_cost": current_total_cost,
            "projected_total_cost": projected_total,
            "avg_cost_per_message": avg_cost_per_msg,
            "estimated_remaining_messages": estimated_remaining_messages,
            "token_budget_utilization": token_budget_ratio,
            "trajectory": trajectory,
            "warning": "approaching_budget" if token_budget_ratio > 0.8 else "normal"
        }
    
    def get_cost_summary(self) -> Dict:
        if not self.cost_history:
            return {
                "total_queries": 0,
                "total_cost": 0.0,
                "avg_cost_per_query": 0.0
            }
        
        total_cost = sum(item["total_cost"] for item in self.cost_history if "total_cost" in item)
        total_query_tokens = sum(item["query_tokens"] for item in self.cost_history if "query_tokens" in item)
        
        return {
            "total_queries": len(self.cost_history),
            "total_cost": total_cost,
            "avg_cost_per_query": total_cost / len(self.cost_history) if self.cost_history else 0.0,
            "total_query_tokens": total_query_tokens,
            "query_history_window": len(self.cost_history)
        }
