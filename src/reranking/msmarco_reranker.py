"""
MS MARCO Cross-Encoder Reranker Module

This module provides reranking functionality using MS MARCO trained cross-encoders
for efficient and effective document reranking in RAG pipelines.
"""

import numpy as np
from typing import List, Union, Tuple
import logging

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    raise ImportError(
        "sentence-transformers is required for MS MARCO reranker. "
        "Install with: pip install sentence-transformers"
    )


class MSMARCOReranker:
    """
    MS MARCO Cross-Encoder reranker for document reranking.
    
    Uses pre-trained cross-encoders from MS MARCO dataset for efficient
    point-wise document scoring and reranking.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
        max_length: int = 512,
        device: str = "auto"
    ):
        """
        Initialize MS MARCO cross-encoder reranker.
        
        Args:
            model_name: MS MARCO cross-encoder model name
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            device: Device for computation ('auto', 'cuda', 'cpu')
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
        # Load model
        try:
            self.model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=device
            )
            self.logger.info(f"Loaded MS MARCO reranker: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def score(self, query: str, documents: List[str]) -> List[float]:
        """
        Score query-document pairs using cross-encoder.
        
        Args:
            query: Query string
            documents: List of document strings
            
        Returns:
            List of relevance scores (one per document)
        """
        if not documents:
            return []
        
        try:
            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]
            
            # Score all pairs
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False
            )
            
            # Convert to list of floats
            if isinstance(scores, np.ndarray):
                scores = scores.tolist()
            elif not isinstance(scores, list):
                scores = [float(scores)]
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error scoring documents: {e}")
            # Return zero scores as fallback
            return [0.0] * len(documents)
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        document_ids: List[int], 
        k: int = 100
    ) -> List[int]:
        """
        Rerank documents and return top-k document IDs.
        
        Args:
            query: Query string
            documents: List of document texts
            document_ids: List of document IDs (same order as documents)
            k: Number of top documents to return
            
        Returns:
            List of reranked document IDs (top-k)
        """
        if len(documents) != len(document_ids):
            raise ValueError("Documents and document_ids must have same length")
        
        # Score documents
        scores = self.score(query, documents)
        
        # Create scored pairs and sort by score (descending)
        scored_pairs = list(zip(scores, document_ids))
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k document IDs
        reranked_ids = [doc_id for _, doc_id in scored_pairs[:k]]
        
        return reranked_ids
    
    def batch_score(
        self, 
        queries: List[str], 
        documents_list: List[List[str]]
    ) -> List[List[float]]:
        """
        Score multiple queries with their respective document sets efficiently.
        
        Args:
            queries: List of query strings
            documents_list: List of document lists (one per query)
            
        Returns:
            List of score lists (one per query)
        """
        if len(queries) != len(documents_list):
            raise ValueError("Queries and documents_list must have same length")
        
        all_pairs = []
        pair_to_query_idx = []
        pair_to_doc_idx = []
        
        # Create all query-document pairs with tracking indices
        for q_idx, (query, documents) in enumerate(zip(queries, documents_list)):
            for d_idx, doc in enumerate(documents):
                all_pairs.append([query, doc])
                pair_to_query_idx.append(q_idx)
                pair_to_doc_idx.append(d_idx)
        
        if not all_pairs:
            return []
        
        # Score all pairs in batches
        try:
            all_scores = self.model.predict(
                all_pairs,
                batch_size=self.batch_size,
                show_progress_bar=len(all_pairs) > 1000
            )
            
            # Group scores back by query
            result_scores = [[] for _ in range(len(queries))]
            
            for i, score in enumerate(all_scores):
                q_idx = pair_to_query_idx[i]
                result_scores[q_idx].append(float(score))
            
            return result_scores
            
        except Exception as e:
            self.logger.error(f"Error in batch scoring: {e}")
            # Return zero scores as fallback
            return [[0.0] * len(docs) for docs in documents_list]
    
    def batch_rerank(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        document_ids_list: List[List[int]],
        k: int = 100
    ) -> List[List[int]]:
        """
        Rerank multiple queries efficiently.
        
        Args:
            queries: List of query strings
            documents_list: List of document lists (one per query)
            document_ids_list: List of document ID lists (one per query)
            k: Number of top documents to return per query
            
        Returns:
            List of reranked document ID lists (one per query)
        """
        # Validate input
        if not (len(queries) == len(documents_list) == len(document_ids_list)):
            raise ValueError("All input lists must have same length")
        
        for docs, ids in zip(documents_list, document_ids_list):
            if len(docs) != len(ids):
                raise ValueError("Each documents list must match its document_ids list")
        
        # Batch score all queries
        all_scores = self.batch_score(queries, documents_list)
        
        # Rerank each query individually
        results = []
        for scores, doc_ids in zip(all_scores, document_ids_list):
            # Create scored pairs and sort by score (descending)
            scored_pairs = list(zip(scores, doc_ids))
            scored_pairs.sort(key=lambda x: x[0], reverse=True)
            
            # Take top-k
            reranked_ids = [doc_id for _, doc_id in scored_pairs[:k]]
            results.append(reranked_ids)
        
        return results
    
    def get_score_statistics(self, scores: List[float]) -> dict:
        """
        Get statistics about score distribution.
        
        Args:
            scores: List of scores
            
        Returns:
            Dictionary with score statistics
        """
        if not scores:
            return {}
        
        scores_array = np.array(scores)
        
        return {
            'mean': float(np.mean(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'median': float(np.median(scores_array)),
            'q25': float(np.percentile(scores_array, 25)),
            'q75': float(np.percentile(scores_array, 75))
        }
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "device": str(self.model.device) if hasattr(self.model, 'device') else 'unknown',
            "model_type": "MS MARCO Cross-Encoder"
        }
    
    def set_batch_size(self, batch_size: int) -> None:
        """Update batch size for processing."""
        self.batch_size = batch_size
        self.logger.info(f"Updated batch size to {batch_size}")
    
    def warm_up(self, sample_query: str = "test query", sample_doc: str = "test document") -> float:
        """
        Warm up the model with a sample prediction to reduce first-call latency.
        
        Args:
            sample_query: Sample query for warm-up
            sample_doc: Sample document for warm-up
            
        Returns:
            Sample prediction score
        """
        self.logger.info("Warming up MS MARCO reranker...")
        
        try:
            score = self.score(sample_query, [sample_doc])[0]
            self.logger.info(f"Model warm-up complete. Sample score: {score:.4f}")
            return score
        except Exception as e:
            self.logger.warning(f"Model warm-up failed: {e}")
            return 0.0


# Available MS MARCO models
AVAILABLE_MODELS = {
    'mini-lm-6': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    'mini-lm-12': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
    'bert-base': 'cross-encoder/ms-marco-electra-base',
    'distilbert-base': 'cross-encoder/ms-marco-distilbert-base',
    'roberta-base': 'cross-encoder/ms-marco-roberta-base'
}


def create_msmarco_reranker(
    model_type: str = "mini-lm-6", 
    **kwargs
) -> MSMARCOReranker:
    """
    Factory function to create MS MARCO reranker with predefined model types.
    
    Args:
        model_type: Model type key (see AVAILABLE_MODELS)
        **kwargs: Additional arguments for MSMARCOReranker
        
    Returns:
        Initialized MSMARCOReranker
    """
    if model_type not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    model_name = AVAILABLE_MODELS[model_type]
    return MSMARCOReranker(model_name=model_name, **kwargs)