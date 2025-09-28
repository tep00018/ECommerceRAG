"""
Webis Set-Encoder Reranker Module

This module provides reranking functionality using Webis Set-Encoder models
for permutation-invariant inter-passage attention in document reranking.
"""

import numpy as np
from typing import List, Union
import logging

try:
    from lightning_ir import CrossEncoderModule, CrossEncoderOutput
except ImportError:
    raise ImportError(
        "lightning-ir is required for Webis reranker. "
        "Install with: pip install lightning-ir"
    )


class WebisReranker:
    """
    Webis Set-Encoder reranker for document reranking.
    
    Uses permutation-invariant attention mechanisms to model inter-document
    relationships for improved reranking performance.
    """
    
    def __init__(
        self,
        model_name: str = "webis/set-encoder-large",
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Initialize Webis Set-Encoder reranker.
        
        Args:
            model_name: Webis model name (set-encoder-base, set-encoder-large, etc.)
            batch_size: Batch size for processing
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.logger = logging.getLogger(__name__)
        
        # Load model
        try:
            self.model = CrossEncoderModule(model_name)
            self.model.eval()
            self.logger.info(f"Loaded Webis reranker: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def score(self, query: str, documents: List[str]) -> List[float]:
        """
        Score query-document pairs using set-encoder.
        
        Args:
            query: Query string
            documents: List of document strings
            
        Returns:
            List of relevance scores (one per document)
        """
        if not documents:
            return []
        
        try:
            # Score all documents at once (set-wise approach)
            raw_output = self.model.score(query, documents)
            
            # Extract scores safely
            scores = self._extract_scores(raw_output)
            
            # Convert to list of floats
            if isinstance(scores, np.ndarray):
                scores = scores.tolist() if scores.ndim > 0 else [float(scores)]
            elif not isinstance(scores, list):
                scores = [float(scores)]
            
            # Ensure we have one score per document
            if len(scores) != len(documents):
                self.logger.warning(
                    f"Score count mismatch: {len(scores)} scores for {len(documents)} documents"
                )
                # If single score returned, replicate it
                if len(scores) == 1:
                    scores = scores * len(documents)
                else:
                    # Fallback: pad with zeros or truncate
                    scores = (scores + [0.0] * len(documents))[:len(documents)]
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error scoring documents: {e}")
            # Return zero scores as fallback
            return [0.0] * len(documents)
    
    def _extract_scores(self, raw_output) -> Union[np.ndarray, List[float]]:
        """
        Extract scores from model output.
        
        Args:
            raw_output: Raw model output
            
        Returns:
            Extracted scores
        """
        if isinstance(raw_output, CrossEncoderOutput):
            if hasattr(raw_output, 'scores'):
                scores = raw_output.scores
            else:
                raise AttributeError("CrossEncoderOutput has no `scores` attribute.")
        else:
            scores = raw_output
        
        # Convert tensor to numpy if needed
        if hasattr(scores, 'cpu'):
            scores = scores.squeeze().cpu().numpy()
        
        return scores
    
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
        Score multiple queries with their respective document sets.
        
        Args:
            queries: List of query strings
            documents_list: List of document lists (one per query)
            
        Returns:
            List of score lists (one per query)
        """
        if len(queries) != len(documents_list):
            raise ValueError("Queries and documents_list must have same length")
        
        all_scores = []
        
        for query, documents in zip(queries, documents_list):
            scores = self.score(query, documents)
            all_scores.append(scores)
        
        return all_scores
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "model_type": "Webis Set-Encoder"
        }