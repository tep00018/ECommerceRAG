"""
Base Pipeline Abstract Class for Neural Retriever-Reranker Pipelines

This module defines the abstract base class for all RAG pipeline implementations
in the Neural Retriever-Reranker framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path


class BasePipeline(ABC):
    """Abstract base class for RAG pipelines."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base pipeline with configuration.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.logger = self._setup_logger()
        self.node_df = None
        self.retriever = None
        self.reranker = None
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the pipeline."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    @abstractmethod
    def build_retriever(self) -> None:
        """Build the retrieval component."""
        pass
    
    @abstractmethod
    def build_reranker(self) -> None:
        """Build the reranking component."""
        pass
    
    @abstractmethod
    def retrieve(self, query: str, k: int = 100) -> List[int]:
        """
        Retrieve top-k candidates for a query.
        
        Args:
            query: Input query string
            k: Number of candidates to retrieve
            
        Returns:
            List of candidate node IDs
        """
        pass
    
    @abstractmethod
    def rerank(self, query: str, candidate_ids: List[int], k: int = 100) -> List[int]:
        """
        Rerank candidates using the reranking component.
        
        Args:
            query: Input query string
            candidate_ids: List of candidate node IDs
            k: Number of top candidates to return
            
        Returns:
            List of reranked node IDs
        """
        pass
    
    def process_query(self, query: str, retrieve_k: int = 100, rerank_k: int = 100) -> List[int]:
        """
        Process a single query through the full pipeline.
        
        Args:
            query: Input query string
            retrieve_k: Number of candidates to retrieve
            rerank_k: Number of candidates to return after reranking
            
        Returns:
            List of final ranked node IDs
        """
        # Retrieve candidates
        candidate_ids = self.retrieve(query, retrieve_k)
        
        # Rerank candidates
        final_ids = self.rerank(query, candidate_ids, rerank_k)
        
        return final_ids
    
    def evaluate_batch(self, queries: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate pipeline on a batch of queries.
        
        Args:
            queries: List of query dictionaries with 'query' and 'correct_answer' keys
            
        Returns:
            Dictionary of evaluation metrics
        """
        from ..evaluation.metrics import compute_metrics_batch
        
        results = []
        
        self.logger.info(f"Processing {len(queries)} queries...")
        
        for i, query_data in enumerate(queries):
            if i % 100 == 0:
                self.logger.info(f"Processed {i}/{len(queries)} queries")
                
            query_text = query_data['query']
            correct_answers = query_data['correct_answer']
            
            # Process query through pipeline
            predicted_ids = self.process_query(query_text)
            
            results.append({
                'predicted': predicted_ids,
                'ground_truth': correct_answers
            })
        
        # Compute batch metrics
        metrics = compute_metrics_batch(results)
        
        return metrics
    
    def load_data(self, data_path: Path) -> None:
        """Load node data required for the pipeline."""
        from ..utils.data_loader import load_node_data
        
        self.logger.info(f"Loading node data from {data_path}")
        self.node_df = load_node_data(data_path)
        self.logger.info(f"Loaded {len(self.node_df)} nodes")
    
    def save_results(self, results: List[Dict], output_path: Path) -> None:
        """Save pipeline results to file."""
        import pandas as pd
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Results saved to {output_path}")