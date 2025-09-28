"""
FRWSR Pipeline: FAISS Retrieval and Webis Set-encoder Reranking

This module implements the FRWSR pipeline that combines E5-Large embeddings,
FAISS-HNSW retrieval, and Webis Set-Encoder/Large cross-encoder reranking.
"""

import ast
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path

from .base_pipeline import BasePipeline
from ..retrieval.faiss_retriever import FAISSRetriever
from ..reranking.webis_reranker import WebisReranker


class FRWSRPipeline(BasePipeline):
    """
    FRWSR Pipeline: FAISS Retrieval and Webis Set-encoder Reranking
    
    This pipeline combines:
    - E5-Large embeddings for dense retrieval
    - FAISS-HNSW for efficient approximate nearest neighbor search
    - Webis Set-Encoder/Large for cross-encoder reranking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FRWSR pipeline.
        
        Args:
            config: Pipeline configuration containing model paths and parameters
        """
        super().__init__(config)
        self.pipeline_name = "FRWSR"
        
        # Initialize components
        self.build_retriever()
        self.build_reranker()
    
    def build_retriever(self) -> None:
        """Build FAISS retriever with E5-Large embeddings."""
        retriever_config = self.config.get('retriever', {})
        
        self.retriever = FAISSRetriever(
            model_name=retriever_config.get('model_name', 'intfloat/e5-large'),
            index_path=retriever_config.get('index_path'),
            index_type=retriever_config.get('index_type', 'HNSW'),
            **retriever_config.get('index_params', {})
        )
        
        self.logger.info("FAISS retriever initialized")
    
    def build_reranker(self) -> None:
        """Build Webis Set-encoder reranker."""
        reranker_config = self.config.get('reranker', {})
        
        self.reranker = WebisReranker(
            model_name=reranker_config.get('model_name', 'webis/set-encoder-large'),
            batch_size=reranker_config.get('batch_size', 16),
            max_length=reranker_config.get('max_length', 512)
        )
        
        self.logger.info("Webis Set-encoder reranker initialized")
    
    def retrieve(self, query: str, k: int = 100) -> List[int]:
        """
        Retrieve top-k candidates using FAISS.
        
        Args:
            query: Input query string
            k: Number of candidates to retrieve
            
        Returns:
            List of candidate node IDs
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized. Call build_retriever() first.")
        
        candidate_ids = self.retriever.search(query, k=k)
        return candidate_ids
    
    def rerank(self, query: str, candidate_ids: List[int], k: int = 100) -> List[int]:
        """
        Rerank candidates using Webis Set-encoder.
        
        Args:
            query: Input query string
            candidate_ids: List of candidate node IDs
            k: Number of top candidates to return
            
        Returns:
            List of reranked node IDs
        """
        if self.reranker is None:
            raise ValueError("Reranker not initialized. Call build_reranker() first.")
        
        if self.node_df is None:
            raise ValueError("Node data not loaded. Call load_data() first.")
        
        # Get document texts for candidates
        docs_and_ids = self._get_candidate_documents(candidate_ids)
        
        if not docs_and_ids:
            return []
        
        doc_texts = [doc['combined_text'] for doc in docs_and_ids['docs']]
        doc_ids = docs_and_ids['ids']
        
        # Score documents using set-encoder
        scores = self.reranker.score(query, doc_texts)
        
        # Create scored pairs and sort by score (descending)
        scored_pairs = list(zip(scores, doc_ids))
        scored_pairs.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k reranked node IDs
        reranked_ids = [node_id for _, node_id in scored_pairs[:k]]
        
        return reranked_ids
    
    def _get_candidate_documents(self, candidate_ids: List[int]) -> Dict[str, List]:
        """
        Retrieve document texts for candidate node IDs while preserving order.
        
        Args:
            candidate_ids: List of node IDs
            
        Returns:
            Dictionary with 'docs' and 'ids' keys containing aligned documents and IDs
        """
        # Create lookup dictionary for efficient node retrieval
        node_lookup = {
            row['node_id']: row 
            for _, row in self.node_df.iterrows()
        }
        
        docs, ids = [], []
        
        for node_id in candidate_ids:
            if node_id in node_lookup:
                docs.append(node_lookup[node_id])
                ids.append(node_id)
        
        return {'docs': docs, 'ids': ids}
    
    def evaluate_on_dataset(
        self, 
        query_file: Path, 
        output_file: Path, 
        save_partial: bool = True,
        partial_interval: int = 10
    ) -> pd.DataFrame:
        """
        Evaluate pipeline on a dataset and save results.
        
        Args:
            query_file: Path to query dataset CSV file
            output_file: Path to save evaluation results
            save_partial: Whether to save partial results during processing
            partial_interval: Interval for saving partial results
            
        Returns:
            DataFrame with evaluation results
        """
        from ..evaluation.metrics import compute_metrics
        from tqdm import tqdm
        
        # Load queries
        queries_df = pd.read_csv(query_file)
        
        # Preprocess answer columns if they're strings
        if 'correct_answer' in queries_df.columns:
            queries_df['correct_answer'] = queries_df['correct_answer'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        
        results_list = []
        partial_file = output_file.with_suffix('.partial.csv')
        
        self.logger.info(f"Evaluating on {len(queries_df)} queries")
        
        with tqdm(total=len(queries_df), desc=f"Processing {self.pipeline_name}") as pbar:
            for idx, row in queries_df.iterrows():
                query_text = row['query']
                query_id = row.get('query_id', idx + 1)
                correct_answers = [int(x) for x in row['correct_answer']]
                
                # Process query through pipeline
                try:
                    predicted_ids = self.process_query(query_text)
                    
                    # Compute metrics
                    metrics = compute_metrics(predicted_ids, correct_answers)
                    
                    # Store results
                    result = {
                        'query_id': query_id,
                        'query': query_text,
                        f'{self.pipeline_name}_answer': predicted_ids,
                        'correct_answer': correct_answers,
                        **metrics
                    }
                    
                    results_list.append(result)
                    
                    # Log progress for first few queries
                    if idx < 5:
                        self.logger.info(f"Query {idx}: {query_text[:50]}...")
                        self.logger.info(f"Predicted: {predicted_ids[:5]}")
                        self.logger.info(f"Hit@1: {metrics['hit@1']}, MRR: {metrics['MRR']:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing query {idx}: {e}")
                    # Add empty result to maintain alignment
                    results_list.append({
                        'query_id': query_id,
                        'query': query_text,
                        f'{self.pipeline_name}_answer': [],
                        'correct_answer': correct_answers,
                        'hit@1': 0, 'hit@5': 0, 'hit@10': 0, 'hit@20': 0,
                        'recall@20': 0, 'MRR': 0
                    })
                
                # Save partial results
                if save_partial and idx % partial_interval == 0:
                    pd.DataFrame(results_list).to_csv(partial_file, index=False)
                
                pbar.update(1)
        
        # Create final results DataFrame
        results_df = pd.DataFrame(results_list)
        
        # Save final results
        results_df.to_csv(output_file, index=False)
        
        # Clean up partial file
        if partial_file.exists():
            partial_file.unlink()
        
        # Log summary metrics
        self._log_summary_metrics(results_df)
        
        return results_df
    
    def _log_summary_metrics(self, results_df: pd.DataFrame) -> None:
        """Log summary evaluation metrics."""
        metric_cols = ['hit@1', 'hit@5', 'hit@10', 'hit@20', 'recall@20', 'MRR']
        summary_metrics = results_df[metric_cols].mean()
        
        self.logger.info(f"\n=== {self.pipeline_name} Pipeline Summary Metrics ===")
        for metric, value in summary_metrics.items():
            self.logger.info(f"{metric}: {value:.4f}")