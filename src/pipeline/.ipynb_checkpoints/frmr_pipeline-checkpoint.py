"""
FRMR Pipeline: FAISS Retrieval and MS MARCO Reranking

This module implements the FRMR pipeline based on the actual code from
FRMR_faiss_hnsw_retriever_ce_compositequeries_040125.py that combines E5-Large embeddings,
FAISS-HNSW retrieval, and MS MARCO MiniLM-L-6-v2 cross-encoder reranking.
"""

import ast
import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder

from .base_pipeline import BasePipeline


class FRMRPipeline(BasePipeline):
    """
    FRMR Pipeline: FAISS Retrieval and MS MARCO Reranking
    
    This pipeline combines:
    - E5-Large embeddings for dense retrieval
    - FAISS-HNSW for efficient approximate nearest neighbor search (M=64, efConstruction=100, efSearch=200)
    - MS MARCO MiniLM-L-6-v2 for fast cross-encoder reranking
    
    Based on the actual implementation that achieved Hit@1: 51.25%, MRR: 0.6128
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize FRMR pipeline.
        
        Args:
            config: Pipeline configuration containing model paths and parameters
        """
        super().__init__(config)
        self.pipeline_name = "FRMR"
        
        # Load E5-Large model for embeddings
        self.embedding_model = SentenceTransformer('intfloat/e5-large')
        self.logger.info("Loaded E5-Large embedding model")
        
        # Load MS MARCO cross-encoder for reranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.logger.info("Loaded MS MARCO MiniLM-L-6-v2 reranker")
        
        # FAISS index will be loaded separately
        self.faiss_index = None
        
    def load_faiss_index(self, index_path: str) -> None:
        """Load FAISS HNSW index from file."""
        self.faiss_index = faiss.read_index(index_path)
        self.logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} embeddings")
        
        # Verify HNSW parameters
        if hasattr(self.faiss_index, 'hnsw'):
            self.logger.info(f"HNSW parameters - M: {self.faiss_index.hnsw.M}, "
                           f"efSearch: {self.faiss_index.hnsw.efSearch}")
    
    def retrieve(self, query: str, k: int = 100) -> List[int]:
        """
        Retrieve top-k candidates using FAISS HNSW.
        
        Args:
            query: Input query string
            k: Number of candidates to retrieve
            
        Returns:
            List of candidate node IDs
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not loaded. Call load_faiss_index() first.")
        
        # Embed the query using E5-Large
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index for top-k results
        distances, indices = self.faiss_index.search(query_embedding, k)
        retrieved_indices = indices[0]
        
        # Map indices to node IDs
        retrieved_node_ids = [int(self.node_df.iloc[idx]['node_id']) for idx in retrieved_indices]
        
        return retrieved_node_ids
    
    def rerank(self, query: str, candidate_ids: List[int], k: int = 100) -> List[int]:
        """
        Rerank candidates using MS MARCO cross-encoder.
        
        Args:
            query: Input query string
            candidate_ids: List of candidate node IDs
            k: Number of top candidates to return
            
        Returns:
            List of reranked node IDs
        """
        if self.node_df is None:
            raise ValueError("Node data not loaded. Call load_data() first.")
        
        # Get retrieved node IDs from node_df that match the predicted IDs
        retrieved_node_ids = self.node_df[self.node_df['node_id'].isin(candidate_ids)]['node_id'].tolist()
        
        # Create input pairs for cross-encoder
        input_pairs = []
        valid_node_ids = []
        
        for node_id in retrieved_node_ids:
            # Find the node in the dataframe
            node_row = self.node_df[self.node_df['node_id'] == node_id]
            if not node_row.empty:
                combined_text = node_row.iloc[0]['combined_text']
                input_pairs.append((query, combined_text))
                valid_node_ids.append(node_id)
        
        if not input_pairs:
            return []
        
        # Score using cross-encoder
        scores = self.reranker.predict(input_pairs)
        
        # Sort by score (descending) and return top-k
        ranked_results = [x for _, x in sorted(zip(scores, valid_node_ids), reverse=True)]
        return ranked_results[:k]
    
    def evaluate_on_dataset(
        self, 
        query_file: Path, 
        output_file: Path, 
        save_partial: bool = True,
        partial_interval: int = 10
    ) -> pd.DataFrame:
        """
        Evaluate pipeline on a dataset matching the original implementation.
        
        Args:
            query_file: Path to query dataset CSV file
            output_file: Path to save evaluation results
            save_partial: Whether to save partial results during processing
            partial_interval: Interval for saving partial results
            
        Returns:
            DataFrame with evaluation results
        """
        from ..evaluation.metrics import compute_metrics
        
        # Load queries
        queries_df = pd.read_csv(query_file)
        
        # Preprocess answer columns if they're strings
        if 'answer_ids' in queries_df.columns:
            queries_df['answer_ids'] = queries_df['answer_ids'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        
        results_list = []
        partial_file = output_file.with_suffix('.partial.csv')
        
        self.logger.info(f"Evaluating on {len(queries_df)} queries")
        
        with tqdm(total=len(queries_df), desc=f"Processing {self.pipeline_name}") as pbar:
            for idx, row in queries_df.iterrows():
                query_text = row['query']
                query_id = row.get('id', idx + 1)
                correct_answers = [int(x) for x in row['answer_ids']]
                
                try:
                    # Retrieve candidates
                    candidate_ids = self.retrieve(query_text, k=100)
                    
                    # Rerank candidates
                    final_ids = self.rerank(query_text, candidate_ids, k=100)
                    
                    # Compute metrics
                    metrics = compute_metrics(final_ids, correct_answers)
                    
                    # Store results
                    result = {
                        'query': query_text,
                        'FAISSHNSW_answer': final_ids,
                        'correct_answer': correct_answers,
                        **metrics
                    }
                    
                    results_list.append(result)
                    
                    # Log progress for first few queries
                    if idx < 5:
                        self.logger.info(f"Query {idx}: {query_text[:50]}...")
                        self.logger.info(f"Predicted: {final_ids[:5]}")
                        self.logger.info(f"Hit@1: {metrics['hit@1']}, MRR: {metrics['MRR']:.3f}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing query {idx}: {e}")
                    # Add empty result to maintain alignment
                    results_list.append({
                        'query': query_text,
                        'FAISSHNSW_answer': [],
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
    
    def batch_process(
        self, 
        queries: List[str], 
        retrieve_k: int = 100, 
        rerank_k: int = 100,
        batch_size: int = 16
    ) -> List[List[int]]:
        """
        Process multiple queries efficiently in batches.
        
        Args:
            queries: List of query strings
            retrieve_k: Number of candidates to retrieve per query
            rerank_k: Number of candidates to return after reranking
            batch_size: Batch size for processing
            
        Returns:
            List of result lists (one per query)
        """
        results = []
        
        self.logger.info(f"Processing {len(queries)} queries in batches of {batch_size}")
        
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            batch_results = []
            
            for query in batch_queries:
                result = self.process_query(query, retrieve_k, rerank_k)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            if i % (batch_size * 10) == 0:
                self.logger.info(f"Processed {i + len(batch_queries)}/{len(queries)} queries")
        
        return results
    
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
        
        # Log performance characteristics
        self.logger.info("\n=== Pipeline Characteristics ===")
        self.logger.info("✅ Optimized for speed/accuracy tradeoff")
        self.logger.info("⚡ Fast MS MARCO cross-encoder reranking")
        self.logger.info("🚀 Production-ready performance")
        self.logger.info("💰 Cost-effective for large-scale deployment")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        info = {
            'name': self.pipeline_name,
            'description': 'FAISS Retrieval and MS MARCO Reranking Pipeline',
            'retriever': {
                'type': 'FAISS',
                'model': self.retriever.model_name if self.retriever else 'Not initialized',
                'index_type': self.retriever.index_type if self.retriever else 'Not initialized'
            },
            'reranker': {
                'type': 'MS MARCO Cross-Encoder',
                'model': self.reranker.model_name if self.reranker else 'Not initialized'
            },
            'characteristics': [
                'Best speed/accuracy tradeoff',
                'Production-ready performance', 
                'Fast cross-encoder reranking',
                'Cost-effective deployment'
            ],
            'use_cases': [
                'Customer-facing applications',
                'Real-time query processing',
                'Production systems with moderate resources'
            ]
        }
        
        return info