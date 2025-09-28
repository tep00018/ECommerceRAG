"""
BARMR Pipeline: BM25 Augmented Retrieval and MS MARCO Reranking

This module implements the BARMR pipeline based on the actual code from
BARMR_bm25_retriever_minilm_l6_cross_encoder.py that combines BM25 sparse retrieval
with graph augmentation and MS MARCO cross-encoder reranking.
"""

import ast
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# BM25 and text processing
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sentence_transformers import CrossEncoder

from .base_pipeline import BasePipeline

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass


class BARMRPipeline(BasePipeline):
    """
    BARMR Pipeline: BM25 Augmented Retrieval and MS MARCO Reranking
    
    This pipeline combines:
    - BM25 sparse retrieval with optimized hyperparameters (k1=1.016, b=0.886, threshold=21)
    - 1-hop graph expansion using 'also-bought' and 'also-viewed' relationships
    - MS MARCO MiniLM-L-6-v2 cross-encoder for reranking
    
    Based on the actual implementation that achieved Hit@1: 49.67%, MRR: 0.6037
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize BARMR pipeline.
        
        Args:
            config: Pipeline configuration containing model paths and parameters
        """
        super().__init__(config)
        self.pipeline_name = "BARMR"
        
        # BM25 hyperparameters from the actual implementation
        self.k1 = 1.016564434220879
        self.b = 0.8856501982953431
        self.similarity_threshold = 21
        self.top_n = 100
        
        # Text processing setup
        self.stop_words = set(stopwords.words('english'))
        
        # Load MS MARCO cross-encoder for reranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.logger.info("Loaded MS MARCO MiniLM-L-6-v2 reranker")
        
        # BM25 and data will be loaded separately
        self.bm25 = None
        self.node_tokens = None
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text with stopword removal."""
        if not isinstance(text, str):
            text = str(text)
        tokens = word_tokenize(text.lower())
        return [word for word in tokens if word.lower() not in self.stop_words]
    
    def load_bm25_data(self, tokens_file: str) -> None:
        """Load preprocessed tokens and build BM25 index."""
        with open(tokens_file, "rb") as file:
            self.node_tokens = pickle.load(file)
        
        # Initialize BM25 with custom hyperparameters
        self.bm25 = BM25Okapi(self.node_tokens, k1=self.k1, b=self.b)
        self.logger.info(f"Built BM25 index with {len(self.node_tokens)} documents")
    
    def retrieve(self, query: str, k: int = 100) -> List[int]:
        """
        Retrieve top-k candidates using BM25 with similarity threshold.
        
        Args:
            query: Input query string
            k: Number of candidates to retrieve
            
        Returns:
            List of candidate node IDs
        """
        if self.bm25 is None:
            raise ValueError("BM25 not loaded. Call load_bm25_data() first.")
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Apply similarity threshold and get top candidates
        top_n_idx = [idx for idx, score in enumerate(scores) if score >= self.similarity_threshold]
        top_n_idx.sort(key=lambda idx: scores[idx], reverse=True)
        
        # Get corresponding node IDs
        top_n_node_ids = [int(self.node_df.iloc[idx]['node_id']) for idx in top_n_idx[:k]]
        
        return top_n_node_ids
    
    def apply_graph_augmentation(self, initial_node_ids: List[int]) -> List[int]:
        """
        Apply 1-hop graph augmentation using also-buy and also-view relationships.
        
        Args:
            initial_node_ids: Initial BM25 retrieval results
            
        Returns:
            Augmented list of node IDs
        """
        augmented_nodes = set(initial_node_ids)
        
        for node_id in initial_node_ids:
            # Find the node in the dataframe
            node_row = self.node_df[self.node_df['node_id'] == node_id]
            
            if not node_row.empty:
                row = node_row.iloc[0]
                
                # Add also_buy neighbors
                if 'also_buy' in row and pd.notna(row['also_buy']):
                    try:
                        also_buy = ast.literal_eval(row['also_buy']) if isinstance(row['also_buy'], str) else row['also_buy']
                        if isinstance(also_buy, list):
                            augmented_nodes.update(also_buy)
                    except (ValueError, SyntaxError):
                        pass
                
                # Add also_view neighbors  
                if 'also_view' in row and pd.notna(row['also_view']):
                    try:
                        also_view = ast.literal_eval(row['also_view']) if isinstance(row['also_view'], str) else row['also_view']
                        if isinstance(also_view, list):
                            augmented_nodes.update(also_view)
                    except (ValueError, SyntaxError):
                        pass
        
        return list(augmented_nodes)
    
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
        
        if not retrieved_node_ids:
            return []
        
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
    
    def process_query(self, query: str, retrieve_k: int = 100, rerank_k: int = 100, 
                     use_augmentation: bool = True) -> List[int]:
        """
        Process a query through the full BARMR pipeline.
        
        Args:
            query: Input query string
            retrieve_k: Number of candidates to retrieve with BM25
            rerank_k: Number of candidates to return after reranking
            use_augmentation: Whether to apply graph augmentation
            
        Returns:
            List of final ranked node IDs
        """
        # Step 1: BM25 retrieval
        initial_candidates = self.retrieve(query, retrieve_k)
        
        # Step 2: Graph augmentation (optional)
        if use_augmentation:
            augmented_candidates = self.apply_graph_augmentation(initial_candidates)
        else:
            augmented_candidates = initial_candidates
        
        # Step 3: Cross-encoder reranking
        final_results = self.rerank(query, augmented_candidates, rerank_k)
        
        return final_results
    
    def load_data(self, data_path: Path) -> None:
        """Load node data and build retrieval index."""
        # Load node data using parent method
        super().load_data(data_path)
        
        # Extract graph edges for augmentation
        graph_edges = self._extract_graph_edges()
        
        # Prepare documents and build BM25 index
        documents = self.node_df['combined_text'].tolist()
        node_ids = self.node_df['node_id'].tolist()
        
        self.logger.info("Building BM25 index with graph augmentation...")
        self.retriever.build_index(documents, node_ids, graph_edges)
        
        self.logger.info(f"Index built with {len(documents)} documents and {len(graph_edges)} graph nodes")
    
    def _extract_graph_edges(self) -> Dict[int, Dict[str, List[int]]]:
        """
        Extract graph edges from node data for augmentation.
        
        Returns:
            Dictionary mapping node_id to edge information
        """
        graph_edges = {}
        
        for _, row in self.node_df.iterrows():
            node_id = int(row['node_id'])
            edges = {}
            
            # Extract also_buy relationships
            if 'also_buy' in row and pd.notna(row['also_buy']):
                try:
                    also_buy = ast.literal_eval(row['also_buy']) if isinstance(row['also_buy'], str) else row['also_buy']
                    if isinstance(also_buy, list):
                        edges['also_buy'] = [int(x) for x in also_buy]
                except (ValueError, SyntaxError):
                    edges['also_buy'] = []
            else:
                edges['also_buy'] = []
            
            # Extract also_view relationships
            if 'also_view' in row and pd.notna(row['also_view']):
                try:
                    also_view = ast.literal_eval(row['also_view']) if isinstance(row['also_view'], str) else row['also_view']
                    if isinstance(also_view, list):
                        edges['also_view'] = [int(x) for x in also_view]
                except (ValueError, SyntaxError):
                    edges['also_view'] = []
            else:
                edges['also_view'] = []
            
            # Only store nodes that have edges
            if edges['also_buy'] or edges['also_view']:
                graph_edges[node_id] = edges
        
        return graph_edges
    
    def retrieve(self, query: str, k: int = 100) -> List[int]:
        """
        Retrieve top-k candidates using BM25 with graph augmentation.
        
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
        Rerank candidates using MS MARCO cross-encoder.
        
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
        
        # Rerank using MS MARCO cross-encoder
        reranked_ids = self.reranker.rerank(query, doc_texts, doc_ids, k=k)
        
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
    
    def analyze_retrieval_process(self, query: str, k: int = 100) -> Dict[str, Any]:
        """
        Analyze the retrieval process for interpretability.
        
        Args:
            query: Input query string
            k: Number of candidates to analyze
            
        Returns:
            Dictionary with retrieval analysis
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized.")
        
        # Get query coverage analysis
        coverage = self.retriever.analyze_query_coverage(query)
        
        # Get BM25 scores and initial results
        scores, node_ids = self.retriever.get_scores(query)
        initial_results = node_ids[:k]
        
        # Get augmented results
        augmented_results = self.retrieve(query, k)
        
        # Analyze augmentation effect
        initial_set = set(initial_results)
        augmented_set = set(augmented_results)
        
        added_by_augmentation = augmented_set - initial_set
        original_preserved = list(initial_set & augmented_set)
        
        analysis = {
            'query': query,
            'query_coverage': coverage,
            'bm25_results': {
                'count': len(initial_results),
                'top_scores': scores[:10],  # Top 10 scores
                'node_ids': initial_results[:10]  # Top 10 node IDs
            },
            'augmentation_effect': {
                'original_count': len(initial_results),
                'augmented_count': len(augmented_results),
                'nodes_added': len(added_by_augmentation),
                'nodes_preserved': len(original_preserved),
                'added_nodes': list(added_by_augmentation)[:20]  # Sample of added nodes
            },
            'graph_utilization': self._analyze_graph_utilization(initial_results)
        }
        
        return analysis
    
    def _analyze_graph_utilization(self, node_ids: List[int]) -> Dict[str, Any]:
        """
        Analyze how graph edges are utilized in augmentation.
        
        Args:
            node_ids: List of initial node IDs
            
        Returns:
            Graph utilization statistics
        """
        if not self.retriever.graph_edges:
            return {"status": "No graph edges available"}
        
        total_also_buy = 0
        total_also_view = 0
        nodes_with_edges = 0
        
        for node_id in node_ids:
            if node_id in self.retriever.graph_edges:
                nodes_with_edges += 1
                edges = self.retriever.graph_edges[node_id]
                total_also_buy += len(edges.get('also_buy', []))
                total_also_view += len(edges.get('also_view', []))
        
        return {
            'total_initial_nodes': len(node_ids),
            'nodes_with_edges': nodes_with_edges,
            'utilization_rate': nodes_with_edges / len(node_ids) if node_ids else 0,
            'total_also_buy_edges': total_also_buy,
            'total_also_view_edges': total_also_view,
            'avg_edges_per_node': (total_also_buy + total_also_view) / nodes_with_edges if nodes_with_edges > 0 else 0
        }
    
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
                    
                    # Log progress for first few queries with analysis
                    if idx < 3:
                        self.logger.info(f"Query {idx}: {query_text[:50]}...")
                        self.logger.info(f"Predicted: {predicted_ids[:5]}")
                        self.logger.info(f"Hit@1: {metrics['hit@1']}, MRR: {metrics['MRR']:.3f}")
                        
                        # Add retrieval analysis for first query
                        if idx == 0:
                            analysis = self.analyze_retrieval_process(query_text)
                            self.logger.info(f"Graph utilization: {analysis['graph_utilization']['utilization_rate']:.2%}")
                    
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
        
        # Log pipeline characteristics
        self.logger.info("\n=== Pipeline Characteristics ===")
        self.logger.info("🔍 Interpretable sparse retrieval with BM25")
        self.logger.info("🕸️ Graph-enhanced with 1-hop expansion")
        self.logger.info("📈 Leverages also-bought/also-viewed relationships")
        self.logger.info("⚡ Fast cross-encoder reranking")
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        retriever_stats = self.retriever.get_index_stats() if self.retriever else {}
        
        info = {
            'name': self.pipeline_name,
            'description': 'BM25 Augmented Retrieval and MS MARCO Reranking Pipeline',
            'retriever': {
                'type': 'BM25 + Graph Augmentation',
                'graph_augmentation': retriever_stats.get('graph_augmentation', False),
                'nodes_with_edges': retriever_stats.get('nodes_with_edges', 0),
                'total_edges': retriever_stats.get('total_edges', 0)
            },
            'reranker': {
                'type': 'MS MARCO Cross-Encoder',
                'model': self.reranker.model_name if self.reranker else 'Not initialized'
            },
            'characteristics': [
                'Interpretable sparse retrieval',
                'Graph-enhanced with knowledge relationships',
                'Fast cross-encoder reranking',
                'Keyword-based with semantic enhancement'
            ],
            'use_cases': [
                'Interpretable retrieval decisions',
                'Exact keyword matching priority',
                'Leveraging product relationships',
                'Explainable recommendation systems'
            ]
        }
        
        return info