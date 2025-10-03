"""
BARMR Pipeline Updates for Integration with Updated BM25Retriever

This shows the key methods that need to be updated in your barmr_pipeline.py 
to work seamlessly with the new BM25Retriever.
"""

import ast
import pickle
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from sentence_transformers import CrossEncoder

# Import the updated BM25Retriever
from .bm25_retriever import BM25Retriever
from .cross_encoder import MSMARCOReranker


class BARMRPipeline:
    """
    BARMR Pipeline: BM25 Augmented Retrieval and MS MARCO Reranking
    
    Updated to use the new BM25Retriever class that matches the ipynb implementation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize BARMR pipeline."""
        self.pipeline_name = "BARMR"
        self.config = config
        
        # Initialize BM25 retriever with optimized hyperparameters
        self.retriever = BM25Retriever(
            k1=1.016564434220879,
            b=0.8856501982953431,
            similarity_threshold=21.0,
            graph_augmentation=True,
            remove_stopwords=True
        )
        
        # Initialize MS MARCO cross-encoder for reranking
        # Option 1: Use CrossEncoder directly
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Option 2: Use the MSMARCOReranker wrapper
        # self.reranker = MSMARCOReranker(
        #     model_name='cross-encoder/ms-marco-MiniLM-L-6-v2',
        #     batch_size=32
        # )
        
        self.node_df = None
        
    def load_data_and_build_index(
        self, 
        data_path: Path,
        tokens_file: Path
    ) -> None:
        """
        Load node data and build BM25 index.
        
        Args:
            data_path: Path to node data CSV
            tokens_file: Path to preprocessed tokens pickle file
        """
        # Load node data
        self.node_df = pd.read_csv(data_path)
        
        # Load preprocessed tokens
        with open(tokens_file, 'rb') as f:
            node_tokens = pickle.load(f)
        
        # Extract node IDs
        node_ids = self.node_df['node_id'].tolist()
        
        # Extract graph edges
        graph_edges = self._extract_graph_edges()
        
        # Build BM25 index
        self.retriever.build_index(
            node_tokens=node_tokens,
            node_ids=node_ids,
            graph_edges=graph_edges
        )
        
        print(f"Index built with {len(node_tokens)} documents")
        print(f"Graph has {len(graph_edges)} nodes with edges")
    
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
        
        This now directly calls the BM25Retriever.search() method which handles:
        - Query tokenization
        - Stopword removal
        - BM25 scoring
        - Similarity threshold filtering
        - 1-hop graph augmentation
        
        Args:
            query: Input query string
            k: Number of candidates to retrieve
            
        Returns:
            List of candidate node IDs (after augmentation)
        """
        return self.retriever.search(query, k=k)
    
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
            raise ValueError("Node data not loaded")
        
        # Create lookup for efficient node retrieval
        node_lookup = {
            int(row['node_id']): row['combined_text'] 
            for _, row in self.node_df.iterrows()
        }
        
        # Build query-document pairs
        input_pairs = []
        valid_node_ids = []
        
        for node_id in candidate_ids:
            if node_id in node_lookup:
                combined_text = node_lookup[node_id]
                input_pairs.append([query, combined_text])
                valid_node_ids.append(node_id)
        
        if not input_pairs:
            return []
        
        # Score using cross-encoder
        scores = self.reranker.predict(input_pairs)
        
        # Sort by score (descending) and return top-k
        ranked_results = [
            node_id for _, node_id in 
            sorted(zip(scores, valid_node_ids), reverse=True)
        ]
        
        return ranked_results[:k]
    
    def process_query(
        self, 
        query: str, 
        retrieve_k: int = 100, 
        rerank_k: int = 100
    ) -> List[int]:
        """
        Process a query through the full BARMR pipeline.
        
        Args:
            query: Input query string
            retrieve_k: Number of candidates to retrieve (will be augmented)
            rerank_k: Number of final results after reranking
            
        Returns:
            List of final ranked node IDs
        """
        # Step 1: BM25 retrieval with 1-hop augmentation
        # The retriever.search() method handles both BM25 and augmentation
        augmented_candidates = self.retrieve(query, retrieve_k)
        
        # Step 2: Cross-encoder reranking
        final_results = self.rerank(query, augmented_candidates, rerank_k)
        
        return final_results


# Example usage showing the complete workflow
def example_usage():
    """Example of how to use the updated pipeline."""
    
    # 1. Initialize pipeline
    config = {}  # Add any config parameters you need
    pipeline = BARMRPipeline(config)
    
    # 2. Load data and build index
    pipeline.load_data_and_build_index(
        data_path=Path("data/nodes.csv"),
        tokens_file=Path("data/node_tokens.pkl")
    )
    
    # 3. Process queries
    query = "wireless bluetooth headphones"
    results = pipeline.process_query(
        query=query,
        retrieve_k=100,  # BM25 will retrieve up to 100, then augment
        rerank_k=100     # Return top 100 after reranking
    )
    
    print(f"Top 10 results for '{query}':")
    for i, node_id in enumerate(results[:10], 1):
        print(f"{i}. Node ID: {node_id}")