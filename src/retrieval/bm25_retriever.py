"""
BM25 Retriever Module with 1-Hop Graph Augmentation

This module provides BM25-based sparse retrieval with 1-hop graph expansion
matching the implementation from BM25_CE_CompositeQueries_040125_ipynb.ipynb
"""

import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
import logging

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError(
        "rank_bm25 is required for BM25 retrieval. "
        "Install with: pip install rank-bm25"
    )

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    raise ImportError(
        "nltk is required for text processing. "
        "Install with: pip install nltk"
    )


class BM25Retriever:
    """
    BM25-based sparse retriever with 1-hop graph augmentation.
    
    Implements BM25 ranking with similarity threshold filtering and 1-hop graph 
    expansion using 'also_buy' and 'also_view' relationships.
    
    This matches the implementation that achieved Hit@1: 49.67%, MRR: 0.6037
    """
    
    def __init__(
        self,
        k1: float = 1.016564434220879,
        b: float = 0.8856501982953431,
        similarity_threshold: float = 21.0,
        graph_augmentation: bool = True,
        remove_stopwords: bool = True
    ):
        """
        Initialize BM25 retriever with optimized hyperparameters.
        
        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (field length normalization)
            similarity_threshold: Minimum BM25 score threshold for candidates
            graph_augmentation: Whether to use 1-hop graph augmentation
            remove_stopwords: Whether to remove stopwords from queries
        """
        self.k1 = k1
        self.b = b
        self.similarity_threshold = similarity_threshold
        self.graph_augmentation = graph_augmentation
        self.remove_stopwords = remove_stopwords
        
        self.logger = logging.getLogger(__name__)
        
        # Setup NLTK
        self._setup_nltk()
        
        # Initialize components
        self.bm25 = None
        self.node_ids = None
        self.node_tokens = None  # Preprocessed tokenized corpus
        self.graph_edges = None  # Dict mapping node_id to {'also_buy': [...], 'also_view': [...]}
        
        # Text processing components
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
    
    def _setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            self.logger.warning(f"NLTK data download issue: {e}")
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text with lowercasing (matches ipynb tokenize function).
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not isinstance(text, str):
            text = str(text)
        return word_tokenize(text.lower())
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove stopwords from tokens (matches ipynb remove_stopwords function).
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of tokens with stopwords removed
        """
        return [word for word in tokens if word.lower() not in self.stop_words]
    
    def build_index(
        self, 
        node_tokens: List[List[str]], 
        node_ids: List[int],
        graph_edges: Optional[Dict[int, Dict[str, List[int]]]] = None
    ) -> None:
        """
        Build BM25 index from preprocessed tokens.
        
        Args:
            node_tokens: List of tokenized documents (already preprocessed)
            node_ids: List of node IDs corresponding to documents
            graph_edges: Dictionary mapping node_id to edge information
                        Format: {node_id: {'also_buy': [...], 'also_view': [...]}}
        """
        if len(node_tokens) != len(node_ids):
            raise ValueError("node_tokens and node_ids must have same length")
        
        self.node_tokens = node_tokens
        self.node_ids = node_ids
        self.graph_edges = graph_edges if graph_edges else {}
        
        # Build BM25 index with custom hyperparameters
        self.bm25 = BM25Okapi(node_tokens, k1=self.k1, b=self.b)
        
        self.logger.info(f"Built BM25 index with {len(node_tokens)} documents")
        self.logger.info(f"Hyperparameters: k1={self.k1}, b={self.b}, threshold={self.similarity_threshold}")
        
        if self.graph_augmentation and self.graph_edges:
            self.logger.info(f"Graph augmentation enabled with {len(self.graph_edges)} nodes having edges")
    
    def search(self, query: str, k: int = 100) -> List[int]:
        """
        Search using BM25 with optional 1-hop graph augmentation.
        
        Implementation matches the ipynb notebook:
        1. Tokenize and remove stopwords from query
        2. Get BM25 scores
        3. Filter by similarity threshold
        4. If graph_augmentation enabled, apply 1-hop expansion
        
        Args:
            query: Query string
            k: Maximum number of results to return
            
        Returns:
            List of node IDs
        """
        if self.bm25 is None:
            raise ValueError("No index built. Call build_index() first.")
        
        # Tokenize query (matches ipynb: tokenize then remove_stopwords)
        query_tokens = self.tokenize(query)
        
        if self.remove_stopwords:
            query_tokens = self.remove_stopwords_from_tokens(query_tokens)
        
        if not query_tokens:
            self.logger.warning("Empty query after tokenization")
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Apply similarity threshold and get top candidates (matches ipynb)
        top_n_idx = [idx for idx, score in enumerate(scores) if score >= self.similarity_threshold]
        top_n_idx.sort(key=lambda idx: scores[idx], reverse=True)
        
        # Limit to top k before augmentation
        top_n_idx = top_n_idx[:k]
        
        # Get corresponding node IDs
        initial_node_ids = [int(self.node_ids[idx]) for idx in top_n_idx]
        
        # Apply graph augmentation if enabled
        if self.graph_augmentation and self.graph_edges:
            augmented_node_ids = self._apply_1hop_augmentation(initial_node_ids)
            return augmented_node_ids
        
        return initial_node_ids
    
    def _apply_1hop_augmentation(self, initial_node_ids: List[int]) -> List[int]:
        """
        Apply 1-hop graph augmentation (matches ipynb append_node_ids function).
        
        For each node in initial results:
        - Add all nodes from 'also_buy'
        - Add all nodes from 'also_view'
        - Use set to avoid duplicates
        
        Args:
            initial_node_ids: Initial BM25 results
            
        Returns:
            List of augmented node IDs (original + 1-hop neighbors)
        """
        appended_ids = set(initial_node_ids)  # Using set to avoid duplicates
        
        for node_id in initial_node_ids:
            if node_id in self.graph_edges:
                edges = self.graph_edges[node_id]
                
                # Add all also_buy neighbors
                if 'also_buy' in edges and edges['also_buy']:
                    appended_ids.update(edges['also_buy'])
                
                # Add all also_view neighbors
                if 'also_view' in edges and edges['also_view']:
                    appended_ids.update(edges['also_view'])
        
        return list(appended_ids)
    
    def get_scores(self, query: str) -> Tuple[List[float], List[int]]:
        """
        Get BM25 scores for all documents given a query.
        
        Args:
            query: Query string
            
        Returns:
            Tuple of (scores, node_ids) sorted by score (descending)
        """
        if self.bm25 is None:
            raise ValueError("No index built. Call build_index() first.")
        
        # Tokenize query
        query_tokens = self.tokenize(query)
        
        if self.remove_stopwords:
            query_tokens = self.remove_stopwords_from_tokens(query_tokens)
        
        if not query_tokens:
            return [], []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Sort by score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        sorted_scores = [float(scores[idx]) for idx in sorted_indices]
        sorted_node_ids = [int(self.node_ids[idx]) for idx in sorted_indices]
        
        return sorted_scores, sorted_node_ids
    
    def save_index(self, save_path: Path) -> None:
        """Save BM25 index and metadata to disk."""
        if self.bm25 is None:
            raise ValueError("No index to save. Build index first.")
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save BM25 index
        index_file = save_path / "bm25.pkl"
        with open(index_file, 'wb') as f:
            pickle.dump(self.bm25, f)
        
        # Save metadata
        metadata = {
            'node_ids': self.node_ids,
            'node_tokens': self.node_tokens,
            'graph_edges': self.graph_edges,
            'config': {
                'k1': self.k1,
                'b': self.b,
                'similarity_threshold': self.similarity_threshold,
                'graph_augmentation': self.graph_augmentation,
                'remove_stopwords': self.remove_stopwords
            }
        }
        
        metadata_file = save_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"BM25 index saved to {save_path}")
    
    def load_index(self, index_path: Path) -> None:
        """Load BM25 index and metadata from disk."""
        # Load BM25 index
        index_file = index_path / "bm25.pkl"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        with open(index_file, 'rb') as f:
            self.bm25 = pickle.load(f)
        
        # Load metadata
        metadata_file = index_path / "metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.node_ids = metadata['node_ids']
        self.node_tokens = metadata['node_tokens']
        self.graph_edges = metadata['graph_edges']
        
        # Restore configuration
        config = metadata['config']
        self.k1 = config['k1']
        self.b = config['b']
        self.similarity_threshold = config['similarity_threshold']
        self.graph_augmentation = config['graph_augmentation']
        self.remove_stopwords = config['remove_stopwords']
        
        self.logger.info(f"Loaded BM25 index with {len(self.node_ids)} documents")
        
        if self.graph_augmentation:
            edge_count = len(self.graph_edges) if self.graph_edges else 0
            self.logger.info(f"Graph augmentation enabled with {edge_count} nodes having edges")
    
    def get_index_stats(self) -> dict:
        """Get statistics about the loaded index."""
        if self.bm25 is None:
            return {"status": "No index loaded"}
        
        stats = {
            "total_documents": len(self.node_ids),
            "k1": self.k1,
            "b": self.b,
            "similarity_threshold": self.similarity_threshold,
            "graph_augmentation": self.graph_augmentation,
            "remove_stopwords": self.remove_stopwords
        }
        
        if self.graph_augmentation and self.graph_edges:
            total_edges = sum(
                len(edges.get('also_buy', [])) + len(edges.get('also_view', []))
                for edges in self.graph_edges.values()
            )
            stats.update({
                "nodes_with_edges": len(self.graph_edges),
                "total_edges": total_edges
            })
        
        return stats