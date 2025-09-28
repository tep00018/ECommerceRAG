"""
BM25 Retriever Module

This module provides BM25-based sparse retrieval with optional graph augmentation
for knowledge graph-enhanced information retrieval.
"""

import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Set, Optional, Tuple
from pathlib import Path
from collections import Counter
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
    from nltk.stem import PorterStemmer
except ImportError:
    raise ImportError(
        "nltk is required for text processing. "
        "Install with: pip install nltk"
    )


class BM25Retriever:
    """
    BM25-based sparse retriever with optional graph augmentation.
    
    Supports traditional BM25 ranking with optional 1-hop graph expansion
    using knowledge graph relationships like 'also-bought' and 'also-viewed'.
    """
    
    def __init__(
        self,
        tokenizer: str = "simple",
        remove_stopwords: bool = True,
        use_stemming: bool = False,
        k1: float = 1.5,
        b: float = 0.75,
        graph_augmentation: bool = False,
        max_expansion: int = 50
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            tokenizer: Tokenization method ('simple' or 'nltk')
            remove_stopwords: Whether to remove stopwords
            use_stemming: Whether to use stemming
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (field length normalization)
            graph_augmentation: Whether to use graph augmentation
            max_expansion: Maximum nodes to add per candidate in graph expansion
        """
        self.tokenizer_type = tokenizer
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.k1 = k1
        self.b = b
        self.graph_augmentation = graph_augmentation
        self.max_expansion = max_expansion
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize NLTK components if needed
        if tokenizer == "nltk" or remove_stopwords or use_stemming:
            self._setup_nltk()
        
        # Initialize components
        self.bm25 = None
        self.node_ids = None
        self.documents = None
        self.tokenized_corpus = None
        self.graph_edges = None
        
        # Text processing components
        self.stopwords = set(stopwords.words('english')) if remove_stopwords else set()
        self.stemmer = PorterStemmer() if use_stemming else None
    
    def _setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize and process text.
        
        Args:
            text: Input text string
            
        Returns:
            List of processed tokens
        """
        if not isinstance(text, str):
            text = str(text)
        
        # Tokenize
        if self.tokenizer_type == "nltk":
            tokens = word_tokenize(text.lower())
        else:
            # Simple tokenization
            tokens = text.lower().split()
        
        # Remove punctuation and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalnum()]
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # Apply stemming
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        return tokens
    
    def build_index(
        self, 
        documents: List[str], 
        node_ids: List[int],
        graph_edges: Optional[Dict[int, Dict[str, List[int]]]] = None
    ) -> None:
        """
        Build BM25 index from documents.
        
        Args:
            documents: List of document texts
            node_ids: Corresponding node IDs
            graph_edges: Graph structure for augmentation
                        Format: {node_id: {'also_buy': [...], 'also_view': [...]}}
        """
        self.logger.info(f"Building BM25 index for {len(documents)} documents")
        
        if len(documents) != len(node_ids):
            raise ValueError("Documents and node_ids must have same length")
        
        # Store documents and IDs
        self.documents = documents
        self.node_ids = np.array(node_ids)
        self.graph_edges = graph_edges or {}
        
        # Tokenize all documents
        self.logger.info("Tokenizing documents...")
        self.tokenized_corpus = [self._tokenize_text(doc) for doc in documents]
        
        # Build BM25 index
        self.logger.info("Building BM25 index...")
        self.bm25 = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b
        )
        
        self.logger.info(f"BM25 index built with {len(self.tokenized_corpus)} documents")
        
        if self.graph_augmentation and graph_edges:
            self.logger.info(f"Graph augmentation enabled with {len(graph_edges)} nodes having edges")
    
    def search(self, query: str, k: int = 100) -> List[int]:
        """
        Search for top-k most similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of node IDs ranked by BM25 score
        """
        if self.bm25 is None:
            raise ValueError("No index built. Call build_index() first.")
        
        # Tokenize query
        tokenized_query = self._tokenize_text(query)
        
        if not tokenized_query:
            self.logger.warning("Empty query after tokenization")
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Convert to node IDs
        initial_results = [int(self.node_ids[idx]) for idx in top_indices]
        
        # Apply graph augmentation if enabled
        if self.graph_augmentation and self.graph_edges:
            augmented_results = self._apply_graph_augmentation(initial_results, k)
            return augmented_results
        
        return initial_results
    
    def _apply_graph_augmentation(self, initial_results: List[int], k: int) -> List[int]:
        """
        Apply graph augmentation by adding 1-hop neighbors.
        
        Args:
            initial_results: Initial BM25 results
            k: Target number of results
            
        Returns:
            Augmented result list
        """
        augmented_nodes = set(initial_results)
        
        # Add 1-hop neighbors for each initial result
        for node_id in initial_results:
            if node_id in self.graph_edges:
                edges = self.graph_edges[node_id]
                
                # Add also-bought neighbors
                if 'also_buy' in edges:
                    neighbors = edges['also_buy'][:self.max_expansion]
                    augmented_nodes.update(neighbors)
                
                # Add also-viewed neighbors
                if 'also_view' in edges:
                    neighbors = edges['also_view'][:self.max_expansion]
                    augmented_nodes.update(neighbors)
        
        # Convert back to list, preserving initial order and adding new nodes
        result_list = []
        
        # First, add initial results in their original order
        for node_id in initial_results:
            if node_id in augmented_nodes:
                result_list.append(node_id)
                augmented_nodes.remove(node_id)
        
        # Then add remaining augmented nodes
        result_list.extend(list(augmented_nodes))
        
        # Return top-k results
        return result_list[:k]
    
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
        tokenized_query = self._tokenize_text(query)
        
        if not tokenized_query:
            return [], []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
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
            'documents': self.documents,
            'tokenized_corpus': self.tokenized_corpus,
            'graph_edges': self.graph_edges,
            'config': {
                'tokenizer_type': self.tokenizer_type,
                'remove_stopwords': self.remove_stopwords,
                'use_stemming': self.use_stemming,
                'k1': self.k1,
                'b': self.b,
                'graph_augmentation': self.graph_augmentation,
                'max_expansion': self.max_expansion
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
        self.documents = metadata['documents']
        self.tokenized_corpus = metadata['tokenized_corpus']
        self.graph_edges = metadata['graph_edges']
        
        # Restore configuration
        config = metadata['config']
        self.tokenizer_type = config['tokenizer_type']
        self.remove_stopwords = config['remove_stopwords']
        self.use_stemming = config['use_stemming']
        self.k1 = config['k1']
        self.b = config['b']
        self.graph_augmentation = config['graph_augmentation']
        self.max_expansion = config['max_expansion']
        
        self.logger.info(f"Loaded BM25 index with {len(self.documents)} documents")
        
        if self.graph_augmentation:
            edge_count = len(self.graph_edges) if self.graph_edges else 0
            self.logger.info(f"Graph augmentation enabled with {edge_count} nodes having edges")
    
    def get_index_stats(self) -> dict:
        """Get statistics about the loaded index."""
        if self.bm25 is None:
            return {"status": "No index loaded"}
        
        stats = {
            "total_documents": len(self.documents),
            "tokenizer": self.tokenizer_type,
            "remove_stopwords": self.remove_stopwords,
            "use_stemming": self.use_stemming,
            "k1": self.k1,
            "b": self.b,
            "graph_augmentation": self.graph_augmentation
        }
        
        if self.graph_augmentation and self.graph_edges:
            total_edges = sum(
                len(edges.get('also_buy', [])) + len(edges.get('also_view', []))
                for edges in self.graph_edges.values()
            )
            stats.update({
                "nodes_with_edges": len(self.graph_edges),
                "total_edges": total_edges,
                "max_expansion": self.max_expansion
            })
        
        return stats
    
    def analyze_query_coverage(self, query: str) -> dict:
        """
        Analyze how well the corpus covers the query terms.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary with coverage statistics
        """
        if self.bm25 is None:
            return {}
        
        tokenized_query = self._tokenize_text(query)
        
        if not tokenized_query:
            return {"error": "Empty query after tokenization"}
        
        # Get document frequencies for query terms
        doc_freqs = {}
        total_docs = len(self.documents)
        
        for term in tokenized_query:
            # Count how many documents contain this term
            term_count = sum(1 for doc_tokens in self.tokenized_corpus if term in doc_tokens)
            doc_freqs[term] = {
                'document_frequency': term_count,
                'idf': np.log(total_docs / (term_count + 1))
            }
        
        return {
            "query_terms": tokenized_query,
            "total_documents": total_docs,
            "term_coverage": doc_freqs,
            "covered_terms": sum(1 for stats in doc_freqs.values() if stats['document_frequency'] > 0),
            "total_terms": len(tokenized_query)
        }