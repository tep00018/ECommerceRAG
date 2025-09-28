"""
FAISS Retriever Module

This module provides FAISS-based dense retrieval using sentence transformers
for embedding generation and FAISS indices for efficient similarity search.
"""

import faiss
import numpy as np
from typing import List, Optional
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pickle
import logging


class FAISSRetriever:
    """FAISS-based dense retriever with sentence transformer embeddings."""
    
    def __init__(
        self,
        model_name: str = "intfloat/e5-large",
        index_path: Optional[Path] = None,
        index_type: str = "HNSW",
        dimension: int = 1024,
        **index_params
    ):
        """
        Initialize FAISS retriever.
        
        Args:
            model_name: Sentence transformer model name
            index_path: Path to pre-built FAISS index
            index_type: Type of FAISS index ('FLAT' or 'HNSW')
            dimension: Embedding dimension
            **index_params: Additional index parameters (M, efConstruction, efSearch for HNSW)
        """
        self.model_name = model_name
        self.index_type = index_type
        self.dimension = dimension
        self.index_params = index_params
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentence transformer
        self.encoder = SentenceTransformer(model_name)
        self.logger.info(f"Loaded sentence transformer: {model_name}")
        
        # Load or create FAISS index
        if index_path and index_path.exists():
            self.load_index(index_path)
        else:
            self.index = None
            self.node_ids = None
            
    def build_index(self, documents: List[str], node_ids: List[int]) -> None:
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of document texts
            node_ids: Corresponding node IDs
        """
        self.logger.info(f"Building FAISS index for {len(documents)} documents")
        
        # Generate embeddings
        embeddings = self.encoder.encode(
            documents, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        embeddings = embeddings.astype('float32')
        
        # Create FAISS index
        if self.index_type == "FLAT":
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "HNSW":
            # HNSW parameters
            M = self.index_params.get('M', 64)
            efConstruction = self.index_params.get('efConstruction', 100)
            
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            self.index.hnsw.efConstruction = efConstruction
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        # Set search parameters for HNSW
        if self.index_type == "HNSW":
            efSearch = self.index_params.get('efSearch', 200)
            self.index.hnsw.efSearch = efSearch
        
        # Store node IDs
        self.node_ids = np.array(node_ids)
        
        self.logger.info(f"Built {self.index_type} index with {self.index.ntotal} vectors")
    
    def save_index(self, save_path: Path) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_file = save_path / "faiss.index"
        faiss.write_index(self.index, str(index_file))
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'index_type': self.index_type,
            'dimension': self.dimension,
            'index_params': self.index_params,
            'node_ids': self.node_ids
        }
        
        metadata_file = save_path / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        self.logger.info(f"Index saved to {save_path}")
    
    def load_index(self, index_path: Path) -> None:
        """Load FAISS index and metadata from disk."""
        # Load FAISS index
        index_file = index_path / "faiss.index"
        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        self.index = faiss.read_index(str(index_file))
        
        # Load metadata
        metadata_file = index_path / "metadata.pkl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.model_name = metadata['model_name']
        self.index_type = metadata['index_type']
        self.dimension = metadata['dimension']
        self.index_params = metadata['index_params']
        self.node_ids = metadata['node_ids']
        
        # Set search parameters for HNSW
        if self.index_type == "HNSW":
            efSearch = self.index_params.get('efSearch', 200)
            self.index.hnsw.efSearch = efSearch
        
        self.logger.info(f"Loaded {self.index_type} index with {self.index.ntotal} vectors")
    
    def search(self, query: str, k: int = 100) -> List[int]:
        """
        Search for top-k most similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of node IDs ranked by similarity
        """
        if self.index is None:
            raise ValueError("No index loaded. Build or load an index first.")
        
        # Encode query
        query_embedding = self.encoder.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding.astype('float32')
        
        # Search index
        scores, indices = self.index.search(query_embedding, k)
        
        # Convert indices to node IDs
        retrieved_node_ids = [int(self.node_ids[idx]) for idx in indices[0]]
        
        return retrieved_node_ids
    
    def get_index_stats(self) -> dict:
        """Get statistics about the loaded index."""
        if self.index is None:
            return {"status": "No index loaded"}
        
        stats = {
            "index_type": self.index_type,
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "model_name": self.model_name
        }
        
        if self.index_type == "HNSW":
            stats.update({
                "M": self.index.hnsw.M,
                "efConstruction": self.index.hnsw.efConstruction,
                "efSearch": self.index.hnsw.efSearch
            })
        
        return stats