#!/usr/bin/env python3
"""
Build FAISS FLAT Index for Dense Retrieval

This script creates a FAISS FLAT (brute-force) index from pre-computed
E5-Large embeddings. The FLAT index guarantees exact nearest neighbor
search results.

Usage:
    python scripts/build_faiss_flat.py --embeddings data/embeddings/intfloat_e5_large_embeddings.npy
    python scripts/build_faiss_flat.py --embeddings embeddings.npy --data-file data/nodes/nodes.csv
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import numpy as np
import pandas as pd
import faiss


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'faiss_flat_build_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ]
    )
    return logging.getLogger(__name__)


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """Load embeddings from numpy file."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    logger.info(f"Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    
    # Ensure float32
    if embeddings.dtype != np.float32:
        logger.info(f"Converting embeddings from {embeddings.dtype} to float32")
        embeddings = embeddings.astype('float32')
    
    return embeddings


def build_flat_index(embeddings: np.ndarray, normalize: bool = True) -> faiss.IndexFlatIP:
    """
    Build FAISS FLAT index with inner product similarity.
    
    Args:
        embeddings: Numpy array of embeddings (N x D)
        normalize: Whether to L2-normalize embeddings for cosine similarity
        
    Returns:
        FAISS FLAT index
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Building FAISS FLAT index...")
    logger.info(f"  Index type: IndexFlatIP (Inner Product)")
    logger.info(f"  Normalization: {normalize}")
    logger.info(f"  Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"  Number of vectors: {embeddings.shape[0]:,}")
    
    # Normalize for cosine similarity
    if normalize:
        logger.info("Normalizing embeddings (L2 norm)...")
        faiss.normalize_L2(embeddings)
    
    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings
    logger.info("Adding embeddings to index...")
    index.add(embeddings)
    
    logger.info(f"✓ Index built successfully with {index.ntotal:,} vectors")
    
    return index


def save_index(index: faiss.IndexFlatIP, output_path: Path) -> None:
    """Save FAISS index to disk."""
    logger = logging.getLogger(__name__)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving index to {output_path}")
    faiss.write_index(index, str(output_path))
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"✓ Index saved successfully ({file_size_mb:.1f} MB)")


def validate_index(
    index: faiss.IndexFlatIP,
    embeddings: np.ndarray,
    num_test_queries: int = 10
) -> bool:
    """Validate index by performing test searches."""
    logger = logging.getLogger(__name__)
    
    logger.info("Validating index...")
    
    # Check index size
    if index.ntotal != embeddings.shape[0]:
        logger.error(f"Index size mismatch: {index.ntotal} vs {embeddings.shape[0]}")
        return False
    
    # Test searches with random queries
    test_indices = np.random.choice(embeddings.shape[0], size=num_test_queries, replace=False)
    
    for i, test_idx in enumerate(test_indices):
        query = embeddings[test_idx:test_idx+1].copy()
        faiss.normalize_L2(query)
        
        distances, indices = index.search(query, k=5)
        
        # First result should be the query itself
        if indices[0][0] != test_idx:
            logger.error(f"Test query {i}: Expected index {test_idx}, got {indices[0][0]}")
            return False
        
        # Distance to itself should be ~1.0 (for normalized vectors)
        if not (0.99 <= distances[0][0] <= 1.01):
            logger.error(f"Test query {i}: Unexpected self-distance {distances[0][0]}")
            return False
    
    logger.info(f"✓ Index validation passed ({num_test_queries} test queries)")
    return True


def create_index_metadata(
    index: faiss.IndexFlatIP,
    embeddings_path: Path,
    data_file: Optional[Path],
    output_path: Path
) -> None:
    """Create metadata file for the index."""
    import json
    
    logger = logging.getLogger(__name__)
    
    metadata = {
        'index_type': 'FAISS FLAT (IndexFlatIP)',
        'similarity_metric': 'Inner Product (cosine with normalized vectors)',
        'num_vectors': int(index.ntotal),
        'dimension': int(index.d),
        'embeddings_source': str(embeddings_path),
        'data_source': str(data_file) if data_file else None,
        'created_at': datetime.now().isoformat(),
        'index_file': output_path.name,
        'is_normalized': True
    }
    
    metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Metadata saved to {metadata_path}")


@click.command()
@click.option(
    '--embeddings', '-e',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to embeddings numpy file (.npy)'
)
@click.option(
    '--data-file', '-d',
    type=click.Path(exists=True, path_type=Path),
    help='Path to original node data CSV (for metadata only)'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='data/indices/',
    help='Output directory for index'
)
@click.option(
    '--index-name',
    default='faiss_flat_index',
    help='Name for the index file'
)
@click.option(
    '--no-normalize',
    is_flag=True,
    help='Skip L2 normalization (use raw inner product instead of cosine)'
)
@click.option(
    '--skip-validation',
    is_flag=True,
    help='Skip index validation'
)
@click.option(
    '--force', '-f',
    is_flag=True,
    help='Overwrite existing index'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def main(
    embeddings: Path,
    data_file: Optional[Path],
    output_dir: Path,
    index_name: str,
    no_normalize: bool,
    skip_validation: bool,
    force: bool,
    verbose: bool
):
    """Build FAISS FLAT index for exact nearest neighbor search."""
    
    logger = setup_logging(verbose)
    
    try:
        logger.info("=== FAISS FLAT Index Building ===")
        logger.info(f"Embeddings: {embeddings}")
        logger.info(f"Output directory: {output_dir}")
        
        # Setup output path
        output_path = output_dir / index_name
        if not output_path.suffix:
            output_path = output_path.with_suffix('.index')
        
        # Check for existing index
        if output_path.exists() and not force:
            logger.warning(f"Index already exists: {output_path}")
            if not click.confirm("Overwrite existing index?"):
                logger.info("Aborted")
                return
        
        # Load embeddings
        embeddings_array = load_embeddings(embeddings)
        
        # Build index
        index = build_flat_index(embeddings_array, normalize=not no_normalize)
        
        # Validate
        if not skip_validation:
            if not validate_index(index, embeddings_array):
                logger.error("Index validation failed")
                sys.exit(1)
        
        # Save index
        save_index(index, output_path)
        
        # Create metadata
        create_index_metadata(index, embeddings, data_file, output_path)
        
        logger.info("=== Index Building Complete ===")
        logger.info(f"Index file: {output_path}")
        logger.info(f"Vectors: {index.ntotal:,}")
        logger.info(f"Dimension: {index.d}")
        logger.info(f"\nUsage in retrieval:")
        logger.info(f"  index = faiss.read_index('{output_path}')")
        logger.info(f"  distances, indices = index.search(query_vectors, k=20)")
        
    except Exception as e:
        logger.error(f"Index building failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()