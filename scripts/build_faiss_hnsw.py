#!/usr/bin/env python3
"""
Build FAISS HNSW Index for Approximate Nearest Neighbor Search

This script creates a FAISS HNSW (Hierarchical Navigable Small World) index
from pre-computed E5-Large embeddings. The HNSW index provides fast approximate
nearest neighbor search with tunable accuracy/speed tradeoffs.

Usage:
    python scripts/build_faiss_hnsw.py --embeddings data/embeddings/intfloat_e5_large_embeddings.npy
    python scripts/build_faiss_hnsw.py --embeddings embeddings.npy --M 64 --ef-construction 100 --ef-search 200
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

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
                f'faiss_hnsw_build_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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


def build_hnsw_index(
    embeddings: np.ndarray,
    M: int = 64,
    efConstruction: int = 100,
    efSearch: int = 200,
    normalize: bool = True
) -> faiss.IndexHNSWFlat:
    """
    Build FAISS HNSW index with specified parameters.
    
    Args:
        embeddings: Numpy array of embeddings (N x D)
        M: Number of neighbors per layer (higher = better recall, slower build)
        efConstruction: Size of dynamic candidate list during construction
        efSearch: Size of dynamic candidate list during search
        normalize: Whether to L2-normalize embeddings for cosine similarity
        
    Returns:
        FAISS HNSW index
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Building FAISS HNSW index...")
    logger.info(f"  Index type: IndexHNSWFlat")
    logger.info(f"  M: {M} (neighbors per layer)")
    logger.info(f"  efConstruction: {efConstruction}")
    logger.info(f"  efSearch: {efSearch}")
    logger.info(f"  Normalization: {normalize}")
    logger.info(f"  Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"  Number of vectors: {embeddings.shape[0]:,}")
    
    # Normalize for cosine similarity
    if normalize:
        logger.info("Normalizing embeddings (L2 norm)...")
        faiss.normalize_L2(embeddings)
    
    # Create index
    dimension = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dimension, M)
    
    # Set construction parameter
    index.hnsw.efConstruction = efConstruction
    
    # Add embeddings
    logger.info("Adding embeddings to index (this may take several minutes)...")
    index.add(embeddings)
    
    # Set search parameter
    index.hnsw.efSearch = efSearch
    
    logger.info(f"✓ Index built successfully with {index.ntotal:,} vectors")
    logger.info(f"  Graph has {index.hnsw.max_level + 1} levels")
    
    return index


def save_index(index: faiss.IndexHNSWFlat, output_path: Path) -> None:
    """Save FAISS index to disk."""
    logger = logging.getLogger(__name__)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving index to {output_path}")
    faiss.write_index(index, str(output_path))
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"✓ Index saved successfully ({file_size_mb:.1f} MB)")


def validate_index(
    index: faiss.IndexHNSWFlat,
    embeddings: np.ndarray,
    num_test_queries: int = 100,
    k: int = 20
) -> Tuple[bool, float]:
    """
    Validate index by comparing with brute force search.
    
    Returns:
        Tuple of (is_valid, recall@k)
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Validating index with {num_test_queries} test queries...")
    
    # Check index size
    if index.ntotal != embeddings.shape[0]:
        logger.error(f"Index size mismatch: {index.ntotal} vs {embeddings.shape[0]}")
        return False, 0.0
    
    # Build a small FLAT index for ground truth
    logger.info("Building ground truth index (FLAT)...")
    flat_index = faiss.IndexFlatIP(embeddings.shape[1])
    flat_embeddings = embeddings.copy()
    faiss.normalize_L2(flat_embeddings)
    flat_index.add(flat_embeddings)
    
    # Test queries
    test_indices = np.random.choice(embeddings.shape[0], size=num_test_queries, replace=False)
    test_queries = embeddings[test_indices].copy()
    faiss.normalize_L2(test_queries)
    
    # Search both indices
    _, hnsw_results = index.search(test_queries, k)
    _, flat_results = flat_index.search(test_queries, k)
    
    # Calculate recall@k
    recalls = []
    for hnsw_result, flat_result in zip(hnsw_results, flat_results):
        hnsw_set = set(hnsw_result)
        flat_set = set(flat_result)
        recall = len(hnsw_set & flat_set) / k
        recalls.append(recall)
    
    mean_recall = np.mean(recalls)
    
    logger.info(f"✓ Recall@{k}: {mean_recall:.4f}")
    
    if mean_recall < 0.95:
        logger.warning(f"Low recall detected: {mean_recall:.4f}")
        logger.warning("Consider increasing efSearch or efConstruction parameters")
    
    return True, mean_recall


def benchmark_index(
    index: faiss.IndexHNSWFlat,
    embeddings: np.ndarray,
    num_queries: int = 100,
    k: int = 20
) -> dict:
    """Benchmark index search performance."""
    import time
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Benchmarking index with {num_queries} queries...")
    
    # Prepare queries
    test_indices = np.random.choice(embeddings.shape[0], size=num_queries, replace=False)
    test_queries = embeddings[test_indices].copy()
    faiss.normalize_L2(test_queries)
    
    # Warm-up
    index.search(test_queries[:10], k)
    
    # Benchmark
    start_time = time.time()
    distances, indices = index.search(test_queries, k)
    elapsed_time = time.time() - start_time
    
    qps = num_queries / elapsed_time
    avg_latency_ms = (elapsed_time / num_queries) * 1000
    
    logger.info(f"✓ Benchmark results:")
    logger.info(f"  Queries per second: {qps:.1f}")
    logger.info(f"  Average latency: {avg_latency_ms:.2f} ms")
    
    return {
        'qps': qps,
        'avg_latency_ms': avg_latency_ms,
        'num_queries': num_queries,
        'k': k
    }


def create_index_metadata(
    index: faiss.IndexHNSWFlat,
    embeddings_path: Path,
    data_file: Optional[Path],
    output_path: Path,
    M: int,
    efConstruction: int,
    efSearch: int,
    recall: Optional[float],
    benchmark_results: Optional[dict]
) -> None:
    """Create metadata file for the index."""
    import json
    
    logger = logging.getLogger(__name__)
    
    metadata = {
        'index_type': 'FAISS HNSW (IndexHNSWFlat)',
        'similarity_metric': 'Inner Product (cosine with normalized vectors)',
        'num_vectors': int(index.ntotal),
        'dimension': int(index.d),
        'parameters': {
            'M': M,
            'efConstruction': efConstruction,
            'efSearch': efSearch
        },
        'graph_levels': int(index.hnsw.max_level + 1),
        'embeddings_source': str(embeddings_path),
        'data_source': str(data_file) if data_file else None,
        'created_at': datetime.now().isoformat(),
        'index_file': output_path.name,
        'is_normalized': True
    }
    
    if recall is not None:
        metadata['recall_at_20'] = float(recall)
    
    if benchmark_results:
        metadata['benchmark'] = benchmark_results
    
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
    default='faiss_hnsw_index',
    help='Name for the index file'
)
@click.option(
    '--M',
    type=int,
    default=64,
    help='Number of neighbors per layer (16-48 typical, 64 for high recall)'
)
@click.option(
    '--ef-construction',
    type=int,
    default=100,
    help='Size of dynamic list during construction (higher = better quality, slower)'
)
@click.option(
    '--ef-search',
    type=int,
    default=200,
    help='Size of dynamic list during search (higher = better recall, slower)'
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
    '--skip-benchmark',
    is_flag=True,
    help='Skip performance benchmarking'
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
    m: int,
    ef_construction: int,
    ef_search: int,
    no_normalize: bool,
    skip_validation: bool,
    skip_benchmark: bool,
    force: bool,
    verbose: bool
):
    """Build FAISS HNSW index for approximate nearest neighbor search."""
    
    logger = setup_logging(verbose)
    
    try:
        logger.info("=== FAISS HNSW Index Building ===")
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
        index = build_hnsw_index(
            embeddings_array,
            M=m,
            efConstruction=ef_construction,
            efSearch=ef_search,
            normalize=not no_normalize
        )
        
        # Validate
        recall = None
        if not skip_validation:
            is_valid, recall = validate_index(index, embeddings_array)
            if not is_valid:
                logger.error("Index validation failed")
                sys.exit(1)
        
        # Benchmark
        benchmark_results = None
        if not skip_benchmark:
            benchmark_results = benchmark_index(index, embeddings_array)
        
        # Save index
        save_index(index, output_path)
        
        # Create metadata
        create_index_metadata(
            index, embeddings, data_file, output_path,
            m, ef_construction, ef_search, recall, benchmark_results
        )
        
        logger.info("=== Index Building Complete ===")
        logger.info(f"Index file: {output_path}")
        logger.info(f"Vectors: {index.ntotal:,}")
        logger.info(f"Dimension: {index.d}")
        logger.info(f"Parameters: M={m}, efConstruction={ef_construction}, efSearch={ef_search}")
        if recall:
            logger.info(f"Recall@20: {recall:.4f}")
        logger.info(f"\nUsage in retrieval:")
        logger.info(f"  index = faiss.read_index('{output_path}')")
        logger.info(f"  # Optionally adjust efSearch: index.hnsw.efSearch = 300")
        logger.info(f"  distances, indices = index.search(query_vectors, k=20)")
        
    except Exception as e:
        logger.error(f"Index building failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()