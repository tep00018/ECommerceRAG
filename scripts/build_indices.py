#!/usr/bin/env python3
"""
Index Building Script

This script builds FAISS and BM25 indices for the Neural Retriever-Reranker
RAG pipelines from processed node data.

Usage:
    python scripts/build_indices.py --data-file data/nodes/amazon_stark_nodes_processed.csv
    python scripts/build_indices.py --index-type faiss --model intfloat/e5-large
    python scripts/build_indices.py --index-type bm25 --graph-augmentation
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import click
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from retrieval.faiss_retriever import FAISSRetriever
from retrieval.bm25_retriever import BM25Retriever
from utils.data_loader import load_node_data, prepare_documents_for_indexing
from utils.preprocessing import clean_amazon_fields, get_preprocessor


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'index_building_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def build_faiss_index(
    documents: list,
    node_ids: list,
    output_path: Path,
    model_name: str = "intfloat/e5-large",
    index_type: str = "HNSW",
    index_params: Dict[str, Any] = None
) -> None:
    """
    Build FAISS index from documents.
    
    Args:
        documents: List of document texts
        node_ids: List of corresponding node IDs
        output_path: Path to save the index
        model_name: Sentence transformer model name
        index_type: Type of FAISS index ('FLAT' or 'HNSW')
        index_params: Index-specific parameters
    """
    logger = logging.getLogger(__name__)
    
    if index_params is None:
        index_params = {
            'M': 64,
            'efConstruction': 100,
            'efSearch': 200
        } if index_type == "HNSW" else {}
    
    logger.info(f"Building FAISS {index_type} index with {model_name}")
    logger.info(f"Index parameters: {index_params}")
    
    # Initialize retriever
    retriever = FAISSRetriever(
        model_name=model_name,
        index_type=index_type,
        **index_params
    )
    
    # Build index
    logger.info(f"Processing {len(documents)} documents...")
    retriever.build_index(documents, node_ids)
    
    # Save index
    output_path.mkdir(parents=True, exist_ok=True)
    retriever.save_index(output_path)
    
    # Log index statistics
    stats = retriever.get_index_stats()
    logger.info(f"FAISS index built successfully: {stats}")


def build_bm25_index(
    documents: list,
    node_ids: list,
    output_path: Path,
    graph_edges: Dict[int, Dict[str, list]] = None,
    bm25_params: Dict[str, Any] = None,
    graph_augmentation: bool = False
) -> None:
    """
    Build BM25 index from documents.
    
    Args:
        documents: List of document texts
        node_ids: List of corresponding node IDs
        output_path: Path to save the index
        graph_edges: Graph structure for augmentation
        bm25_params: BM25-specific parameters
        graph_augmentation: Whether to enable graph augmentation
    """
    logger = logging.getLogger(__name__)
    
    if bm25_params is None:
        bm25_params = {
            'k1': 1.5,
            'b': 0.75,
            'tokenizer': 'simple',
            'remove_stopwords': True,
            'use_stemming': False
        }
    
    logger.info(f"Building BM25 index with graph augmentation: {graph_augmentation}")
    logger.info(f"BM25 parameters: {bm25_params}")
    
    # Initialize retriever
    retriever = BM25Retriever(
        graph_augmentation=graph_augmentation,
        max_expansion=50,
        **bm25_params
    )
    
    # Build index
    logger.info(f"Processing {len(documents)} documents...")
    retriever.build_index(documents, node_ids, graph_edges)
    
    # Save index
    output_path.mkdir(parents=True, exist_ok=True)
    retriever.save_index(output_path)
    
    # Log index statistics
    stats = retriever.get_index_stats()
    logger.info(f"BM25 index built successfully: {stats}")


def extract_graph_edges(node_df: pd.DataFrame) -> Dict[int, Dict[str, list]]:
    """
    Extract graph edges from node data for BM25 augmentation.
    
    Args:
        node_df: DataFrame with node data
        
    Returns:
        Dictionary mapping node_id to edge information
    """
    logger = logging.getLogger(__name__)
    logger.info("Extracting graph edges...")
    
    graph_edges = {}
    
    for _, row in tqdm(node_df.iterrows(), total=len(node_df), desc="Extracting edges"):
        node_id = int(row['node_id'])
        edges = {}
        
        # Extract also_buy relationships
        if 'also_buy' in row and pd.notna(row['also_buy']):
            try:
                import ast
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
                import ast
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
    
    logger.info(f"Extracted edges for {len(graph_edges)} nodes")
    return graph_edges


def validate_index(index_path: Path, index_type: str, sample_queries: list = None) -> bool:
    """
    Validate that the built index works correctly.
    
    Args:
        index_path: Path to the index
        index_type: Type of index ('faiss' or 'bm25')
        sample_queries: List of sample queries for testing
        
    Returns:
        True if validation passes, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Validating {index_type} index at {index_path}")
    
    try:
        if index_type == "faiss":
            retriever = FAISSRetriever()
            retriever.load_index(index_path)
        elif index_type == "bm25":
            retriever = BM25Retriever()
            retriever.load_index(index_path)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Test with sample queries
        if sample_queries is None:
            sample_queries = [
                "high quality wireless headphones",
                "outdoor camping gear",
                "kitchen appliances"
            ]
        
        for query in sample_queries:
            try:
                results = retriever.search(query, k=10)
                if not results:
                    logger.warning(f"No results for query: {query}")
                else:
                    logger.info(f"Query '{query}' returned {len(results)} results")
            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
                return False
        
        logger.info("Index validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Index validation failed: {e}")
        return False


@click.command()
@click.option(
    '--data-file', '-d',
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to processed node data CSV file'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='data/indices/',
    help='Output directory for indices'
)
@click.option(
    '--index-type', '-t',
    type=click.Choice(['faiss', 'bm25', 'both']),
    default='both',
    help='Type of index to build'
)
@click.option(
    '--model-name', '-m',
    default='intfloat/e5-large',
    help='Sentence transformer model for FAISS (ignored for BM25)'
)
@click.option(
    '--faiss-type',
    type=click.Choice(['FLAT', 'HNSW']),
    default='HNSW',
    help='FAISS index type'
)
@click.option(
    '--graph-augmentation/--no-graph-augmentation',
    default=True,
    help='Enable graph augmentation for BM25'
)
@click.option(
    '--sample-size', '-s',
    type=int,
    help='Use only a sample of the data for testing'
)
@click.option(
    '--validate/--no-validate',
    default=True,
    help='Validate indices after building'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def main(
    data_file: Path,
    output_dir: Path,
    index_type: str,
    model_name: str,
    faiss_type: str,
    graph_augmentation: bool,
    sample_size: Optional[int],
    validate: bool,
    verbose: bool
):
    """Build search indices for RAG pipelines."""
    
    logger = setup_logging(verbose)
    
    try:
        logger.info("=== Neural Retriever-Reranker Index Building ===")
        logger.info(f"Data file: {data_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Index type: {index_type}")
        
        # Load data
        logger.info("Loading node data...")
        node_df = load_node_data(data_file)
        
        # Apply sample if requested
        if sample_size:
            if sample_size < len(node_df):
                node_df = node_df.sample(n=sample_size, random_state=42)
                logger.info(f"Using sample of {len(node_df)} nodes")
        
        # Clean data
        logger.info("Cleaning Amazon-specific fields...")
        node_df = clean_amazon_fields(node_df)
        
        # Prepare documents
        logger.info("Preparing documents for indexing...")
        documents, node_ids = prepare_documents_for_indexing(node_df)
        
        logger.info(f"Prepared {len(documents)} documents for indexing")
        
        # Build indices
        if index_type in ['faiss', 'both']:
            logger.info("Building FAISS index...")
            
            faiss_output = output_dir / f"faiss_{model_name.replace('/', '_')}_{faiss_type.lower()}"
            
            # FAISS index parameters
            faiss_params = {
                'M': 64,
                'efConstruction': 100,
                'efSearch': 200
            } if faiss_type == 'HNSW' else {}
            
            build_faiss_index(
                documents=documents,
                node_ids=node_ids,
                output_path=faiss_output,
                model_name=model_name,
                index_type=faiss_type,
                index_params=faiss_params
            )
            
            # Validate FAISS index
            if validate:
                validate_index(faiss_output, 'faiss')
        
        if index_type in ['bm25', 'both']:
            logger.info("Building BM25 index...")
            
            # Extract graph edges if augmentation is enabled
            graph_edges = None
            if graph_augmentation:
                graph_edges = extract_graph_edges(node_df)
            
            bm25_output = output_dir / f"bm25{'_augmented' if graph_augmentation else ''}"
            
            build_bm25_index(
                documents=documents,
                node_ids=node_ids,
                output_path=bm25_output,
                graph_edges=graph_edges,
                graph_augmentation=graph_augmentation
            )
            
            # Validate BM25 index
            if validate:
                validate_index(bm25_output, 'bm25')
        
        logger.info("=== Index Building Complete ===")
        
        # Print summary
        logger.info("\n=== Summary ===")
        logger.info(f"📊 Processed {len(documents):,} documents")
        
        if index_type in ['faiss', 'both']:
            faiss_path = output_dir / f"faiss_{model_name.replace('/', '_')}_{faiss_type.lower()}"
            logger.info(f"✅ FAISS index: {faiss_path}")
        
        if index_type in ['bm25', 'both']:
            bm25_path = output_dir / f"bm25{'_augmented' if graph_augmentation else ''}"
            logger.info(f"✅ BM25 index: {bm25_path}")
            if graph_augmentation:
                edge_count = len(graph_edges) if graph_edges else 0
                logger.info(f"   Graph edges: {edge_count:,} nodes with relationships")
        
        logger.info("\nNext steps:")
        logger.info("1. Update pipeline configs to point to the new indices")
        logger.info("2. Run evaluation: python scripts/run_evaluation.py")
        
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        raise


if __name__ == "__main__":
    main()