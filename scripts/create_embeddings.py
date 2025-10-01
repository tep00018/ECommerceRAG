#!/usr/bin/env python3
"""
Create E5-Large Embeddings for Amazon STaRK Nodes

This script generates dense vector embeddings for product nodes using the
E5-Large model. Embeddings are saved with checkpointing support for
resumable processing of large datasets.

Usage:
    python scripts/create_embeddings.py --data-file data/nodes/amazon_stark_nodes_processed.csv
    python scripts/create_embeddings.py --data-file data/nodes/nodes.csv --output-dir data/embeddings/
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

import click
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'embedding_creation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ]
    )
    return logging.getLogger(__name__)


def check_gpu_availability() -> Tuple[bool, str]:
    """Check if GPU is available and return device info."""
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        return True, device_name
    return False, "CPU"


def load_node_texts(data_file: Path, text_column: str = 'combined_text') -> Tuple[list, pd.DataFrame]:
    """
    Load node texts from CSV file.
    
    Args:
        data_file: Path to the node data CSV
        text_column: Name of column containing combined text
        
    Returns:
        Tuple of (texts list, full DataFrame)
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading node data from {data_file}")
    df = pd.read_csv(data_file)
    
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in data file. Available columns: {df.columns.tolist()}")
    
    texts = df[text_column].fillna("").tolist()
    
    logger.info(f"Loaded {len(texts):,} texts")
    logger.info(f"Average text length: {np.mean([len(t) for t in texts]):.1f} characters")
    
    return texts, df


def create_embeddings_with_checkpointing(
    texts: list,
    model_name: str,
    checkpoint_dir: Path,
    final_output_path: Path,
    chunk_size: int = 50000,
    batch_size: int = 128,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Create embeddings with checkpointing for large datasets.
    
    Args:
        texts: List of text strings to embed
        model_name: Sentence transformer model name
        checkpoint_dir: Directory for checkpoint files
        final_output_path: Path for final embeddings file
        chunk_size: Number of texts per checkpoint
        batch_size: Batch size for encoding
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        NumPy array of embeddings
    """
    logger = logging.getLogger(__name__)
    
    # Check if final embeddings already exist
    if final_output_path.exists():
        logger.info(f"Loading existing embeddings from {final_output_path}")
        return np.load(final_output_path)
    
    # Create checkpoint directory
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    logger.info(f"Model loaded successfully on {device}")
    
    all_embeddings = []
    num_chunks = (len(texts) + chunk_size - 1) // chunk_size
    
    logger.info(f"Processing {len(texts):,} texts in {num_chunks} chunks of {chunk_size:,}")
    
    for chunk_idx in range(num_chunks):
        checkpoint_path = checkpoint_dir / f'embeddings_chunk_{chunk_idx}.npy'
        
        # Check if checkpoint exists
        if checkpoint_path.exists():
            logger.info(f"Loading cached chunk {chunk_idx + 1}/{num_chunks} from checkpoint")
            chunk_embeddings = np.load(checkpoint_path)
        else:
            # Process this chunk
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(texts))
            chunk_texts = texts[start_idx:end_idx]
            
            logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}: texts {start_idx:,} to {end_idx:,}")
            
            chunk_embeddings = model.encode(
                chunk_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                device=device,
                normalize_embeddings=False  # We'll normalize later if needed
            )
            
            # Save checkpoint
            np.save(checkpoint_path, chunk_embeddings)
            logger.info(f"Saved checkpoint for chunk {chunk_idx + 1}")
        
        all_embeddings.append(chunk_embeddings)
    
    # Combine all chunks
    logger.info("Combining all chunks...")
    embeddings = np.vstack(all_embeddings)
    
    # Convert to float32 for FAISS compatibility
    embeddings = embeddings.astype('float32')
    
    # Save final embeddings
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(final_output_path, embeddings)
    logger.info(f"Saved final embeddings to {final_output_path}")
    
    return embeddings


def validate_embeddings(embeddings: np.ndarray, num_texts: int) -> bool:
    """Validate that embeddings were created correctly."""
    logger = logging.getLogger(__name__)
    
    logger.info("Validating embeddings...")
    
    if embeddings.shape[0] != num_texts:
        logger.error(f"Embedding count mismatch: {embeddings.shape[0]} vs {num_texts} texts")
        return False
    
    if embeddings.dtype != np.float32:
        logger.error(f"Incorrect dtype: {embeddings.dtype} (expected float32)")
        return False
    
    if np.any(np.isnan(embeddings)):
        logger.error("Embeddings contain NaN values")
        return False
    
    if np.any(np.isinf(embeddings)):
        logger.error("Embeddings contain infinite values")
        return False
    
    logger.info(f"✓ Embeddings validated successfully")
    logger.info(f"  Shape: {embeddings.shape}")
    logger.info(f"  Dtype: {embeddings.dtype}")
    logger.info(f"  Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    
    return True


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
    default='data/embeddings/',
    help='Output directory for embeddings'
)
@click.option(
    '--model-name', '-m',
    default='intfloat/e5-large',
    help='Sentence transformer model name'
)
@click.option(
    '--text-column', '-t',
    default='combined_text',
    help='Name of column containing text to embed'
)
@click.option(
    '--chunk-size', '-c',
    type=int,
    default=50000,
    help='Number of texts per checkpoint'
)
@click.option(
    '--batch-size', '-b',
    type=int,
    default=128,
    help='Batch size for encoding'
)
@click.option(
    '--force', '-f',
    is_flag=True,
    help='Overwrite existing embeddings'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def main(
    data_file: Path,
    output_dir: Path,
    model_name: str,
    text_column: str,
    chunk_size: int,
    batch_size: int,
    force: bool,
    verbose: bool
):
    """Create E5-Large embeddings for Amazon STaRK nodes."""
    
    logger = setup_logging(verbose)
    
    try:
        logger.info("=== E5-Large Embedding Creation ===")
        logger.info(f"Data file: {data_file}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Model: {model_name}")
        
        # Check GPU
        cuda_available, device_info = check_gpu_availability()
        logger.info(f"Device: {device_info}")
        device = 'cuda' if cuda_available else 'cpu'
        
        # Load texts
        texts, df = load_node_texts(data_file, text_column)
        
        # Setup paths
        checkpoint_dir = output_dir / 'checkpoints'
        model_slug = model_name.replace('/', '_').replace('-', '_')
        final_output = output_dir / f'{model_slug}_embeddings.npy'
        
        # Check for existing embeddings
        if final_output.exists() and not force:
            logger.warning(f"Embeddings already exist: {final_output}")
            if not click.confirm("Overwrite existing embeddings?"):
                logger.info("Loading existing embeddings...")
                embeddings = np.load(final_output)
                logger.info(f"Loaded embeddings shape: {embeddings.shape}")
                return
        
        # Create embeddings
        embeddings = create_embeddings_with_checkpointing(
            texts=texts,
            model_name=model_name,
            checkpoint_dir=checkpoint_dir,
            final_output_path=final_output,
            chunk_size=chunk_size,
            batch_size=batch_size,
            device=device
        )
        
        # Validate
        if not validate_embeddings(embeddings, len(texts)):
            logger.error("Embedding validation failed")
            sys.exit(1)
        
        logger.info("=== Embedding Creation Complete ===")
        logger.info(f"Output file: {final_output}")
        logger.info(f"File size: {final_output.stat().st_size / 1024 / 1024:.1f} MB")
        logger.info(f"\nNext steps:")
        logger.info(f"1. Build FAISS FLAT index: python scripts/build_faiss_flat.py --embeddings {final_output}")
        logger.info(f"2. Build FAISS HNSW index: python scripts/build_faiss_hnsw.py --embeddings {final_output}")
        
    except Exception as e:
        logger.error(f"Embedding creation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()