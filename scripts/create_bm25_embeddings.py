#!/usr/bin/env python3
"""
Create BM25 Sparse Embeddings (Tokenized Documents)

This script tokenizes documents for BM25 retrieval using the same preprocessing
as the original ipynb implementation.

Usage:
    python scripts/create_bm25_embeddings.py --data-file data/nodes/amazon_stark_nodes_processed.csv
"""
import sys
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import click
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text with lowercasing (matches ipynb implementation).
    
    Args:
        text: Input text string
        
    Returns:
        List of tokens
    """
    if not isinstance(text, str):
        return []
    # Match ipynb: word_tokenize(text.lower())
    return word_tokenize(text.lower())


def preprocess_documents(
    texts: List[str],
    remove_stops: bool = True
) -> Tuple[List[List[str]], dict]:
    """
    Preprocess documents for BM25 indexing.
    
    This matches the ipynb preprocessing:
    1. Tokenize with lowercasing
    2. Remove stopwords
    
    Args:
        texts: List of document texts
        remove_stops: Whether to remove stopwords
        
    Returns:
        Tuple of (tokenized_docs, statistics)
    """
    logger = logging.getLogger(__name__)
    stop_words = set(stopwords.words('english')) if remove_stops else set()
    
    tokenized_docs = []
    total_tokens = 0
    empty_docs = 0
    
    logger.info(f"Preprocessing {len(texts)} documents...")
    logger.info(f"Remove stopwords: {remove_stops}")
    
    for text in tqdm(texts, desc="Tokenizing"):
        # Tokenize
        tokens = tokenize_text(text)
        
        # Remove stopwords if enabled
        if remove_stops:
            tokens = [w for w in tokens if w not in stop_words]
        
        tokenized_docs.append(tokens)
        total_tokens += len(tokens)
        
        if not tokens:
            empty_docs += 1
    
    stats = {
        'num_documents': len(tokenized_docs),
        'total_tokens': total_tokens,
        'avg_tokens_per_doc': total_tokens / len(tokenized_docs) if tokenized_docs else 0,
        'empty_documents': empty_docs,
        'stopwords_removed': remove_stops
    }
    
    return tokenized_docs, stats


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
    help='Output directory for tokenized documents'
)
@click.option(
    '--text-column', '-t',
    default='combined_text',
    help='Name of the text column to tokenize'
)
@click.option(
    '--remove-stopwords/--keep-stopwords',
    default=True,
    help='Whether to remove stopwords (default: remove)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def main(
    data_file: Path,
    output_dir: Path,
    text_column: str,
    remove_stopwords: bool,
    verbose: bool
):
    """Create BM25 tokenized documents from node data."""
    
    logger = setup_logging(verbose)
    
    logger.info("=== BM25 Tokenization ===")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Text column: {text_column}")
    
    # Load data
    logger.info("Loading node data...")
    df = pd.read_csv(data_file)
    
    # Extract texts and node IDs
    texts = df[text_column].fillna("").tolist()
    node_ids = df['node_id'].tolist()
    
    logger.info(f"Loaded {len(texts):,} documents")
    
    # Process documents
    tokenized_docs, stats = preprocess_documents(texts, remove_stops=remove_stopwords)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'bm25_tokenized_documents.pkl'
    
    # Save tokenized documents with metadata
    data = {
        'tokenized_documents': tokenized_docs,
        'node_ids': node_ids,
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'source_file': str(data_file),
            'text_column': text_column,
            'statistics': stats
        }
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"\n✅ Tokenized documents saved to: {output_path}")
    logger.info("\n=== Statistics ===")
    logger.info(f"📄 Total documents: {stats['num_documents']:,}")
    logger.info(f"🔤 Total tokens: {stats['total_tokens']:,}")
    logger.info(f"📊 Avg tokens/doc: {stats['avg_tokens_per_doc']:.1f}")
    logger.info(f"⚠️  Empty documents: {stats['empty_documents']:,}")
    logger.info(f"🚫 Stopwords removed: {stats['stopwords_removed']}")
    
    logger.info("\nNext step: Run build_indices.py to create BM25 index")


if __name__ == "__main__":
    main()