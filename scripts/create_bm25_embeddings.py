# Create the script
cat > scripts/create_bm25_embeddings.py << 'EOF'
#!/usr/bin/env python3
"""
Create BM25 Sparse Embeddings (Tokenized Documents)

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
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def tokenize_text(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return word_tokenize(text.lower())


def preprocess_documents(
    texts: List[str],
    remove_stops: bool = True
) -> Tuple[List[List[str]], dict]:
    logger = logging.getLogger(__name__)
    stop_words = set(stopwords.words('english')) if remove_stops else set()
    
    tokenized_docs = []
    total_tokens = 0
    
    logger.info(f"Preprocessing {len(texts)} documents...")
    
    for text in tqdm(texts, desc="Tokenizing"):
        tokens = tokenize_text(text)
        if remove_stops:
            tokens = [w for w in tokens if w not in stop_words]
        tokenized_docs.append(tokens)
        total_tokens += len(tokens)
    
    stats = {
        'num_documents': len(tokenized_docs),
        'total_tokens': total_tokens,
        'avg_tokens_per_doc': total_tokens / len(tokenized_docs) if tokenized_docs else 0
    }
    
    return tokenized_docs, stats


@click.command()
@click.option('--data-file', '-d', type=click.Path(exists=True, path_type=Path), required=True)
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), default='data/embeddings/')
@click.option('--text-column', '-t', default='combined_text')
@click.option('--verbose', '-v', is_flag=True)
def main(data_file: Path, output_dir: Path, text_column: str, verbose: bool):
    logger = setup_logging(verbose)
    
    logger.info("=== BM25 Tokenization ===")
    
    # Load data
    df = pd.read_csv(data_file)
    texts = df[text_column].fillna("").tolist()
    node_ids = df['node_id'].tolist()
    
    # Process
    tokenized_docs, stats = preprocess_documents(texts)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'bm25_tokenized_documents.pkl'
    
    data = {
        'tokenized_documents': tokenized_docs,
        'node_ids': node_ids,
        'metadata': {'created_at': datetime.now().isoformat(), 'statistics': stats}
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    logger.info(f"Saved to {output_path}")
    logger.info(f"Documents: {stats['num_documents']:,}")
    logger.info(f"Avg tokens/doc: {stats['avg_tokens_per_doc']:.1f}")


if __name__ == "__main__":
    main()
EOF

# Make it executable
chmod +x scripts/create_bm25_embeddings.py

# Add it to git
git add scripts/create_bm25_embeddings.py