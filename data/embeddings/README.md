# Embeddings Directory

This directory stores vector embeddings and tokenized documents for retrieval.

## Files

### Dense Embeddings (FAISS)
- **File**: `e5_large_embeddings.npy`
- **Size**: ~4GB
- **Format**: NumPy array (N x 1024, float32)
- **Model**: intfloat/e5-large
- **Creation**: `python scripts/create_embeddings.py --data-file data/nodes/amazon_stark_nodes_processed.csv`

### Sparse Embeddings (BM25)
- **File**: `bm25_tokenized_documents.pkl`
- **Size**: ~3GB
- **Format**: Pickled list of token lists
- **Preprocessing**: Tokenized, stopwords removed
- **Creation**: `python scripts/create_bm25_embeddings.py --data-file data/nodes/amazon_stark_nodes_processed.csv`

## Creating Embeddings

## Quick Start
```bash
# For FAISS pipelines
python scripts/create_embeddings.py --data-file data/nodes/amazon_stark_nodes_processed.csv

# For BM25 pipeline
python scripts/create_bm25_embeddings.py --data-file data/nodes/amazon_stark_nodes_processed.csv
