# Neural Retriever-Reranker RAG Pipelines

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Official implementation of "Neural Retriever–Reranker Pipelines for Retrieval Augmented Generation over Knowledge Graphs in e-Commerce Applications"**

This repository contains three state-of-the-art RAG pipeline implementations evaluated on the Amazon STaRK Semi-structured Knowledge Base, achieving **20.4% improvement in Hit@1** and **14.5% improvement in Mean Reciprocal Rank (MRR)** over published benchmarks.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-retriever-reranker-rag.git
cd neural-retriever-reranker-rag

# Install dependencies
pip install -r requirements.txt

# Download and prepare data (see Data Setup below)
python scripts/download_data.py

# Run evaluation on FRWSR pipeline
python scripts/run_evaluation.py --config configs/frwsr_config.yaml --dataset validation
```

## 📊 Pipeline Variants

This repository implements three high-performance RAG pipelines based on actual research implementations:

### 🔥 FRWSR (FAISS + Webis Set-Encoder)
**Best Performance**: Hit@1: 54.75% | MRR: 0.6403

- **Retrieval**: E5-Large embeddings with FAISS-HNSW indexing
- **Reranking**: Webis Set-Encoder/Large with permutation-invariant attention
- **Use Case**: Maximum accuracy applications where computational cost is secondary

### ⚡ FRMR (FAISS + MS MARCO)
**Best Speed/Accuracy Tradeoff**: Hit@1: 51.25% | MRR: 0.6128

- **Retrieval**: E5-Large embeddings with FAISS-HNSW indexing (M=64, efConstruction=100, efSearch=200)
- **Reranking**: MS MARCO MiniLM-L-6-v2 cross-encoder  
- **Speed**: 1.89 it/s (189x faster than FRWSR)
- **Use Case**: Production systems requiring fast response times

### 🔍 BARMR (BM25 + Graph Augmentation)
**Graph-Enhanced Sparse**: Hit@1: 49.67% | MRR: 0.6037

- **Retrieval**: BM25 sparse retrieval (k1=1.017, b=0.886, threshold=21) with 1-hop graph expansion
- **Reranking**: MS MARCO MiniLM-L-6-v2 cross-encoder
- **Graph**: Leverages "also-bought" and "also-viewed" relationships
- **Use Case**: Systems prioritizing interpretability and exact keyword matching

## 🏗️ Repository Structure

```
neural-retriever-reranker-rag/
├── src/                          # Source code
│   ├── pipeline/                 # Pipeline implementations
│   ├── retrieval/               # Retrieval components
│   ├── reranking/               # Reranking components
│   ├── evaluation/              # Metrics and evaluation
│   └── utils/                   # Utilities and data loading
├── configs/                     # Configuration files
├── scripts/                     # Execution scripts
├── data/                        # Data directory (see Data Setup)
├── examples/                    # Usage examples
└── docs/                        # Documentation
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM for full dataset processing

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/neural-retriever-reranker-rag.git
cd neural-retriever-reranker-rag
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt

# For GPU support (optional but recommended)
pip install faiss-gpu torch[cuda]
```

## 📁 Data Setup

### Required Data Files

Download and place the following files in the `data/` directory:

1. **Node Data** (`data/nodes/amazon_stark_nodes_processed.csv`)
   - Processed Amazon STaRK product nodes with combined text fields
   - Contains: `node_id`, `combined_text`, and metadata columns

2. **Query Datasets** (`data/queries/`)
   - `validation_queries.csv` - 910 queries for validation
   - `evaluation_queries_filtered.csv` - 6,380 queries (≤20 ground truth answers)
   - `evaluation_queries_full.csv` - 9,100 queries (complete dataset)

3. **FAISS Indices** (`data/indices/`)
   - Pre-built FAISS indices for faster startup (optional)
   - Can be generated automatically if not provided

### Data Download Script

```bash
# Automated data download and preprocessing
python scripts/download_data.py --output-dir data/

# Or download manually from:
# https://huggingface.co/datasets/snap-stanford/stark
```

### File Structure After Setup
```
data/
├── nodes/
│   └── amazon_stark_nodes_processed.csv
├── queries/
│   ├── validation_queries.csv
│   ├── evaluation_queries_filtered.csv
│   └── evaluation_queries_full.csv
├── indices/
│   └── faiss_e5_large_hnsw/
│       ├── faiss.index
│       └── metadata.pkl
└── results/
    └── (evaluation outputs will be saved here)
```

## 🚀 Usage

### Basic Evaluation

```bash
# Evaluate FRWSR pipeline on validation set
python scripts/run_evaluation.py \
    --config configs/frwsr_config.yaml \
    --dataset validation

# Evaluate FRMR pipeline on filtered evaluation set
python scripts/run_evaluation.py \
    --config configs/frmr_config.yaml \
    --dataset evaluation_filtered

# Evaluate BARMR pipeline on full evaluation set  
python scripts/run_evaluation.py \
    --config configs/barmr_config.yaml \
    --dataset evaluation_full
```

### Programmatic Usage

```python
from src.pipeline.frwsr_pipeline import FRWSRPipeline
import yaml

# Load configuration
with open('configs/frwsr_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
pipeline = FRWSRPipeline(config)
pipeline.load_data('data/nodes/amazon_stark_nodes_processed.csv')

# Process single query
query = "What are some high-quality, pocket-sized map sets?"
results = pipeline.process_query(query, retrieve_k=100, rerank_k=20)
print(f"Top results: {results[:5]}")
```

### Building Custom Indices

```python
from src.retrieval.faiss_retriever import FAISSRetriever
from src.utils.data_loader import load_node_data, prepare_documents_for_indexing

# Load node data
node_df = load_node_data('data/nodes/amazon_stark_nodes_processed.csv')
documents, node_ids = prepare_documents_for_indexing(node_df)

# Build FAISS index
retriever = FAISSRetriever(
    model_name="intfloat/e5-large",
    index_type="HNSW",
    M=64,
    efConstruction=100,
    efSearch=200
)

retriever.build_index(documents, node_ids)
retriever.save_index('data/indices/custom_faiss_index/')
```

## 🌟 Branch Structure

Each pipeline implementation is available on dedicated branches:

- **`main`**: Core framework and documentation
- **`frwsr-pipeline`**: FAISS + Webis Set-Encoder implementation
- **`frmr-pipeline`**: FAISS + MS MARCO implementation
- **`barmr-pipeline`**: BM25 + MS MARCO implementation

```bash
# Switch to specific pipeline branch
git checkout frwsr-pipeline  # For FRWSR implementation
git checkout frmr-pipeline   # For FRMR implementation  
git checkout barmr-pipeline  # For BARMR implementation
```

## 📈 Performance Benchmarks

Results on Amazon STaRK evaluation dataset (6,380 queries):

| Pipeline | Hit@1 | Hit@5 | Hit@20 | Recall@20 | MRR |
|----------|-------|-------|--------|-----------|-----|
| **FRWSR** | **54.75%** | **75.25%** | **84.21%** | **61.30%** | **0.6403** |
| **FRMR** | 51.25% | 73.56% | 82.15% | 59.51% | 0.6128 |
| **BARMR** | 49.67% | 73.74% | 81.89% | 57.89% | 0.6037 |
| Best Baseline* | 45.49% | 71.17% | 78.33% | 55.35% | 0.5591 |

*Best published baseline: Claude 3 Re-ranker from Wu et al. (2024)

### Key Improvements
- **+20.4% Hit@1** over best published baseline
- **+14.5% MRR** over best published baseline
- **+7.8% Recall@20** over best published baseline

## 🔧 Configuration

Each pipeline can be customized via YAML configuration files:

```yaml
# Example: configs/frwsr_config.yaml
pipeline:
  name: "FRWSR"

retriever:
  model_name: "intfloat/e5-large"
  index_type: "HNSW"
  retrieve_k: 100

reranker:  
  model_name: "webis/set-encoder-large"
  rerank_k: 100

evaluation:
  metrics: ["hit@1", "hit@5", "hit@20", "recall@20", "MRR"]
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Usage Examples](docs/usage.md) 
- [API Reference](docs/api_reference.md)
- [Configuration Guide](docs/configuration.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{rumble2025neural,
  title={Neural Retriever–Reranker Pipelines for Retrieval Augmented Generation over Knowledge Graphs in e-Commerce Applications},
  author={Rumble, Teri and Gazdík, Zbyněk and Zarrin, Javad and Ahluwalia, Jagdeep},
  journal={Journal of the ACM},
  volume={37},
  number={4},
  pages={111:1--111:24},
  year={2025},
  publisher={ACM}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Amazon STaRK Dataset](https://stark.stanford.edu/) by Stanford SNAP Group
- [sentence-transformers](https://github.com/UKPLab/sentence-transformers) library
- [FAISS](https://github.com/facebookresearch/faiss) by Facebook AI Research
- [Lightning IR](https://github.com/webis-de/lightning-ir) by Webis Research Group

## ❓ Support

- 📧 Email: [corresponding-author@university.edu](mailto:corresponding-author@university.edu)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/neural-retriever-reranker-rag/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/neural-retriever-reranker-rag/discussions)

---

⭐ **If you find this work helpful, please consider starring the repository!** ⭐