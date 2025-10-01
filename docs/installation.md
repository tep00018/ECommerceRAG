## Installation Guide
Complete installation instructions for the ECommerceRAG Neural Retriever-Reranker pipelines.  

### Prerequisites
Before beginning, ensure you have:

- Python 3.8 or higher
- Git
- 16GB+ RAM (32GB recommended)
- 50GB+ free disk space
- GPU with CUDA support (optional but strongly recommended)
- Internet connection

### Step 1: Clone Repository
```bash
# Navigate to your working directory
cd ~/work

# Clone the repository
git clone https://github.com/tep00018/ECommerceRAG.git

# Enter the repository
cd ECommerceRAG

# Verify repository structure
ls -la
```

Expected output should show directories: src/, scripts/, configs/, data/, docs/, etc.


### Step 2: Set Up Python Environment
#### Step 2A: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate     # On Windows
```

#### Step 2B: Install Dependencies
```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

#### Step 2C: Verify Installation
```bash
# Check Python version
python --version

# Check key packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import faiss; print('FAISS: Installed')"

# Check GPU availability (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Step 2D: Download NLTK Data
```bash
# Required for BM25 tokenization
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 3: Download Amazon STaRK Dataset
#### Step 3A: Automated Download (Recommended)
```bash
# Download and process dataset
python scripts/download_stark_nodes.py --output-dir data/nodes/ --verbose

# This downloads ~3.5GB and takes 15-30 minutes
# Creates: data/nodes/amazon_stark_nodes_processed.csv
```

#### Step 3B: Manual Download (If Step 3A Fails)
If automated download fails:  
1. Open notebooks/download_stark_amazon_skb.ipynb in Google Colab  
2. Run all cells in the notebook  
3. Download the resulting CSV file  
4. Upload to data/nodes/amazon_stark_nodes_processed.csv  

#### Step 3C: Verify Data Download
```bash
# Check file exists and size
ls -lh data/nodes/amazon_stark_nodes_processed.csv
# Should show ~500MB file

# Check row count
wc -l data/nodes/amazon_stark_nodes_processed.csv
# Should show ~1,035,542 lines

# View first few lines
head -n 3 data/nodes/amazon_stark_nodes_processed.csv
```

#### Step 3D: Verify Query Files
```bash
# Query files should already be in repository
ls -lh data/queries/

# Should see:
# - validation_queries.csv (910 queries)
# - evaluation_filtered_queries.csv (6,380 queries)
# - evaluation_full_queries.csv (9,100 queries)
```

### Step 4: Create Embeddings
Choose the appropriate substep based on which pipeline(s) you plan to run:

FRWSR and FRMR: Require Step 4A (dense embeddings)
BARMR: Requires Step 4B (sparse embeddings)
All pipelines: Complete both Step 4A and Step 4B

#### Step 4A: Create Dense Embeddings (For FRWSR and FRMR)
```bash
# Create E5-Large dense vector embeddings
python scripts/create_embeddings.py \
  --data-file data/nodes/amazon_stark_nodes_processed.csv \
  --output-dir data/embeddings/ \
  --model-name intfloat/e5-large \
  --batch-size 128 \
  --chunk-size 50000 \
  --verbose

# Duration: 2-4 hours with GPU, 8-12 hours without GPU
# Creates: data/embeddings/intfloat_e5_large_embeddings.npy (~4GB)
```

Note: If you encounter out-of-memory errors, reduce batch size:
```bash
python scripts/create_embeddings.py \
  --data-file data/nodes/amazon_stark_nodes_processed.csv \
  --output-dir data/embeddings/ \
  --batch-size 32 \
  --chunk-size 10000 \
  --verbose
```

Verify embedding creation:
```bash
# Check file exists
ls -lh data/embeddings/intfloat_e5_large_embeddings.npy
# Should show ~4GB file

# Verify with Python
python -c "import numpy as np; emb = np.load('data/embeddings/intfloat_e5_large_embeddings.npy'); print(f'Shape: {emb.shape}, Dtype: {emb.dtype}')"
# Should show: Shape: (1035542, 1024), Dtype: float32
```

#### Step 4B: Create Sparse Embeddings (For BARMR)
```bash
# Create BM25 tokenized documents
python scripts/create_bm25_embeddings.py \
  --data-file data/nodes/amazon_stark_nodes_processed.csv \
  --output-dir data/embeddings/ \
  --verbose

# Duration: 10-15 minutes
# Creates: data/embeddings/bm25_tokenized_documents.pkl (~500MB)
```

Verify tokenization:
```bash
# Check file exists
ls -lh data/embeddings/bm25_tokenized_documents.pkl
# Should show ~500MB file
```

### Step 5: Build Indices
Choose the appropriate substeps based on which pipeline(s) you plan to run.  
#### Step 5A: Build FAISS HNSW Index (For FRWSR and FRMR)
```bash
# Build FAISS HNSW index with optimized parameters
python scripts/build_faiss_hnsw.py \
  --embeddings data/embeddings/intfloat_e5_large_embeddings.npy \
  --output-dir data/indices/ \
  --index-name faiss_hnsw_index \
  --M 64 \
  --ef-construction 100 \
  --ef-search 200 \
  --verbose

# Duration: 1-2 hours
# Creates: data/indices/faiss_hnsw_index.index
```

Verify FAISS index:
```bash
# Check file exists
ls -lh data/indices/faiss_hnsw_index.index

# Test loading the index
python -c "import faiss; idx = faiss.read_index('data/indices/faiss_hnsw_index.index'); print(f'Index loaded: {idx.ntotal} vectors')"
# Should show: Index loaded: 1035542 vectors
```

#### Step 5B: Build FAISS FLAT Index (Optional - For Exact Search)
```bash
# Build FAISS FLAT index for exact nearest neighbor search
python scripts/build_faiss_flat.py \
  --embeddings data/embeddings/intfloat_e5_large_embeddings.npy \
  --output-dir data/indices/ \
  --index-name faiss_flat_index \
  --verbose

# Duration: 30-45 minutes
# Creates: data/indices/faiss_flat_index.index
# Note: FLAT is slower but guarantees higher recall
```

#### Step 5C: Build BM25 Index with Graph Augmentation (For BARMR)
```bash
# Build BM25 index with graph structure
python scripts/build_indices.py \
  --data-file data/nodes/amazon_stark_nodes_processed.csv \
  --index-type bm25 \
  --graph-augmentation \
  --output-dir data/indices/ \
  --verbose

# Duration: 30-45 minutes
# Creates: data/indices/bm25_augmented/
```

Verify BM25 index:
```bash
# Check directory exists
ls -lh data/indices/bm25_augmented/
```

### Step 6: Run Smoke Tests  
Verify that all components are working correctly:
```bash
# Run smoke test
python scripts/smoke_test.py
```

Expected output:
============================================================
SMOKE TEST - ECommerceRAG
============================================================

Testing imports...
✓ All imports successful

Checking data files...
✓ data/queries/validation_queries.csv
✓ data/queries/evaluation_filtered_queries.csv
✓ data/queries/evaluation_full_queries.csv

Testing pipeline configs...
✓ configs/frmr_config.yaml
✓ configs/frwsr_config.yaml
✓ configs/barmr_config.yaml

============================================================
✓ ALL TESTS PASSED
============================================================
```

If tests fail, verify:  
- All dependencies installed correctly  
- Data files downloaded successfully  
- Embeddings and indices created  

### Step 7: Test Individual Pipelines
#### Step 7A: Test FRMR Pipeline (Fastest)
```bash
# Run on 10 validation queries as a quick test
python scripts/run_evaluation.py \
  --config configs/frmr_config.yaml \
  --dataset validation \
  --max-queries 10 \
  --output-dir data/results/test/

# Duration: ~5 minutes
```

#### Step 7B: Test FRWSR Pipeline (Highest Accuracy)
```bash
# Run on 10 validation queries
python scripts/run_evaluation.py \
  --config configs/frwsr_config.yaml \
  --dataset validation \
  --max-queries 10 \
  --output-dir data/results/test/

# Duration: ~10-15 minutes (slower reranker)
```

#### Step 7C: Test BARMR Pipeline (Sparse Retrieval)
```bash
# Run on 10 validation queries
python scripts/run_evaluation.py \
  --config configs/barmr_config.yaml \
  --dataset validation \
  --max-queries 10 \
  --output-dir data/results/test/

# Duration: ~8-10 minutes
```

### Step 8: Run Full Validation
After confirming pipelines work, run full validation set (910 queries):
#### Step 8A: FRMR Validation
```bash
python scripts/run_evaluation.py \
  --config configs/frmr_config.yaml \
  --dataset validation \
  --output-dir data/results/frmr/

# Duration: 1-2 hours
```

#### Step 8B: FRWSR Validation
```bash
python scripts/run_evaluation.py \
  --config configs/frwsr_config.yaml \
  --dataset validation \
  --output-dir data/results/frwsr/

# Duration: 3-4 hours
```

#### Step 8C: BARMR Validation
```bash
python scripts/run_evaluation.py \
  --config configs/barmr_config.yaml \
  --dataset validation \
  --output-dir data/results/barmr/

# Duration: 2-3 hours
```

#### Troubleshooting
##### GPU Not Detected
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install PyTorch with CUDA support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

##### Out of Memory Errors
Reduce batch sizes:
```bash
# For embeddings
python scripts/create_embeddings.py \
  --batch-size 32 \
  --chunk-size 10000 \
  [other options]

# For evaluation
python scripts/run_evaluation.py \
  --batch-size 16 \
  [other options]
```
  
##### FAISS Installation Issues
```bash
# Uninstall and reinstall FAISS
pip uninstall faiss-cpu faiss-gpu

# For CPU only
pip install faiss-cpu

# For GPU
pip install faiss-gpu
```

##### Download Failures
If automated download fails, use manual method (Step 3B) with the Colab notebook.


#### Disk Space Requirements
Total disk space needed:  
- Repository: ~50MB  
- Amazon STaRK nodes CSV: ~500MB  
- Dense embeddings (E5-Large): ~4GB  
- Sparse embeddings (BM25): ~500MB  
- FAISS HNSW index: ~4GB  
- FAISS FLAT index: ~4GB (optional)  
- BM25 index: ~500MB  
- Results and logs: ~1GB  
##### Total: ~15GB minimum, 20GB recommended

#### Expected Timelines
Installation timeline with GPU:  
1. Repository setup: 5-10 minutes  
2. Data download: 15-30 minutes  
3. Dense embeddings: 2-4 hours  
4. Sparse embeddings: 10-15 minutes  
5. Index building: 2-3 hours  
6. Testing: 1 hour  
##### Total: 6-9 hours  

Without GPU (CPU only):  
1. Repository setup: 5-10 minutes  
2. Data download: 15-30 minutes  
3. Dense embeddings: 8-12 hours  
4. Sparse embeddings: 10-15 minutes  
5. Index building: 3-4 hours  
6. Testing: 2 hours  
##### Total: 14-19 hours  

#### Next Steps 
After successful installation: run full evaluations on the complete dataset

#### Support
If you encounter issues:
- Review the troubleshooting section above
- Ensure all prerequisites are met
- Verify disk space and memory availability

#### Citation
Repository: https://github.com/tep00018/ECommerceRAG 
Paper: "Neural Retriever–Reranker Pipelines for Retrieval Augmented Generation over Knowledge Graphs in e-Commerce Applications" (2025)  