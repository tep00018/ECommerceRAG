#!/usr/bin/env python3
"""
Data Download and Preprocessing Script

This script downloads and preprocesses the Amazon STaRK dataset for use
with the Neural Retriever-Reranker RAG pipelines.

Usage:
    python scripts/download_data.py --output-dir data/
    python scripts/download_data.py --components nodes,queries --output-dir data/
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

import click
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    from datasets import load_dataset
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    click.echo("Warning: datasets library not available. Manual download required.")

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.preprocessing import preprocess_node_text, create_combined_text


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'data_download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def download_stark_dataset() -> Dict[str, Any]:
    """
    Download the Amazon STaRK dataset from Hugging Face.
    
    Returns:
        Dictionary containing the dataset splits
    """
    if not HUGGINGFACE_AVAILABLE:
        raise ImportError(
            "datasets library is required for automatic download. "
            "Install with: pip install datasets"
        )
    
    logger = logging.getLogger(__name__)
    logger.info("Downloading Amazon STaRK dataset from Hugging Face...")
    
    try:
        # Load the dataset
        dataset = load_dataset("snap-stanford/stark")
        logger.info("Dataset downloaded successfully")
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        raise


def process_node_data(dataset: Dict[str, Any], output_path: Path) -> None:
    """
    Process and save node data with combined text fields.
    
    Args:
        dataset: Raw dataset from Hugging Face
        output_path: Path to save processed node data
    """
    logger = logging.getLogger(__name__)
    logger.info("Processing node data...")
    
    # Extract node data (assuming it's in the 'nodes' split)
    if 'nodes' in dataset:
        nodes_data = dataset['nodes']
    else:
        # Fallback: look for any split that contains node data
        for split_name, split_data in dataset.items():
            if 'node_id' in split_data.column_names:
                nodes_data = split_data
                logger.info(f"Using split '{split_name}' for node data")
                break
        else:
            raise ValueError("Could not find node data in dataset")
    
    # Convert to pandas DataFrame
    nodes_df = nodes_data.to_pandas()
    logger.info(f"Loaded {len(nodes_df)} nodes")
    
    # Process text fields
    logger.info("Creating combined text fields...")
    
    # Define text fields to combine
    text_fields = [
        'title', 'description', 'brand', 'category', 'global_category',
        'features', 'reviews', 'qa'
    ]
    
    combined_texts = []
    
    with tqdm(total=len(nodes_df), desc="Processing nodes") as pbar:
        for idx, row in nodes_df.iterrows():
            combined_text = create_combined_text(row, text_fields)
            combined_texts.append(combined_text)
            pbar.update(1)
    
    # Add combined text to DataFrame
    nodes_df['combined_text'] = combined_texts
    
    # Save processed data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nodes_df.to_csv(output_path, index=False)
    
    logger.info(f"Processed node data saved to {output_path}")
    logger.info(f"Dataset shape: {nodes_df.shape}")


def process_query_data(dataset: Dict[str, Any], output_dir: Path) -> None:
    """
    Process and save query data with proper splits.
    
    Args:
        dataset: Raw dataset from Hugging Face
        output_dir: Directory to save query splits
    """
    logger = logging.getLogger(__name__)
    logger.info("Processing query data...")
    
    # Extract query data
    if 'queries' in dataset:
        queries_data = dataset['queries']
    else:
        # Look for query-related splits
        for split_name, split_data in dataset.items():
            if 'query' in split_data.column_names:
                queries_data = split_data
                logger.info(f"Using split '{split_name}' for query data")
                break
        else:
            raise ValueError("Could not find query data in dataset")
    
    # Convert to pandas DataFrame
    queries_df = queries_data.to_pandas()
    logger.info(f"Loaded {len(queries_df)} queries")
    
    # Ensure required columns exist
    required_columns = ['query', 'correct_answer']
    missing_columns = [col for col in required_columns if col not in queries_df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Add query_id if not present
    if 'query_id' not in queries_df.columns:
        queries_df['query_id'] = range(1, len(queries_df) + 1)
    
    # Filter queries by ground truth length for different evaluation sets
    logger.info("Creating query splits...")
    
    # Calculate ground truth lengths
    gt_lengths = queries_df['correct_answer'].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    
    # Create splits
    splits = {
        'validation': queries_df.sample(n=min(910, len(queries_df)), random_state=42),
        'evaluation_filtered': queries_df[gt_lengths <= 20],
        'evaluation_full': queries_df
    }
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_df in splits.items():
        output_path = output_dir / f"{split_name}_queries.csv"
        split_df.to_csv(output_path, index=False)
        logger.info(f"Saved {split_name} split: {len(split_df)} queries -> {output_path}")


def create_sample_indices(nodes_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create sample FAISS indices for quick testing.
    
    Args:
        nodes_df: Processed node DataFrame
        output_dir: Directory to save indices
    """
    logger = logging.getLogger(__name__)
    logger.info("Creating sample indices...")
    
    try:
        # Import required modules
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from retrieval.faiss_retriever import FAISSRetriever
        from utils.data_loader import prepare_documents_for_indexing
        
        # Prepare documents (use sample for faster processing)
        sample_size = min(10000, len(nodes_df))
        sample_df = nodes_df.sample(n=sample_size, random_state=42)
        
        documents, node_ids = prepare_documents_for_indexing(sample_df)
        
        logger.info(f"Building FAISS index with {len(documents)} documents...")
        
        # Build FAISS HNSW index
        retriever = FAISSRetriever(
            model_name="intfloat/e5-large",
            index_type="HNSW",
            M=64,
            efConstruction=100,
            efSearch=200
        )
        
        retriever.build_index(documents, node_ids)
        
        # Save index
        index_path = output_dir / "faiss_e5_large_hnsw_sample"
        retriever.save_index(index_path)
        
        logger.info(f"Sample FAISS index saved to {index_path}")
        
    except Exception as e:
        logger.warning(f"Failed to create sample indices: {e}")
        logger.info("You can build indices later using scripts/build_indices.py")


def create_combined_text(row: pd.Series, text_fields: List[str]) -> str:
    """
    Create combined text from multiple fields.
    
    Args:
        row: DataFrame row
        text_fields: List of field names to combine
        
    Returns:
        Combined text string
    """
    text_parts = []
    
    for field in text_fields:
        if field in row and pd.notna(row[field]):
            value = row[field]
            
            # Handle different data types
            if isinstance(value, str):
                if value.strip():
                    text_parts.append(f"{field.title()}: {value.strip()}")
            elif isinstance(value, list):
                if value:
                    # Join list items
                    list_text = " ".join(str(item) for item in value if str(item).strip())
                    if list_text:
                        text_parts.append(f"{field.title()}: {list_text}")
            else:
                # Convert to string
                str_value = str(value).strip()
                if str_value and str_value.lower() not in ['nan', 'none', '']:
                    text_parts.append(f"{field.title()}: {str_value}")
    
    return " ".join(text_parts)


@click.command()
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='data/',
    help='Output directory for processed data'
)
@click.option(
    '--components', '-c',
    default='nodes,queries',
    help='Components to download (nodes,queries,indices)'
)
@click.option(
    '--sample-size', '-s',
    type=int,
    help='Process only a sample of the data (for testing)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
@click.option(
    '--force', '-f',
    is_flag=True,
    help='Overwrite existing files'
)
def main(output_dir: Path, components: str, sample_size: Optional[int], verbose: bool, force: bool):
    """Download and preprocess Amazon STaRK dataset."""
    
    logger = setup_logging(verbose)
    
    try:
        logger.info("=== Amazon STaRK Data Download and Preprocessing ===")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Components: {components}")
        
        # Parse components
        component_list = [comp.strip().lower() for comp in components.split(',')]
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check for existing files
        existing_files = []
        if (output_dir / 'nodes' / 'amazon_stark_nodes_processed.csv').exists():
            existing_files.append('nodes')
        if (output_dir / 'queries' / 'validation_queries.csv').exists():
            existing_files.append('queries')
        
        if existing_files and not force:
            logger.warning(f"Existing files found for: {existing_files}")
            logger.info("Use --force to overwrite existing files")
            if not click.confirm("Continue with non-existing components?"):
                return
        
        # Download dataset
        if not HUGGINGFACE_AVAILABLE:
            logger.error("Automatic download not available. Please download manually:")
            logger.info("1. Visit: https://huggingface.co/datasets/snap-stanford/stark")
            logger.info("2. Download the dataset files")
            logger.info("3. Place them in the appropriate directories")
            return
        
        dataset = download_stark_dataset()
        
        # Process components
        if 'nodes' in component_list:
            node_output = output_dir / 'nodes' / 'amazon_stark_nodes_processed.csv'
            if not node_output.exists() or force:
                process_node_data(dataset, node_output)
            else:
                logger.info(f"Skipping nodes (file exists): {node_output}")
        
        if 'queries' in component_list:
            query_output_dir = output_dir / 'queries'
            if not (query_output_dir / 'validation_queries.csv').exists() or force:
                process_query_data(dataset, query_output_dir)
            else:
                logger.info(f"Skipping queries (files exist): {query_output_dir}")
        
        if 'indices' in component_list:
            # Load processed nodes for index creation
            node_file = output_dir / 'nodes' / 'amazon_stark_nodes_processed.csv'
            if node_file.exists():
                logger.info("Loading processed nodes for index creation...")
                nodes_df = pd.read_csv(node_file)
                
                if sample_size:
                    nodes_df = nodes_df.sample(n=min(sample_size, len(nodes_df)), random_state=42)
                    logger.info(f"Using sample of {len(nodes_df)} nodes")
                
                indices_dir = output_dir / 'indices'
                create_sample_indices(nodes_df, indices_dir)
            else:
                logger.warning("Node data not found. Cannot create indices.")
        
        # Create README files
        create_data_readme(output_dir)
        
        logger.info("=== Data Download and Preprocessing Complete ===")
        
        # Print summary
        logger.info("\n=== Summary ===")
        for component in component_list:
            if component == 'nodes':
                node_file = output_dir / 'nodes' / 'amazon_stark_nodes_processed.csv'
                if node_file.exists():
                    node_count = len(pd.read_csv(node_file, usecols=[0]))
                    logger.info(f"✅ Nodes: {node_count:,} processed")
                else:
                    logger.info("❌ Nodes: Failed to process")
            
            elif component == 'queries':
                query_dir = output_dir / 'queries'
                if (query_dir / 'validation_queries.csv').exists():
                    val_count = len(pd.read_csv(query_dir / 'validation_queries.csv', usecols=[0]))
                    eval_count = len(pd.read_csv(query_dir / 'evaluation_full_queries.csv', usecols=[0]))
                    logger.info(f"✅ Queries: {val_count:,} validation, {eval_count:,} evaluation")
                else:
                    logger.info("❌ Queries: Failed to process")
            
            elif component == 'indices':
                index_dir = output_dir / 'indices' / 'faiss_e5_large_hnsw_sample'
                if index_dir.exists():
                    logger.info("✅ Indices: Sample FAISS index created")
                else:
                    logger.info("❌ Indices: Failed to create")
        
        logger.info(f"\nData ready in: {output_dir.absolute()}")
        logger.info("Next steps:")
        logger.info("1. Run: python scripts/run_evaluation.py --config configs/frwsr_config.yaml --dataset validation")
        logger.info("2. Or follow the README.md for detailed usage instructions")
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def create_data_readme(output_dir: Path) -> None:
    """Create README files in data directories."""
    
    # Main data README (we already created this)
    readme_content = """# Data Directory

This directory contains the processed Amazon STaRK dataset files.

## Generated Files

- `nodes/amazon_stark_nodes_processed.csv` - Processed product nodes (~1M entries)
- `queries/validation_queries.csv` - Validation queries (910 entries)  
- `queries/evaluation_queries_filtered.csv` - Evaluation queries with ≤20 answers
- `queries/evaluation_queries_full.csv` - Complete evaluation set (9,100 queries)
- `indices/faiss_e5_large_hnsw_sample/` - Sample FAISS index for testing

## File Formats

### Node Data
- `node_id`: Unique identifier
- `combined_text`: Preprocessed text combining all product fields
- Additional metadata columns preserved from original dataset

### Query Data  
- `query_id`: Unique identifier
- `query`: Natural language query
- `correct_answer`: List of relevant node IDs

## Usage

Load data using the utilities in `src/utils/data_loader.py`:

```python
from src.utils.data_loader import load_node_data, load_query_data

nodes = load_node_data('data/nodes/amazon_stark_nodes_processed.csv')
queries = load_query_data('data/queries/validation_queries.csv')
```

For more information, see the main README.md file.
"""
    
    readme_path = output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)


if __name__ == "__main__":
    main()