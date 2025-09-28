"""
Data Loading Utilities

This module provides utilities for loading and preprocessing data for the
Neural Retriever-Reranker RAG pipelines.
"""

import ast
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging


def load_node_data(file_path: Path) -> pd.DataFrame:
    """
    Load node data from CSV file.
    
    Args:
        file_path: Path to node data CSV file
        
    Returns:
        DataFrame containing node data
    """
    logger = logging.getLogger(__name__)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Node data file not found: {file_path}")
    
    logger.info(f"Loading node data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} nodes")
        
        # Validate required columns
        required_columns = ['node_id', 'combined_text']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading node data: {e}")
        raise


def load_query_data(file_path: Path) -> pd.DataFrame:
    """
    Load query data from CSV file and preprocess answer columns.
    
    Args:
        file_path: Path to query data CSV file
        
    Returns:
        DataFrame containing query data with processed answer columns
    """
    logger = logging.getLogger(__name__)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Query data file not found: {file_path}")
    
    logger.info(f"Loading query data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} queries")
        
        # Validate required columns
        required_columns = ['query']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Process answer columns if they exist
        if 'correct_answer' in df.columns:
            df['correct_answer'] = df['correct_answer'].apply(
                lambda x: _safe_literal_eval(x) if isinstance(x, str) else x
            )
        
        # Add query_id if not present
        if 'query_id' not in df.columns:
            df['query_id'] = range(1, len(df) + 1)
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading query data: {e}")
        raise


def load_retrieval_results(file_path: Path) -> pd.DataFrame:
    """
    Load retrieval results from CSV file and preprocess list columns.
    
    Args:
        file_path: Path to retrieval results CSV file
        
    Returns:
        DataFrame containing retrieval results
    """
    logger = logging.getLogger(__name__)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Retrieval results file not found: {file_path}")
    
    logger.info(f"Loading retrieval results from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} retrieval results")
        
        # Process list columns that might be stored as strings
        list_columns = [col for col in df.columns if 'answer' in col.lower()]
        
        for col in list_columns:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: _safe_literal_eval(x) if isinstance(x, str) else x
                )
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading retrieval results: {e}")
        raise


def _safe_literal_eval(x):
    """Safely evaluate string representations of Python literals."""
    if pd.isna(x) or x == '':
        return []
    
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        # If literal_eval fails, try to parse as a simple list format
        if isinstance(x, str):
            # Remove brackets and split by comma
            x = x.strip('[]')
            if not x:
                return []
            
            try:
                # Try to convert to integers
                return [int(item.strip()) for item in x.split(',') if item.strip()]
            except ValueError:
                # If that fails, keep as strings
                return [item.strip() for item in x.split(',') if item.strip()]
        
        return []


def prepare_documents_for_indexing(node_df: pd.DataFrame) -> Tuple[List[str], List[int]]:
    """
    Prepare documents and node IDs for FAISS indexing.
    
    Args:
        node_df: DataFrame containing node data
        
    Returns:
        Tuple of (documents, node_ids) for indexing
    """
    logger = logging.getLogger(__name__)
    
    # Validate required columns
    if 'combined_text' not in node_df.columns:
        raise ValueError("Node data must contain 'combined_text' column")
    
    if 'node_id' not in node_df.columns:
        raise ValueError("Node data must contain 'node_id' column")
    
    # Filter out rows with missing text or node_id
    valid_rows = node_df.dropna(subset=['combined_text', 'node_id'])
    
    if len(valid_rows) != len(node_df):
        logger.warning(f"Filtered out {len(node_df) - len(valid_rows)} rows with missing data")
    
    documents = valid_rows['combined_text'].tolist()
    node_ids = valid_rows['node_id'].astype(int).tolist()
    
    logger.info(f"Prepared {len(documents)} documents for indexing")
    
    return documents, node_ids


def split_data(
    df: pd.DataFrame, 
    train_ratio: float = 0.8, 
    val_ratio: float = 0.1, 
    test_ratio: float = 0.1,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/validation/test sets.
    
    Args:
        df: Input DataFrame
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    logger = logging.getLogger(__name__)
    
    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    n = len(df_shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_shuffled[:train_end]
    val_df = df_shuffled[train_end:val_end]
    test_df = df_shuffled[val_end:]
    
    logger.info(f"Split data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    
    return train_df, val_df, test_df


def save_split_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    output_dir: Path,
    prefix: str = ""
) -> None:
    """
    Save train/validation/test splits to CSV files.
    
    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        output_dir: Directory to save files
        prefix: Optional prefix for filenames
    """
    logger = logging.getLogger(__name__)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if prefix:
        prefix = f"{prefix}_"
    
    # Save splits
    train_path = output_dir / f"{prefix}train.csv"
    val_path = output_dir / f"{prefix}validation.csv"
    test_path = output_dir / f"{prefix}test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved data splits to {output_dir}")


def create_document_lookup(node_df: pd.DataFrame) -> Dict[int, Dict]:
    """
    Create a lookup dictionary for fast node retrieval.
    
    Args:
        node_df: DataFrame containing node data
        
    Returns:
        Dictionary mapping node_id to node data
    """
    lookup = {}
    
    for _, row in node_df.iterrows():
        node_id = int(row['node_id'])
        lookup[node_id] = row.to_dict()
    
    return lookup


def validate_data_consistency(node_df: pd.DataFrame, query_df: pd.DataFrame) -> bool:
    """
    Validate consistency between node and query data.
    
    Args:
        node_df: Node data DataFrame
        query_df: Query data DataFrame
        
    Returns:
        True if data is consistent, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    # Check for duplicate node IDs
    if node_df['node_id'].duplicated().any():
        logger.error("Duplicate node IDs found in node data")
        return False
    
    # If query data has correct answers, check if all referenced nodes exist
    if 'correct_answer' in query_df.columns:
        all_node_ids = set(node_df['node_id'].astype(int))
        
        for idx, row in query_df.iterrows():
            correct_answers = row['correct_answer']
            if isinstance(correct_answers, list):
                missing_nodes = [nid for nid in correct_answers if nid not in all_node_ids]
                if missing_nodes:
                    logger.warning(f"Query {idx}: Missing nodes {missing_nodes}")
    
    logger.info("Data consistency check passed")
    return True