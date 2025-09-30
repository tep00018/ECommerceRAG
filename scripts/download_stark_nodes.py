#!/usr/bin/env python3
"""
Download and process STARK Amazon SKB dataset.

This script downloads the raw STARK Amazon dataset from Hugging Face,
extracts node information with edges, cleans the data, and creates
the processed CSV file needed for the RAG pipeline.

Usage:
    python scripts/download_stark_nodes.py --output-dir data/nodes/
"""

import sys
import logging
import pickle
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f'stark_download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            )
        ]
    )
    return logging.getLogger(__name__)


def download_and_extract_stark_data(cache_dir: Path) -> Path:
    """Download and extract STARK Amazon dataset."""
    logger = logging.getLogger(__name__)
    
    logger.info("Downloading STARK Amazon dataset from Hugging Face...")
    logger.info("This is a 3.5GB download and may take several minutes...")
    
    # Download the processed data
    repo_id = "snap-stanford/stark"
    filename = "skb/amazon/processed.zip"
    
    file_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir
    )
    
    logger.info(f"Download complete: {file_path}")
    
    # Extract
    extract_dir = cache_dir / "amazon_skb_extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Extracting dataset...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    logger.info("Extraction complete")
    return extract_dir


def load_stark_data(extract_dir: Path):
    """Load node info, edges, and edge types from extracted files."""
    logger = logging.getLogger(__name__)
    
    processed_dir = extract_dir / "processed"
    
    logger.info("Loading node information...")
    with open(processed_dir / "node_info.pkl", 'rb') as f:
        node_info = pickle.load(f)
    logger.info(f"Loaded {len(node_info):,} nodes")
    
    logger.info("Loading edge index...")
    edge_index = torch.load(processed_dir / "edge_index.pt")
    logger.info(f"Loaded edge_index with shape {edge_index.shape}")
    
    logger.info("Loading edge types...")
    edge_types = torch.load(processed_dir / "edge_types.pt")
    logger.info(f"Loaded {len(edge_types):,} edges")
    
    with open(processed_dir / "edge_type_dict.pkl", 'rb') as f:
        edge_type_mapping = pickle.load(f)
    logger.info(f"Edge type mapping: {edge_type_mapping}")
    
    return node_info, edge_index, edge_types, edge_type_mapping


def build_edge_dictionary(node_info, edge_index, edge_types):
    """Build dictionary mapping nodes to their edges by type."""
    logger = logging.getLogger(__name__)
    
    edge_type_dict = {
        0: 'also_buy',
        1: 'also_view',
        2: 'has_brand',
        3: 'has_category',
        4: 'has_color'
    }
    
    logger.info("Building edge dictionary...")
    edges = edge_index.numpy()
    edge_types_np = edge_types.numpy()
    edge_list = list(zip(edges[0], edges[1], edge_types_np))
    
    edge_dict = {
        node_id: {etype: [] for etype in edge_type_dict.values()}
        for node_id in node_info.keys()
    }
    
    for src, dst, etype in tqdm(edge_list, desc="Processing edges"):
        edge_name = edge_type_dict.get(etype, "Unknown")
        if src in edge_dict:
            edge_dict[src][edge_name].append(int(dst))
        if dst in edge_dict:
            edge_dict[dst][edge_name].append(int(src))
    
    logger.info(f"Built edge dictionary for {len(edge_dict):,} nodes")
    return edge_dict


def create_dataframe(node_info, edge_dict):
    """Create DataFrame with node info, reviews, and edges."""
    logger = logging.getLogger(__name__)
    
    logger.info("Building DataFrame with node info, reviews, and edges...")
    node_data = []
    
    for node_id, node_info_dict in tqdm(node_info.items(), desc="Processing nodes"):
        title = node_info_dict.get("title", "")
        description = node_info_dict.get("description", "")
        feature = node_info_dict.get("feature", "")
        global_category = node_info_dict.get("global_category", "")
        categories = ", ".join(node_info_dict.get("category", [])) \
            if isinstance(node_info_dict.get("category"), list) else ""
        brand = node_info_dict.get("brand", "")
        price = node_info_dict.get("price", "")
        rank = node_info_dict.get("rank", "")
        
        reviews = node_info_dict.get("review", [])
        if isinstance(reviews, list):
            review_texts = [
                str(review.get("reviewText", ""))
                for review in reviews if isinstance(review, dict)
            ]
            review_ratings = [
                review.get("overall", None)
                for review in reviews if isinstance(review, dict)
            ]
        else:
            review_texts = []
            review_ratings = []
        
        node_edges = edge_dict.get(node_id, {})
        
        node_data.append({
            "node_id": node_id,
            "title": title,
            "description": description,
            "feature": feature,
            "global_category": global_category,
            "categories": categories,
            "brand": brand,
            "price": price,
            "rank": rank,
            "reviews": " | ".join(filter(None, review_texts)),
            "ratings": review_ratings,
            "also_buy": node_edges.get("also_buy", []),
            "also_view": node_edges.get("also_view", []),
            "has_brand": node_edges.get("has_brand", []),
            "has_category": node_edges.get("has_category", []),
            "has_color": node_edges.get("has_color", [])
        })
    
    df = pd.DataFrame(node_data)
    logger.info(f"Created DataFrame with {len(df):,} rows and {len(df.columns)} columns")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning operations to the DataFrame."""
    logger = logging.getLogger(__name__)
    logger.info("Cleaning data...")
    
    # Lowercase text columns
    text_columns = ['title', 'feature', 'description', 'global_category', 
                    'categories', 'brand', 'reviews']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].str.lower()
    
    # Clean description and feature fields
    import string
    import re
    
    def clean_text_field(desc):
        if isinstance(desc, list):
            desc = ' '.join(desc)
        elif not isinstance(desc, str):
            desc = ""
        desc = desc.translate(str.maketrans('', '', string.punctuation + "[]"))
        return ' '.join(desc.split())
    
    df['description'] = df['description'].apply(clean_text_field)
    df['feature'] = df['feature'].apply(clean_text_field)
    
    # Clean price
    df['price'] = df['price'].replace(r'[^\d.]', '', regex=True)
    df['price'] = df['price'].replace('', pd.NA)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Clean rank
    df['rank'] = df['rank'].apply(lambda x: str(x).replace(',', '')) \
                           .apply(lambda x: re.findall(r'\d+', x)) \
                           .apply(lambda x: x[0] if x else "Unknown")
    
    # Calculate avg_rating and rating_count
    df['avg_rating'] = df['ratings'].apply(
        lambda x: sum(eval(x)) / len(eval(x)) if isinstance(x, str) and len(eval(x)) > 0 else 0
    )
    df['rating_count'] = df['ratings'].apply(
        lambda x: len(eval(x)) if isinstance(x, str) else 0
    )
    
    # Remove duplicates from edge columns
    edge_columns = ['also_buy', 'also_view', 'has_brand', 'has_category', 'has_color']
    for col in edge_columns:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: list(set(eval(x))) if isinstance(x, str) else []
            )
    
    logger.info("Cleaning complete")
    return df


def create_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    """Create combined_text column."""
    logger = logging.getLogger(__name__)
    logger.info("Creating combined_text column...")
    
    def combine_text(row):
        parts = []
        
        if pd.notna(row.get('title')) and str(row['title']).strip():
            parts.append(f"Title: {row['title']}")
        if pd.notna(row.get('description')) and str(row['description']).strip():
            parts.append(f"Description: {row['description']}")
        if pd.notna(row.get('feature')) and str(row['feature']).strip():
            parts.append(f"Features: {row['feature']}")
        if pd.notna(row.get('brand')) and str(row['brand']).strip():
            parts.append(f"Brand: {row['brand']}")
        if pd.notna(row.get('reviews')) and str(row['reviews']).strip():
            parts.append(f"Reviews: {row['reviews']}")
        if pd.notna(row.get('price')):
            parts.append(f"Price: {row['price']}")
        if pd.notna(row.get('global_category')) and str(row['global_category']).strip():
            parts.append(f"Global Category: {row['global_category']}")
        if pd.notna(row.get('categories')) and str(row['categories']).strip():
            parts.append(f"Categories: {row['categories']}")
        if pd.notna(row.get('rank')) and str(row['rank']).strip() and str(row['rank']) != 'Unknown':
            parts.append(f"Rank: {row['rank']}")
        if pd.notna(row.get('avg_rating')) and row['avg_rating'] > 0:
            parts.append(f"Rating: {row['avg_rating']}")
        
        return ". ".join(parts)
    
    df['combined_text'] = df.apply(combine_text, axis=1)
    
    logger.info(f"Average text length: {df['combined_text'].str.len().mean():.0f} characters")
    return df


@click.command()
@click.option(
    '--output-dir', '-o',
    type=click.Path(path_type=Path),
    default='data/nodes/',
    help='Output directory for processed node data'
)
@click.option(
    '--cache-dir', '-c',
    type=click.Path(path_type=Path),
    default='.cache/',
    help='Cache directory for downloads'
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
def main(output_dir: Path, cache_dir: Path, verbose: bool, force: bool):
    """Download and process STARK Amazon SKB dataset."""
    
    logger = setup_logging(verbose)
    
    try:
        logger.info("=== STARK Amazon SKB Download and Processing ===")
        
        # Create directories
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "amazon_stark_nodes_processed.csv"
        
        # Check for existing file
        if output_file.exists() and not force:
            logger.warning(f"Output file already exists: {output_file}")
            if not click.confirm("Overwrite existing file?"):
                logger.info("Aborted")
                return
        
        # Download and extract
        extract_dir = download_and_extract_stark_data(cache_dir)
        
        # Load data
        node_info, edge_index, edge_types, edge_type_mapping = load_stark_data(extract_dir)
        
        # Build edge dictionary
        edge_dict = build_edge_dictionary(node_info, edge_index, edge_types)
        
        # Create DataFrame
        df = create_dataframe(node_info, edge_dict)
        
        # Clean data
        df = clean_dataframe(df)
        
        # Create combined text
        df = create_combined_text(df)
        
        # Save
        logger.info(f"Saving to {output_file}...")
        df.to_csv(output_file, index=False)
        
        logger.info("=== Processing Complete ===")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Total nodes: {len(df):,}")
        logger.info(f"Total columns: {len(df.columns)}")
        logger.info(f"File size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()