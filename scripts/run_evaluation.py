#!/usr/bin/env python3
"""
Neural Retriever-Reranker RAG Pipeline Evaluation Script

This script runs evaluation for the specified pipeline configuration
on the Amazon STaRK dataset.

Usage:
    python scripts/run_evaluation.py --config configs/frwsr_config.yaml --dataset validation
    python scripts/run_evaluation.py --config configs/frmr_config.yaml --dataset evaluation_filtered
    python scripts/run_evaluation.py --config configs/barmr_config.yaml --dataset evaluation_full
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import click
import yaml
import pandas as pd

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.frwsr_pipeline import FRWSRPipeline
from pipeline.frmr_pipeline import FRMRPipeline  
from pipeline.barmr_pipeline import BARMRPipeline
from utils.data_loader import load_node_data, load_query_data


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'pipeline_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load pipeline configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_pipeline_class(pipeline_name: str):
    """Get pipeline class based on pipeline name."""
    pipeline_map = {
        'FRWSR': FRWSRPipeline,
        'FRMR': FRMRPipeline,
        'BARMR': BARMRPipeline
    }
    
    if pipeline_name not in pipeline_map:
        raise ValueError(f"Unknown pipeline: {pipeline_name}. Available: {list(pipeline_map.keys())}")
    
    return pipeline_map[pipeline_name]


def validate_paths(config: Dict[str, Any], base_dir: Path) -> None:
    """Validate that all required data paths exist."""
    data_config = config['data']
    
    # Check node file
    node_file = base_dir / data_config['node_file']
    if not node_file.exists():
        raise FileNotFoundError(f"Node data file not found: {node_file}")
    
    # Check query files
    for split_name, query_file in data_config['query_splits'].items():
        query_path = base_dir / query_file
        if not query_path.exists():
            raise FileNotFoundError(f"Query file '{split_name}' not found: {query_path}")
    
    # Check/create results directory
    results_dir = base_dir / data_config['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)


@click.command()
@click.option(
    '--config', '-c', 
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help='Path to pipeline configuration YAML file'
)
@click.option(
    '--dataset', '-d',
    type=click.Choice(['validation', 'evaluation_filtered', 'evaluation_full']),
    default='validation',
    help='Dataset split to evaluate on'
)
@click.option(
    '--output-suffix', '-s',
    type=str,
    default='',
    help='Suffix to add to output filename'
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Perform dry run without actual evaluation'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose logging'
)
def main(config: Path, dataset: str, output_suffix: str, dry_run: bool, verbose: bool):
    """Run pipeline evaluation on specified dataset."""
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    logger = setup_logging(log_level)
    
    try:
        logger.info("=== Neural Retriever-Reranker Pipeline Evaluation ===")
        logger.info(f"Configuration: {config}")
        logger.info(f"Dataset: {dataset}")
        logger.info(f"Dry run: {dry_run}")
        
        # Load configuration
        logger.info("Loading configuration...")
        config_data = load_config(config)
        pipeline_name = config_data['pipeline']['name']
        
        # Set base directory (assuming script is run from repo root)
        base_dir = Path.cwd()
        
        # Validate paths
        logger.info("Validating data paths...")
        validate_paths(config_data, base_dir)
        
        if dry_run:
            logger.info("Dry run completed successfully!")
            return
        
        # Initialize pipeline
        logger.info(f"Initializing {pipeline_name} pipeline...")
        pipeline_class = get_pipeline_class(pipeline_name)
        pipeline = pipeline_class(config_data)
        
        # Load node data
        logger.info("Loading node data...")
        node_file = base_dir / config_data['data']['node_file']
        pipeline.load_data(node_file)
        
        # Load query data
        query_file = base_dir / config_data['data']['query_splits'][dataset]
        logger.info(f"Loading {dataset} queries from {query_file}")
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{pipeline_name.lower()}_{dataset}_{timestamp}"
        if output_suffix:
            output_filename += f"_{output_suffix}"
        output_filename += ".csv"
        
        output_path = base_dir / config_data['data']['results_dir'] / output_filename
        
        # Run evaluation
        logger.info(f"Starting evaluation on {dataset} dataset...")
        logger.info(f"Results will be saved to: {output_path}")
        
        results_df = pipeline.evaluate_on_dataset(
            query_file=query_file,
            output_file=output_path,
            save_partial=config_data['evaluation'].get('save_partial', True),
            partial_interval=config_data['evaluation'].get('partial_interval', 10)
        )
        
        # Log summary statistics
        logger.info("=== Evaluation Complete ===")
        logger.info(f"Processed {len(results_df)} queries")
        logger.info(f"Results saved to: {output_path}")
        
        # Display key metrics
        metric_cols = ['hit@1', 'hit@5', 'hit@20', 'recall@20', 'MRR']
        available_metrics = [col for col in metric_cols if col in results_df.columns]
        
        if available_metrics:
            summary_metrics = results_df[available_metrics].mean()
            logger.info("\n=== Summary Metrics ===")
            for metric, value in summary_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()