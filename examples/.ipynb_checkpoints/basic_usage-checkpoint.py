#!/usr/bin/env python3
"""
Basic Usage Example for Neural Retriever-Reranker RAG Pipelines

This example demonstrates how to use the different pipeline implementations
for query processing and evaluation.
"""

import sys
from pathlib import Path
import yaml
import pandas as pd

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.frwsr_pipeline import FRWSRPipeline
from pipeline.frmr_pipeline import FRMRPipeline
from pipeline.barmr_pipeline import BARMRPipeline


def basic_pipeline_usage():
    """Demonstrate basic pipeline usage with a simple query."""
    
    print("=== Neural Retriever-Reranker RAG Pipeline Examples ===\n")
    
    # Example configuration (you would normally load from YAML)
    base_config = {
        'data': {
            'node_file': 'data/nodes/amazon_stark_nodes_processed.csv'
        },
        'retriever': {
            'model_name': 'intfloat/e5-large',
            'index_path': 'data/indices/faiss_e5_large_hnsw/',
            'retrieve_k': 20
        },
        'reranker': {
            'rerank_k': 10
        },
        'logging': {
            'level': 'INFO'
        }
    }
    
    # Sample query
    sample_query = "What are some high-quality wireless noise-cancelling headphones?"
    
    print(f"Sample Query: {sample_query}\n")
    
    # Example 1: FRWSR Pipeline (FAISS + Webis Set-Encoder)
    print("1. FRWSR Pipeline (Best Accuracy)")
    print("-" * 40)
    
    try:
        frwsr_config = base_config.copy()
        frwsr_config['pipeline'] = {'name': 'FRWSR'}
        frwsr_config['reranker']['model_name'] = 'webis/set-encoder-large'
        
        frwsr_pipeline = FRWSRPipeline(frwsr_config)
        
        # Check if data and indices are available
        node_file = Path(frwsr_config['data']['node_file'])
        index_path = Path(frwsr_config['retriever']['index_path'])
        
        if node_file.exists() and index_path.exists():
            frwsr_pipeline.load_data(node_file)
            results = frwsr_pipeline.process_query(sample_query, rerank_k=5)
            print(f"✅ FRWSR Results: {results}")
            
            # Get pipeline info
            info = frwsr_pipeline.get_pipeline_info()
            print(f"   Characteristics: {', '.join(info['characteristics'])}")
        else:
            print("❌ Data or indices not found. Run data setup first.")
            print(f"   Missing: {node_file if not node_file.exists() else index_path}")
        
    except Exception as e:
        print(f"❌ FRWSR Error: {e}")
    
    print()
    
    # Example 3: BARMR Pipeline (BM25 + Graph Augmentation)
    print("3. BARMR Pipeline (Interpretable + Graph-Enhanced)")
    print("-" * 52)
    
    try:
        barmr_config = base_config.copy()
        barmr_config['pipeline'] = {'name': 'BARMR'}
        barmr_config['retriever'] = {
            'type': 'BM25',
            'bm25_params': {
                'k1': 1.016564434220879,
                'b': 0.8856501982953431,
                'similarity_threshold': 21
            },
            'graph_augmentation': {
                'enabled': True,
                'max_expansion': 50
            }
        }
        barmr_config['reranker']['model_name'] = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        
        barmr_pipeline = BARMRPipeline(barmr_config)
        
        # Demo without actual data loading
        info = barmr_pipeline.get_pipeline_info()
        print(f"✅ BARMR Pipeline initialized")
        print(f"   BM25 hyperparameters: k1={barmr_pipeline.k1:.3f}, b={barmr_pipeline.b:.3f}")
        print(f"   Similarity threshold: {barmr_pipeline.similarity_threshold}")
        print(f"   Characteristics: {', '.join(info['characteristics'])}")
        
    except Exception as e:
        print(f"❌ BARMR Error: {e}")
    
    print()


def actual_implementation_details():
    """Show details from the actual implementations."""
    
    print("=== Actual Implementation Details ===\n")
    
    implementations = {
        'FRWSR': {
            'file': 'FRWSR_RAG_Pipeline.txt',
            'retriever': 'E5-Large + FAISS-HNSW',
            'reranker': 'webis/set-encoder-large',
            'key_features': [
                'Set-wise cross-encoder with inter-passage attention',
                'Permutation-invariant attention mechanisms',
                'Context-aware individual document scoring'
            ]
        },
        'FRMR': {
            'file': 'FRMR_faiss_hnsw_retriever_ce_compositequeries_040125.py', 
            'retriever': 'E5-Large + FAISS-HNSW (M=64, efConstruction=100, efSearch=200)',
            'reranker': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'key_features': [
                'Optimized HNSW parameters for speed/accuracy balance',
                'Point-wise cross-encoder for fast reranking',
                'Production-ready with 1.89 it/s processing speed'
            ]
        },
        'BARMR': {
            'file': 'BARMR_bm25_retriever_minilm_l6_cross_encoder.py',
            'retriever': 'BM25 (k1=1.017, b=0.886, threshold=21) + Graph augmentation',
            'reranker': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'key_features': [
                'Optimized BM25 hyperparameters from extensive tuning',
                '1-hop graph expansion with also-buy/also-view edges',
                'Similarity threshold filtering for quality candidates'
            ]
        }
    }
    
    for pipeline_name, details in implementations.items():
        print(f"📁 {pipeline_name} Pipeline")
        print(f"   Source: {details['file']}")
        print(f"   Retriever: {details['retriever']}")
        print(f"   Reranker: {details['reranker']}")
        print("   Key Features:")
        for feature in details['key_features']:
            print(f"   • {feature}")
        print()


def hyperparameter_details():
    """Show the actual hyperparameters from implementations."""
    
    print("=== Hyperparameter Details ===\n")
    
    hyperparams = {
        'FAISS-HNSW (FRMR/FRWSR)': {
            'M': 64,
            'efConstruction': 100, 
            'efSearch': 200,
            'model': 'intfloat/e5-large',
            'dimension': 1024
        },
        'BM25 (BARMR)': {
            'k1': 1.016564434220879,
            'b': 0.8856501982953431,
            'similarity_threshold': 21,
            'top_n': 100,
            'tokenizer': 'NLTK with stopword removal'
        },
        'MS MARCO Reranker (FRMR/BARMR)': {
            'model': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
            'batch_size': 32,
            'max_length': 512,
            'processing_speed': '1.89 it/s (FRMR), 0.44 it/s (BARMR)'
        },
        'Webis Set-Encoder (FRWSR)': {
            'model': 'webis/set-encoder-large',
            'batch_size': 16,
            'max_length': 512,
            'processing_speed': '0.01 it/s (high accuracy, slower)'
        }
    }
    
    for component, params in hyperparams.items():
        print(f"⚙️  {component}")
        for param, value in params.items():
            print(f"   {param}: {value}")
        print()


def performance_analysis():
    """Detailed performance analysis from actual results."""
    
    print("=== Performance Analysis ===\n")
    
    results = {
        'FRWSR': {
            'hit@1': 54.75, 'hit@5': 75.25, 'hit@20': 84.21, 'recall@20': 61.30, 'mrr': 64.03,
            'speed': '0.01 it/s', 'strength': 'Highest accuracy', 'weakness': 'Slowest processing'
        },
        'FRMR': {
            'hit@1': 51.25, 'hit@5': 73.56, 'hit@20': 82.15, 'recall@20': 59.51, 'mrr': 61.28,
            'speed': '1.89 it/s', 'strength': 'Best balance', 'weakness': '3.5% accuracy loss vs FRWSR'
        },
        'BARMR': {
            'hit@1': 49.67, 'hit@5': 73.74, 'hit@20': 81.89, 'recall@20': 57.89, 'mrr': 60.37,
            'speed': '0.44 it/s', 'strength': 'Interpretable + graph-enhanced', 'weakness': 'Lowest Hit@1'
        }
    }
    
    print(f"{'Pipeline':<8} {'Hit@1':<8} {'Hit@5':<8} {'Hit@20':<8} {'Recall@20':<10} {'MRR':<8} {'Speed':<12}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"{name:<8} {metrics['hit@1']:<7.1f}% {metrics['hit@5']:<7.1f}% "
              f"{metrics['hit@20']:<7.1f}% {metrics['recall@20']:<9.1f}% "
              f"{metrics['mrr']:<7.1f}% {metrics['speed']:<12}")
    
    print("\n💡 Key Insights:")
    for name, metrics in results.items():
        print(f"• {name}: {metrics['strength']} ({metrics['weakness']})")
    
    print()
    
    # Speed comparison
    print("⚡ Speed Comparison:")
    print("• FRMR is 189x faster than FRWSR")
    print("• FRMR is 4.3x faster than BARMR")  
    print("• FRWSR trades speed for 3.5% higher Hit@1 vs FRMR")
    print("• BARMR provides interpretability with competitive accuracy")
    print()


def configuration_example():
    """Show how to load and use configuration files."""
    
    print("=== Configuration File Usage ===\n")
    
    config_files = [
        'configs/frwsr_config.yaml',
        'configs/frmr_config.yaml', 
        'configs/barmr_config.yaml'
    ]
    
    for config_file in config_files:
        config_path = Path(config_file)
        
        if config_path.exists():
            print(f"Loading {config_file}:")
            
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                pipeline_name = config['pipeline']['name']
                print(f"✅ {pipeline_name}: {config['pipeline']['description']}")
                
                # Show key settings
                if 'retriever' in config:
                    retriever_info = config['retriever']
                    if 'model_name' in retriever_info:
                        print(f"   Retriever: {retriever_info['model_name']}")
                    if 'index_type' in retriever_info:
                        print(f"   Index: {retriever_info['index_type']}")
                
                if 'reranker' in config:
                    reranker_info = config['reranker']
                    if 'model_name' in reranker_info:
                        print(f"   Reranker: {reranker_info['model_name']}")
                
            except Exception as e:
                print(f"❌ Error loading {config_file}: {e}")
        else:
            print(f"❌ {config_file} not found")
        
        print()


def evaluation_example():
    """Show how to run evaluation on sample data."""
    
    print("=== Evaluation Example ===\n")
    
    # Sample query data for demonstration
    sample_queries = [
        {
            'query_id': 1,
            'query': 'wireless bluetooth headphones with noise cancellation',
            'correct_answer': [12345, 67890, 11111]
        },
        {
            'query_id': 2, 
            'query': 'waterproof outdoor camping tent for 2 people',
            'correct_answer': [22222, 33333]
        },
        {
            'query_id': 3,
            'query': 'stainless steel kitchen knife set',
            'correct_answer': [44444, 55555, 66666, 77777]
        }
    ]
    
    print(f"Sample evaluation with {len(sample_queries)} queries:")
    print()
    
    for query_data in sample_queries:
        print(f"Query {query_data['query_id']}: {query_data['query']}")
        print(f"Ground truth: {len(query_data['correct_answer'])} relevant items")
        print()
    
    # Show how to compute metrics
    try:
        from evaluation.metrics import compute_metrics
        
        # Simulate pipeline results
        predicted_results = [12345, 99999, 67890, 88888, 11111]
        ground_truth = [12345, 67890, 11111]
        
        metrics = compute_metrics(predicted_results, ground_truth)
        
        print("Example metrics calculation:")
        print(f"✅ Hit@1: {metrics['hit@1']}")
        print(f"✅ Hit@5: {metrics['hit@5']}")
        print(f"✅ Recall@20: {metrics['recall@20']:.3f}")
        print(f"✅ MRR: {metrics['MRR']:.3f}")
        
    except Exception as e:
        print(f"❌ Metrics calculation error: {e}")
    
    print()


def performance_comparison():
    """Show expected performance characteristics of different pipelines."""
    
    print("=== Performance Comparison ===\n")
    
    pipelines = {
        'FRWSR': {
            'description': 'FAISS + Webis Set-Encoder',
            'hit@1': 54.75,
            'hit@5': 75.25,
            'mrr': 0.6403,
            'speed': 'Slow (0.01 it/s)',
            'use_case': 'Maximum accuracy applications'
        },
        'FRMR': {
            'description': 'FAISS + MS MARCO',
            'hit@1': 51.25,
            'hit@5': 73.56,
            'mrr': 0.6128,
            'speed': 'Fast (1.89 it/s)', 
            'use_case': 'Production systems'
        },
        'BARMR': {
            'description': 'BM25 + Graph Augmentation',
            'hit@1': 49.67,
            'hit@5': 73.74,
            'mrr': 0.6037,
            'speed': 'Medium (0.44 it/s)',
            'use_case': 'Interpretable retrieval'
        }
    }
    
    print(f"{'Pipeline':<8} {'Hit@1':<8} {'Hit@5':<8} {'MRR':<8} {'Speed':<16} {'Best For'}")
    print("-" * 75)
    
    for name, stats in pipelines.items():
        print(f"{name:<8} {stats['hit@1']:<7.1f}% {stats['hit@5']:<7.1f}% "
              f"{stats['mrr']:<8.3f} {stats['speed']:<16} {stats['use_case']}")
    
    print()
    print("Key Insights:")
    print("• FRWSR: Best accuracy with set-wise cross-encoder")
    print("• FRMR: Best speed/accuracy tradeoff for production")  
    print("• BARMR: Graph relationships enhance sparse retrieval")
    print()


def data_setup_guide():
    """Show what data files are needed and how to set them up."""
    
    print("=== Data Setup Guide ===\n")
    
    required_files = {
        'Node Data': {
            'path': 'data/nodes/amazon_stark_nodes_processed.csv',
            'description': 'Processed Amazon product nodes with combined text',
            'size': '~1M nodes (~3GB)'
        },
        'Query Data': {
            'path': 'data/queries/',
            'description': 'Query datasets for evaluation',
            'files': [
                'validation_queries.csv (910 queries)',
                'evaluation_queries_filtered.csv (6,380 queries)', 
                'evaluation_queries_full.csv (9,100 queries)'
            ]
        },
        'FAISS Index': {
            'path': 'data/indices/faiss_e5_large_hnsw/',
            'description': 'Pre-built FAISS index for fast startup',
            'size': '~4GB'
        }
    }
    
    print("Required data files:")
    print()
    
    for category, info in required_files.items():
        print(f"📁 {category}")
        print(f"   Path: {info['path']}")
        print(f"   Description: {info['description']}")
        
        if 'size' in info:
            print(f"   Size: {info['size']}")
        
        if 'files' in info:
            for file_info in info['files']:
                print(f"   • {file_info}")
        
        # Check if files exist
        path = Path(info['path'])
        if path.exists():
            print("   ✅ Available")
        else:
            print("   ❌ Missing")
        
        print()
    
    print("Setup commands:")
    print("1. Download data: python scripts/download_data.py")
    print("2. Build indices: python scripts/build_indices.py --data-file data/nodes/amazon_stark_nodes_processed.csv")
    print("3. Run evaluation: python scripts/run_evaluation.py --config configs/frwsr_config.yaml")
    print()


def troubleshooting_guide():
    """Common issues and solutions."""
    
    print("=== Troubleshooting Guide ===\n")
    
    issues = [
        {
            'issue': 'ModuleNotFoundError: No module named sentence_transformers',
            'solution': 'pip install sentence-transformers'
        },
        {
            'issue': 'ModuleNotFoundError: No module named lightning_ir',
            'solution': 'pip install lightning-ir  # Required for Webis rerankers'
        },
        {
            'issue': 'CUDA out of memory',
            'solution': 'Reduce batch_size in config or use CPU: device="cpu"'
        },
        {
            'issue': 'FileNotFoundError: Node data not found',
            'solution': 'Run: python scripts/download_data.py --components nodes'
        },
        {
            'issue': 'Index loading fails',
            'solution': 'Rebuild indices: python scripts/build_indices.py'
        },
        {
            'issue': 'Slow performance',
            'solution': 'Use FRMR pipeline or reduce retrieve_k/rerank_k values'
        }
    ]
    
    for i, item in enumerate(issues, 1):
        print(f"{i}. {item['issue']}")
        print(f"   Solution: {item['solution']}")
        print()


def main():
    """Run all examples."""
    
    print("🚀 Neural Retriever-Reranker RAG Pipeline Examples")
    print("=" * 60)
    print()
    
    # Run example sections
    try:
        basic_pipeline_usage()
        print()
        
        configuration_example()
        print()
        
        evaluation_example()
        print()
        
        actual_implementation_details()
        print()
        
        hyperparameter_details()
        print()
        
        performance_analysis()
        print()
        
        data_setup_guide()
        print()
        
        troubleshooting_guide()
        
    except KeyboardInterrupt:
        print("\nExamples interrupted by user.")
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("This is normal if data files are not set up yet.")
        print("Follow the data setup guide above to get started.")
    
    print("🎉 Examples complete!")
    print()
    print("Next steps:")
    print("1. Set up data: python scripts/download_data.py")
    print("2. Try evaluation: python scripts/run_evaluation.py --help")
    print("3. Read the full documentation in README.md")


if __name__ == "__main__":
    main()
    
    # Example 2: FRMR Pipeline (FAISS + MS MARCO)
    print("2. FRMR Pipeline (Best Speed/Accuracy Tradeoff)")
    print("-" * 50)
    
    try:
        frmr_config = base_config.copy()
        frmr_config['pipeline'] = {'name': 'FRMR'}
        frmr_config['reranker']['model_name'] = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        
        frmr_pipeline = FRMRPipeline(frmr_config)
        
        # Demo without actual data loading
        info = frmr_pipeline.get_pipeline_info()
        print(f"✅ FRMR Pipeline initialized")
        print(f"   Use cases: {', '.join(info['use_cases'])}")
        
    except Exception as e:
        print(f"❌ FRMR Error: {e}")
    
    print()