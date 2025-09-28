"""
Evaluation Metrics Module

This module provides standard information retrieval evaluation metrics
including Hit@k, Recall@k, and Mean Reciprocal Rank (MRR).
"""

from typing import List, Dict, Any, Tuple
import numpy as np


def compute_metrics(retrieved: List[int], ground_truth: List[int]) -> Dict[str, float]:
    """
    Compute standard IR metrics for a single query.
    
    Args:
        retrieved: List of retrieved document IDs (ranked order)
        ground_truth: List of relevant document IDs
        
    Returns:
        Dictionary containing metric values
    """
    if not retrieved:
        return {
            'hit@1': 0, 'hit@5': 0, 'hit@10': 0, 'hit@20': 0,
            'hit@30': 0, 'hit@50': 0, 'hit@75': 0, 'hit@100': 0,
            'recall@20': 0, 'recall@30': 0, 'recall@50': 0,
            'recall@75': 0, 'recall@100': 0, 'MRR': 0
        }
    
    ground_truth_set = set(ground_truth)
    
    # Compute Hit@k metrics
    hit_at_k = {}
    for k in [1, 5, 10, 20, 30, 50, 75, 100]:
        hit_at_k[f'hit@{k}'] = int(any(
            r in ground_truth_set for r in retrieved[:k]
        ))
    
    # Compute Recall@k metrics
    recall_at_k = {}
    if len(ground_truth) > 0:  # Avoid division by zero
        for k in [20, 30, 50, 75, 100]:
            relevant_in_k = sum(1 for r in retrieved[:k] if r in ground_truth_set)
            recall_at_k[f'recall@{k}'] = relevant_in_k / len(ground_truth_set)
    else:
        for k in [20, 30, 50, 75, 100]:
            recall_at_k[f'recall@{k}'] = 0
    
    # Compute Mean Reciprocal Rank (MRR)
    mrr = 0
    for i, r in enumerate(retrieved):
        if r in ground_truth_set:
            mrr = 1.0 / (i + 1)
            break
    
    # Combine all metrics
    metrics = {**hit_at_k, **recall_at_k, 'MRR': mrr}
    
    return metrics


def compute_metrics_batch(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Compute averaged metrics over a batch of queries.
    
    Args:
        results: List of result dictionaries, each containing 'predicted' and 'ground_truth'
        
    Returns:
        Dictionary of averaged metric values
    """
    if not results:
        return {}
    
    # Compute metrics for each query
    all_metrics = []
    for result in results:
        predicted = result['predicted']
        ground_truth = result['ground_truth']
        metrics = compute_metrics(predicted, ground_truth)
        all_metrics.append(metrics)
    
    # Average metrics across all queries
    metric_names = all_metrics[0].keys()
    averaged_metrics = {}
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics]
        averaged_metrics[metric_name] = np.mean(values)
    
    return averaged_metrics


def hit_at_k(retrieved: List[int], ground_truth: List[int], k: int) -> int:
    """
    Compute Hit@k metric.
    
    Args:
        retrieved: List of retrieved document IDs
        ground_truth: List of relevant document IDs
        k: Cutoff value
        
    Returns:
        1 if any relevant document is in top-k, 0 otherwise
    """
    if not retrieved:
        return 0
    
    ground_truth_set = set(ground_truth)
    return int(any(r in ground_truth_set for r in retrieved[:k]))


def recall_at_k(retrieved: List[int], ground_truth: List[int], k: int) -> float:
    """
    Compute Recall@k metric.
    
    Args:
        retrieved: List of retrieved document IDs
        ground_truth: List of relevant document IDs
        k: Cutoff value
        
    Returns:
        Proportion of relevant documents found in top-k
    """
    if not retrieved or not ground_truth:
        return 0.0
    
    ground_truth_set = set(ground_truth)
    relevant_in_k = sum(1 for r in retrieved[:k] if r in ground_truth_set)
    
    return relevant_in_k / len(ground_truth_set)


def mean_reciprocal_rank(retrieved: List[int], ground_truth: List[int]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    Args:
        retrieved: List of retrieved document IDs
        ground_truth: List of relevant document IDs
        
    Returns:
        Reciprocal rank of first relevant document, or 0 if none found
    """
    if not retrieved:
        return 0.0
    
    ground_truth_set = set(ground_truth)
    
    for i, r in enumerate(retrieved):
        if r in ground_truth_set:
            return 1.0 / (i + 1)
    
    return 0.0


def precision_at_k(retrieved: List[int], ground_truth: List[int], k: int) -> float:
    """
    Compute Precision@k metric.
    
    Args:
        retrieved: List of retrieved document IDs
        ground_truth: List of relevant document IDs
        k: Cutoff value
        
    Returns:
        Proportion of top-k retrieved documents that are relevant
    """
    if not retrieved:
        return 0.0
    
    ground_truth_set = set(ground_truth)
    relevant_in_k = sum(1 for r in retrieved[:k] if r in ground_truth_set)
    
    return relevant_in_k / min(k, len(retrieved))


def ndcg_at_k(retrieved: List[int], ground_truth: List[int], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG@k).
    
    Args:
        retrieved: List of retrieved document IDs
        ground_truth: List of relevant document IDs
        k: Cutoff value
        
    Returns:
        NDCG@k score
    """
    if not retrieved or not ground_truth:
        return 0.0
    
    ground_truth_set = set(ground_truth)
    
    # Compute DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in ground_truth_set:
            dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0
    
    # Compute IDCG@k (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(ground_truth))))
    
    # Return NDCG@k
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ranking_quality(
    results: List[Dict[str, Any]], 
    metrics: List[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation of ranking quality with multiple metrics.
    
    Args:
        results: List of result dictionaries
        metrics: List of metrics to compute (if None, computes all standard metrics)
        
    Returns:
        Dictionary with individual and averaged metrics
    """
    if metrics is None:
        metrics = ['hit@1', 'hit@5', 'hit@10', 'hit@20', 'recall@20', 'MRR']
    
    # Compute metrics for each query
    individual_results = []
    for result in results:
        predicted = result['predicted']
        ground_truth = result['ground_truth']
        query_metrics = compute_metrics(predicted, ground_truth)
        individual_results.append(query_metrics)
    
    # Compute averages
    averaged_metrics = {}
    for metric in metrics:
        if metric in individual_results[0]:
            values = [r[metric] for r in individual_results]
            averaged_metrics[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    return {
        'individual': individual_results,
        'averaged': averaged_metrics
    }