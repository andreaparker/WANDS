#!/usr/bin/env python
"""
Evaluation module for WANDS benchmark.

This module calculates evaluation metrics and generates visualizations.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from utils import setup_logging, create_directory, load_json, save_json, load_env_variables

logger = setup_logging(__name__)

def calculate_ndcg(
    ranking: List[str], 
    relevance: Dict[str, float], 
    k: int
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain.
    
    Args:
        ranking: List of item IDs
        relevance: Dictionary of item ID to relevance score
        k: Number of items to consider
        
    Returns:
        nDCG@k score
    """
    ranking = ranking[:k]
    
    dcg = 0
    for i, item_id in enumerate(ranking):
        rel = relevance.get(item_id, 0)
        dcg += (2 ** rel - 1) / np.log2(i + 2)
        
    ideal_ranking = sorted(relevance.keys(), key=lambda x: relevance[x], reverse=True)[:k]
    idcg = 0
    for i, item_id in enumerate(ideal_ranking):
        rel = relevance[item_id]
        idcg += (2 ** rel - 1) / np.log2(i + 2)
        
    if idcg == 0:
        return 0
    return dcg / idcg

def calculate_mrr(
    ranking: List[str], 
    relevance: Dict[str, float]
) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        ranking: List of item IDs
        relevance: Dictionary of item ID to relevance score
        
    Returns:
        MRR score
    """
    for i, item_id in enumerate(ranking):
        if relevance.get(item_id, 0) > 0:
            return 1 / (i + 1)
            
    return 0

def calculate_precision_recall(
    ranking: List[str], 
    relevance: Dict[str, float], 
    k: int
) -> Tuple[float, float]:
    """
    Calculate Precision@k and Recall@k.
    
    Args:
        ranking: List of item IDs
        relevance: Dictionary of item ID to relevance score
        k: Number of items to consider
        
    Returns:
        Tuple of (Precision@k, Recall@k)
    """
    ranking = ranking[:k]
    
    relevant_retrieved = sum(1 for item_id in ranking if relevance.get(item_id, 0) > 0)
    total_relevant = sum(1 for rel in relevance.values() if rel > 0)
    
    precision = relevant_retrieved / k if k > 0 else 0
    recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
    
    return precision, recall

def evaluate_queries(
    queries: List[Dict[str, Any]], 
    k_values: List[int] = [3, 5, 10]
) -> Dict[str, Any]:
    """
    Evaluate queries using various metrics.
    
    Args:
        queries: List of query dictionaries
        k_values: List of k values for metrics
        
    Returns:
        Dictionary of evaluation results
    """
    results = {
        "overall": {
            f"ndcg@{k}": 0 for k in k_values
        },
        "by_query_type": {},
        "by_space_type": {},
        "confidence": {
            "avg": 0,
            "by_query_type": {}
        }
    }
    
    results["overall"]["mrr"] = 0
    for k in k_values:
        results["overall"][f"precision@{k}"] = 0
        results["overall"][f"recall@{k}"] = 0
    
    valid_queries = 0
    total_confidence = 0
    
    query_types = set()
    query_type_counts = {}
    query_type_confidence = {}
    
    for query in queries:
        if not query.get("gold_standard_ranking"):
            continue
            
        query_type = query.get("query_type", "unknown")
        query_types.add(query_type)
        
        query_type_counts[query_type] = query_type_counts.get(query_type, 0) + 1
        
        confidence = query.get("relevance_confidence_avg", 0)
        total_confidence += confidence
        
        if query_type not in query_type_confidence:
            query_type_confidence[query_type] = []
        query_type_confidence[query_type].append(confidence)
        
        relevance = {
            c["product_id"]: c["similarity_score"]
            for c in query.get("stage1_candidates", [])
        }
        
        ranking = query["gold_standard_ranking"]
        
        for k in k_values:
            ndcg = calculate_ndcg(ranking, relevance, k)
            results["overall"][f"ndcg@{k}"] += ndcg
            
        mrr = calculate_mrr(ranking, relevance)
        results["overall"]["mrr"] += mrr
        
        for k in k_values:
            precision, recall = calculate_precision_recall(ranking, relevance, k)
            results["overall"][f"precision@{k}"] += precision
            results["overall"][f"recall@{k}"] += recall
            
        valid_queries += 1
    
    if valid_queries > 0:
        for metric in results["overall"]:
            results["overall"][metric] /= valid_queries
            
        results["confidence"]["avg"] = total_confidence / valid_queries
    
    for query_type in query_types:
        results["by_query_type"][query_type] = {
            f"ndcg@{k}": 0 for k in k_values
        }
        results["by_query_type"][query_type]["mrr"] = 0
        results["by_query_type"][query_type]["count"] = query_type_counts[query_type]
        
        for k in k_values:
            results["by_query_type"][query_type][f"precision@{k}"] = 0
            results["by_query_type"][query_type][f"recall@{k}"] = 0
    
    for query_type, confidences in query_type_confidence.items():
        results["confidence"]["by_query_type"][query_type] = sum(confidences) / len(confidences)
    
    return results

def generate_visualizations(
    gpt4_results: Dict[str, Any],
    gpt35_results: Dict[str, Any],
    output_dir: Path
) -> None:
    """
    Generate visualizations for evaluation results.
    
    Args:
        gpt4_results: GPT-4 evaluation results
        gpt35_results: GPT-3.5 evaluation results
        output_dir: Output directory
    """
    vis_dir = output_dir / "visualizations"
    create_directory(vis_dir)
    
    k_values = [3, 5, 10]
    gpt4_ndcg = [gpt4_results["overall"][f"ndcg@{k}"] for k in k_values]
    gpt35_ndcg = [gpt35_results["overall"][f"ndcg@{k}"] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(k_values))
    plt.bar(x - width/2, gpt4_ndcg, width, label="GPT-4")
    plt.bar(x + width/2, gpt35_ndcg, width, label="GPT-3.5")
    plt.xlabel("k")
    plt.ylabel("nDCG@k")
    plt.title("nDCG@k Comparison")
    plt.xticks(x, [f"k={k}" for k in k_values])
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "ndcg_comparison.png")
    plt.close()
    
    query_types = list(gpt4_results["confidence"]["by_query_type"].keys())
    gpt4_conf = [gpt4_results["confidence"]["by_query_type"][qt] for qt in query_types]
    gpt35_conf = [gpt35_results["confidence"]["by_query_type"][qt] for qt in query_types]
    
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(query_types))
    plt.bar(x - width/2, gpt4_conf, width, label="GPT-4")
    plt.bar(x + width/2, gpt35_conf, width, label="GPT-3.5")
    plt.xlabel("Query Type")
    plt.ylabel("Average Confidence")
    plt.title("Confidence by Query Type")
    plt.xticks(x, query_types, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "confidence_by_query_type.png")
    plt.close()
    
    metrics = ["mrr", "precision@5", "recall@5", "ndcg@5"]
    gpt4_metrics = [gpt4_results["overall"][m] for m in metrics]
    gpt35_metrics = [gpt35_results["overall"][m] for m in metrics]
    
    plt.figure(figsize=(10, 6))
    width = 0.35
    x = np.arange(len(metrics))
    plt.bar(x - width/2, gpt4_metrics, width, label="GPT-4")
    plt.bar(x + width/2, gpt35_metrics, width, label="GPT-3.5")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.title("Overall Metrics Comparison")
    plt.xticks(x, metrics, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(vis_dir / "overall_metrics_comparison.png")
    plt.close()
    
    summary = {
        "Metric": ["nDCG@3", "nDCG@5", "nDCG@10", "MRR", "Precision@5", "Recall@5", "Avg Confidence"],
        "GPT-4": [
            gpt4_results["overall"]["ndcg@3"],
            gpt4_results["overall"]["ndcg@5"],
            gpt4_results["overall"]["ndcg@10"],
            gpt4_results["overall"]["mrr"],
            gpt4_results["overall"]["precision@5"],
            gpt4_results["overall"]["recall@5"],
            gpt4_results["confidence"]["avg"]
        ],
        "GPT-3.5": [
            gpt35_results["overall"]["ndcg@3"],
            gpt35_results["overall"]["ndcg@5"],
            gpt35_results["overall"]["ndcg@10"],
            gpt35_results["overall"]["mrr"],
            gpt35_results["overall"]["precision@5"],
            gpt35_results["overall"]["recall@5"],
            gpt35_results["confidence"]["avg"]
        ]
    }
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(vis_dir / "summary_metrics.csv", index=False)
    
    fig = go.Figure(data=[
        go.Bar(name="GPT-4", x=summary["Metric"], y=summary["GPT-4"]),
        go.Bar(name="GPT-3.5", x=summary["Metric"], y=summary["GPT-3.5"])
    ])
    fig.update_layout(
        title="GPT-4 vs GPT-3.5 Metrics Comparison",
        xaxis_title="Metric",
        yaxis_title="Score",
        barmode="group"
    )
    fig.write_html(vis_dir / "metrics_comparison.html")

def main():
    """Main function to run the evaluation."""
    ranking_dir = Path("../ranking_results")
    output_dir = Path("..")
    
    env_vars = load_env_variables()
    k_values = [int(k) for k in env_vars["METRICS_K_VALUES"].split(",")]
    
    gpt4_queries = load_json(ranking_dir / "gpt4_ranking_results.json")["queries"]
    gpt35_queries = load_json(ranking_dir / "gpt35_ranking_results.json")["queries"]
    
    gpt4_results = evaluate_queries(gpt4_queries, k_values)
    gpt35_results = evaluate_queries(gpt35_queries, k_values)
    
    save_json(gpt4_results, ranking_dir / "gpt4_evaluation_results.json")
    save_json(gpt35_results, ranking_dir / "gpt35_evaluation_results.json")
    
    generate_visualizations(gpt4_results, gpt35_results, output_dir)
    
    logger.info("Evaluation complete")
    logger.info(f"GPT-4 nDCG@5: {gpt4_results['overall']['ndcg@5']:.4f}")
    logger.info(f"GPT-3.5 nDCG@5: {gpt35_results['overall']['ndcg@5']:.4f}")
    logger.info(f"GPT-4 MRR: {gpt4_results['overall']['mrr']:.4f}")
    logger.info(f"GPT-3.5 MRR: {gpt35_results['overall']['mrr']:.4f}")
    
if __name__ == "__main__":
    main()
