#!/usr/bin/env python
"""
LLM ranking module for WANDS benchmark.

This module implements pairwise ranking using GPT-4 and GPT-3.5.
"""
import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
from openai import OpenAI
from utils import setup_logging, create_directory, load_json, save_json, load_env_variables

logger = setup_logging(__name__)

class LLMRanker:
    """LLM ranker for pairwise ranking."""

    def __init__(self, model_name: str, api_key: str):
        """
        Initialize LLM ranker.

        Args:
            model_name: Name of the LLM model
            api_key: OpenAI API key
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def _get_product_text(self, product: Dict[str, Any]) -> str:
        """
        Get product text for ranking.

        Args:
            product: Product dictionary

        Returns:
            Product text
        """
        return (
            f"Product ID: {product['product_id']}\n"
            f"Name: {product['product_name']}\n"
            f"Category: {product['product_class']}\n"
            f"Description: {product['product_description']}\n"
            f"Features: {product['processed_clean_features_str']}\n"
            f"Price: ${product['price']}"
        )

    def _create_pairwise_prompt(
        self,
        query: str,
        product_a: Dict[str, Any],
        product_b: Dict[str, Any]
    ) -> str:
        """
        Create prompt for pairwise ranking.

        Args:
            query: Query text
            product_a: First product
            product_b: Second product

        Returns:
            Prompt text
        """
        return (
            f"You are a search relevance expert. Your task is to determine which product is more relevant to the given query.\n\n"
            f"Query: {query}\n\n"
            f"Product A:\n{self._get_product_text(product_a)}\n\n"
            f"Product B:\n{self._get_product_text(product_b)}\n\n"
            f"Please analyze both products and determine which one is more relevant to the query. "
            f"Consider factors such as product name, description, features, and price. "
            f"Respond with 'A' if Product A is more relevant, 'B' if Product B is more relevant, or 'tie' if they are equally relevant. "
            f"Also provide a confidence score from 0 to 10, where 0 means completely uncertain and 10 means absolutely certain. "
            f"Format your response as: 'Choice: [A/B/tie], Confidence: [0-10]'"
        )

    def compare_products(
        self,
        query: str,
        product_a: Dict[str, Any],
        product_b: Dict[str, Any]
    ) -> Tuple[str, int]:
        """
        Compare two products for relevance.

        Args:
            query: Query text
            product_a: First product
            product_b: Second product

        Returns:
            Tuple of (choice, confidence)
        """
        prompt = self._create_pairwise_prompt(query, product_a, product_b)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a search relevance expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=100
            )

            response_text = response.choices[0].message.content.strip()

            choice = "tie"
            confidence = 5

            if "Choice:" in response_text and "Confidence:" in response_text:
                choice_part = response_text.split("Choice:")[1].split(",")[0].strip().lower()
                confidence_part = response_text.split("Confidence:")[1].strip()

                if "a" in choice_part:
                    choice = "A"
                elif "b" in choice_part:
                    choice = "B"
                else:
                    choice = "tie"

                try:
                    confidence = int(confidence_part)
                except:
                    confidence = 5

            return choice, confidence

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return "tie", 5

    def rank_candidates(
        self,
        query: Dict[str, Any],
        products: Dict[str, Dict[str, Any]],
        max_comparisons: int = 100
    ) -> Tuple[List[str], List[Dict[str, Any]], float]:
        """
        Rank candidates using pairwise comparisons.

        Args:
            query: Query dictionary
            products: Dictionary of product dictionaries
            max_comparisons: Maximum number of comparisons

        Returns:
            Tuple of (ranked product IDs, pairwise comparisons, average confidence)
        """
        candidates = query["stage1_candidates"]
        candidate_ids = [c["product_id"] for c in candidates]

        scores = {pid: 0 for pid in candidate_ids}

        comparisons = []

        if len(candidate_ids) > 20:
            candidate_ids = candidate_ids[:20]

        num_comparisons = 0
        total_confidence = 0

        for i in range(len(candidate_ids)):
            for j in range(i + 1, len(candidate_ids)):
                if num_comparisons >= max_comparisons:
                    break

                product_a = products[candidate_ids[i]]
                product_b = products[candidate_ids[j]]

                choice, confidence = self.compare_products(query["query_text"], product_a, product_b)

                if choice == "A":
                    scores[candidate_ids[i]] += 1
                elif choice == "B":
                    scores[candidate_ids[j]] += 1
                else:  # tie
                    scores[candidate_ids[i]] += 0.5
                    scores[candidate_ids[j]] += 0.5

                comparisons.append({
                    "product_a_id": candidate_ids[i],
                    "product_b_id": candidate_ids[j],
                    "choice": choice,
                    "confidence": confidence
                })

                num_comparisons += 1
                total_confidence += confidence

                time.sleep(0.5)

        avg_confidence = total_confidence / num_comparisons if num_comparisons > 0 else 0

        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        return ranked_ids, comparisons, avg_confidence

def process_queries(
    ranker: LLMRanker,
    queries: List[Dict[str, Any]],
    products: Dict[str, Dict[str, Any]],
    max_comparisons: int = 100
) -> List[Dict[str, Any]]:
    """
    Process queries and rank candidates.

    Args:
        ranker: LLM ranker
        queries: List of query dictionaries
        products: Dictionary of product dictionaries
        max_comparisons: Maximum number of comparisons per query

    Returns:
        List of query dictionaries with rankings
    """
    logger.info(f"Processing {len(queries)} queries with {ranker.model_name}")

    for query in tqdm(queries):
        if not query.get("stage1_candidates"):
            logger.warning(f"No candidates for query {query['query_id']}")
            continue

        ranked_ids, comparisons, avg_confidence = ranker.rank_candidates(
            query, products, max_comparisons
        )

        query["gold_standard_ranking"] = ranked_ids
        query["pairwise_comparisons"] = comparisons
        query["relevance_confidence_avg"] = round(avg_confidence, 2)
        query["ranking_model"] = ranker.model_name

    return queries

def main():
    """Main function to run the ranking."""
    input_dir = Path("../data/processed")
    query_dir = Path("../queries/retrieval")
    output_dir = Path("../ranking_results")
    create_directory(output_dir)

    env_vars = load_env_variables()

    if not env_vars["OPENAI_API_KEY"]:
        logger.error("OpenAI API key not found. Please set OPENAI_API_KEY in .env file.")
        return

    products_df = pd.read_csv(input_dir / "processed_products.csv")

    products = {str(row["product_id"]): row.to_dict() for _, row in products_df.iterrows()}

    queries = load_json(query_dir / "retrieval_results.json")["queries"]

    gpt4_ranker = LLMRanker(model_name=env_vars["GPT4_MODEL"], api_key=env_vars["OPENAI_API_KEY"])
    gpt35_ranker = LLMRanker(model_name=env_vars["GPT35_MODEL"], api_key=env_vars["OPENAI_API_KEY"])

    gpt4_queries = process_queries(gpt4_ranker, queries.copy(), products)

    save_json({"queries": gpt4_queries}, output_dir / "gpt4_ranking_results.json")
    logger.info(f"Saved GPT-4 ranking results for {len(gpt4_queries)} queries")

    gpt35_queries = process_queries(gpt35_ranker, queries.copy(), products)

    save_json({"queries": gpt35_queries}, output_dir / "gpt35_ranking_results.json")
    logger.info(f"Saved GPT-3.5 ranking results for {len(gpt35_queries)} queries")

if __name__ == "__main__":
    main()
