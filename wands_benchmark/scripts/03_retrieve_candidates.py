#!/usr/bin/env python
"""
SPLADE retrieval module for WANDS benchmark.

This module implements sparse retrieval using SPLADE.
"""
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils import setup_logging, create_directory, load_json, save_json, load_env_variables

logger = setup_logging(__name__)

class SpladeRetriever:
    """SPLADE retriever for sparse vector search."""

    def __init__(self, query_model_name: str = "naver/efficient-splade-V-large-query",
                 doc_model_name: str = "naver/efficient-splade-V-large-doc"):
        """
        Initialize SPLADE retriever with separate query and document models.

        Args:
            query_model_name: Name of the SPLADE query model
            doc_model_name: Name of the SPLADE document model
        """
        self.query_model_name = query_model_name
        self.doc_model_name = doc_model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading SPLADE query model: {query_model_name}")
        self.query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        self.query_model = AutoModelForMaskedLM.from_pretrained(query_model_name).to(self.device)
        self.query_model.eval()

        # Load document model
        logger.info(f"Loading SPLADE document model: {doc_model_name}")
        self.doc_tokenizer = AutoTokenizer.from_pretrained(doc_model_name)
        self.doc_model = AutoModelForMaskedLM.from_pretrained(doc_model_name).to(self.device)
        self.doc_model.eval()

        self.doc_embeddings = {}
        self.doc_ids = []

    def _get_query_embedding(self, text: str) -> np.ndarray:
        """
        Get sparse embedding for query text using the query model.

        Args:
            text: Query text to embed

        Returns:
            Sparse embedding
        """
        inputs = self.query_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.query_model(**inputs)

        logits = outputs.logits
        relu_log = torch.log(1 + torch.relu(logits))

        sparse_vector = torch.max(relu_log, dim=1)[0].squeeze()

        return sparse_vector.cpu().numpy()

    def _get_doc_embedding(self, text: str) -> np.ndarray:
        """
        Get sparse embedding for document text using the document model.

        Args:
            text: Document text to embed

        Returns:
            Sparse embedding
        """
        inputs = self.doc_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.doc_model(**inputs)

        logits = outputs.logits
        relu_log = torch.log(1 + torch.relu(logits))

        sparse_vector = torch.max(relu_log, dim=1)[0].squeeze()

        return sparse_vector.cpu().numpy()

    def index_documents(self, documents: List[Dict[str, Any]], text_field: str = "searchable_text"):
        """
        Index documents for retrieval.

        Args:
            documents: List of document dictionaries
            text_field: Field to use for indexing
        """
        logger.info(f"Indexing {len(documents)} documents")

        self.doc_embeddings = {}
        self.doc_ids = []

        for doc in tqdm(documents):
            doc_id = str(doc["product_id"])
            text = doc[text_field]

            if doc_id in self.doc_embeddings:
                continue

            embedding = self._get_doc_embedding(text)

            self.doc_embeddings[doc_id] = embedding
            self.doc_ids.append(doc_id)

        logger.info(f"Indexed {len(self.doc_embeddings)} documents")

    def save_embeddings(self, file_path: str):
        """
        Save document embeddings to file.

        Args:
            file_path: Path to save embeddings
        """
        data = {
            "doc_ids": self.doc_ids,
            "embeddings": {doc_id: embedding.tolist() for doc_id, embedding in self.doc_embeddings.items()}
        }

        with open(file_path, "w") as f:
            json.dump(data, f)

        logger.info(f"Saved embeddings to {file_path}")

    def load_embeddings(self, file_path: str):
        """
        Load document embeddings from file.

        Args:
            file_path: Path to load embeddings from
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        self.doc_ids = data["doc_ids"]
        self.doc_embeddings = {doc_id: np.array(embedding) for doc_id, embedding in data["embeddings"].items()}

        logger.info(f"Loaded embeddings for {len(self.doc_embeddings)} documents")

    def retrieve(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieve documents for query.

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        query_embedding = self._get_query_embedding(query)

        scores = {}
        for doc_id in self.doc_ids:
            doc_embedding = self.doc_embeddings[doc_id]
            score = np.dot(query_embedding, doc_embedding)
            scores[doc_id] = float(score)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = [
            {"product_id": doc_id, "similarity_score": score}
            for doc_id, score in sorted_scores[:top_k]
        ]

        return results

def process_queries(
    retriever: SpladeRetriever,
    queries: List[Dict[str, Any]],
    top_k: int = 50
) -> List[Dict[str, Any]]:
    """
    Process queries and retrieve candidates.

    Args:
        retriever: SPLADE retriever
        queries: List of query dictionaries
        top_k: Number of candidates to retrieve

    Returns:
        List of query dictionaries with candidates
    """
    logger.info(f"Processing {len(queries)} queries")

    for query in tqdm(queries):
        query_text = query["query_text"]

        candidates = retriever.retrieve(query_text, top_k=top_k)

        query["stage1_candidates"] = candidates
        query["num_candidates"] = len(candidates)
        query["stage1_retriever_model"] = f"query:{retriever.query_model_name}, doc:{retriever.doc_model_name}"

    return queries

def main():
    """Main function to run the retrieval."""
    input_dir = Path("../data/processed")
    query_dir = Path("../queries/generated")
    output_dir = Path("../queries/retrieval")
    embedding_dir = Path("../embeddings/splade")
    create_directory(output_dir)
    create_directory(embedding_dir)

    env_vars = load_env_variables()

    products_df = pd.read_csv(input_dir / "processed_products.csv")

    products = products_df.to_dict(orient="records")

    queries = load_json(query_dir / "generated_queries.json")["queries"]

    retriever = SpladeRetriever(
        query_model_name=env_vars["SPLADE_QUERY_MODEL_NAME"],
        doc_model_name=env_vars["SPLADE_DOC_MODEL_NAME"]
    )

    embedding_file = embedding_dir / "product_embeddings.json"
    if embedding_file.exists():
        retriever.load_embeddings(embedding_file)
    else:
        retriever.index_documents(products)

        retriever.save_embeddings(embedding_file)

    processed_queries = process_queries(retriever, queries)

    save_json({"queries": processed_queries}, output_dir / "retrieval_results.json")
    logger.info(f"Saved retrieval results for {len(processed_queries)} queries")

if __name__ == "__main__":
    main()
