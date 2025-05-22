#!/usr/bin/env python
"""
Query generation module for WANDS benchmark.

This module generates synthetic queries from product attributes.
"""
import os
import json
import random
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from utils import setup_logging, create_directory, load_json, save_json

logger = setup_logging(__name__)

QUERY_TYPES = {
    "mini": {
        "template": "{category}",
        "description": "Short, simple phrases without numbers",
        "examples": ["area rugs for living room", "vintage-style furniture", "dog clothes"]
    },
    "regular": {
        "template": "{attribute} {category} {optional_filter}",
        "description": "More complex, structured queries",
        "examples": ["gas fireplace under $300", "leather sofa with recliners", "queen size platform bed with storage"]
    },
    "advanced": {
        "template": "I'm looking for {attribute_1} {category}, {attribute_2}, {optional_filter}.",
        "description": "Natural language, complex conditions",
        "examples": [
            "I'm looking for a modern patio stool, preferably backless, in black.",
            "Do you have faux floral orchid arrangements suitable for residential decor?",
            "Can you recommend a sturdy farmhouse style TV stand for a 65-inch TV?"
        ]
    }
}

ADJECTIVE_ORDER = [
    "opinion",  # beautiful, ugly, delicious
    "size",     # big, small, tall, tiny
    "age",      # young, old, ancient, new
    "shape",    # round, square, flat, circular
    "color",    # red, blue, green, purple
    "origin",   # American, French, Asian
    "material", # wooden, metal, cotton, silk
    "purpose"   # dining table, running shoes
]

def extract_attributes(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Extract unique attributes from product dataset.

    Args:
        df: Product DataFrame

    Returns:
        Dictionary of attribute categories to attribute values
    """
    attributes = {
        "color": set(),
        "material": set(),
        "style": set(),
        "category": set(),
        "shape": set(),
        "pattern": set(),
        "finish": set()
    }

    for _, row in df.iterrows():
        features = row['standardized_features_map']
        if not isinstance(features, dict):
            try:
                if isinstance(features, str):
                    features = json.loads(features.replace("'", "\""))
                else:
                    continue
            except:
                continue

        if 'Color' in features:
            attributes['color'].add(features['Color'].lower())
        if 'Material' in features:
            attributes['material'].add(features['Material'].lower())
        if 'Style' in features:
            attributes['style'].add(features['Style'].lower())
        if 'Shape' in features:
            attributes['shape'].add(features['Shape'].lower())
        if 'Pattern' in features:
            attributes['pattern'].add(features['Pattern'].lower())
        if 'Finish' in features:
            attributes['finish'].add(features['Finish'].lower())

        if pd.notna(row['product_class']):
            attributes['category'].add(row['product_class'].lower())

    return {k: list(v) for k, v in attributes.items()}

def generate_price_filter() -> str:
    """
    Generate a price filter string.

    Returns:
        Price filter string
    """
    price_templates = [
        "under ${price}",
        "less than ${price}",
        "${price} or less",
        "within ${price} budget",
        "cheaper than ${price}",
        "not more than ${price}"
    ]

    price_points = [50, 100, 150, 200, 250, 300, 500, 1000]
    template = random.choice(price_templates)
    price = random.choice(price_points)

    return template.replace("{price}", str(price))

def generate_negation_filter(attributes: Dict[str, List[str]]) -> str:
    """
    Generate a negation filter string.

    Args:
        attributes: Dictionary of attributes

    Returns:
        Negation filter string
    """
    negation_templates = [
        "without {attribute}",
        "not {attribute}",
        "no {attribute}",
        "excluding {attribute}"
    ]

    attribute_types = ["color", "material", "style", "pattern", "finish"]
    attribute_type = random.choice(attribute_types)

    if not attributes[attribute_type]:
        return ""

    attribute_value = random.choice(attributes[attribute_type])
    template = random.choice(negation_templates)

    return template.replace("{attribute}", attribute_value)

def generate_query_text(
    query_type: str,
    attributes: Dict[str, List[str]],
    include_price: bool = False,
    include_negation: bool = False
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a query text and its parameters.

    Args:
        query_type: Type of query to generate (mini, regular, advanced)
        attributes: Dictionary of attributes
        include_price: Whether to include price filter
        include_negation: Whether to include negation

    Returns:
        Tuple of (query text, query parameters)
    """
    category = random.choice(attributes['category']) if attributes['category'] else "furniture"
    color = random.choice(attributes['color']) if attributes['color'] and random.random() > 0.5 else ""
    material = random.choice(attributes['material']) if attributes['material'] and random.random() > 0.5 else ""
    style = random.choice(attributes['style']) if attributes['style'] and random.random() > 0.5 else ""

    attribute_parts = []
    if style:
        attribute_parts.append(style)
    if color:
        attribute_parts.append(color)
    if material:
        attribute_parts.append(material)

    attribute_phrase = " ".join(attribute_parts)

    filters = []
    if include_price:
        filters.append(generate_price_filter())

    if include_negation:
        filters.append(generate_negation_filter(attributes))

    filter_phrase = " ".join(filters)

    query_text = ""
    if query_type == "mini":
        query_text = f"{category}"
        if attribute_phrase:
            query_text = f"{attribute_phrase} {query_text}"

    elif query_type == "regular":
        query_text = f"{attribute_phrase} {category}"
        if filter_phrase:
            query_text = f"{query_text} {filter_phrase}"

    elif query_type == "advanced":
        query_text = f"I'm looking for {attribute_phrase} {category}"
        if filter_phrase:
            query_text = f"{query_text}, {filter_phrase}"
        query_text = f"{query_text}."

    query_text = " ".join(query_text.split())

    query_params = {
        "category": category,
        "attributes": {
            "color": color,
            "material": material,
            "style": style
        }
    }

    return query_text, query_params

def analyze_query_specifics(
    query_text: str,
    query_params: Dict[str, Any],
    include_price: bool,
    include_negation: bool
) -> Dict[str, Any]:
    """
    Analyze query and determine space types, negation, and hard filters.

    Args:
        query_text: Query text
        query_params: Query parameters
        include_price: Whether price filter was included
        include_negation: Whether negation was included

    Returns:
        Query specifics dictionary
    """
    space_types = {
        "TextSimilaritySpace": True,
        "NumberSpace": False,
        "CategoricalSimilaritySpace": False
    }

    number_patterns = [
        r"\$\d+", r"\d+\s*dollars",  # Prices
        r"\d+(\.\d+)?\s*(stars?|ratings?)",  # Ratings
        r"\d+(\.\d+)?\s*(inch|inches|in|feet|ft|cm|m|meter)",  # Dimensions
        r"\d+(\.\d+)?\s*(lbs?|pounds|kg|kilograms)"  # Weights
    ]

    if include_price or any(re.search(pattern, query_text, re.IGNORECASE) for pattern in number_patterns):
        space_types["NumberSpace"] = True

    if query_params["attributes"]["color"] or query_params["attributes"]["material"] or query_params["attributes"]["style"]:
        space_types["CategoricalSimilaritySpace"] = True

    query_specifics = {
        "space_types": space_types,
        "has_negation": include_negation,
        "has_hard_filters": "must" in query_text.lower() or "only" in query_text.lower() or "exactly" in query_text.lower()
    }

    return query_specifics

def generate_queries(
    df: pd.DataFrame,
    num_mini: int = 50,
    num_regular: int = 100,
    num_advanced: int = 50
) -> List[Dict[str, Any]]:
    """
    Generate synthetic queries from product attributes.

    Args:
        df: Product DataFrame
        num_mini: Number of mini queries to generate
        num_regular: Number of regular queries to generate
        num_advanced: Number of advanced queries to generate

    Returns:
        List of query dictionaries
    """
    attributes = extract_attributes(df)

    queries = []

    for i in range(num_mini):
        include_price = False
        include_negation = False

        query_text, query_params = generate_query_text(
            "mini", attributes, include_price, include_negation
        )

        query_specifics = analyze_query_specifics(
            query_text, query_params, include_price, include_negation
        )

        query = {
            "query_id": f"mini-{i:03d}",
            "query_text": query_text,
            "query_params": query_params,
            "query_type": "basic_category",
            "num_results": 0,  # Will be filled after retrieval
            "sl_specifics": query_specifics
        }

        queries.append(query)

    for i in range(num_regular):
        include_price = random.random() > 0.7
        include_negation = random.random() > 0.8

        query_text, query_params = generate_query_text(
            "regular", attributes, include_price, include_negation
        )

        query_specifics = analyze_query_specifics(
            query_text, query_params, include_price, include_negation
        )

        query = {
            "query_id": f"regular-{i:03d}",
            "query_text": query_text,
            "query_params": query_params,
            "query_type": "attribute_based",
            "num_results": 0,  # Will be filled after retrieval
            "sl_specifics": query_specifics
        }

        queries.append(query)

    for i in range(num_advanced):
        include_price = random.random() > 0.5
        include_negation = random.random() > 0.7

        query_text, query_params = generate_query_text(
            "advanced", attributes, include_price, include_negation
        )

        query_specifics = analyze_query_specifics(
            query_text, query_params, include_price, include_negation
        )

        query = {
            "query_id": f"advanced-{i:03d}",
            "query_text": query_text,
            "query_params": query_params,
            "query_type": "nlq",
            "num_results": 0,  # Will be filled after retrieval
            "sl_specifics": query_specifics
        }

        queries.append(query)

    return queries

def main():
    """Main function to run the query generation."""
    input_dir = Path("../data/processed")
    output_dir = Path("../queries/generated")
    create_directory(output_dir)

    products_df = pd.read_csv(input_dir / "processed_products.csv")

    queries = generate_queries(products_df)

    save_json({"queries": queries}, output_dir / "generated_queries.json")
    logger.info(f"Generated {len(queries)} queries")

if __name__ == "__main__":
    main()
