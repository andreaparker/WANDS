#!/usr/bin/env python
"""
Data preparation module for WANDS benchmark.

This module handles loading, cleaning, and preprocessing the WANDS dataset.
It implements SUPERKEY processing and price generation.
"""
import os
import json
import random
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import Counter
import wordsegment
import spacy
from utils import setup_logging, create_directory

wordsegment.load()
nlp = spacy.load('en_core_web_sm')

logger = setup_logging(__name__)

JUNK_VALUES = [
    "none", "n/a", "na", "not applicable", "does not apply", "other", "unknown",
    "unspecified", "no", "yes", "true", "false", "select", "choose", "varies"
]

JUNK_FEATURE_TERMS = [
    "yes", "no", "true", "false", "available", "not available", "included",
    "not included", "standard", "custom", "special", "regular", "default"
]

SUPERKEY_CONFIG = {
    "ProductType": ["product_type"],
    "Color": ["color", "ds_color", "primary_color", "ds_color_group"],
    "Material": ["primary_material", "material", "ds_material", "frame_material", "top_material_details"],
    "Style": ["ds_primary_product_style", "style", "ds_secondary_product_style"],
    "DSWoodTone": ["ds_wood_tone"],
    "Finish": ["finish"],
    "Upholstery": ["upholstered", "upholstery_material", "upholstery_fill_material"],
    "Pattern": ["pattern"],
    "Shape": ["shape"]
}

CORE_SUPERKEYS_ORDERED = [
    "ProductType", "Color", "Material", "Style", "DSWoodTone", "Finish", "Upholstery",
    "Pattern", "Shape", "OverallHeight", "OverallWidth", "OverallDepth", "WeightCapacity"
]

NUMERIC_PLACEHOLDER = "[NUM]"
KEY_SIMILARITY_THRESHOLD = 0.85  # Threshold for semantic similarity
MIN_KEY_FREQUENCY = 5  # Minimum frequency for key selection

PRICE_CEILING = {
    'Furniture / Living Room Furniture': 500.0,
    'Furniture / Kitchen & Dining Furniture': 300.0,
    'Furniture / Bedroom Furniture': 500.0,
    'Furniture / Office Furniture': 300.0,
    'Furniture / Small Spaces': 200.0,
    'Furniture / Entry & Mudroom Furniture': 300.0,
    'Furniture / Furniture Sale': 250.0,
    'Furniture / Game Tables & Game Room Furniture': 500.0,

    'Home Improvement / Bathroom Remodel & Bathroom Fixtures': 500.0,
    'Home Improvement / Flooring, Walls & Ceiling': 100.0,
    'Home Improvement / Hardware': 50.0,
    'Home Improvement / Kitchen Remodel & Kitchen Fixtures': 5000.0,
    'Home Improvement / Doors & Door Hardware': 150.0,

    'Outdoor / Outdoor & Patio Furniture': 500.0,
    'Outdoor / Garden': 150.0,
    'Outdoor / Outdoor Décor': 50.0,
    'Outdoor / Outdoor Shades': 150.0,
    'Outdoor / Outdoor Recreation': 150.0,
    'Outdoor / Hot Tubs & Saunas': 2000.0,
    'Outdoor / Outdoor Cooking & Tableware': 150.0,
    'Outdoor / Outdoor Fencing & Flooring': 200.0,
    'Outdoor / Outdoor Heating': 200.0,
    'Outdoor / Outdoor Sale': 150.0,

    'Bed & Bath / Bedding': 100.0,
    'Bed & Bath / Mattresses & Foundations': 500.0,
    'Bed & Bath / Bedding Essentials': 50.0,
    'Bed & Bath / Shower Curtains & Accessories': 30.0,
    'Bed & Bath / Bathroom Accessories & Organization': 30.0,
    'Bed & Bath / Bath Rugs & Towels': 50.0,
    'Bed & Bath / Dorm Décor & Back to School Essentials': 100.0,

    'Unknown / Unknown': 50.0
}

def load_dataset(file_path: str, sep: str = '\t') -> pd.DataFrame:
    """
    Load dataset from CSV file with appropriate separator.

    Args:
        file_path: Path to the CSV file
        sep: Separator used in the CSV file (default: tab)

    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path, sep=sep)
        logger.info(f"Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset from {file_path}: {e}")
        raise

def clean_text(text: str) -> str:
    """
    Clean text by removing special characters and normalizing whitespace.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""

    cleaned = text.lower().strip()
    return cleaned

def segment_key(key: str) -> str:
    """
    Segment concatenated words in key using wordsegment.

    Args:
        key: Key to segment

    Returns:
        Segmented key
    """
    if " " in key:
        return key

    segments = wordsegment.segment(key)
    return "_".join(segments)

def is_numeric_value(value: str) -> bool:
    """
    Check if a value is numeric.

    Args:
        value: Value to check

    Returns:
        True if value is numeric, False otherwise
    """
    if re.match(r'^\d+(\.\d+)?(\s*[a-zA-Z]+)?$', value):
        return True

    if re.match(r'^\d+(\.\d+)?\s+', value):
        return True

    return False

def calculate_key_similarity(key1: str, key2: str) -> float:
    """
    Calculate semantic similarity between two keys using spaCy.

    Args:
        key1: First key
        key2: Second key

    Returns:
        Similarity score between 0 and 1
    """
    key1 = key1.replace('_', ' ')
    key2 = key2.replace('_', ' ')

    doc1 = nlp(key1)
    doc2 = nlp(key2)

    return doc1.similarity(doc2)

def parse_product_features(features_str: str) -> Dict[str, str]:
    """
    Parse product features string into a dictionary.

    Args:
        features_str: String containing product features

    Returns:
        Dictionary of feature name to feature value
    """
    if not isinstance(features_str, str) or not features_str:
        return {}

    features = {}
    for feature in features_str.split('|'):
        feature = feature.strip()
        if not feature or ':' not in feature:
            continue

        parts = feature.split(':', 1)

        raw_key = parts[0].strip().lower().replace(' ', '_')
        key = segment_key(raw_key)

        value = parts[1].strip() if len(parts) > 1 else ""

        if value.lower() in JUNK_VALUES:
            continue

        if is_numeric_value(value):
            value = NUMERIC_PLACEHOLDER

        features[key] = value

    return features

def cluster_similar_keys(feature_keys: List[str]) -> Dict[str, List[str]]:
    """
    Cluster semantically similar keys.
    
    Args:
        feature_keys: List of feature keys
        
    Returns:
        Dictionary mapping representative keys to lists of similar keys
    """
    # Calculate key frequencies
    key_freq = Counter(feature_keys)
    
    sorted_keys = sorted(key_freq.keys(), key=lambda k: key_freq[k], reverse=True)
    
    frequent_keys = [k for k in sorted_keys if key_freq[k] >= MIN_KEY_FREQUENCY]
    
    clusters = {}
    assigned_keys = set()
    
    for key in frequent_keys:
        if key in assigned_keys:
            continue
            
        clusters[key] = [key]
        assigned_keys.add(key)
        
        for other_key in sorted_keys:
            if other_key in assigned_keys:
                continue
                
            # Check similarity
            similarity = calculate_key_similarity(key, other_key)
            if similarity >= KEY_SIMILARITY_THRESHOLD:
                clusters[key].append(other_key)
                assigned_keys.add(other_key)
    
    return clusters

def standardize_features(features: Dict[str, str], key_clusters: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Standardize features using clustered keys.
    
    Args:
        features: Dictionary of feature name to feature value
        key_clusters: Dictionary mapping representative keys to lists of similar keys
        
    Returns:
        Dictionary of standardized features
    """
    standardized = {}
    
    superkeys = create_superkeys(features)
    if superkeys:
        for superkey, value in superkeys.items():
            standardized[superkey] = value
    
    for rep_key, similar_keys in key_clusters.items():
        for key in similar_keys:
            if key in features:
                if rep_key not in standardized:
                    standardized[rep_key] = features[key]
                    break
    
    return standardized

def create_superkeys(features: Dict[str, str]) -> Dict[str, str]:
    """
    Create standardized superkeys from product features.

    Args:
        features: Dictionary of product features

    Returns:
        Dictionary of superkeys
    """
    superkeys = {}

    for superkey, feature_keys in SUPERKEY_CONFIG.items():
        values = []
        for key in feature_keys:
            if key in features and features[key] and features[key].lower() not in JUNK_VALUES:
                values.append(features[key])

        if values:
            superkeys[superkey] = values[0]  # Take the first non-empty value

    return superkeys

def generate_price(product_class: str, category_hierarchy: str) -> float:
    """
    Generate a realistic price based on product category.

    Args:
        product_class: Product class
        category_hierarchy: Category hierarchy

    Returns:
        Generated price
    """
    ceiling = 50.0  # Default ceiling

    for cat, cat_ceiling in PRICE_CEILING.items():
        if cat in category_hierarchy:
            ceiling = cat_ceiling
            break

    base_price = ceiling * 0.7

    jitter = random.uniform(-0.5, 0.5)
    price = base_price * (1 + jitter)

    endings = [0.00, 0.25, 0.49, 0.75, 0.99]
    dollars = int(price)
    cents = random.choice(endings)
    final_price = dollars + cents

    return max(4.99, final_price)  # Ensure price is at least $4.99

def filter_low_quality_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out low-quality records.

    Args:
        df: DataFrame to filter

    Returns:
        Filtered DataFrame
    """
    df['desc_word_count'] = df['product_description'].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0
    )

    df['features_word_count'] = df['clean_product_features'].apply(
        lambda x: len(str(x).split()) if isinstance(x, str) else 0
    )

    df['desc_char_count'] = df['product_description'].apply(
        lambda x: len(str(x)) if isinstance(x, str) else 0
    )

    df['features_char_count'] = df['clean_product_features'].apply(
        lambda x: len(str(x)) if isinstance(x, str) else 0
    )

    filtered_df = df[
        ((df['desc_word_count'] > 5) & (df['desc_char_count'] >= 15)) |
        ((df['features_word_count'] > 5) & (df['features_char_count'] >= 15))
    ]

    filtered_df = filtered_df.drop(
        ['desc_word_count', 'features_word_count', 'desc_char_count', 'features_char_count'],
        axis=1
    )

    logger.info(f"Filtered out {len(df) - len(filtered_df)} low-quality records")
    return filtered_df

def process_dataset(input_path: str, output_path: str, sep: str = '\t') -> None:
    """
    Process the WANDS dataset.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        sep: Separator used in the CSV file (default: tab)
    """
    df = load_dataset(input_path, sep)

    df = filter_low_quality_records(df)

    # Parse product features
    df['parsed_features'] = df['clean_product_features'].apply(parse_product_features)
    
    all_keys = []
    for features in df['parsed_features']:
        all_keys.extend(features.keys())
    
    # Cluster similar keys
    key_clusters = cluster_similar_keys(all_keys)
    
    logger.info(f"Identified {len(key_clusters)} key clusters:")
    for rep_key, similar_keys in key_clusters.items():
        logger.info(f"  {rep_key}: {similar_keys}")
    
    # Create standardized features map using clustered keys
    df['standardized_features_map'] = df['parsed_features'].apply(
        lambda features: standardize_features(features, key_clusters)
    )

    df['processed_clean_features_str'] = df['standardized_features_map'].apply(
        lambda x: ' | '.join([f"{k}: {v}" for k, v in x.items()])
    )

    if 'price' not in df.columns or df['price'].isna().any():
        df['price'] = df.apply(
            lambda row: generate_price(row['product_class'], row['category hierarchy']),
            axis=1
        )

    df['searchable_text'] = df.apply(
        lambda row: (
            f"{row['product_name']} "
            f"{row['product_description']} "
            f"{row['processed_clean_features_str']}"
        ),
        axis=1
    )

    df.to_csv(output_path, index=False)
    logger.info(f"Saved processed dataset to {output_path}")

def main():
    """Main function to run the data preparation."""
    input_dir = Path("../data/raw")
    output_dir = Path("../data/processed")
    attachments_dir = Path("/home/ubuntu/attachments/6e94ade9-72ef-4214-8828-a108131e3e86")
    create_directory(output_dir)

    process_dataset(
        input_path=attachments_dir / "good_canonical_wands_20250519_033834_good_head_2000.csv",
        output_path=output_dir / "processed_products.csv",
        sep="\t"  # Using tab delimiter as specified
    )

if __name__ == "__main__":
    main()
