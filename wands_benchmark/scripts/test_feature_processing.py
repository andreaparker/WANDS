#!/usr/bin/env python
"""
Test script for enhanced product features processing.
"""
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import setup_logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("prepare_dataset", 
                                             os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                         "01_prepare_dataset.py"))
prepare_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare_dataset)

parse_product_features = prepare_dataset.parse_product_features
segment_key = prepare_dataset.segment_key
is_numeric_value = prepare_dataset.is_numeric_value
calculate_key_similarity = prepare_dataset.calculate_key_similarity
cluster_similar_keys = prepare_dataset.cluster_similar_keys
standardize_features = prepare_dataset.standardize_features

logger = setup_logging(__name__)

def test_word_segmentation():
    """Test word segmentation."""
    test_keys = [
        "producttype", "capacityquarts", "primarycolor", "mainmaterial"
    ]
    expected = [
        "product_type", "capacity_quarts", "primary_color", "main_material"
    ]
    
    for key, expected_result in zip(test_keys, expected):
        result = segment_key(key)
        logger.info(f"Segmenting '{key}' -> '{result}' (Expected: '{expected_result}')")
        assert result == expected_result

def test_numeric_value_detection():
    """Test numeric value detection."""
    test_values = [
        "5", "10.5", "3.7 lbs", "5 feet", "5feet", "not a number", "red"
    ]
    expected = [
        True, True, True, True, True, False, False
    ]
    
    for value, expected_result in zip(test_values, expected):
        result = is_numeric_value(value)
        logger.info(f"Is '{value}' numeric? {result} (Expected: {expected_result})")
        assert result == expected_result

def test_key_similarity():
    """Test key similarity calculation."""
    test_pairs = [
        ("color", "colour"),
        ("primary_color", "color"),
        ("material", "material_type"),
        ("product_type", "item_type"),
        ("price", "cost"),
        ("color", "material")
    ]
    
    for key1, key2 in test_pairs:
        similarity = calculate_key_similarity(key1, key2)
        logger.info(f"Similarity between '{key1}' and '{key2}': {similarity:.2f}")

def test_feature_parsing():
    """Test feature parsing with segmentation and numeric replacement."""
    test_features = (
        "capacityquarts: 7 | producttype: slow cooker | "
        "primarycolor: red | height: 10.5 inches | weight: 5 lbs"
    )
    
    features = parse_product_features(test_features)
    logger.info(f"Parsed features: {features}")
    
    assert "capacity_quarts" in features
    assert "product_type" in features
    assert "primary_color" in features
    
    assert features["capacity_quarts"] == "[NUM]"
    assert features["height"] == "[NUM]"
    assert features["weight"] == "[NUM]"

def main():
    """Run all tests."""
    logger.info("Testing word segmentation...")
    test_word_segmentation()
    
    logger.info("\nTesting numeric value detection...")
    test_numeric_value_detection()
    
    logger.info("\nTesting key similarity calculation...")
    test_key_similarity()
    
    logger.info("\nTesting feature parsing...")
    test_feature_parsing()
    
    logger.info("\nAll tests passed!")

if __name__ == "__main__":
    main()
