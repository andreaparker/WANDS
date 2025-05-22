#!/usr/bin/env python
"""
Script to create a Jupyter notebook for product feature analysis.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import os
from pathlib import Path

notebook_dir = Path('../notebooks')
notebook_dir.mkdir(parents=True, exist_ok=True)

nb = new_notebook()

nb.cells.append(new_markdown_cell('# WANDS Product Feature Analysis\n\nThis notebook demonstrates the enhanced product feature processing implemented in the WANDS benchmark system. It includes:\n\n1. Word segmentation for concatenated lowercase words in keys\n2. Semantic clustering of similar keys\n3. Numeric value replacement for BERT/SPLADE compatibility\n4. Key frequency analysis and standardization'))

nb.cells.append(new_code_cell('''import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re
import wordsegment
import spacy
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))))
from scripts.utils import setup_logging

wordsegment.load()
nlp = spacy.load('en_core_web_sm')

logger = setup_logging('product_feature_analysis')'''))

nb.cells.append(new_markdown_cell('## 1. Load the Dataset\n\nFirst, we\'ll load the WANDS dataset and examine the product features.'))
nb.cells.append(new_code_cell('''# Path to the dataset
data_path = Path('../data/raw/good_canonical_wands_20250519_033834_good_head_2000.csv')

df = pd.read_csv(data_path, sep='\\t')

print(f"Dataset shape: {df.shape}")
df.head()'''))

nb.cells.append(new_markdown_cell('## 2. Examine Product Features\n\nLet\'s look at the product features column and see what kind of data we\'re working with.'))
nb.cells.append(new_code_cell('''# Check if product_features column exists
feature_column = 'product_features' if 'product_features' in df.columns else 'clean_product_features'

for i, features in enumerate(df[feature_column].dropna().head(5)):
    print(f"Example {i+1}:")
    print(features)
    print("---")'''))

nb.cells.append(new_markdown_cell('## 3. Word Segmentation\n\nWe\'ll implement word segmentation to split concatenated lowercase words in keys.'))
nb.cells.append(new_code_cell('''def segment_key(key):
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

test_keys = [
    "producttype", 
    "capacityquarts", 
    "primarycolor", 
    "mainmaterial",
    "heightinches",
    "weightcapacity"
]

for key in test_keys:
    segmented = segment_key(key)
    print(f"{key} -> {segmented}")'''))

nb.cells.append(new_markdown_cell('## 4. Numeric Value Detection\n\nWe\'ll implement a function to detect numeric values in product features.'))
nb.cells.append(new_code_cell('''def is_numeric_value(value):
    """
    Check if a value is numeric.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is numeric, False otherwise
    """
    if re.match(r'^\\d+(\\.\\d+)?(\\s*[a-zA-Z]+)?$', value):
        return True
    
    if re.match(r'^\\d+(\\.\\d+)?\\s+', value):
        return True
        
    return False

test_values = [
    "5", 
    "10.5", 
    "3.7 lbs", 
    "5 feet", 
    "5feet", 
    "not a number", 
    "red"
]

for value in test_values:
    is_numeric = is_numeric_value(value)
    print(f"{value} -> {'Numeric' if is_numeric else 'Not numeric'}")'''))

nb.cells.append(new_markdown_cell('## 5. Parse Product Features\n\nNow we\'ll implement the function to parse product features with word segmentation and numeric value replacement.'))
nb.cells.append(new_code_cell('''# Constants for feature processing
NUMERIC_PLACEHOLDER = "[NUM]"
JUNK_VALUES = [
    "none", "n/a", "na", "not applicable", "does not apply", "other", "unknown", 
    "unspecified", "no", "yes", "true", "false", "select", "choose", "varies"
]

def parse_product_features(features_str):
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

test_features = (
    "capacityquarts: 7 | producttype: slow cooker | "
    "primarycolor: red | height: 10.5 inches | weight: 5 lbs"
)

parsed_features = parse_product_features(test_features)
print("Parsed features:")
for key, value in parsed_features.items():
    print(f"  {key}: {value}")'''))

nb.cells.append(new_markdown_cell('## 6. Semantic Key Similarity\n\nWe\'ll implement a function to calculate semantic similarity between keys using spaCy.'))
nb.cells.append(new_code_cell('''def calculate_key_similarity(key1, key2):
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
    print(f"Similarity between '{key1}' and '{key2}': {similarity:.2f}")'''))

nb.cells.append(new_markdown_cell('## 7. Process Real Product Features\n\nNow let\'s process the actual product features from our dataset.'))
nb.cells.append(new_code_cell('''# Parse product features for the first 100 rows
sample_size = 100
parsed_features_list = []

for features_str in df[feature_column].dropna().head(sample_size):
    parsed = parse_product_features(features_str)
    parsed_features_list.append(parsed)
    
all_keys = []
for features in parsed_features_list:
    all_keys.extend(features.keys())
    
key_freq = Counter(all_keys)

print("Top 20 most frequent keys:")
for key, count in key_freq.most_common(20):
    print(f"  {key}: {count}")'''))

nb.cells.append(new_markdown_cell('## 8. Cluster Similar Keys\n\nNow we\'ll implement key clustering to group semantically similar keys.'))
nb.cells.append(new_code_cell('''KEY_SIMILARITY_THRESHOLD = 0.85  # Threshold for semantic similarity
MIN_KEY_FREQUENCY = 3  # Minimum frequency for key selection (lower for demo)

def cluster_similar_keys(feature_keys):
    """
    Cluster semantically similar keys.
    
    Args:
        feature_keys: List of feature keys
        
    Returns:
        Dictionary mapping representative keys to lists of similar keys
    """
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
                
            similarity = calculate_key_similarity(key, other_key)
            if similarity >= KEY_SIMILARITY_THRESHOLD:
                clusters[key].append(other_key)
                assigned_keys.add(other_key)
    
    return clusters

key_clusters = cluster_similar_keys(all_keys)

print(f"Identified {len(key_clusters)} key clusters:")
for rep_key, similar_keys in key_clusters.items():
    if len(similar_keys) > 1:  # Only show clusters with multiple keys
        print(f"  {rep_key}: {similar_keys}")'''))

nb.cells.append(new_markdown_cell('## 9. Visualize Key Clusters\n\nLet\'s visualize the key clusters to better understand the relationships between keys.'))
nb.cells.append(new_code_cell('''# Create a similarity matrix for the top 30 most frequent keys
top_keys = [key for key, _ in key_freq.most_common(30)]
similarity_matrix = np.zeros((len(top_keys), len(top_keys)))

for i, key1 in enumerate(top_keys):
    for j, key2 in enumerate(top_keys):
        similarity = calculate_key_similarity(key1, key2)
        similarity_matrix[i, j] = similarity

plt.figure(figsize=(12, 10))
sns.heatmap(similarity_matrix, annot=False, cmap='viridis', 
            xticklabels=top_keys, yticklabels=top_keys)
plt.title('Semantic Similarity Between Top 30 Feature Keys')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()'''))

nb.cells.append(new_markdown_cell('## 10. Analyze Numeric Values\n\nLet\'s analyze the numeric values in the product features to understand what we\'re replacing with `[NUM]`.'))
nb.cells.append(new_code_cell('''# Extract numeric values from product features
numeric_values = []
numeric_keys = []

for features_str in df[feature_column].dropna().head(sample_size):
    for feature in features_str.split('|'):
        feature = feature.strip()
        if not feature or ':' not in feature:
            continue
        
        parts = feature.split(':', 1)
        key = parts[0].strip().lower()
        value = parts[1].strip() if len(parts) > 1 else ""
        
        if is_numeric_value(value):
            numeric_values.append(value)
            numeric_keys.append(key)
            
numeric_key_freq = Counter(numeric_keys)
print("Top 10 keys with numeric values:")
for key, count in numeric_key_freq.most_common(10):
    print(f"  {key}: {count}")
    
print("\\nExample numeric values:")
for value in numeric_values[:20]:
    print(f"  {value}")'''))

nb.cells.append(new_markdown_cell('## 11. Standardize Features\n\nFinally, let\'s implement the standardization of features using the clustered keys.'))
nb.cells.append(new_code_cell('''def standardize_features(features, key_clusters):
    """
    Standardize features using clustered keys.
    
    Args:
        features: Dictionary of feature name to feature value
        key_clusters: Dictionary mapping representative keys to lists of similar keys
        
    Returns:
        Dictionary of standardized features
    """
    standardized = {}
    
    for rep_key, similar_keys in key_clusters.items():
        for key in similar_keys:
            if key in features:
                if rep_key not in standardized:
                    standardized[rep_key] = features[key]
                    break
    
    for key, value in features.items():
        if key not in [k for keys in key_clusters.values() for k in keys]:
            standardized[key] = value
    
    return standardized

test_features = {
    "color": "red",
    "primary_color": "blue",
    "material": "wood",
    "height": "[NUM]",
    "weight": "[NUM]"
}

test_clusters = {
    "color": ["color", "primary_color", "main_color"],
    "material": ["material", "main_material", "material_type"],
    "weight": ["weight", "weight_lbs", "product_weight"]
}

standardized = standardize_features(test_features, test_clusters)
print("Standardized features:")
for key, value in standardized.items():
    print(f"  {key}: {value}")'''))

nb.cells.append(new_markdown_cell('## 12. Process the Full Dataset\n\nNow let\'s process the full dataset and see the results.'))
nb.cells.append(new_code_cell('''# Process a larger sample
sample_size = 500  # Increase for more comprehensive analysis
sample_df = df.head(sample_size).copy()

sample_df['parsed_features'] = sample_df[feature_column].apply(parse_product_features)

all_keys = []
for features in sample_df['parsed_features']:
    all_keys.extend(features.keys())

key_clusters = cluster_similar_keys(all_keys)

sample_df['standardized_features_map'] = sample_df['parsed_features'].apply(
    lambda features: standardize_features(features, key_clusters)
)

sample_df['processed_clean_features_str'] = sample_df['standardized_features_map'].apply(
    lambda x: ' | '.join([f"{k}: {v}" for k, v in x.items()])
)

print("Original vs. Processed Features:")
for i in range(5):
    print(f"\\nExample {i+1}:")
    print("Original:")
    print(sample_df[feature_column].iloc[i])
    print("\\nProcessed:")
    print(sample_df['processed_clean_features_str'].iloc[i])
    print("---")'''))

nb.cells.append(new_markdown_cell('## 13. Summary and Conclusions\n\nThe enhanced product feature processing pipeline successfully:\n\n1. Segments concatenated words in keys (e.g., \"capacityquarts\" → \"capacity_quarts\")\n2. Replaces numeric values with \"[NUM]\" for better BERT/SPLADE compatibility\n3. Clusters semantically similar keys (e.g., \"color\", \"colour\", \"primary_color\")\n4. Standardizes features using the most frequent key in each cluster\n\nThese enhancements improve the quality and consistency of the product features, making them more suitable for search and retrieval tasks. The standardized features can be used to create better SUPERKEYS and improve the overall performance of the WANDS benchmark system.'))

nb.metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'codemirror_mode': {
            'name': 'ipython',
            'version': 3
        },
        'file_extension': '.py',
        'mimetype': 'text/x-python',
        'name': 'python',
        'nbconvert_exporter': 'python',
        'pygments_lexer': 'ipython3',
        'version': '3.9.0'
    }
}

notebook_path = Path('../notebooks/product_feature_analysis.ipynb')
with open(notebook_path, 'w') as f:
    nbf.write(nb, f)

print(f'Notebook created successfully at {notebook_path}')
