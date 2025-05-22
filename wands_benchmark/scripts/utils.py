#!/usr/bin/env python
"""
Utility functions for WANDS benchmark.
"""
import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        name: Name of the logger
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    return logger

def create_directory(path: Path) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Path to create
    """
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        
def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded JSON data
    """
    with open(file_path, 'r') as f:
        return json.load(f)
        
def save_json(data: Dict[str, Any], file_path: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        file_path: Path to JSON file
    """
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
        
def word_count(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Text to count words in
        
    Returns:
        Number of words
    """
    if not isinstance(text, str):
        return 0
    return len(text.split())

def load_env_variables() -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Returns:
        Dictionary of environment variables
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "SPLADE_QUERY_MODEL_NAME": os.getenv("SPLADE_QUERY_MODEL_NAME", "naver/efficient-splade-V-large-query"),
        "SPLADE_DOC_MODEL_NAME": os.getenv("SPLADE_DOC_MODEL_NAME", "naver/efficient-splade-V-large-doc"),
        "GPT4_MODEL": os.getenv("GPT4_MODEL", "gpt-4"),
        "GPT35_MODEL": os.getenv("GPT35_MODEL", "gpt-3.5-turbo"),
        "METRICS_K_VALUES": os.getenv("METRICS_K_VALUES", "3,5,10"),
    }
