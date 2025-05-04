# Utility functions shared across modules
import os
import json
import pickle
import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory_path: Path to the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")

def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save an object to a pickle file.
    
    Args:
        obj: The object to save.
        filepath: Path where to save the object.
    """
    create_directory_if_not_exists(os.path.dirname(filepath))
    
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    
    logger.info(f"Object saved to {filepath}")

def load_pickle(filepath: str) -> Any:
    """
    Load an object from a pickle file.
    
    Args:
        filepath: Path to the pickle file.
        
    Returns:
        The loaded object.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    
    logger.info(f"Object loaded from {filepath}")
    return obj

def save_json(data: Dict, filepath: str) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: The data to save.
        filepath: Path where to save the data.
    """
    create_directory_if_not_exists(os.path.dirname(filepath))
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Data saved to {filepath}")

def load_json(filepath: str) -> Dict:
    """
    Load data from a JSON file.
    
    Args:
        filepath: Path to the JSON file.
        
    Returns:
        The loaded data.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Data loaded from {filepath}")
    return data

def format_nutriscore(nutriscore: str) -> str:
    """
    Format nutriscore to a standardized format (lowercase).
    
    Args:
        nutriscore: The nutriscore to format.
        
    Returns:
        Formatted nutriscore.
    """
    if pd.isna(nutriscore) or not nutriscore:
        return "unknown"
    
    return str(nutriscore).lower()

def get_nutriscore_color(nutriscore: str) -> str:
    """
    Get the color associated with a nutriscore.
    
    Args:
        nutriscore: The nutriscore.
        
    Returns:
        Color name (for Bootstrap classes).
    """
    nutriscore = format_nutriscore(nutriscore)
    
    if nutriscore in ['a', 'b']:
        return "success"
    elif nutriscore in ['c', 'd']:
        return "warning"
    elif nutriscore == 'e':
        return "danger"
    else:
        return "secondary"

def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        filepath: Path to the file.
        
    Returns:
        Human-readable file size.
    """
    if not os.path.exists(filepath):
        return "0 B"
    
    size_bytes = os.path.getsize(filepath)
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} PB"
