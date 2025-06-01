# Utility functions shared across modules
import os
import json
import pickle
import logging
import re
from typing import Any, Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

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

def clean_text(text: str) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text: The text to clean.
        
    Returns:
        Cleaned text.
    """
    if not text or pd.isna(text):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove special characters but keep letters, numbers, and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_allergens(allergens_text: str) -> List[str]:
    """
    Extract allergens from allergens text.
    
    Args:
        allergens_text: The allergens text.
        
    Returns:
        List of allergens.
    """
    if not allergens_text or pd.isna(allergens_text):
        return []
    
    # Common allergen patterns
    allergen_patterns = [
        r'gluten', r'milk', r'eggs?', r'nuts?', r'peanuts?', r'soy', r'fish',
        r'shellfish', r'sesame', r'sulfites?', r'mustard', r'celery', r'lupin'
    ]
    
    allergens = []
    text_lower = str(allergens_text).lower()
    
    for pattern in allergen_patterns:
        if re.search(pattern, text_lower):
            # Extract the base allergen name
            match = re.search(pattern, text_lower)
            if match:
                allergen = match.group().rstrip('s')  # Remove plural 's'
                if allergen not in allergens:
                    allergens.append(allergen)
    
    return allergens

def parse_ingredients(ingredients_text: str) -> List[str]:
    """
    Parse ingredients from ingredients text.
    
    Args:
        ingredients_text: The ingredients text.
        
    Returns:
        List of ingredients.
    """
    if not ingredients_text or pd.isna(ingredients_text):
        return []
    
    # Split by common delimiters
    ingredients = re.split(r'[,;.]', str(ingredients_text))
    
    # Clean each ingredient
    cleaned_ingredients = []
    for ingredient in ingredients:
        # Remove percentage indicators
        ingredient = re.sub(r'\([^)]*%[^)]*\)', '', ingredient)
        ingredient = re.sub(r'\d+%', '', ingredient)
        
        # Clean and normalize
        ingredient = clean_text(ingredient).strip()
        
        if ingredient and len(ingredient) > 1:
            cleaned_ingredients.append(ingredient)
    
    return cleaned_ingredients[:20]  # Limit to first 20 ingredients

def create_spark_session(app_name: str = "FoodRecommendation") -> SparkSession:
    """
    Create a Spark session with optimized configuration.
    
    Args:
        app_name: Name of the Spark application.
        
    Returns:
        SparkSession object.
    """
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.driver.memory", "4g")
            .config("spark.driver.maxResultSize", "2g")
            .config("spark.executor.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .getOrCreate())

def validate_nutrition_value(value: Union[str, float, int]) -> float:
    """
    Validate and convert nutrition values to float.
    
    Args:
        value: The nutrition value to validate.
        
    Returns:
        Validated float value or 0.0 if invalid.
    """
    if pd.isna(value) or value is None or value == '':
        return 0.0
    
    try:
        # Handle string values
        if isinstance(value, str):
            # Remove any non-numeric characters except decimal point
            cleaned_value = re.sub(r'[^\d.]', '', value)
            if not cleaned_value:
                return 0.0
            return float(cleaned_value)
        
        # Handle numeric values
        return float(value)
    
    except (ValueError, TypeError):
        return 0.0

def calculate_nutriscore_points(energy: float, sugars: float, saturated_fat: float, 
                               sodium: float, fiber: float, protein: float,
                               fruits_vegetables: float = 0.0) -> int:
    """
    Calculate Nutri-Score points based on nutrition values.
    
    Args:
        energy: Energy value per 100g (kJ)
        sugars: Sugars per 100g (g)
        saturated_fat: Saturated fat per 100g (g)
        sodium: Sodium per 100g (mg)
        fiber: Fiber per 100g (g)
        protein: Protein per 100g (g)
        fruits_vegetables: Fruits/vegetables percentage (%)
        
    Returns:
        Nutri-Score points (lower is better)
    """
    points = 0
    
    # Negative points (bad nutrients)
    # Energy points
    if energy <= 335:
        points += 0
    elif energy <= 670:
        points += 1
    elif energy <= 1005:
        points += 2
    elif energy <= 1340:
        points += 3
    elif energy <= 1675:
        points += 4
    elif energy <= 2010:
        points += 5
    elif energy <= 2345:
        points += 6
    elif energy <= 2680:
        points += 7
    elif energy <= 3015:
        points += 8
    elif energy <= 3350:
        points += 9
    else:
        points += 10
    
    # Sugars points
    if sugars <= 4.5:
        points += 0
    elif sugars <= 9:
        points += 1
    elif sugars <= 13.5:
        points += 2
    elif sugars <= 18:
        points += 3
    elif sugars <= 22.5:
        points += 4
    elif sugars <= 27:
        points += 5
    elif sugars <= 31:
        points += 6
    elif sugars <= 36:
        points += 7
    elif sugars <= 40:
        points += 8
    elif sugars <= 45:
        points += 9
    else:
        points += 10
    
    # Saturated fat points
    if saturated_fat <= 1:
        points += 0
    elif saturated_fat <= 2:
        points += 1
    elif saturated_fat <= 3:
        points += 2
    elif saturated_fat <= 4:
        points += 3
    elif saturated_fat <= 5:
        points += 4
    elif saturated_fat <= 6:
        points += 5
    elif saturated_fat <= 7:
        points += 6
    elif saturated_fat <= 8:
        points += 7
    elif saturated_fat <= 9:
        points += 8
    elif saturated_fat <= 10:
        points += 9
    else:
        points += 10
    
    # Sodium points
    if sodium <= 90:
        points += 0
    elif sodium <= 180:
        points += 1
    elif sodium <= 270:
        points += 2
    elif sodium <= 360:
        points += 3
    elif sodium <= 450:
        points += 4
    elif sodium <= 540:
        points += 5
    elif sodium <= 630:
        points += 6
    elif sodium <= 720:
        points += 7
    elif sodium <= 810:
        points += 8
    elif sodium <= 900:
        points += 9
    else:
        points += 10
    
    # Positive points (good nutrients)
    # Fiber points
    if fiber <= 0.9:
        fiber_points = 0
    elif fiber <= 1.9:
        fiber_points = 1
    elif fiber <= 2.8:
        fiber_points = 2
    elif fiber <= 3.7:
        fiber_points = 3
    elif fiber <= 4.7:
        fiber_points = 4
    else:
        fiber_points = 5
    
    # Protein points
    if protein <= 1.6:
        protein_points = 0
    elif protein <= 3.2:
        protein_points = 1
    elif protein <= 4.8:
        protein_points = 2
    elif protein <= 6.4:
        protein_points = 3
    elif protein <= 8.0:
        protein_points = 4
    else:
        protein_points = 5
    
    # Fruits/vegetables points
    if fruits_vegetables <= 40:
        fv_points = 0
    elif fruits_vegetables <= 60:
        fv_points = 1
    elif fruits_vegetables <= 80:
        fv_points = 2
    else:
        fv_points = 5
    
    # Apply condition for protein points
    if points >= 11 and fv_points < 5:
        # Only count fiber and fruits/vegetables points
        points -= (fiber_points + fv_points)
    else:
        # Count all positive points
        points -= (fiber_points + protein_points + fv_points)
    
    return max(0, points)

def nutriscore_points_to_grade(points: int) -> str:
    """
    Convert Nutri-Score points to letter grade.
    
    Args:
        points: Nutri-Score points.
        
    Returns:
        Letter grade (a, b, c, d, e).
    """
    if points <= -1:
        return 'a'
    elif points <= 2:
        return 'b'
    elif points <= 10:
        return 'c'
    elif points <= 18:
        return 'd'
    else:
        return 'e'

def get_data_quality_metrics(df) -> Dict:
    """
    Calculate data quality metrics for a Spark DataFrame.
    
    Args:
        df: Spark DataFrame to analyze.
        
    Returns:
        Dictionary containing quality metrics.
    """
    total_rows = df.count()
    total_cols = len(df.columns)
    
    metrics = {
        'total_rows': total_rows,
        'total_columns': total_cols,
        'column_metrics': {}
    }
    
    for col_name in df.columns:
        col_metrics = {
            'null_count': df.filter(col(col_name).isNull()).count(),
            'non_null_count': df.filter(col(col_name).isNotNull()).count(),
            'distinct_count': df.select(col_name).distinct().count()
        }
        
        col_metrics['null_percentage'] = (col_metrics['null_count'] / total_rows) * 100
        col_metrics['fill_rate'] = 100 - col_metrics['null_percentage']
        
        metrics['column_metrics'][col_name] = col_metrics
    
    return metrics

def log_processing_step(step_name: str, start_time: float, end_time: float, 
                       rows_processed: int = None) -> None:
    """
    Log processing step information.
    
    Args:
        step_name: Name of the processing step.
        start_time: Start time timestamp.
        end_time: End time timestamp.
        rows_processed: Number of rows processed (optional).
    """
    duration = end_time - start_time
    
    log_message = f"Step '{step_name}' completed in {duration:.2f} seconds"
    
    if rows_processed:
        rate = rows_processed / duration if duration > 0 else 0
        log_message += f" ({rows_processed:,} rows, {rate:.0f} rows/sec)"
    
    logger.info(log_message)
