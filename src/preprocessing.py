# Data cleaning and preprocessing functions with PySpark
import os
import pickle
import logging
from typing import List, Dict, Optional, Union, Tuple

import numpy as np
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, StringType, MapType, IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import SparseVector, DenseVector
from pyspark.ml import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Spark Session
def get_spark_session(app_name="FoodRecommenderPreprocessing"):
    """
    Initialize and return a Spark session.
    
    Args:
        app_name: Name of the Spark application.
        
    Returns:
        SparkSession object.
    """
    return (SparkSession.builder
            .appName(app_name)
            .config("spark.driver.memory", "4g")  # Adjust based on your system
            .config("spark.executor.memory", "4g")  # Adjust based on your system
            .config("spark.sql.shuffle.partitions", "8")  # Adjust based on your data size
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")  # Enable Arrow optimization
            .getOrCreate())

def load_data(spark, data_path: str) -> DataFrame:
    """
    Load data from parquet file using PySpark.
    
    Args:
        spark: SparkSession object.
        data_path: Path to the parquet file.
        
    Returns:
        Loaded DataFrame.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = spark.read.parquet(data_path)
    logger.info(f"Data loaded successfully from {data_path}")
    logger.info(f"Number of rows: {df.count()}, Number of columns: {len(df.columns)}")
    return df

def select_relevant_columns(df: DataFrame, columns: List[str] = None) -> DataFrame:
    """
    Select relevant columns from the DataFrame.
    
    Args:
        df: Input DataFrame.
        columns: List of columns to select. If None, uses default set.
        
    Returns:
        DataFrame with selected columns.
    """
    if columns is None:
        columns = [
            'code', 'product_name', 'categories_en', 
            'ingredients_text_en', 'nutriscore_grade',
            # Add other nutritional columns if needed
        ]
    
    # Check which relevant columns are actually present in the DataFrame
    existing_columns = [col for col in columns if col in df.columns]
    logger.info(f"Selecting columns: {existing_columns}")
    
    return df.select(existing_columns)

def clean_data(df: DataFrame, essential_cols: List[str] = None) -> DataFrame:
    """
    Applies cleaning steps to the DataFrame.
    - Handles missing values
    - Drops rows with missing essential data
    
    Args:
        df: Input DataFrame.
        essential_cols: List of columns that must not be null.
        
    Returns:
        Cleaned DataFrame.
    """
    logger.info("Applying data cleaning...")
    
    # Define essential columns if not provided
    if essential_cols is None:
        essential_cols = ['product_name', 'ingredients_text_en']
    
    # Get the count of null values in each column
    null_counts = []
    for column in df.columns:
        null_count = df.filter(F.col(column).isNull()).count()
        null_counts.append((column, null_count, (null_count / df.count()) * 100))
    
    # Print the null value percentages
    logger.info("Missing value percentage per column:")
    for col, count, percentage in null_counts:
        logger.info(f"{col}: {count} nulls ({percentage:.2f}%)")
    
    # Drop rows with missing essential columns
    for col in essential_cols:
        if col in df.columns:
            df = df.filter(F.col(col).isNotNull())
    
    logger.info(f"Shape after dropping rows with missing essential info: {df.count()} rows")
    
    # Fill missing nutriscore_grade with 'unknown'
    if 'nutriscore_grade' in df.columns:
        df = df.fillna({'nutriscore_grade': 'unknown'})
        logger.info("Filled missing 'nutriscore_grade' with 'unknown'.")
    
    return df

def normalize_text(df: DataFrame, columns: List[str] = None) -> DataFrame:
    """
    Normalizes text columns (e.g., lowercase).
    
    Args:
        df: Input DataFrame.
        columns: Text columns to normalize.
        
    Returns:
        DataFrame with normalized text.
    """
    logger.info(f"Normalizing text columns...")
    
    if columns is None:
        columns = ['product_name', 'categories_en', 'ingredients_text_en']
    
    for col in columns:
        if col in df.columns:
            df = df.withColumn(col, F.lower(F.col(col)))
            logger.info(f"Column '{col}' converted to lowercase.")
    
    return df

def tokenize_ingredients(df: DataFrame, text_col: str = 'ingredients_text_en') -> DataFrame:
    """
    Tokenizes ingredients text and removes stopwords using PySpark ML.
    
    Args:
        df: Input DataFrame.
        text_col: Column containing ingredients text.
        
    Returns:
        DataFrame with tokenized ingredients added.
    """
    logger.info(f"Tokenizing features in column: {text_col}...")
    
    if text_col not in df.columns:
        logger.warning(f"Warning: Column '{text_col}' not found in DataFrame.")
        return df
    
    # Replace null values with empty string
    df = df.fillna({text_col: ""})
    
    # Create a tokenizer
    tokenizer = Tokenizer(inputCol=text_col, outputCol="words")
    
    # Create a StopWordsRemover
    remover = StopWordsRemover(inputCol="words", outputCol="ingredients_tokens")
    
    # Build the pipeline
    pipeline = Pipeline(stages=[tokenizer, remover])
    
    # Fit and transform the data
    df_transformed = pipeline.fit(df).transform(df)
    
    # Create document string by joining tokens
    df_transformed = df_transformed.withColumn(
        'ingredients_doc', 
        F.concat_ws(" ", F.col("ingredients_tokens"))
    )
    
    logger.info("Tokenized ingredients and removed stopwords.")
    
    return df_transformed

def process_categories(df: DataFrame, categories_col: str = 'categories_en') -> DataFrame:
    """
    Process categories column - split comma-separated values.
    
    Args:
        df: Input DataFrame.
        categories_col: Column containing categories.
        
    Returns:
        DataFrame with processed categories.
    """
    logger.info(f"Processing categories in column: {categories_col}...")
    
    if categories_col not in df.columns:
        logger.warning(f"Warning: Column '{categories_col}' not found in DataFrame.")
        return df
    
    # Split categories string into array, trim whitespace, and filter empty values
    df = df.withColumn(
        'categories_list',
        F.expr(f"filter(transform(split({categories_col}, ','), x -> trim(x)), x -> length(x) > 0)")
    )
    
    logger.info("Split categories into lists.")
    
    return df

def create_tfidf_vectors(df: DataFrame, text_col: str = 'ingredients_doc', 
                         max_features: int = 5000, min_df: int = 5) -> Tuple[DataFrame, CountVectorizer, IDF]:
    """
    Create TF-IDF vectors for text data using PySpark ML.
    
    Args:
        df: Input DataFrame.
        text_col: Column containing text to vectorize.
        max_features: Maximum number of features for TF-IDF.
        min_df: Minimum document frequency for terms.
        
    Returns:
        Tuple of (DataFrame, CountVectorizer, IDF)
    """
    logger.info(f"Creating TF-IDF vectors for column: {text_col}...")
    
    if text_col not in df.columns:
        logger.warning(f"Warning: Column '{text_col}' not found in DataFrame.")
        return df, None, None
    
    # Replace null values with empty string
    df = df.fillna({text_col: ""})
    
    # Create a tokenizer for the document column
    tokenizer = Tokenizer(inputCol=text_col, outputCol=f"{text_col}_words")
    
    # Create a CountVectorizer
    cv = CountVectorizer(inputCol=f"{text_col}_words", outputCol="tf", 
                         vocabSize=max_features, minDF=min_df)
    
    # Create an IDF model
    idf = IDF(inputCol="tf", outputCol="tfidf_features")
    
    # Build and apply the pipeline
    pipeline = Pipeline(stages=[tokenizer, cv, idf])
    model = pipeline.fit(df)
    df_vectorized = model.transform(df)
    
    # Get the vocabulary size
    vocabulary_size = len(model.stages[1].vocabulary)
    logger.info(f"Created TF-IDF matrix with vocabulary size: {vocabulary_size}")
    
    return df_vectorized, model.stages[1], model.stages[2]

def save_processed_data(df: DataFrame, output_path: str, vectorizer=None, model_path: str = None) -> None:
    """
    Save processed DataFrame and models.
    
    Args:
        df: Processed DataFrame.
        output_path: Path for output parquet file.
        vectorizer: CountVectorizer model (optional).
        model_path: Path to save the model (optional).
    """
    logger.info("Saving processed data and models...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the cleaned DataFrame
    df.write.mode("overwrite").parquet(output_path)
    logger.info(f"Saved cleaned data to: {output_path}")
    
    # Save the model if provided
    if vectorizer is not None and model_path is not None:
        model_output_path = os.path.join(model_path, "cv_model")
        vectorizer.save(model_output_path)
        logger.info(f"Saved CountVectorizer model to: {model_output_path}")
    
    logger.info("Saving complete!")

def preprocess_pipeline(spark, data_path: str, output_path: str = None, model_path: str = None) -> DataFrame:
    """
    Run the full preprocessing pipeline using PySpark.
    
    Args:
        spark: SparkSession object.
        data_path: Path to the input data file.
        output_path: Path to save the processed data (optional).
        model_path: Path to save the model (optional).
        
    Returns:
        Processed DataFrame.
    """
    logger.info("Starting preprocessing pipeline with PySpark...")
    
    # Set default output path if not provided
    if output_path is None:
        output_dir = os.path.dirname(data_path)
        output_path = os.path.join(output_dir, "cleaned_food_data.parquet")
    
    # Set default model path if not provided
    if model_path is None:
        model_path = os.path.dirname(data_path)
    
    # 1. Load data
    df = load_data(spark, data_path)
    
    # 2. Select relevant columns
    df = select_relevant_columns(df)
    
    # 3. Clean data (handle missing values)
    df = clean_data(df)
    
    # 4. Normalize text
    df = normalize_text(df)
    
    # 5. Tokenize ingredients
    df = tokenize_ingredients(df)
    
    # 6. Process categories
    df = process_categories(df)
    
    # 7. Create TF-IDF vectors
    df, cv_model, _ = create_tfidf_vectors(df)
    
    # 8. Save processed data and model
    save_processed_data(df, output_path, cv_model, model_path)
    
    logger.info("Preprocessing pipeline complete!")
    return df

# Example usage
if __name__ == "__main__":
    try:
        # Initialize Spark session
        spark = get_spark_session()
        
        # Default paths
        data_path = "../data/food.parquet"
        output_path = "../data/cleaned_food_data.parquet"
        model_path = "../data"
        
        # Run preprocessing pipeline
        processed_df = preprocess_pipeline(spark, data_path, output_path, model_path)
        
        # Show sample of processed data
        logger.info(f"Processed dataset row count: {processed_df.count()}")
        logger.info("Sample processed data:")
        processed_df.select('product_name', 'ingredients_tokens', 'categories_list').show(5, truncate=False)
        
        # Stop Spark session
        spark.stop()
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        
        # Make sure to stop Spark session in case of error
        try:
            spark.stop()
        except:
            pass