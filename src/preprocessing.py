# Data cleaning and preprocessing functions
import pandas as pd
import numpy as np
import os
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Dict, Optional, Union, Tuple

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from parquet file.
    
    Args:
        data_path: Path to the parquet file.
        
    Returns:
        Loaded dataframe.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_parquet(data_path)
    print(f"Data loaded successfully from {data_path}")
    print(f"Shape: {df.shape}")
    return df

def select_relevant_columns(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Select relevant columns from the dataframe.
    
    Args:
        df: Input dataframe.
        columns: List of columns to select. If None, uses default set.
        
    Returns:
        Dataframe with selected columns.
    """
    if columns is None:
        columns = [
            'code', 'product_name', 'categories_en', 
            'ingredients_text_en', 'nutriscore_grade',
            # Add other nutritional columns if needed
        ]
    
    # Check which relevant columns are actually present in the DataFrame
    existing_columns = [col for col in columns if col in df.columns]
    print(f"Selecting columns: {existing_columns}")
    
    return df[existing_columns].copy()

def clean_data(df: pd.DataFrame, essential_cols: List[str] = None) -> pd.DataFrame:
    """
    Applies cleaning steps to the DataFrame.
    - Handles missing values
    - Drops rows with missing essential data
    
    Args:
        df: Input dataframe.
        essential_cols: List of columns that must not be null.
        
    Returns:
        Cleaned dataframe.
    """
    print("Applying data cleaning...")
    
    # Check missing values percentage
    missing_percentage = df.isnull().sum() * 100 / len(df)
    print("Missing value percentage per column:")
    print(missing_percentage)
    
    # Define essential columns if not provided
    if essential_cols is None:
        essential_cols = ['product_name', 'ingredients_text_en']
    
    # Drop rows with missing essential info (e.g., product_name, ingredients)
    df_cleaned = df.dropna(subset=[col for col in essential_cols if col in df.columns]).copy()
    
    print(f"Shape after dropping rows with missing essential info: {df_cleaned.shape}")
    
    # Fill missing nutriscore_grade with a placeholder like 'unknown' or drop them
    if 'nutriscore_grade' in df_cleaned.columns:
        df_cleaned['nutriscore_grade'] = df_cleaned['nutriscore_grade'].fillna('unknown')
        print("Filled missing 'nutriscore_grade' with 'unknown'.")
    
    return df_cleaned

def normalize_text(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """
    Normalizes text columns (e.g., lowercase).
    
    Args:
        df: Input dataframe.
        columns: Text columns to normalize.
        
    Returns:
        Dataframe with normalized text.
    """
    print(f"Normalizing text columns...")
    
    if columns is None:
        columns = ['product_name', 'categories_en', 'ingredients_text_en']
    
    for col in columns:
        if col in df.columns:
            # Ensure the column is string type before applying .str methods
            df[col] = df[col].astype(str).str.lower()
            print(f"Column '{col}' converted to lowercase.")
    
    return df

def tokenize_ingredients(df: pd.DataFrame, text_col: str = 'ingredients_text_en') -> pd.DataFrame:
    """
    Tokenizes ingredients text and removes stopwords.
    
    Args:
        df: Input dataframe.
        text_col: Column containing ingredients text.
        
    Returns:
        Dataframe with tokenized ingredients added.
    """
    print(f"Tokenizing features in column: {text_col}...")
    
    if text_col not in df.columns:
        print(f"Warning: Column '{text_col}' not found in dataframe.")
        return df
    
    # Initialize stopwords
    stop_words = set(stopwords.words('english'))
    
    # Function to tokenize and remove stopwords
    def tokenize_and_clean(text):
        if pd.isna(text) or text == 'nan':
            return []
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Remove stopwords and non-alphabetic tokens
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
        return tokens
    
    # Apply tokenization and stopword removal
    df['ingredients_tokens'] = df[text_col].apply(tokenize_and_clean)
    print("Tokenized ingredients and removed stopwords.")
    
    # Create document strings for vectorization
    df['ingredients_doc'] = df['ingredients_tokens'].apply(lambda tokens: ' '.join(tokens))
    
    return df

def process_categories(df: pd.DataFrame, categories_col: str = 'categories_en') -> pd.DataFrame:
    """
    Process categories column - split comma-separated values.
    
    Args:
        df: Input dataframe.
        categories_col: Column containing categories.
        
    Returns:
        Dataframe with processed categories.
    """
    print(f"Processing categories in column: {categories_col}...")
    
    if categories_col not in df.columns:
        print(f"Warning: Column '{categories_col}' not found in dataframe.")
        return df
    
    # Function to split categories and clean them
    def clean_categories(cats_string):
        if pd.isna(cats_string) or cats_string == 'nan':
            return []
        # Split by commas and clean each category
        categories = [cat.strip() for cat in cats_string.split(',')]
        # Remove empty categories
        categories = [cat for cat in categories if cat]
        return categories
    
    df['categories_list'] = df[categories_col].apply(clean_categories)
    print("Split categories into lists.")
    
    return df

def create_tfidf_vectors(df: pd.DataFrame, text_col: str = 'ingredients_doc', 
                        max_features: int = 5000, min_df: int = 5) -> Tuple[pd.DataFrame, TfidfVectorizer, np.ndarray]:
    """
    Create TF-IDF vectors for text data.
    
    Args:
        df: Input dataframe.
        text_col: Column containing text to vectorize.
        max_features: Maximum number of features for TF-IDF.
        min_df: Minimum document frequency for terms.
        
    Returns:
        Tuple of (dataframe, tfidf_vectorizer, tfidf_matrix)
    """
    print(f"Creating TF-IDF vectors for column: {text_col}...")
    
    if text_col not in df.columns:
        print(f"Warning: Column '{text_col}' not found in dataframe.")
        return df, None, None
    
    # Create TF-IDF vectors
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_col])
    
    print(f"Created TF-IDF matrix with shape: {tfidf_matrix.shape}")
    print(f"Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    
    return df, tfidf_vectorizer, tfidf_matrix

def create_category_encoding(df: pd.DataFrame, categories_col: str = 'categories_list',
                           max_categories: int = 1000) -> pd.DataFrame:
    """
    Create one-hot encoding for categories.
    
    Args:
        df: Input dataframe.
        categories_col: Column containing category lists.
        max_categories: Maximum number of unique categories to use for one-hot encoding.
        
    Returns:
        Dataframe with category encoding added.
    """
    print(f"Creating encoding for categories in column: {categories_col}...")
    
    if categories_col not in df.columns:
        print(f"Warning: Column '{categories_col}' not found in dataframe.")
        return df
    
    # Get all unique categories
    all_categories = set()
    for cat_list in df[categories_col]:
        if isinstance(cat_list, list):
            all_categories.update(cat_list)
    
    print(f"Total unique categories found: {len(all_categories)}")
    
    # Function to create one-hot encoding for categories
    def one_hot_categories(categories, all_cats):
        encoding = {cat: 1 if cat in categories else 0 for cat in all_cats}
        return encoding
    
    # Apply one-hot encoding (only if the number of categories is manageable)
    if len(all_categories) < max_categories:
        df['categories_onehot'] = df[categories_col].apply(
            lambda cats: one_hot_categories(cats, all_categories))
        print("Created one-hot encoding for categories.")
    else:
        print(f"Too many unique categories ({len(all_categories)}) for one-hot encoding. " +
              f"Consider using embeddings or dimensionality reduction.")
    
    return df

def save_processed_data(df: pd.DataFrame, output_path: str, 
                      tfidf_vectorizer: TfidfVectorizer = None,
                      similarity_matrix: np.ndarray = None,
                      max_matrix_size: int = 1000) -> None:
    """
    Save processed dataframe and models.
    
    Args:
        df: Processed dataframe.
        output_path: Base path for output files.
        tfidf_vectorizer: Fitted TF-IDF vectorizer.
        similarity_matrix: Pre-computed similarity matrix.
        max_matrix_size: Maximum number of rows for saving full similarity matrix.
    """
    print("Saving processed data and models...")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the cleaned dataframe
    df.to_parquet(output_path)
    print(f"Saved cleaned data to: {output_path}")
    
    # Save the TF-IDF vectorizer
    if tfidf_vectorizer is not None:
        vectorizer_path = os.path.join(os.path.dirname(output_path), "tfidf_vectorizer.pkl")
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(tfidf_vectorizer, f)
        print(f"Saved TF-IDF vectorizer to: {vectorizer_path}")
    
    # Save similarity matrix if provided and not too large
    if similarity_matrix is not None and len(df) <= max_matrix_size:
        matrix_path = os.path.join(os.path.dirname(output_path), "cosine_sim_matrix.pkl")
        with open(matrix_path, 'wb') as f:
            pickle.dump(similarity_matrix, f)
        print(f"Saved similarity matrix to: {matrix_path}")
    elif similarity_matrix is not None:
        print(f"Similarity matrix too large ({len(df)} > {max_matrix_size}). Not saving.")
    
    print("Saving complete!")

def preprocess_pipeline(data_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.
    
    Args:
        data_path: Path to the input data file.
        output_path: Path to save the processed data (optional).
        
    Returns:
        Processed dataframe.
    """
    print("Starting preprocessing pipeline...")
    
    # Set default output path if not provided
    if output_path is None:
        output_dir = os.path.dirname(data_path)
        output_path = os.path.join(output_dir, "cleaned_food_data.parquet")
    
    # 1. Load data
    df = load_data(data_path)
    
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
    df, tfidf_vectorizer, tfidf_matrix = create_tfidf_vectors(df)
    
    # 8. Create category encoding
    df = create_category_encoding(df)
    
    # 9. Save processed data and models
    if tfidf_vectorizer is not None:
        save_processed_data(df, output_path, tfidf_vectorizer, None)  # Don't save similarity matrix by default
    
    print("Preprocessing pipeline complete!")
    return df

# Example usage
if __name__ == "__main__":
    try:
        # Default paths
        data_path = "../data/food.parquet"
        output_path = "../data/cleaned_food_data.parquet"
        
        # Run preprocessing pipeline
        processed_df = preprocess_pipeline(data_path, output_path)
        
        print(f"Processed dataset shape: {processed_df.shape}")
        print("Sample processed data:")
        print(processed_df[['product_name', 'ingredients_tokens', 'categories_list']].head())
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()