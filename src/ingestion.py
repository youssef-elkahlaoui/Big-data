# Data ingestion scripts (e.g., downloading from API, initial loading)
import os
import requests
import gzip
import shutil
import json
import pandas as pd
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenFoodFactsIngestion:
    """Class for downloading and ingesting Open Food Facts data."""
    
    def __init__(self, data_dir="../data"):
        """
        Initialize the data ingestion process.
        
        Args:
            data_dir: Directory to store downloaded and processed data.
        """
        self.data_dir = data_dir
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def download_parquet_data(self, force_download=False):
        """
        Download the Open Food Facts dataset in Parquet format.
        
        Args:
            force_download: If True, redownload even if file exists.
            
        Returns:
            Path to downloaded file.
        """
        url = "https://challenges.openfoodfacts.org/data/en.openfoodfacts.org.products.parquet"
        output_file = os.path.join(self.data_dir, "food.parquet")
        
        if os.path.exists(output_file) and not force_download:
            logger.info(f"Parquet file already exists at {output_file}. Use force_download=True to redownload.")
            return output_file
        
        logger.info(f"Downloading Open Food Facts dataset from {url}")
        
        # Stream download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_file, 'wb') as f, tqdm(
            desc="Downloading", 
            total=total_size, 
            unit='B', 
            unit_scale=True,
            unit_divisor=1024
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Download complete. File saved to {output_file}")
        return output_file
    
    def download_json_data(self, force_download=False):
        """
        Download the Open Food Facts dataset in JSON format.
        
        Args:
            force_download: If True, redownload even if file exists.
            
        Returns:
            Path to downloaded and extracted file.
        """
        url = "https://challenges.openfoodfacts.org/data/en.openfoodfacts.org.products.json.gz"
        compressed_file = os.path.join(self.data_dir, "food_data.json.gz")
        output_file = os.path.join(self.data_dir, "food_data.json")
        
        if os.path.exists(output_file) and not force_download:
            logger.info(f"JSON file already exists at {output_file}. Use force_download=True to redownload.")
            return output_file
        
        logger.info(f"Downloading Open Food Facts dataset from {url}")
        
        # Stream download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(compressed_file, 'wb') as f, tqdm(
            desc="Downloading", 
            total=total_size, 
            unit='B', 
            unit_scale=True,
            unit_divisor=1024
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024*1024):  # 1MB chunks
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        logger.info(f"Download complete. Extracting from {compressed_file}")
        
        # Extract the gzipped file
        with gzip.open(compressed_file, 'rb') as f_in:
            with open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove compressed file after extraction
        os.remove(compressed_file)
        
        logger.info(f"Extraction complete. File saved to {output_file}")
        return output_file
    
    def fetch_categories(self, limit=100):
        """
        Fetch categories from Open Food Facts API.
        
        Args:
            limit: Maximum number of categories to fetch.
            
        Returns:
            List of category dictionaries.
        """
        url = f"https://world.openfoodfacts.org/categories.json?limit={limit}"
        logger.info(f"Fetching {limit} categories from API")
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            categories = data.get('tags', [])
            logger.info(f"Successfully fetched {len(categories)} categories")
            return categories
        else:
            logger.error(f"Failed to fetch categories. Status code: {response.status_code}")
            return []
    
    def fetch_products_by_category(self, category, page=1, page_size=50):
        """
        Fetch products from a specific category using Open Food Facts API.
        
        Args:
            category: Category tag ID.
            page: Page number for pagination.
            page_size: Number of products per page.
            
        Returns:
            List of product dictionaries.
        """
        url = f"https://world.openfoodfacts.org/category/{category}/{page}.json?page_size={page_size}"
        logger.info(f"Fetching products for category '{category}', page {page}")
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            products = data.get('products', [])
            logger.info(f"Successfully fetched {len(products)} products")
            return products
        else:
            logger.error(f"Failed to fetch products. Status code: {response.status_code}")
            return []
    
    def convert_json_to_parquet(self, json_file, output_file=None):
        """
        Convert JSON data to Parquet format for more efficient processing.
        
        Args:
            json_file: Path to the JSON file.
            output_file: Path for the output Parquet file.
            
        Returns:
            Path to the Parquet file.
        """
        if output_file is None:
            output_file = os.path.join(self.data_dir, "food.parquet")
        
        logger.info(f"Converting JSON data to Parquet format...")
        
        # Read JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if the data is an object with a 'products' key or a list of products
        if isinstance(data, dict) and 'products' in data:
            products = data['products']
        else:
            products = data
        
        # Convert to DataFrame
        df = pd.DataFrame(products)
        
        # Write to Parquet
        df.to_parquet(output_file)
        
        logger.info(f"Conversion complete. Parquet file saved to {output_file}")
        return output_file
    
    def load_data(self, format='parquet', file_path=None):
        """
        Load data from Parquet or JSON file.
        
        Args:
            format: 'parquet' or 'json'.
            file_path: Path to the file. If None, use default location.
            
        Returns:
            Pandas DataFrame with the data.
        """
        if file_path is None:
            if format.lower() == 'parquet':
                file_path = os.path.join(self.data_dir, "food.parquet")
            else:
                file_path = os.path.join(self.data_dir, "food_data.json")
        
        logger.info(f"Loading data from {file_path}")
        
        if format.lower() == 'parquet':
            if not os.path.exists(file_path):
                logger.warning(f"Parquet file not found. Downloading it...")
                file_path = self.download_parquet_data()
            
            df = pd.read_parquet(file_path)
        else:  # json
            if not os.path.exists(file_path):
                logger.warning(f"JSON file not found. Downloading it...")
                file_path = self.download_json_data()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if the data is an object with a 'products' key or a list of products
            if isinstance(data, dict) and 'products' in data:
                products = data['products']
            else:
                products = data
            
            df = pd.DataFrame(products)
        
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df


if __name__ == "__main__":
    # Example usage
    ingestion = OpenFoodFactsIngestion()
    
    # Download and load the dataset (will use existing file if available)
    try:
        # Option 1: Download Parquet directly (recommended for efficiency)
        parquet_file = ingestion.download_parquet_data()
        df = ingestion.load_data(format='parquet', file_path=parquet_file)
        
        # Display basic info
        print(f"Loaded dataset with {len(df)} products and {len(df.columns)} features")
        print("\nSample columns:", list(df.columns)[:10])
        print("\nSample data:")
        print(df[['code', 'product_name', 'categories_en']].head())
        
        # Option 2: Fetch specific categories and products (commented out by default)
        # categories = ingestion.fetch_categories(limit=10)
        # if categories:
        #     category_id = categories[0]['id']
        #     products = ingestion.fetch_products_by_category(category_id)
        #     print(f"Fetched {len(products)} products for category {category_id}")
        
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
