# Data migration script from CSV to MongoDB
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.models import DatabaseManager, ProductModel
from app.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataMigrator:
    """Handles migration of data from CSV files to MongoDB."""
    
    def __init__(self, csv_path: str, mongo_uri: str, db_name: str):
        """Initialize the data migrator.
        
        Args:
            csv_path: Path to the CSV file
            mongo_uri: MongoDB connection URI
            db_name: Database name
        """
        self.csv_path = csv_path
        self.db_manager = DatabaseManager(mongo_uri, db_name)
        self.product_model = ProductModel(self.db_manager)
        
    def migrate_csv_to_mongodb(self, batch_size: int = 1000, 
                              skip_existing: bool = True) -> Dict[str, int]:
        """Migrate CSV data to MongoDB.
        
        Args:
            batch_size: Number of records to process in each batch
            skip_existing: Whether to skip products that already exist
            
        Returns:
            Dictionary with migration statistics
        """
        logger.info(f"Starting migration from {self.csv_path}")        # Read CSV file
        logger.info("Reading CSV file...")
        try:
            # Read CSV with error handling for malformed lines
            df = pd.read_csv(
                self.csv_path, 
                on_bad_lines='skip',  # Skip malformed lines
                engine='python',      # Use Python engine for better error handling
                encoding='utf-8'
            )
            logger.info(f"Loaded {len(df)} records from CSV")
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            return {"error": str(e)}
        
        # Clean and prepare data
        logger.info("Cleaning data...")
        df_cleaned = self._clean_dataframe(df)
        logger.info(f"Cleaned data: {len(df_cleaned)} records remaining")
        
        # Create indexes
        logger.info("Creating database indexes...")
        self.db_manager.create_indexes()
        
        # Migrate data in batches
        stats = {
            "total_records": len(df_cleaned),
            "processed": 0,
            "inserted": 0,
            "skipped": 0,
            "errors": 0
        }
        
        logger.info(f"Starting batch migration with batch size: {batch_size}")
        
        for i in tqdm(range(0, len(df_cleaned), batch_size), desc="Migrating batches"):
            batch_df = df_cleaned.iloc[i:i + batch_size]
            batch_result = self._migrate_batch(batch_df, skip_existing)
            
            stats["processed"] += batch_result["processed"]
            stats["inserted"] += batch_result["inserted"]
            stats["skipped"] += batch_result["skipped"]
            stats["errors"] += batch_result["errors"]
        
        logger.info(f"Migration completed: {stats}")
        return stats
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the dataframe before migration.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Convert NaN to None for better MongoDB compatibility
        df_clean = df_clean.replace({np.nan: None})
        
        # Ensure code column is not null (required field)
        df_clean = df_clean[df_clean['code'].notna()]
        
        # Convert code to string and remove decimals if present
        df_clean['code'] = df_clean['code'].astype(str).str.replace('.0', '', regex=False)
        
        # Clean numeric columns
        numeric_columns = [
            'nutriscore_score', 'energy_100g', 'fat_100g', 'saturated-fat_100g',
            'carbohydrates_100g', 'sugars_100g', 'proteins_100g', 'salt_100g'
        ]
        
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Clean string columns
        string_columns = [
            'product_name', 'ingredients_text', 'categories', 'brands',
            'labels_tags', 'main_category', 'countries_tags', 'origins',
            'manufacturing_places', 'nutriscore_grade', 'packaging', 'additives_tags'
        ]
        
        for col in string_columns:
            if col in df_clean.columns:
                # Convert to string and strip whitespace
                df_clean[col] = df_clean[col].astype(str).str.strip()
                # Replace 'nan' string with None
                df_clean[col] = df_clean[col].replace(['nan', ''], None)
        
        return df_clean
    
    def _migrate_batch(self, batch_df: pd.DataFrame, 
                      skip_existing: bool = True) -> Dict[str, int]:
        """Migrate a batch of records.
        
        Args:
            batch_df: Dataframe batch to migrate
            skip_existing: Whether to skip existing products
            
        Returns:
            Batch migration statistics
        """
        stats = {"processed": 0, "inserted": 0, "skipped": 0, "errors": 0}
        
        products_to_insert = []
        
        for _, row in batch_df.iterrows():
            stats["processed"] += 1
            
            try:
                # Convert row to dictionary
                product_data = row.to_dict()
                
                # Check if product already exists
                if skip_existing:
                    existing = self.product_model.find_by_code(str(product_data['code']))
                    if existing:
                        stats["skipped"] += 1
                        continue
                
                products_to_insert.append(product_data)
                
            except Exception as e:
                logger.error(f"Error processing record {row.get('code', 'unknown')}: {e}")
                stats["errors"] += 1
        
        # Insert batch
        if products_to_insert:
            try:
                self.product_model.insert_many_products(products_to_insert)
                stats["inserted"] += len(products_to_insert)
            except Exception as e:
                logger.error(f"Error inserting batch: {e}")
                stats["errors"] += len(products_to_insert)
        
        return stats
    
    def validate_migration(self) -> Dict[str, Any]:
        """Validate the migration by comparing counts and sampling data.
        
        Returns:
            Validation results
        """
        logger.info("Validating migration...")
        
        # Count records in CSV
        try:
            df = pd.read_csv(self.csv_path, low_memory=False)
            csv_count = len(df[df['code'].notna()])
        except Exception as e:
            return {"error": f"Failed to read CSV: {e}"}
        
        # Count records in MongoDB
        mongo_count = self.product_model.collection.count_documents({})
        
        # Sample some records for comparison
        sample_products = list(self.product_model.collection.find().limit(5))
        
        # Get database statistics
        stats = self.product_model.get_nutritional_stats()
        
        validation_results = {
            "csv_records": csv_count,
            "mongodb_records": mongo_count,
            "migration_success_rate": (mongo_count / csv_count) * 100 if csv_count > 0 else 0,
            "sample_products": len(sample_products),
            "database_stats": stats
        }
        
        logger.info(f"Validation results: {validation_results}")
        return validation_results
    
    def create_additional_collections(self):
        """Create and populate additional collections for categories, brands, etc."""
        logger.info("Creating additional collections...")
        
        # Get unique categories
        categories = self.product_model.collection.distinct("main_category")
        categories = [cat for cat in categories if cat and cat.strip()]
        
        if categories:
            category_docs = [
                {"name": cat, "product_count": self.product_model.collection.count_documents({"main_category": cat})}
                for cat in categories
            ]
            self.db_manager.db.categories.delete_many({})  # Clear existing
            self.db_manager.db.categories.insert_many(category_docs)
            logger.info(f"Created {len(category_docs)} category documents")
        
        # Get unique brands
        brands = self.product_model.collection.distinct("brands")
        brands = [brand for brand in brands if brand and brand.strip()]
        
        if brands:
            brand_docs = [
                {"name": brand, "product_count": self.product_model.collection.count_documents({"brands": brand})}
                for brand in brands[:1000]  # Limit to top 1000 brands
            ]
            self.db_manager.db.brands.delete_many({})  # Clear existing
            self.db_manager.db.brands.insert_many(brand_docs)
            logger.info(f"Created {len(brand_docs)} brand documents")

def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Migrate CSV data to MongoDB')
    parser.add_argument('--csv-path', required=True, help='Path to CSV file')
    parser.add_argument('--mongo-uri', default='mongodb://localhost:27017/', help='MongoDB URI')
    parser.add_argument('--db-name', default='food_recommendation_db', help='Database name')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size for migration')
    parser.add_argument('--skip-existing', action='store_true', help='Skip existing products')
    parser.add_argument('--validate-only', action='store_true', help='Only run validation')
    
    args = parser.parse_args()
    
    # Initialize migrator
    migrator = DataMigrator(args.csv_path, args.mongo_uri, args.db_name)
    
    try:
        if args.validate_only:
            # Only run validation
            results = migrator.validate_migration()
            print(f"Validation Results: {results}")
        else:
            # Run full migration
            migration_stats = migrator.migrate_csv_to_mongodb(
                batch_size=args.batch_size,
                skip_existing=args.skip_existing
            )
            print(f"Migration completed: {migration_stats}")
            
            # Create additional collections
            migrator.create_additional_collections()
            
            # Validate migration
            validation_results = migrator.validate_migration()
            print(f"Validation: {validation_results}")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)
    finally:
        migrator.db_manager.close_connection()

if __name__ == "__main__":
    main()
