# Database models and schemas for MongoDB
from pymongo import MongoClient, IndexModel, ASCENDING, TEXT
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging
from bson import ObjectId
import re

logger = logging.getLogger(__name__)

class DatabaseManager:
    """MongoDB database manager for the food recommendation system."""
    
    def __init__(self, mongo_uri: str, db_name: str):
        """Initialize the database manager.
        
        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
        """
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self._connect()
        
    def _connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            # Test connection
            self.client.admin.command('ismaster')
            logger.info(f"Successfully connected to MongoDB: {self.db_name}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def close_connection(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def create_indexes(self):
        """Create necessary indexes for optimal performance."""
        try:
            # Products collection indexes
            products = self.db.products
            
            # Text index for search functionality
            products.create_index([
                ("product_name", TEXT),
                ("ingredients_text", TEXT),
                ("categories", TEXT),
                ("brands", TEXT)
            ], name="text_search_index")
            
            # Individual field indexes
            products.create_index("code", unique=True)
            products.create_index("nutriscore_grade")
            products.create_index("main_category")
            products.create_index("brands")
            products.create_index("countries_tags")
            products.create_index("energy_100g")
            
            # Compound indexes for common queries
            products.create_index([
                ("main_category", ASCENDING),
                ("nutriscore_grade", ASCENDING)
            ])
            
            products.create_index([
                ("countries_tags", ASCENDING),
                ("nutriscore_grade", ASCENDING)
            ])
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            raise

class ProductModel:
    """Model for food product documents."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager.db
        self.collection = self.db.products
    
    def insert_product(self, product_data: Dict[str, Any]) -> str:
        """Insert a single product into the database.
        
        Args:
            product_data: Product data dictionary
            
        Returns:
            Inserted document ID
        """
        # Add metadata
        product_data['created_at'] = datetime.utcnow()
        product_data['updated_at'] = datetime.utcnow()
        
        # Clean and validate data
        product_data = self._clean_product_data(product_data)
        
        result = self.collection.insert_one(product_data)
        return str(result.inserted_id)
    
    def insert_many_products(self, products_data: List[Dict[str, Any]]) -> List[str]:
        """Insert multiple products into the database.
        
        Args:
            products_data: List of product data dictionaries
            
        Returns:
            List of inserted document IDs
        """
        now = datetime.utcnow()
        
        # Clean and add metadata to all products
        for product in products_data:
            product['created_at'] = now
            product['updated_at'] = now
            product = self._clean_product_data(product)
        
        result = self.collection.insert_many(products_data, ordered=False)
        return [str(id) for id in result.inserted_ids]
    
    def find_by_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Find a product by its unique code.
        
        Args:
            code: Product code
            
        Returns:
            Product document or None if not found
        """
        return self.collection.find_one({"code": code})
    
    def search_products(self, query: str, limit: int = 50, 
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search products using text search.
        
        Args:
            query: Search query
            limit: Maximum number of results
            filters: Additional filters
            
        Returns:
            List of matching products
        """
        search_filter = {"$text": {"$search": query}}
        
        if filters:
            search_filter.update(filters)
        
        cursor = self.collection.find(
            search_filter,
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(limit)
        
        return list(cursor)
    
    def find_by_category(self, category: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Find products by main category.
        
        Args:
            category: Category name
            limit: Maximum number of results
            
        Returns:
            List of products in the category
        """
        return list(self.collection.find(
            {"main_category": {"$regex": category, "$options": "i"}}
        ).limit(limit))
    
    def find_by_country(self, country: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Find products by country.
        
        Args:
            country: Country name
            limit: Maximum number of results
            
        Returns:
            List of products from the country
        """
        return list(self.collection.find(
            {"countries_tags": {"$regex": country, "$options": "i"}}
        ).limit(limit))
    
    def get_recommendations(self, product_code: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get product recommendations based on similarity.
        
        Args:
            product_code: Reference product code
            limit: Number of recommendations
            
        Returns:
            List of recommended products
        """
        # Get the reference product
        ref_product = self.find_by_code(product_code)
        if not ref_product:
            return []
        
        # Find similar products based on category, nutriscore, and other factors
        query = {
            "code": {"$ne": product_code},  # Exclude the reference product
            "$or": [
                {"main_category": ref_product.get("main_category")},
                {"nutriscore_grade": ref_product.get("nutriscore_grade")},
                {"brands": {"$in": ref_product.get("brands", [])}}
            ]
        }
        
        return list(self.collection.find(query).limit(limit))
    
    def get_nutritional_stats(self) -> Dict[str, Any]:
        """Get nutritional statistics from the database.
        
        Returns:
            Dictionary with nutritional statistics
        """
        pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_energy": {"$avg": "$energy_100g"},
                    "avg_fat": {"$avg": "$fat_100g"},
                    "avg_carbs": {"$avg": "$carbohydrates_100g"},
                    "avg_proteins": {"$avg": "$proteins_100g"},
                    "avg_salt": {"$avg": "$salt_100g"},
                    "total_products": {"$sum": 1}
                }
            }
        ]
        
        result = list(self.collection.aggregate(pipeline))
        return result[0] if result else {}
    
    def get_categories_stats(self) -> List[Dict[str, Any]]:
        """Get category statistics.
        
        Returns:
            List of categories with product counts
        """
        pipeline = [
            {
                "$group": {
                    "_id": "$main_category",
                    "count": {"$sum": 1},
                    "avg_nutriscore": {"$avg": "$nutriscore_score"}
                }
            },
            {"$sort": {"count": -1}},
            {"$limit": 20}
        ]
        
        return list(self.collection.aggregate(pipeline))
    
    def _clean_product_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate product data.
        
        Args:
            product_data: Raw product data
            
        Returns:
            Cleaned product data
        """
        # Convert numeric fields
        numeric_fields = [
            'code', 'nutriscore_score', 'energy_100g', 'fat_100g',
            'saturated-fat_100g', 'carbohydrates_100g', 'sugars_100g',
            'proteins_100g', 'salt_100g'
        ]
        
        for field in numeric_fields:
            if field in product_data and product_data[field] is not None:
                try:
                    product_data[field] = float(product_data[field])
                except (ValueError, TypeError):
                    product_data[field] = None
        
        # Clean string fields
        string_fields = [
            'product_name', 'ingredients_text', 'categories', 'brands',
            'labels_tags', 'main_category', 'countries_tags', 'origins',
            'manufacturing_places', 'nutriscore_grade', 'packaging'
        ]
        
        for field in string_fields:
            if field in product_data and product_data[field] is not None:
                # Convert to string and clean
                value = str(product_data[field]).strip()
                product_data[field] = value if value else None
        
        # Parse tags fields (convert comma-separated strings to arrays)
        tag_fields = ['countries_tags', 'labels_tags', 'additives_tags']
        for field in tag_fields:
            if field in product_data and product_data[field]:
                if isinstance(product_data[field], str):
                    # Split by comma and clean
                    tags = [tag.strip() for tag in product_data[field].split(',')]
                    product_data[field] = [tag for tag in tags if tag]
        
        return product_data

class CategoryModel:
    """Model for food categories."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager.db
        self.collection = self.db.categories
    
    def get_all_categories(self) -> List[str]:
        """Get all unique categories from products.
        
        Returns:
            List of category names
        """
        pipeline = [
            {"$group": {"_id": "$main_category"}},
            {"$sort": {"_id": 1}}
        ]
        
        result = self.db.products.aggregate(pipeline)
        return [doc["_id"] for doc in result if doc["_id"]]

class BrandModel:
    """Model for food brands."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager.db
        self.collection = self.db.brands
    
    def get_popular_brands(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get popular brands by product count.
        
        Args:
            limit: Maximum number of brands to return
            
        Returns:
            List of brands with product counts
        """
        pipeline = [
            {"$match": {"brands": {"$ne": None, "$ne": ""}}},
            {"$group": {
                "_id": "$brands",
                "product_count": {"$sum": 1}
            }},
            {"$sort": {"product_count": -1}},
            {"$limit": limit}
        ]
        
        return list(self.db.products.aggregate(pipeline))
