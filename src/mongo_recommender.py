# MongoDB-based recommendation engine
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import pickle
import json

from app.models import DatabaseManager, ProductModel, CategoryModel, BrandModel

logger = logging.getLogger(__name__)

class MongoFoodRecommender:
    """MongoDB-based food recommendation system."""
    
    def __init__(self, mongo_uri: str, db_name: str, 
                 enable_caching: bool = True):
        """Initialize the MongoDB recommender system.
        
        Args:
            mongo_uri: MongoDB connection URI
            db_name: Database name
            enable_caching: Whether to enable in-memory caching
        """
        self.db_manager = DatabaseManager(mongo_uri, db_name)
        self.product_model = ProductModel(self.db_manager)
        self.category_model = CategoryModel(self.db_manager)
        self.brand_model = BrandModel(self.db_manager)
        
        self.enable_caching = enable_caching
        self._cache = {} if enable_caching else None
        
        # Initialize ML components
        self.tfidf_vectorizer = None
        self.scaler = None
        self.feature_matrix = None
        self.product_codes = None
        
        # Load or build ML models
        self._initialize_ml_models()
        
    def _initialize_ml_models(self):
        """Initialize machine learning models for recommendations."""
        try:
            # Try to load pre-trained models
            self._load_ml_models()
        except Exception as e:
            logger.info(f"Pre-trained models not found, building new ones: {e}")
            self._build_ml_models()
    
    def _load_ml_models(self):
        """Load pre-trained ML models from disk."""
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        
        # Load TF-IDF vectorizer
        with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        
        # Load scaler
        with open(os.path.join(models_dir, 'scaler.pkl'), 'rb') as f:
            self.scaler = pickle.load(f)
          # Load feature matrix
        self.feature_matrix = np.load(os.path.join(models_dir, 'feature_matrix.npy'), allow_pickle=True)
        
        # Load product codes mapping
        with open(os.path.join(models_dir, 'product_codes.json'), 'r') as f:
            self.product_codes = json.load(f)
        
        logger.info("Successfully loaded pre-trained ML models")
    
    def _build_ml_models(self):
        """Build ML models from scratch using MongoDB data."""
        logger.info("Building ML models from MongoDB data...")
        
        # Fetch all products for training
        products = list(self.product_model.collection.find({}, {
            'code': 1, 'product_name': 1, 'ingredients_text': 1, 
            'categories': 1, 'main_category': 1, 'brands': 1,
            'energy_100g': 1, 'fat_100g': 1, 'carbohydrates_100g': 1,
            'proteins_100g': 1, 'salt_100g': 1, 'nutriscore_score': 1
        }))
        
        if len(products) < 100:
            logger.warning("Insufficient data for building ML models")
            return
        
        # Prepare text features
        text_features = []
        numerical_features = []
        product_codes = []
        
        for product in products:
            # Combine text fields
            text_parts = []
            for field in ['product_name', 'ingredients_text', 'categories', 'brands']:
                value = product.get(field, '')
                if value:
                    text_parts.append(str(value))
            
            text_features.append(' '.join(text_parts))
            
            # Extract numerical features
            num_features = []
            for field in ['energy_100g', 'fat_100g', 'carbohydrates_100g', 
                         'proteins_100g', 'salt_100g', 'nutriscore_score']:
                value = product.get(field)
                num_features.append(float(value) if value is not None else 0.0)
            
            numerical_features.append(num_features)
            product_codes.append(str(product['code']))
        
        # Build TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(text_features)
        
        # Normalize numerical features
        self.scaler = StandardScaler()
        numerical_matrix = self.scaler.fit_transform(numerical_features)
        
        # Combine text and numerical features
        from scipy.sparse import hstack
        self.feature_matrix = hstack([tfidf_matrix, numerical_matrix])
        
        # Store product codes mapping
        self.product_codes = {code: idx for idx, code in enumerate(product_codes)}
        
        # Save models
        self._save_ml_models()
        
        logger.info(f"Built ML models with {len(products)} products")
    
    def _save_ml_models(self):
        """Save ML models to disk."""
        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Save TF-IDF vectorizer
        with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.tfidf_vectorizer, f)
        
        # Save scaler
        with open(os.path.join(models_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save feature matrix
        np.save(os.path.join(models_dir, 'feature_matrix.npy'), self.feature_matrix)
        
        # Save product codes mapping
        with open(os.path.join(models_dir, 'product_codes.json'), 'w') as f:
            json.dump(self.product_codes, f)
        
        logger.info("ML models saved to disk")
    
    def search_products(self, query: str, limit: int = 50, 
                       filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for products using text search.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            filters: Additional search filters
            
        Returns:
            List of matching products
        """
        # Check cache first
        cache_key = f"search_{hash(query + str(filters) + str(limit))}"
        if self.enable_caching and cache_key in self._cache:
            return self._cache[cache_key]
        
        results = self.product_model.search_products(query, limit, filters)
        
        # Add to cache
        if self.enable_caching:
            self._cache[cache_key] = results
        
        return results
    
    def get_recommendations(self, product_code: str, 
                          num_recommendations: int = 10,
                          recommendation_type: str = 'similarity') -> List[Dict[str, Any]]:
        """Get product recommendations.
        
        Args:
            product_code: Reference product code
            num_recommendations: Number of recommendations to return
            recommendation_type: Type of recommendation ('similarity', 'category', 'nutritional')
            
        Returns:
            List of recommended products
        """
        # Check cache first
        cache_key = f"recommendations_{product_code}_{num_recommendations}_{recommendation_type}"
        if self.enable_caching and cache_key in self._cache:
            return self._cache[cache_key]
        
        if recommendation_type == 'similarity':
            recommendations = self._get_similarity_recommendations(product_code, num_recommendations)
        elif recommendation_type == 'category':
            recommendations = self._get_category_recommendations(product_code, num_recommendations)
        elif recommendation_type == 'nutritional':
            recommendations = self._get_nutritional_recommendations(product_code, num_recommendations)
        else:
            recommendations = self._get_hybrid_recommendations(product_code, num_recommendations)
        
        # Add to cache
        if self.enable_caching:
            self._cache[cache_key] = recommendations
        
        return recommendations
    
    def _get_similarity_recommendations(self, product_code: str, 
                                      num_recommendations: int) -> List[Dict[str, Any]]:
        """Get recommendations based on content similarity."""
        if not self.feature_matrix or product_code not in self.product_codes:
            # Fallback to simple category-based recommendations
            return self._get_category_recommendations(product_code, num_recommendations)
        
        # Get the product index
        product_idx = self.product_codes[product_code]
        
        # Calculate cosine similarity
        similarities = cosine_similarity(
            self.feature_matrix[product_idx:product_idx+1],
            self.feature_matrix
        ).flatten()
        
        # Get top similar products (excluding the product itself)
        similar_indices = similarities.argsort()[::-1][1:num_recommendations+1]
        
        # Get product codes for similar products
        reverse_mapping = {idx: code for code, idx in self.product_codes.items()}
        similar_codes = [reverse_mapping[idx] for idx in similar_indices if idx in reverse_mapping]
        
        # Fetch product details
        recommendations = []
        for code in similar_codes:
            product = self.product_model.find_by_code(code)
            if product:
                recommendations.append(product)
        
        return recommendations
    
    def _get_category_recommendations(self, product_code: str, 
                                    num_recommendations: int) -> List[Dict[str, Any]]:
        """Get recommendations based on the same category."""
        # Get the reference product
        ref_product = self.product_model.find_by_code(product_code)
        if not ref_product:
            return []
        
        # Find products in the same category
        category = ref_product.get('main_category')
        if not category:
            return []
        
        # Search for products in the same category, excluding the reference product
        query = {
            "main_category": category,
            "code": {"$ne": product_code}
        }
        
        return list(self.product_model.collection.find(query).limit(num_recommendations))
    
    def _get_nutritional_recommendations(self, product_code: str, 
                                       num_recommendations: int) -> List[Dict[str, Any]]:
        """Get recommendations based on similar nutritional profile."""
        # Get the reference product
        ref_product = self.product_model.find_by_code(product_code)
        if not ref_product:
            return []
        
        # Define nutritional similarity criteria
        nutritional_fields = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 'proteins_100g']
        
        # Build query for similar nutritional values (within 20% range)
        query = {"code": {"$ne": product_code}}
        
        for field in nutritional_fields:
            ref_value = ref_product.get(field)
            if ref_value is not None and ref_value > 0:
                tolerance = ref_value * 0.2  # 20% tolerance
                query[field] = {
                    "$gte": ref_value - tolerance,
                    "$lte": ref_value + tolerance
                }
        
        return list(self.product_model.collection.find(query).limit(num_recommendations))
    
    def _get_hybrid_recommendations(self, product_code: str, 
                                  num_recommendations: int) -> List[Dict[str, Any]]:
        """Get hybrid recommendations combining multiple strategies."""
        # Get recommendations from different strategies
        similarity_recs = self._get_similarity_recommendations(product_code, num_recommendations // 2)
        category_recs = self._get_category_recommendations(product_code, num_recommendations // 2)
        
        # Combine and deduplicate
        all_recs = []
        seen_codes = set()
        
        for rec_list in [similarity_recs, category_recs]:
            for rec in rec_list:
                if rec['code'] not in seen_codes:
                    all_recs.append(rec)
                    seen_codes.add(rec['code'])
                
                if len(all_recs) >= num_recommendations:
                    break
            
            if len(all_recs) >= num_recommendations:
                break
        
        return all_recs[:num_recommendations]
    
    def get_product_details(self, product_code: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific product.
        
        Args:
            product_code: Product code
            
        Returns:
            Product details or None if not found
        """
        return self.product_model.find_by_code(product_code)
    
    def get_categories(self) -> List[str]:
        """Get all available product categories.
        
        Returns:
            List of category names
        """
        return self.category_model.get_all_categories()
    
    def get_products_by_category(self, category: str, 
                               limit: int = 50) -> List[Dict[str, Any]]:
        """Get products by category.
        
        Args:
            category: Category name
            limit: Maximum number of products
            
        Returns:
            List of products in the category
        """
        return self.product_model.find_by_category(category, limit)
    
    def get_products_by_country(self, country: str, 
                              limit: int = 50) -> List[Dict[str, Any]]:
        """Get products by country.
        
        Args:
            country: Country name
            limit: Maximum number of products
            
        Returns:
            List of products from the country
        """
        return self.product_model.find_by_country(country, limit)
    
    def get_nutritional_analysis(self, product_codes: List[str]) -> Dict[str, Any]:
        """Get nutritional analysis for a list of products.
        
        Args:
            product_codes: List of product codes
            
        Returns:
            Nutritional analysis results
        """
        products = []
        for code in product_codes:
            product = self.product_model.find_by_code(code)
            if product:
                products.append(product)
        
        if not products:
            return {}
        
        # Calculate averages
        nutritional_fields = ['energy_100g', 'fat_100g', 'carbohydrates_100g', 
                            'proteins_100g', 'salt_100g']
        
        analysis = {'product_count': len(products)}
        
        for field in nutritional_fields:
            values = [p.get(field) for p in products if p.get(field) is not None]
            if values:
                analysis[f'avg_{field}'] = sum(values) / len(values)
                analysis[f'min_{field}'] = min(values)
                analysis[f'max_{field}'] = max(values)
        
        return analysis
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Database statistics
        """
        stats = {
            'total_products': self.product_model.collection.count_documents({}),
            'categories_count': len(self.get_categories()),
            'nutritional_stats': self.product_model.get_nutritional_stats(),
            'top_categories': self.product_model.get_categories_stats()
        }
        
        return stats
    
    def clear_cache(self):
        """Clear the recommendation cache."""
        if self.enable_caching and self._cache:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def recommend_for_recipe(self, country: str, ingredients: List[str], 
                           recipe_description: str = None,
                           nutriscore_filter: List[str] = None,
                           exclude_allergens: List[str] = None,
                           dietary_restrictions: List[str] = None,
                           ecoscore_filter: str = None,
                           packaging_preference: str = None,
                           num_recommendations: int = 15) -> List[Dict[str, Any]]:
        """
        MongoDB-based recipe recommendations implementation.
        
        Args:
            country: Target country for product availability
            ingredients: List of recipe ingredients
            recipe_description: Optional recipe description text
            nutriscore_filter: List of acceptable nutriscore grades (e.g., ['a', 'b'])
            exclude_allergens: List of allergens to exclude
            dietary_restrictions: List of dietary restrictions (vegan, vegetarian, etc.)
            ecoscore_filter: Minimum ecoscore grade
            packaging_preference: Preferred packaging type
            num_recommendations: Number of recommendations to return
            
        Returns:
            List of recommended products with relevance scores
        """
        try:
            # Build MongoDB query pipeline
            pipeline = []
            
            # Step 1: Country filtering
            if country:
                country_match = {
                    "$or": [
                        {"countries_tags": {"$regex": country, "$options": "i"}},
                        {"countries": {"$regex": country, "$options": "i"}}
                    ]
                }
                pipeline.append({"$match": country_match})
            
            # Step 2: Ingredient matching
            if ingredients:
                ingredient_conditions = []
                for ingredient in ingredients:
                    ingredient_conditions.append({
                        "ingredients_text": {"$regex": ingredient, "$options": "i"}
                    })
                
                if ingredient_conditions:
                    pipeline.append({
                        "$match": {
                            "$or": ingredient_conditions
                        }
                    })
            
            # Step 3: Apply filters
            filters = {}
              # Nutriscore filter (convert to lowercase for case-insensitive matching)
            if nutriscore_filter:
                filters["nutriscore_grade"] = {"$in": [grade.lower() for grade in nutriscore_filter]}
            
            # Allergen exclusions
            if exclude_allergens:
                for allergen in exclude_allergens:
                    filters[f"allergens"] = {
                        "$not": {"$regex": allergen, "$options": "i"}
                    }
            
            # Dietary restrictions
            if dietary_restrictions:
                for restriction in dietary_restrictions:
                    restriction_lower = restriction.lower()
                    
                    if restriction_lower == 'vegan':
                        # Exclude animal products
                        animal_keywords = ['milk', 'eggs', 'meat', 'fish', 'chicken', 'beef', 
                                         'pork', 'dairy', 'cheese', 'butter', 'cream', 'whey', 
                                         'casein', 'gelatin', 'honey']
                        filters["ingredients_text"] = {
                            "$not": {"$regex": "|".join(animal_keywords), "$options": "i"}
                        }
                    
                    elif restriction_lower == 'vegetarian':
                        # Exclude meat and fish
                        meat_keywords = ['meat', 'fish', 'chicken', 'beef', 'pork', 'lamb', 
                                       'turkey', 'bacon', 'ham', 'salmon', 'tuna']
                        filters["ingredients_text"] = {
                            "$not": {"$regex": "|".join(meat_keywords), "$options": "i"}
                        }
                    
                    elif restriction_lower in ['gluten-free', 'gluten_free']:
                        # Exclude gluten-containing ingredients
                        gluten_keywords = ['wheat', 'barley', 'rye', 'gluten', 'flour', 'bread', 'pasta']
                        filters["ingredients_text"] = {
                            "$not": {"$regex": "|".join(gluten_keywords), "$options": "i"}
                        }
                    
                    elif restriction_lower in ['dairy-free', 'dairy_free']:
                        # Exclude dairy products
                        dairy_keywords = ['milk', 'dairy', 'cheese', 'butter', 'cream', 'whey', 
                                        'casein', 'lactose']
                        filters["ingredients_text"] = {
                            "$not": {"$regex": "|".join(dairy_keywords), "$options": "i"}
                        }
            
            # Packaging preference
            if packaging_preference:
                filters["packaging"] = {"$regex": packaging_preference, "$options": "i"}
            
            # Apply filters if any
            if filters:
                pipeline.append({"$match": filters})
            
            # Step 4: Calculate relevance scores
            pipeline.append({
                "$addFields": {
                    "ingredient_match_score": {
                        "$divide": [
                            {
                                "$size": {
                                    "$filter": {
                                        "input": ingredients if ingredients else [],
                                        "cond": {
                                            "$regexMatch": {
                                                "input": {"$ifNull": ["$ingredients_text", ""]},
                                                "regex": "$$this",
                                                "options": "i"
                                            }
                                        }
                                    }
                                }
                            },
                            max(len(ingredients), 1) if ingredients else 1
                        ]
                    },
                    "nutriscore_value": {
                        "$switch": {
                            "branches": [
                                {"case": {"$eq": ["$nutriscore_grade", "a"]}, "then": 5},
                                {"case": {"$eq": ["$nutriscore_grade", "b"]}, "then": 4},
                                {"case": {"$eq": ["$nutriscore_grade", "c"]}, "then": 3},
                                {"case": {"$eq": ["$nutriscore_grade", "d"]}, "then": 2},
                                {"case": {"$eq": ["$nutriscore_grade", "e"]}, "then": 1}
                            ],
                            "default": 0
                        }
                    }
                }
            })
            
            # Step 5: Calculate total relevance score
            pipeline.append({
                "$addFields": {
                    "total_relevance_score": {
                        "$add": [
                            {"$multiply": ["$ingredient_match_score", 0.7]},
                            {"$multiply": [{"$divide": ["$nutriscore_value", 5]}, 0.3]}
                        ]
                    }
                }
            })
            
            # Step 6: Sort by relevance and nutritional quality
            pipeline.append({
                "$sort": {
                    "total_relevance_score": -1,
                    "nutriscore_value": -1,
                    "product_name": 1
                }
            })
            
            # Step 7: Limit results
            pipeline.append({"$limit": num_recommendations})
            
            # Execute the aggregation pipeline
            cursor = self.product_model.collection.aggregate(pipeline)
            results = list(cursor)
            
            # Format results
            recommendations = []
            for product in results:
                recommendation = {
                    'code': product.get('code', 'N/A'),
                    'product_name': product.get('product_name', 'N/A'),
                    'brands': product.get('brands', ''),
                    'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
                    'categories': product.get('main_category', ''),
                    'ingredients_text': product.get('ingredients_text', ''),
                    'countries': product.get('countries_tags', ''),
                    'packaging': product.get('packaging', ''),
                    'allergens': product.get('allergens', ''),
                    'relevance_score': float(product.get('total_relevance_score', 0)),
                    'ingredient_match_score': float(product.get('ingredient_match_score', 0)),
                    'nutrition_info': {
                        'energy_100g': product.get('energy_100g', 0),
                        'proteins_100g': product.get('proteins_100g', 0),
                        'carbohydrates_100g': product.get('carbohydrates_100g', 0),
                        'fat_100g': product.get('fat_100g', 0),
                        'sugars_100g': product.get('sugars_100g', 0),
                        'salt_100g': product.get('salt_100g', 0)
                    }
                }
                recommendations.append(recommendation)
            
            logger.info(f"Found {len(recommendations)} recipe recommendations for country: {country}, ingredients: {ingredients}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in recipe-based recommendations: {e}")
            return []
    
    def close(self):
        """Close database connections."""
        self.db_manager.close_connection()
