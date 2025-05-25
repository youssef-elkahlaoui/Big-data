# Recommendation engine logic using PySpark
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import re
from collections import Counter

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import CountVectorizer, IDF, HashingTF
from pyspark.ml.linalg import SparseVector, DenseVector, Vectors
from pyspark.sql.types import StringType, ArrayType, MapType, IntegerType, FloatType, StructType, StructField, BooleanType

class FoodRecommender:
    def __init__(self, data_path: str = "../data/cleaned_food_data.parquet"):
        """
        Initialize the recommender system with preprocessed data using PySpark.
        
        Args:
            data_path: Path to the cleaned and preprocessed data.
        """
        self.data_path = data_path
        self.spark = self._create_spark_session()
        self.df = self._load_data(data_path)
        self.cv_model = self._load_model()
        
    def _create_spark_session(self):
        """Create and return a Spark session."""
        return (SparkSession.builder
                .appName("FoodRecommender")
                .config("spark.driver.memory", "4g")
                .config("spark.executor.memory", "4g")
                .config("spark.sql.shuffle.partitions", "8")
                .getOrCreate())
    
    def _load_data(self, data_path: str) -> DataFrame:
        """Load preprocessed data from Parquet."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        df = self.spark.read.parquet(data_path)
        print(f"Data loaded successfully from {data_path}")
        print(f"Number of rows: {df.count()}, Number of columns: {len(df.columns)}")
        
        # Convert to pandas for quick lookups (optional, remove for very large datasets)
        # self.pdf = df.toPandas()
        
        return df
      def _load_model(self):
        """Initialize vectorization components if needed."""
        # For now, we'll compute similarity on the fly using existing features
        # In a production system, you might want to pre-compute and cache feature vectors
        print("Using on-demand feature computation for similarity.")
        return None
    
    def compute_similarity(self, product_idx: int, num_recommendations: int = 5) -> DataFrame:
        """
        Compute similarity between a product and all other products using PySpark.
        
        Args:
            product_idx: Index of the product in the DataFrame.
            num_recommendations: Number of recommendations to return.
            
        Returns:
            DataFrame with similarity scores.
        """
        # Get the product by index
        product = self.df.limit(product_idx + 1).tail(1)[0]
        
        # Check if TF-IDF features exist
        if "tfidf_features" not in self.df.columns:
            raise ValueError("TF-IDF features not found in the DataFrame. Run preprocessing first.")
        
        # Get the product's TF-IDF vector
        product_vector = product["tfidf_features"]
        
        # Register a UDF to compute cosine similarity
        def cosine_similarity(v1, v2):
            if v1 is None or v2 is None:
                return 0.0
            
            dot = v1.dot(v2)
            norm1 = Vectors.norm(v1, 2)
            norm2 = Vectors.norm(v2, 2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot / (norm1 * norm2))
        
        # Register the UDF
        self.spark.udf.register("cosine_similarity", cosine_similarity, FloatType())
        
        # Create a temporary view of the DataFrame
        self.df.createOrReplaceTempView("products")
        
        # Compute similarities using Spark SQL for better performance
        product_code = product["code"]
        similar_products = self.spark.sql(f"""
            SELECT 
                p.code, 
                p.product_name, 
                p.nutriscore_grade,
                p.categories_en,
                cosine_similarity('{product_vector}', p.tfidf_features) AS similarity_score
            FROM products p
            WHERE p.code != '{product_code}'
            ORDER BY similarity_score DESC
            LIMIT {num_recommendations}
        """)
        
        return similar_products
    
    def advanced_search(self, filters: Dict) -> List[Dict]:
        """
        Advanced search with multiple filtering options.
        
        Args:
            filters: Dictionary containing search filters
                - query: text search
                - country: country filter
                - nutriscore: nutriscore grades list
                - exclude_allergens: allergens to exclude
                - min_ecoscore: minimum ecoscore grade
                - packaging_preference: packaging type preference
                - category: category filter
                - max_results: maximum number of results
        
        Returns:
            List of matching products
        """
        try:
            result_df = self.df
            
            # Text search
            if filters.get('query'):
                query = filters['query'].lower()
                result_df = result_df.filter(
                    F.lower(F.col('product_name')).contains(query) |
                    F.lower(F.col('categories')).contains(query) |
                    F.lower(F.col('ingredients_text')).contains(query)
                )
            
            # Country filter
            if filters.get('country'):
                country = filters['country']
                result_df = result_df.filter(
                    F.lower(F.col('countries')).contains(country.lower())
                )
            
            # Nutriscore filter
            if filters.get('nutriscore'):
                nutriscore_list = filters['nutriscore']
                result_df = result_df.filter(F.col('nutriscore_grade').isin(nutriscore_list))
            
            # Allergen exclusion
            if filters.get('exclude_allergens'):
                for allergen in filters['exclude_allergens']:
                    result_df = result_df.filter(
                        ~F.lower(F.col('allergens')).contains(allergen.lower()) |
                        F.col('allergens').isNull()
                    )
            
            # Ecoscore filter
            if filters.get('min_ecoscore'):
                min_ecoscore = filters['min_ecoscore']
                # Convert letter grades to numbers for comparison
                ecoscore_map = {'a': 5, 'b': 4, 'c': 3, 'd': 2, 'e': 1}
                min_score = ecoscore_map.get(min_ecoscore.lower(), 0)
                
                result_df = result_df.withColumn(
                    'ecoscore_numeric',
                    F.when(F.lower(F.col('ecoscore_grade')) == 'a', 5)
                    .when(F.lower(F.col('ecoscore_grade')) == 'b', 4)
                    .when(F.lower(F.col('ecoscore_grade')) == 'c', 3)
                    .when(F.lower(F.col('ecoscore_grade')) == 'd', 2)
                    .when(F.lower(F.col('ecoscore_grade')) == 'e', 1)
                    .otherwise(0)
                ).filter(F.col('ecoscore_numeric') >= min_score)
            
            # Packaging preference
            if filters.get('packaging_preference'):
                packaging = filters['packaging_preference']
                result_df = result_df.filter(
                    F.lower(F.col('packaging')).contains(packaging.lower())
                )
            
            # Category filter
            if filters.get('category'):
                category = filters['category']
                result_df = result_df.filter(
                    F.lower(F.col('categories')).contains(category.lower())
                )
            
            # Limit results
            max_results = filters.get('max_results', 20)
            result_df = result_df.limit(max_results)
            
            # Convert to list of dictionaries
            products = []
            for row in result_df.collect():
                product = {
                    'code': row.get('code', 'N/A'),
                    'product_name': row.get('product_name', 'N/A'),
                    'nutriscore_grade': row.get('nutriscore_grade', 'unknown'),
                    'ecoscore_grade': row.get('ecoscore_grade', 'unknown'),
                    'categories': row.get('categories', ''),
                    'countries': row.get('countries', ''),
                    'allergens': row.get('allergens', ''),
                    'packaging': row.get('packaging', ''),
                    'energy_100g': row.get('energy_100g', 0),
                    'fat_100g': row.get('fat_100g', 0),
                    'sugars_100g': row.get('sugars_100g', 0),
                    'salt_100g': row.get('salt_100g', 0)
                }
                products.append(product)
            
            return products
            
        except Exception as e:
            print(f"Error in advanced search: {e}")
            return []
    
    def recommend_similar_products(self, 
                                  code: str, 
                                  num_recommendations: int = 5, 
                                  exclude_same_nutriscore: bool = False) -> List[Dict]:
        """
        Recommend similar products based on ingredients.
        
        Args:
            code: Product code.
            num_recommendations: Number of recommendations to return.
            exclude_same_nutriscore: If True, exclude products with the same nutriscore.
            
        Returns:
            List of recommended products with similarity scores.
        """
        # Get the product
        product = self.df.filter(F.col("code") == code).first()
        
        if product is None:
            print(f"Product with code '{code}' not found.")
            return []
        
        # Get the product index
        product_idx = self.df.filter(F.col("code") <= code).count() - 1
        
        # Compute similarities
        similar_products_df = self.compute_similarity(product_idx, num_recommendations * 2)  # Get more to filter
        
        # Filter by nutriscore if requested
        if exclude_same_nutriscore:
            reference_nutriscore = product["nutriscore_grade"]
            similar_products_df = similar_products_df.filter(
                F.col("nutriscore_grade") != reference_nutriscore
            )
        
        # Convert to list of dictionaries
        recommendations = []
        for product in similar_products_df.limit(num_recommendations).collect():
            recommendations.append({
                'code': product.get('code', 'N/A'),
                'product_name': product.get('product_name', 'N/A'),
                'similarity_score': float(product.get('similarity_score', 0)),
                'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
                'categories': product.get('categories_en', '')
            })
        
        return recommendations
    
    def recommend_healthier_alternatives(self, 
                                        code: str, 
                                        num_recommendations: int = 5) -> List[Dict]:
        """
        Recommend healthier alternatives based on nutriscore but with similar ingredients.
        
        Args:
            code: Product code.
            num_recommendations: Number of recommendations to return.
            
        Returns:
            List of healthier alternatives with similarity scores.
        """
        # Get the product
        product = self.df.filter(F.col("code") == code).first()
        
        if product is None:
            print(f"Product with code '{code}' not found.")
            return []
        
        # Get reference nutriscore
        reference_nutriscore = product["nutriscore_grade"]
        
        # Nutriscore order map for comparison (a is best, e is worst)
        nutriscore_order = {
            'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'unknown': 5
        }
        
        # Get product index
        product_idx = self.df.filter(F.col("code") <= code).count() - 1
        
        # Compute similarities
        similar_products_df = self.compute_similarity(product_idx, num_recommendations * 3)  # Get more to filter
        
        # Register UDF to compare nutriscores
        def is_better_nutriscore(score1, score2):
            # Handle missing values
            s1 = score1.lower() if score1 else 'unknown'
            s2 = score2.lower() if score2 else 'unknown'
            
            # If either score is not in our map, consider it unknown
            if s1 not in nutriscore_order or s2 not in nutriscore_order:
                return False
            
            # Lower value means better score (a=0 is best, e=4 is worst)
            return nutriscore_order[s1] < nutriscore_order[s2]
        
        # Register the UDF
        self.spark.udf.register("is_better_nutriscore", is_better_nutriscore, returnType=BooleanType())
        
        # Filter to get only healthier alternatives with good similarity
        healthier_products_df = similar_products_df.filter(
            (is_better_nutriscore(F.col("nutriscore_grade"), F.lit(reference_nutriscore))) &
            (F.col("similarity_score") > 0.2)  # Only include products with reasonable similarity
        )
        
        # Convert to list of dictionaries
        recommendations = []
        for product in healthier_products_df.limit(num_recommendations).collect():
            recommendations.append({
                'code': product.get('code', 'N/A'),
                'product_name': product.get('product_name', 'N/A'),
                'similarity_score': float(product.get('similarity_score', 0)),
                'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
                'categories': product.get('categories_en', '')
            })
        
        # If we didn't find enough healthier alternatives, fill with similar products
        if len(recommendations) < num_recommendations:
            print(f"Only found {len(recommendations)} healthier alternatives. Adding similar products to complete.")
            
            additional_similar = self.recommend_similar_products(
                code,
                num_recommendations - len(recommendations),
                exclude_same_nutriscore=True
            )
            
            # Add the additional similar products
            recommendations.extend(additional_similar)
        
        return recommendations
    
    def recommend_by_category(self, 
                             category: str, 
                             nutriscore: Optional[str] = None, 
                             num_recommendations: int = 5) -> List[Dict]:
        """
        Recommend products in a specific category, optionally with a minimum nutriscore.
        
        Args:
            category: Category name or partial match.
            nutriscore: Minimum nutriscore grade (a is best, e is worst).
            num_recommendations: Number of recommendations to return.
            
        Returns:
            List of recommended products in the category.
        """
        # Filter products by category (case-insensitive partial match)
        category_matches = self.df.filter(F.lower(F.col("categories_en")).contains(category.lower()))
        
        if category_matches.count() == 0:
            print(f"No products found in category '{category}'.")
            return []
        
        # Filter by nutriscore if specified
        if nutriscore is not None:
            nutriscore_order = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
            nutriscore = nutriscore.lower()
            
            if nutriscore in nutriscore_order:
                # Register UDF to filter by nutriscore
                def has_better_or_equal_nutriscore(score, min_score):
                    # Handle missing values
                    s = score.lower() if score else 'unknown'
                    
                    # If score is not in our map, consider it unknown
                    if s not in nutriscore_order:
                        return False
                    
                    # Check if score is better than or equal to min_score
                    return nutriscore_order[s] <= nutriscore_order[min_score]
                
                # Register the UDF
                self.spark.udf.register("has_better_or_equal_nutriscore", has_better_or_equal_nutriscore, BooleanType())
                
                # Apply nutriscore filter
                category_matches = category_matches.filter(has_better_or_equal_nutriscore(F.col("nutriscore_grade"), F.lit(nutriscore)))
                
                if category_matches.count() == 0:
                    print(f"No products found with nutriscore {nutriscore} or better in category '{category}'.")
                    return []
        
        # Sort by nutriscore
        nutriscore_values = {
            'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'unknown': 5
        }
        
        # Register UDF for nutriscore value
        def get_nutriscore_value(score):
            # Handle missing values
            s = score.lower() if score else 'unknown'
            
            # Return numerical value
            return nutriscore_values.get(s, 5)  # Default to 5 (unknown) if not found
        
        # Register the UDF
        self.spark.udf.register("get_nutriscore_value", get_nutriscore_value, IntegerType())
        
        # Order by nutriscore value
        sorted_matches = category_matches.withColumn(
            "nutriscore_value", get_nutriscore_value(F.col("nutriscore_grade"))
        ).orderBy("nutriscore_value")
        
        # Convert to list of dictionaries
        recommendations = []
        for product in sorted_matches.limit(num_recommendations).collect():
            recommendations.append({
                'code': product.get('code', 'N/A'),
                'product_name': product.get('product_name', 'N/A'),
                'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
                'categories': product.get('categories_en', '')
            })
        
        return recommendations
    
    def shutdown(self):
        """Stop the Spark session when done."""
        if hasattr(self, 'spark') and self.spark is not None:
            self.spark.stop()
            print("Spark session stopped.")
    
    def compare_nutrition(self, product_codes: List[str]) -> Dict:
        """
        Compare nutrition facts between multiple products.
        
        Args:
            product_codes: List of product codes to compare
        
        Returns:
            Dictionary containing comparison data
        """
        try:
            # Get products by codes
            products_df = self.df.filter(F.col('code').isin(product_codes))
            
            if products_df.count() == 0:
                return {"error": "No products found"}
            
            # Define nutrition columns to compare
            nutrition_cols = [
                'energy_100g', 'fat_100g', 'saturated_fat_100g', 'carbohydrates_100g',
                'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g', 'sodium_100g'
            ]
            
            comparison_data = {
                'products': [],
                'nutrition_comparison': {},
                'rankings': {}
            }
            
            # Collect product data
            for row in products_df.collect():
                product_data = {
                    'code': row.get('code', 'N/A'),
                    'product_name': row.get('product_name', 'N/A'),
                    'nutriscore_grade': row.get('nutriscore_grade', 'unknown'),
                    'nutrition': {}
                }
                
                for col in nutrition_cols:
                    value = row.get(col, 0)
                    product_data['nutrition'][col] = float(value) if value else 0.0
                
                comparison_data['products'].append(product_data)
            
            # Create comparison tables
            for col in nutrition_cols:
                comparison_data['nutrition_comparison'][col] = []
                values = []
                
                for product in comparison_data['products']:
                    value = product['nutrition'][col]
                    comparison_data['nutrition_comparison'][col].append({
                        'product_name': product['product_name'],
                        'value': value
                    })
                    values.append(value)
                
                # Rank products for this nutrient (lower is better for most)
                if col in ['energy_100g', 'fat_100g', 'saturated_fat_100g', 'sugars_100g', 'salt_100g']:
                    # Lower is better
                    sorted_products = sorted(comparison_data['nutrition_comparison'][col], key=lambda x: x['value'])
                else:
                    # Higher is better (for fiber, proteins)
                    sorted_products = sorted(comparison_data['nutrition_comparison'][col], key=lambda x: x['value'], reverse=True)
                
                comparison_data['rankings'][col] = sorted_products
            
            return comparison_data
            
        except Exception as e:
            print(f"Error in nutrition comparison: {e}")
            return {"error": str(e)}
    
    def get_sustainability_insights(self) -> Dict:
        """
        Get sustainability insights from the dataset.
        
        Returns:
            Dictionary containing sustainability metrics and insights
        """
        try:
            insights = {
                'ecoscore_distribution': {},
                'packaging_analysis': {},
                'country_sustainability': {},
                'recommendations': []
            }
            
            # Ecoscore distribution
            ecoscore_counts = self.df.groupBy('ecoscore_grade').count().collect()
            total_products = self.df.count()
            
            for row in ecoscore_counts:
                grade = row['ecoscore_grade'] if row['ecoscore_grade'] else 'unknown'
                count = row['count']
                percentage = (count / total_products) * 100
                insights['ecoscore_distribution'][grade] = {
                    'count': count,
                    'percentage': round(percentage, 2)
                }
            
            # Packaging analysis
            packaging_df = self.df.filter(F.col('packaging').isNotNull() & (F.col('packaging') != ''))
            
            # Count eco-friendly packaging keywords
            eco_keywords = ['recycled', 'recyclable', 'cardboard', 'paper', 'glass']
            non_eco_keywords = ['plastic', 'polystyrene', 'pvc']
            
            eco_friendly_count = 0
            non_eco_count = 0
            
            for keyword in eco_keywords:
                count = packaging_df.filter(F.lower(F.col('packaging')).contains(keyword)).count()
                eco_friendly_count += count
            
            for keyword in non_eco_keywords:
                count = packaging_df.filter(F.lower(F.col('packaging')).contains(keyword)).count()
                non_eco_count += count
            
            insights['packaging_analysis'] = {
                'eco_friendly_packaging': eco_friendly_count,
                'non_eco_packaging': non_eco_count,
                'eco_friendly_percentage': round((eco_friendly_count / packaging_df.count()) * 100, 2) if packaging_df.count() > 0 else 0
            }
            
            # Country sustainability (top eco-score countries)
            country_eco_df = self.df.filter(
                F.col('countries').isNotNull() & 
                F.col('ecoscore_grade').isNotNull() &
                F.col('ecoscore_grade').isin(['a', 'b'])
            ).groupBy('countries').count().orderBy(F.desc('count')).limit(10)
            
            country_sustainability = []
            for row in country_eco_df.collect():
                country_sustainability.append({
                    'country': row['countries'],
                    'high_eco_products': row['count']
                })
            
            insights['country_sustainability'] = country_sustainability
            
            # Generate sustainability recommendations
            insights['recommendations'] = [
                "Choose products with A or B eco-scores for better environmental impact",
                "Look for products with recyclable or minimal packaging",
                "Consider locally produced items to reduce transportation emissions",
                "Prefer products with organic or sustainable ingredient sourcing"
            ]
            
            return insights
            
        except Exception as e:
            print(f"Error getting sustainability insights: {e}")
            return {"error": str(e)}

    def get_product_by_code(self, product_code: str) -> Optional[Dict]:
        """Enhanced product lookup with more details."""
        try:
            product_row = self.df.filter(F.col('code') == product_code).first()
            
            if not product_row:
                return None
            
            # Convert Row to dictionary with all available fields
            product = {}
            for field in product_row.asDict():
                product[field] = product_row[field]
            
            return product
            
        except Exception as e:
            print(f"Error getting product by code: {e}")
            return None

    def search_products(self, query: str, limit: int = 10) -> List[Dict]:
        """Enhanced product search with better ranking."""
        try:
            query_lower = query.lower()
            
            # Search in multiple fields with different weights
            search_df = self.df.filter(
                F.lower(F.col('product_name')).contains(query_lower) |
                F.lower(F.col('categories')).contains(query_lower) |
                F.lower(F.col('brands')).contains(query_lower) |
                F.lower(F.col('ingredients_text')).contains(query_lower)
            )
            
            # Add relevance scoring
            search_df = search_df.withColumn(
                'relevance_score',
                F.when(F.lower(F.col('product_name')).contains(query_lower), 10).otherwise(0) +
                F.when(F.lower(F.col('brands')).contains(query_lower), 5).otherwise(0) +
                F.when(F.lower(F.col('categories')).contains(query_lower), 3).otherwise(0) +
                F.when(F.lower(F.col('ingredients_text')).contains(query_lower), 1).otherwise(0)
            ).orderBy(F.desc('relevance_score'), F.desc('nutriscore_grade')).limit(limit)
            
            # Convert to list
            products = []
            for row in search_df.collect():
                product = {
                    'code': row.get('code', 'N/A'),
                    'product_name': row.get('product_name', 'N/A'),
                    'nutriscore_grade': row.get('nutriscore_grade', 'unknown'),
                    'categories': row.get('categories', ''),
                    'brands': row.get('brands', ''),
                    'relevance_score': row.get('relevance_score', 0)
                }
                products.append(product)
            
            return products
            
        except Exception as e:
            print(f"Error searching products: {e}")
            return []
        
    def recommend_by_ingredients(self, ingredients: List[str], dietary_restrictions: List[str] = None, exclude_allergens: List[str] = None, num_recommendations: int = 10) -> List[Dict]:
        """
        Recommend products based on desired ingredients and dietary restrictions.
        
        Args:
            ingredients: List of desired ingredients
            dietary_restrictions: List of dietary restrictions (e.g., 'vegan', 'vegetarian', 'gluten-free')
            exclude_allergens: List of allergens to exclude
            num_recommendations: Number of recommendations to return
        
        Returns:
            List of recommended products matching the criteria
        """
        try:
            # Start with all products
            result_df = self.df
            
            # Filter by ingredients (products that contain ANY of the desired ingredients)
            if ingredients:
                ingredient_conditions = []
                for ingredient in ingredients:
                    ingredient_conditions.append(
                        F.lower(F.col('ingredients_text')).contains(ingredient.lower())
                    )
                
                # Combine conditions with OR
                combined_condition = ingredient_conditions[0]
                for condition in ingredient_conditions[1:]:
                    combined_condition = combined_condition | condition
                
                result_df = result_df.filter(combined_condition)
            
            # Apply dietary restrictions
            if dietary_restrictions:
                for restriction in dietary_restrictions:
                    restriction_lower = restriction.lower()
                    
                    if restriction_lower == 'vegan':
                        # Exclude animal products
                        animal_keywords = ['milk', 'eggs', 'meat', 'fish', 'chicken', 'beef', 'pork', 'dairy', 'cheese', 'butter', 'cream', 'whey', 'casein', 'gelatin', 'honey']
                        for keyword in animal_keywords:
                            result_df = result_df.filter(
                                ~F.lower(F.col('ingredients_text')).contains(keyword) |
                                F.col('ingredients_text').isNull()
                            )
                    
                    elif restriction_lower == 'vegetarian':
                        # Exclude meat and fish
                        meat_keywords = ['meat', 'fish', 'chicken', 'beef', 'pork', 'lamb', 'turkey', 'bacon', 'ham', 'salmon', 'tuna']
                        for keyword in meat_keywords:
                            result_df = result_df.filter(
                                ~F.lower(F.col('ingredients_text')).contains(keyword) |
                                F.col('ingredients_text').isNull()
                            )
                    
                    elif restriction_lower == 'gluten-free' or restriction_lower == 'gluten_free':
                        # Exclude gluten-containing ingredients
                        gluten_keywords = ['wheat', 'barley', 'rye', 'gluten', 'flour', 'bread', 'pasta']
                        for keyword in gluten_keywords:
                            result_df = result_df.filter(
                                ~F.lower(F.col('ingredients_text')).contains(keyword) |
                                F.col('ingredients_text').isNull()
                            )
                    
                    elif restriction_lower == 'dairy-free' or restriction_lower == 'dairy_free':
                        # Exclude dairy products
                        dairy_keywords = ['milk', 'dairy', 'cheese', 'butter', 'cream', 'whey', 'casein', 'lactose']
                        for keyword in dairy_keywords:
                            result_df = result_df.filter(
                                ~F.lower(F.col('ingredients_text')).contains(keyword) |
                                F.col('ingredients_text').isNull()
                            )
            
            # Exclude allergens
            if exclude_allergens:
                for allergen in exclude_allergens:
                    result_df = result_df.filter(
                        ~F.lower(F.col('allergens')).contains(allergen.lower()) |
                        F.col('allergens').isNull()
                    )
            
            # Calculate ingredient match score
            if ingredients:
                # Create UDF to calculate ingredient match score
                def calculate_ingredient_score(ingredients_text, target_ingredients):
                    if not ingredients_text:
                        return 0.0
                    
                    ingredients_lower = ingredients_text.lower()
                    score = 0.0
                    
                    for ingredient in target_ingredients:
                        if ingredient.lower() in ingredients_lower:
                            score += 1.0
                    
                    # Normalize by number of target ingredients
                    return score / len(target_ingredients)
                
                # Register UDF
                self.spark.udf.register("calculate_ingredient_score", 
                                       lambda x: calculate_ingredient_score(x, ingredients), 
                                       FloatType())
                
                # Add ingredient score column
                result_df = result_df.withColumn(
                    'ingredient_score',
                    calculate_ingredient_score(F.col('ingredients_text'))
                )
                
                # Order by ingredient score and nutriscore
                result_df = result_df.orderBy(F.desc('ingredient_score'), F.asc('nutriscore_grade'))
            else:
                # Order by nutriscore if no specific ingredients
                result_df = result_df.orderBy(F.asc('nutriscore_grade'))
            
            # Limit results
            result_df = result_df.limit(num_recommendations)
            
            # Convert to list of dictionaries
            recommendations = []
            for row in result_df.collect():
                product = {
                    'code': row.get('code', 'N/A'),
                    'product_name': row.get('product_name', 'N/A'),
                    'nutriscore_grade': row.get('nutriscore_grade', 'unknown'),
                    'ecoscore_grade': row.get('ecoscore_grade', 'unknown'),
                    'categories': row.get('categories', ''),
                    'ingredients_text': row.get('ingredients_text', ''),
                    'allergens': row.get('allergens', ''),
                    'ingredient_score': float(row.get('ingredient_score', 0)) if ingredients else 0.0
                }
                recommendations.append(product)
            
            return recommendations
            
        except Exception as e:
            print(f"Error in ingredient-based recommendations: {e}")
            return []
