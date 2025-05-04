# Recommendation engine logic using PySpark
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import CountVectorizerModel
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
        """Load the CountVectorizer model if available."""
        model_path = os.path.join(os.path.dirname(self.data_path), "cv_model")
        
        if os.path.exists(model_path):
            try:
                cv_model = CountVectorizerModel.load(model_path)
                print("CountVectorizer model loaded successfully.")
                return cv_model
            except Exception as e:
                print(f"Error loading CountVectorizer model: {e}")
        
        print("No CountVectorizer model found. Will use existing features.")
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
    
    def get_product_by_code(self, code: str) -> Optional[Dict]:
        """
        Get product details by code.
        
        Args:
            code: Product code.
            
        Returns:
            Product details as a dictionary or None if not found.
        """
        product = self.df.filter(F.col("code") == code).first()
        
        if product is None:
            return None
        
        # Convert to dictionary
        return {
            'code': product.get('code', 'N/A'),
            'product_name': product.get('product_name', 'N/A'),
            'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
            'categories': product.get('categories_en', '')
        }
    
    def search_products(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for products by name.
        
        Args:
            query: Search query.
            limit: Maximum number of results.
            
        Returns:
            List of matching products.
        """
        # Search by product name (case-insensitive)
        matches = self.df.filter(F.lower(F.col("product_name")).contains(query.lower()))
        
        # Convert to list of dictionaries
        results = []
        for product in matches.limit(limit).collect():
            results.append({
                'code': product.get('code', 'N/A'),
                'product_name': product.get('product_name', 'N/A'),
                'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
                'categories': product.get('categories_en', '')
            })
        
        return results
    
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


# Example usage
if __name__ == "__main__":
    try:
        # Initialize the recommender
        recommender = FoodRecommender()
        
        # Example: Search for products
        search_query = "Nutella"
        search_results = recommender.search_products(search_query)
        
        print(f"\nSearch results for '{search_query}':")
        for i, product in enumerate(search_results[:3]):  # Show top 3
            print(f"{i+1}. {product['product_name']} (Code: {product['code']}, Nutriscore: {product['nutriscore_grade']})")
        
        # Get a product code for recommendations
        if search_results:
            product_code = search_results[0]['code']
            
            # Example: Get similar products
            similar_products = recommender.recommend_similar_products(product_code, num_recommendations=3)
            
            print(f"\nProducts similar to '{search_results[0]['product_name']}':")
            for i, product in enumerate(similar_products):
                print(f"{i+1}. {product['product_name']} (Similarity: {product['similarity_score']:.2f}, Nutriscore: {product['nutriscore_grade']})")
            
            # Example: Get healthier alternatives
            healthier_products = recommender.recommend_healthier_alternatives(product_code, num_recommendations=3)
            
            print(f"\nHealthier alternatives to '{search_results[0]['product_name']}':")
            for i, product in enumerate(healthier_products):
                print(f"{i+1}. {product['product_name']} (Similarity: {product['similarity_score']:.2f}, Nutriscore: {product['nutriscore_grade']})")
        
        # Example: Get products by category with good nutriscore
        category_products = recommender.recommend_by_category("breakfast cereal", nutriscore="b", num_recommendations=3)
        
        print(f"\nRecommended breakfast cereals (nutriscore b or better):")
        for i, product in enumerate(category_products):
            print(f"{i+1}. {product['product_name']} (Nutriscore: {product['nutriscore_grade']})")
        
        # Shutdown Spark session
        recommender.shutdown()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        
        # Make sure to stop Spark session in case of error
        try:
            recommender.shutdown()
        except:
            pass
