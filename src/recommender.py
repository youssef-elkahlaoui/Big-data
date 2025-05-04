# Recommendation engine logic (content-based, alternative)
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional, Union

class FoodRecommender:
    def __init__(self, data_path: str = "../data/cleaned_food_data.parquet"):
        """
        Initialize the recommender system with preprocessed data.
        
        Args:
            data_path: Path to the cleaned and preprocessed data.
        """
        self.data_path = data_path
        self.df = pd.read_parquet(data_path)
        self.load_models()
        
    def load_models(self):
        """Load the TF-IDF vectorizer and similarity matrix if available."""
        vectorizer_path = "../data/tfidf_vectorizer.pkl"
        sim_matrix_path = "../data/cosine_sim_matrix.pkl"
        
        # Load TF-IDF vectorizer
        if os.path.exists(vectorizer_path):
            with open(vectorizer_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            print("TF-IDF vectorizer loaded successfully.")
        else:
            self.tfidf_vectorizer = None
            print("Warning: TF-IDF vectorizer file not found.")
        
        # Load similarity matrix if available (for small datasets)
        if os.path.exists(sim_matrix_path):
            with open(sim_matrix_path, 'rb') as f:
                self.cosine_sim = pickle.load(f)
            print("Pre-computed similarity matrix loaded successfully.")
            self.use_precomputed_sim = True
        else:
            self.use_precomputed_sim = False
            print("No pre-computed similarity matrix found. Will compute similarities on-demand.")
    
    def compute_similarity(self, product_idx: int) -> np.ndarray:
        """
        Compute similarity between a product and all other products.
        
        Args:
            product_idx: Index of the product in the dataframe.
            
        Returns:
            Array of similarity scores.
        """
        if self.use_precomputed_sim:
            return self.cosine_sim[product_idx]
        
        # Compute on-demand similarity using TF-IDF vectors
        product_vec = self.tfidf_vectorizer.transform([self.df['ingredients_doc'].iloc[product_idx]])
        all_vectors = self.tfidf_vectorizer.transform(self.df['ingredients_doc'])
        return cosine_similarity(product_vec, all_vectors)[0]
    
    def get_product_index(self, identifier: Union[str, int]) -> Optional[int]:
        """
        Get product index from code or name.
        
        Args:
            identifier: Product code or name.
            
        Returns:
            Index of the product in the dataframe or None if not found.
        """
        if isinstance(identifier, int) or identifier.isdigit():
            # Search by code
            code = str(identifier)
            matches = self.df[self.df['code'] == code]
            if not matches.empty:
                return matches.index[0]
        
        # Search by name (case-insensitive partial match)
        if isinstance(identifier, str):
            matches = self.df[self.df['product_name'].str.contains(identifier, case=False)]
            if not matches.empty:
                return matches.index[0]
        
        return None
    
    def recommend_similar_products(self, 
                                  identifier: Union[str, int], 
                                  num_recommendations: int = 5, 
                                  exclude_same_nutriscore: bool = False) -> List[Dict]:
        """
        Recommend similar products based on ingredients.
        
        Args:
            identifier: Product code or name.
            num_recommendations: Number of recommendations to return.
            exclude_same_nutriscore: If True, exclude products with the same nutriscore.
            
        Returns:
            List of recommended products with similarity scores.
        """
        product_idx = self.get_product_index(identifier)
        if product_idx is None:
            print(f"Product '{identifier}' not found.")
            return []
        
        # Get similarity scores
        similarity_scores = self.compute_similarity(product_idx)
        
        # Get reference product
        reference_product = self.df.iloc[product_idx]
        reference_nutriscore = reference_product.get('nutriscore_grade', None)
        
        # Get indices of most similar products
        similar_indices = similarity_scores.argsort()[::-1]
        
        # Filter out the reference product and possibly same nutriscore products
        recommendations = []
        for idx in similar_indices:
            if idx == product_idx:
                continue
                
            product = self.df.iloc[idx]
            product_nutriscore = product.get('nutriscore_grade', None)
            
            # Skip products with same nutriscore if requested
            if exclude_same_nutriscore and product_nutriscore == reference_nutriscore:
                continue
                
            recommendations.append({
                'code': product.get('code', 'N/A'),
                'product_name': product.get('product_name', 'N/A'),
                'similarity_score': similarity_scores[idx],
                'nutriscore_grade': product_nutriscore,
                'categories': product.get('categories_en', '')
            })
            
            if len(recommendations) >= num_recommendations:
                break
        
        return recommendations
    
    def recommend_healthier_alternatives(self, 
                                        identifier: Union[str, int], 
                                        num_recommendations: int = 5) -> List[Dict]:
        """
        Recommend healthier alternatives based on nutriscore but with similar ingredients.
        
        Args:
            identifier: Product code or name.
            num_recommendations: Number of recommendations to return.
            
        Returns:
            List of healthier alternatives with similarity scores.
        """
        product_idx = self.get_product_index(identifier)
        if product_idx is None:
            print(f"Product '{identifier}' not found.")
            return []
        
        # Get reference product
        reference_product = self.df.iloc[product_idx]
        reference_nutriscore = reference_product.get('nutriscore_grade', None)
        
        # Nutriscore order from best to worst: a, b, c, d, e, unknown
        nutriscore_order = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'unknown': 5}
        
        if reference_nutriscore not in nutriscore_order:
            print(f"Cannot determine nutriscore for product '{reference_product['product_name']}'.")
            return self.recommend_similar_products(identifier, num_recommendations)
        
        # Get similarity scores
        similarity_scores = self.compute_similarity(product_idx)
        
        # Get indices of most similar products
        similar_indices = similarity_scores.argsort()[::-1]
        
        # Filter products with better nutriscore
        recommendations = []
        for idx in similar_indices:
            if idx == product_idx:
                continue
                
            product = self.df.iloc[idx]
            product_nutriscore = product.get('nutriscore_grade', 'unknown')
            
            # Only include products with better (lower) nutriscore
            if product_nutriscore in nutriscore_order and \
               nutriscore_order[product_nutriscore] < nutriscore_order[reference_nutriscore]:
                
                # Only include products with reasonable similarity (e.g., > 0.2)
                if similarity_scores[idx] > 0.2:
                    recommendations.append({
                        'code': product.get('code', 'N/A'),
                        'product_name': product.get('product_name', 'N/A'),
                        'similarity_score': similarity_scores[idx],
                        'nutriscore_grade': product_nutriscore,
                        'categories': product.get('categories_en', '')
                    })
                
                if len(recommendations) >= num_recommendations:
                    break
        
        # If we didn't find enough healthier alternatives, fill with similar products
        if len(recommendations) < num_recommendations:
            print(f"Only found {len(recommendations)} healthier alternatives. Adding similar products to complete.")
            similar_products = self.recommend_similar_products(
                identifier, 
                num_recommendations - len(recommendations),
                exclude_same_nutriscore=True
            )
            recommendations.extend(similar_products)
        
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
        category_matches = self.df[self.df['categories_en'].str.contains(category, case=False)]
        
        if category_matches.empty:
            print(f"No products found in category '{category}'.")
            return []
        
        # Filter by nutriscore if specified
        if nutriscore is not None:
            nutriscore_order = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}
            
            if nutriscore.lower() in nutriscore_order:
                max_score = nutriscore_order[nutriscore.lower()]
                filtered_matches = category_matches[
                    category_matches['nutriscore_grade'].apply(
                        lambda x: x.lower() in nutriscore_order and nutriscore_order[x.lower()] <= max_score
                    )
                ]
                
                if not filtered_matches.empty:
                    category_matches = filtered_matches
        
        # Sort by nutriscore
        nutriscore_map = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'unknown': 5}
        sorted_matches = category_matches.copy()
        
        if 'nutriscore_grade' in sorted_matches.columns:
            sorted_matches['nutriscore_value'] = sorted_matches['nutriscore_grade'].apply(
                lambda x: nutriscore_map.get(str(x).lower(), 5)
            )
            sorted_matches = sorted_matches.sort_values('nutriscore_value')
        
        # Return the top recommendations
        recommendations = []
        for _, product in sorted_matches.head(num_recommendations).iterrows():
            recommendations.append({
                'code': product.get('code', 'N/A'),
                'product_name': product.get('product_name', 'N/A'),
                'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
                'categories': product.get('categories_en', '')
            })
        
        return recommendations


# Example usage
if __name__ == "__main__":
    try:
        recommender = FoodRecommender()
        
        # Example: Get similar products
        product_name = "Nutella"  # Or a product code
        similar_products = recommender.recommend_similar_products(product_name, num_recommendations=3)
        
        print(f"\nProducts similar to '{product_name}':")
        for i, product in enumerate(similar_products):
            print(f"{i+1}. {product['product_name']} (Similarity: {product['similarity_score']:.2f}, Nutriscore: {product['nutriscore_grade']})")
        
        # Example: Get healthier alternatives
        healthier_products = recommender.recommend_healthier_alternatives(product_name, num_recommendations=3)
        
        print(f"\nHealthier alternatives to '{product_name}':")
        for i, product in enumerate(healthier_products):
            print(f"{i+1}. {product['product_name']} (Similarity: {product['similarity_score']:.2f}, Nutriscore: {product['nutriscore_grade']})")
        
        # Example: Get products by category with good nutriscore
        category_products = recommender.recommend_by_category("breakfast cereal", nutriscore="b", num_recommendations=3)
        
        print(f"\nRecommended breakfast cereals (nutriscore b or better):")
        for i, product in enumerate(category_products):
            print(f"{i+1}. {product['product_name']} (Nutriscore: {product['nutriscore_grade']})")
            
    except Exception as e:
        print(f"Error initializing recommender: {e}")
        print("Make sure to run the preprocessing notebook first to generate the required data files.")
