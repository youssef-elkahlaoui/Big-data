# Flask web application with comprehensive food recommendation features
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
import json
import pandas as pd
from datetime import datetime

# Add the src directory to the path to be able to import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.recommender import FoodRecommender
from src.utils import get_nutriscore_color, get_file_size

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for API endpoints

# Common allergens list for filtering
COMMON_ALLERGENS = [
    'gluten', 'milk', 'eggs', 'nuts', 'peanuts', 'sesame', 'soy', 
    'fish', 'crustaceans', 'molluscs', 'celery', 'mustard', 'lupin', 'sulphites'
]

# Countries with significant presence in Open Food Facts
AVAILABLE_COUNTRIES = [
    'france', 'germany', 'united-states', 'united-kingdom', 'spain', 'italy',
    'belgium', 'netherlands', 'canada', 'switzerland', 'japan', 'australia'
]

# Initialize the recommender
try:
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/cleaned_food_data.parquet')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/food.parquet')
        logger.info(f"Cleaned data not found. Attempting to use raw data: {data_path}")
        
    recommender = FoodRecommender(data_path)
    app.config['RECOMMENDER_LOADED'] = True
    app.config['DATA_PATH'] = data_path
    app.config['DATA_SIZE'] = get_file_size(data_path)
    logger.info(f"Recommender system loaded successfully with data from {data_path} ({app.config['DATA_SIZE']})")
except Exception as e:
    app.config['RECOMMENDER_LOADED'] = False
    app.config['ERROR_MESSAGE'] = str(e)
    logger.error(f"Failed to load recommender: {e}")
    logger.error("The app will run in limited mode. Please run the preprocessing pipeline first.")

@app.route('/')
def index():
    """Home page with search form."""
    recommender_status = "Active" if app.config.get('RECOMMENDER_LOADED', False) else "Inactive"
    data_info = {
        'path': app.config.get('DATA_PATH', 'Not loaded'),
        'size': app.config.get('DATA_SIZE', 'Unknown'),
        'error': app.config.get('ERROR_MESSAGE', None)
    }
    
    # Add Spark info if available
    if app.config.get('RECOMMENDER_LOADED', False):
        data_info['using_spark'] = True
        data_info['spark_version'] = recommender.spark.version
    else:
        data_info['using_spark'] = False
    
    return render_template('index.html', recommender_status=recommender_status, data_info=data_info)

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Search for a product by name."""
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
    else:
        query = request.args.get('query', '').strip()
    
    if not query:
        return render_template('search.html', error="Please enter a search term")
    
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('search.html', 
                               error="Recommender system not loaded. Please run the preprocessing pipeline first.")
    
    # Search for the product in the dataframe
    try:
        products = recommender.search_products(query, limit=10)
        
        if not products:
            return render_template('search.html', 
                                query=query, 
                                error=f"No products found matching '{query}'")
        
        return render_template('search.html', query=query, products=products)
    except Exception as e:
        logger.error(f"Error searching for products: {e}")
        return render_template('search.html', 
                               query=query, 
                               error=f"Error processing your search: {str(e)}")

@app.route('/product/<product_code>')
def product_detail(product_code):
    """Show product details and recommendations."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('error.html', 
                               error="Recommender system not loaded. Please run the preprocessing pipeline first.")
    
    # Get product details
    product = recommender.get_product_by_code(product_code)
    
    if product is None:
        return render_template('error.html', error=f"Product not found: {product_code}")
    
    # Get similar products
    try:
        similar_products = recommender.recommend_similar_products(product_code, num_recommendations=5)
        
        # Get healthier alternatives
        healthier_products = recommender.recommend_healthier_alternatives(product_code, num_recommendations=5)
        
        # Format the product data
        product_data = {
            'code': product.get('code', 'N/A'),
            'product_name': product.get('product_name', 'N/A'),
            'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
            'nutriscore_color': get_nutriscore_color(product.get('nutriscore_grade', 'unknown')),
            'categories': product.get('categories', '')
        }
        
        return render_template('product.html', 
                              product=product_data, 
                              similar_products=similar_products,
                              healthier_products=healthier_products)
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return render_template('error.html', 
                               error=f"Error generating recommendations: {str(e)}")

@app.route('/advanced_search', methods=['GET', 'POST'])
def advanced_search():
    """Advanced search with filtering options."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('advanced_search.html', 
                               error="Recommender system not loaded. Please run the preprocessing pipeline first.",
                               allergens=COMMON_ALLERGENS,
                               countries=AVAILABLE_COUNTRIES)
    
    if request.method == 'GET':
        return render_template('advanced_search.html', 
                               allergens=COMMON_ALLERGENS,
                               countries=AVAILABLE_COUNTRIES)
    
    # Process search filters
    filters = {
        'query': request.form.get('query', '').strip(),
        'country': request.form.get('country', ''),
        'nutriscore': request.form.getlist('nutriscore'),
        'exclude_allergens': request.form.getlist('allergens'),
        'min_ecoscore': request.form.get('min_ecoscore', ''),
        'packaging_preference': request.form.get('packaging', ''),
        'category': request.form.get('category', ''),
        'max_results': int(request.form.get('max_results', 20))
    }
    
    try:
        products = recommender.advanced_search(filters)
        
        if not products:
            return render_template('advanced_search.html', 
                                   filters=filters,
                                   allergens=COMMON_ALLERGENS,
                                   countries=AVAILABLE_COUNTRIES,
                                   error="No products found matching your criteria")
        
        return render_template('advanced_search.html', 
                               filters=filters,
                               products=products,
                               allergens=COMMON_ALLERGENS,
                               countries=AVAILABLE_COUNTRIES)
    except Exception as e:
        logger.error(f"Error in advanced search: {e}")
        return render_template('advanced_search.html', 
                               filters=filters,
                               allergens=COMMON_ALLERGENS,
                               countries=AVAILABLE_COUNTRIES,
                               error=f"Error processing your search: {str(e)}")

@app.route('/recipe_recommendations', methods=['GET', 'POST'])
def recipe_recommendations():
    """Recipe-based recommendations."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('recipe_recommendations.html', 
                               error="Recommender system not loaded. Please run the preprocessing pipeline first.")
    
    if request.method == 'GET':
        return render_template('recipe_recommendations.html')
    
    # Get recipe ingredients
    ingredients = request.form.get('ingredients', '').strip()
    dietary_restrictions = request.form.getlist('dietary_restrictions')
    max_results = int(request.form.get('max_results', 15))
    
    if not ingredients:
        return render_template('recipe_recommendations.html', 
                               error="Please enter at least one ingredient")
    
    try:
        recommendations = recommender.recommend_by_ingredients(
            ingredients, 
            dietary_restrictions=dietary_restrictions,
            num_recommendations=max_results
        )
        
        if not recommendations:
            return render_template('recipe_recommendations.html', 
                                   ingredients=ingredients,
                                   error="No products found matching your ingredients")
        
        return render_template('recipe_recommendations.html', 
                               ingredients=ingredients,
                               dietary_restrictions=dietary_restrictions,
                               recommendations=recommendations)
    except Exception as e:
        logger.error(f"Error in recipe recommendations: {e}")
        return render_template('recipe_recommendations.html', 
                               ingredients=ingredients,
                               error=f"Error generating recommendations: {str(e)}")

@app.route('/nutrition_comparison')
def nutrition_comparison():
    """Compare nutrition facts between products."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('nutrition_comparison.html', 
                               error="Recommender system not loaded.")
    
    product_codes = request.args.getlist('products')
    
    if len(product_codes) < 2:
        return render_template('nutrition_comparison.html', 
                               error="Please select at least 2 products to compare")
    
    try:
        comparison_data = recommender.compare_nutrition(product_codes)
        
        return render_template('nutrition_comparison.html', 
                               comparison_data=comparison_data,
                               product_codes=product_codes)
    except Exception as e:
        logger.error(f"Error in nutrition comparison: {e}")
        return render_template('nutrition_comparison.html', 
                               error=f"Error comparing products: {str(e)}")

@app.route('/sustainability_insights')
def sustainability_insights():
    """Sustainability and eco-score insights."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('sustainability.html', 
                               error="Recommender system not loaded.")
    
    try:
        insights = recommender.get_sustainability_insights()
        
        return render_template('sustainability.html', insights=insights)
    except Exception as e:
        logger.error(f"Error getting sustainability insights: {e}")
        return render_template('sustainability.html', 
                               error=f"Error loading sustainability data: {str(e)}")

@app.route('/category')
def category_search():
    """Search for products by category."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('error.html', 
                               error="Recommender system not loaded. Please run the preprocessing pipeline first.")
    
    category = request.args.get('category', '').strip()
    nutriscore = request.args.get('nutriscore', None)
    
    if not category:
        # Get most common categories for suggestions
        categories = []
        if hasattr(recommender, 'df') and 'categories_list' in recommender.df.columns:
            all_cats = []
            for cats in recommender.df['categories_list']:
                if isinstance(cats, list):
                    all_cats.extend(cats)
            
            from collections import Counter
            cat_counts = Counter(all_cats)
            categories = [cat for cat, count in cat_counts.most_common(20)]
        
        return render_template('category.html', categories=categories)
    
    # Get products by category
    products = recommender.recommend_by_category(category, nutriscore, num_recommendations=10)
    
    if not products:
        return render_template('category.html', 
                              category=category,
                              error=f"No products found in category '{category}'")
    
    return render_template('category.html', 
                          category=category,
                          nutriscore=nutriscore,
                          products=products)

@app.route('/api/search')
def api_search():
    """API endpoint for searching products."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return jsonify({"error": "Recommender system not loaded"}), 503
    
    query = request.args.get('query', '').strip()
    
    if not query:
        return jsonify({"error": "No search query provided"}), 400
    
    # Search for the product in the dataframe
    matches = recommender.df[recommender.df['product_name'].str.contains(query, case=False)]
    
    if matches.empty:
        return jsonify({"error": f"No products found matching '{query}"}), 404
    
    # Return the top 10 matches
    products = []
    for _, product in matches.head(10).iterrows():
        products.append({
            'code': product.get('code', 'N/A'),
            'product_name': product.get('product_name', 'N/A'),
            'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
            'categories': product.get('categories_en', '')
        })
    
    return jsonify({"products": products})

@app.route('/api/recommend')
def api_recommend():
    """API endpoint for getting recommendations."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return jsonify({"error": "Recommender system not loaded"}), 503
    
    product_code = request.args.get('product_code', '').strip()
    recommendation_type = request.args.get('type', 'similar')  # 'similar' or 'healthier'
    
    if not product_code:
        return jsonify({"error": "No product code provided"}), 400
    
    # Get product index
    product_idx = recommender.get_product_index(product_code)
    
    if product_idx is None:
        return jsonify({"error": f"Product not found: {product_code}"}), 404
    
    # Get recommendations based on type
    if recommendation_type == 'healthier':
        products = recommender.recommend_healthier_alternatives(product_code, num_recommendations=5)
    else:
        products = recommender.recommend_similar_products(product_code, num_recommendations=5)
    
    return jsonify({"products": products})

@app.route('/about')
def about():
    """About page with information about the project."""
    return render_template('about.html')

# Health check endpoint for Docker
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring and load balancers."""
    try:
        # Basic health checks
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'services': {
                'recommender': 'available',
                'data': 'loaded' if recommender and hasattr(recommender, 'data') and recommender.data is not None else 'not_loaded'
            }
        }
        return jsonify(health_status), 200
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }), 503

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

def shutdown_spark():
    """Shutdown the Spark session when the Flask app exits."""
    if app.config.get('RECOMMENDER_LOADED', False) and 'recommender' in globals():
        try:
            recommender.shutdown()
            logger.info("Spark session successfully shut down.")
        except Exception as e:
            logger.error(f"Error shutting down Spark session: {e}")

# Register the shutdown function with Flask
@app.teardown_appcontext
def teardown_spark(exception=None):
    shutdown_spark()

if __name__ == '__main__':
    # Initialize necessary directories for the app
    app_dir = os.path.dirname(__file__)
    
    # Create template directory if it doesn't exist
    templates_dir = os.path.join(app_dir, 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        logger.info(f"Created templates directory: {templates_dir}")
    
    # Create static directory and subdirectories if they don't exist
    static_dir = os.path.join(app_dir, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        logger.info(f"Created static directory: {static_dir}")
        
    css_dir = os.path.join(static_dir, 'css')
    if not os.path.exists(css_dir):
        os.makedirs(css_dir)
        logger.info(f"Created CSS directory: {css_dir}")
        
    js_dir = os.path.join(static_dir, 'js')
    if not os.path.exists(js_dir):
        os.makedirs(js_dir)
        logger.info(f"Created JS directory: {js_dir}")
    
    # Initialize recommender system
    try:
        logger.info("Initializing Food Recommender System...")
        recommender = FoodRecommender()
        logger.info("Food Recommender System initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize recommender: {str(e)}")
        recommender = None
    
    # Run the application
    app.run(host='0.0.0.0', port=5000, debug=False)