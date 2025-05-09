# Flask web application
from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd

# Add the src directory to the path to be able to import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.recommender import FoodRecommender
from src.utils import get_nutriscore_color, get_file_size

app = Flask(__name__)

# Initialize the recommender
try:
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/cleaned_food_data.parquet')
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/food.parquet')
        print(f"Cleaned data not found. Attempting to use raw data: {data_path}")
        
    recommender = FoodRecommender(data_path)
    app.config['RECOMMENDER_LOADED'] = True
    app.config['DATA_PATH'] = data_path
    app.config['DATA_SIZE'] = get_file_size(data_path)
    print(f"Recommender system loaded successfully with data from {data_path} ({app.config['DATA_SIZE']})")
except Exception as e:
    app.config['RECOMMENDER_LOADED'] = False
    app.config['ERROR_MESSAGE'] = str(e)
    print(f"Failed to load recommender: {e}")
    print("The app will run in limited mode. Please run the preprocessing pipeline first.")

@app.route('/')
def index():
    """Home page with search form."""
    recommender_status = "Active" if app.config.get('RECOMMENDER_LOADED', False) else "Inactive"
    data_info = {
        'path': app.config.get('DATA_PATH', 'Not loaded'),
        'size': app.config.get('DATA_SIZE', 'Unknown'),
        'error': app.config.get('ERROR_MESSAGE', None)
    }
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
    matches = recommender.df[recommender.df['product_name'].str.contains(query, case=False)]
    
    if matches.empty:
        return render_template('search.html', 
                               query=query, 
                               error=f"No products found matching '{query}'")
    
    # Return the top 10 matches
    products = []
    for _, product in matches.head(10).iterrows():
        products.append({
            'code': product.get('code', 'N/A'),
            'product_name': product.get('product_name', 'N/A'),
            'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
            'categories': product.get('categories_en', '')
        })
    
    return render_template('search.html', query=query, products=products)

@app.route('/product/<product_code>')
def product_detail(product_code):
    """Show product details and recommendations."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('error.html', 
                               error="Recommender system not loaded. Please run the preprocessing pipeline first.")
    
    # Get product index
    product_idx = recommender.get_product_index(product_code)
    
    if product_idx is None:
        return render_template('error.html', error=f"Product not found: {product_code}")
    
    # Get product details
    product = recommender.df.iloc[product_idx]
    
    # Get similar products
    similar_products = recommender.recommend_similar_products(product_code, num_recommendations=5)
    
    # Get healthier alternatives
    healthier_products = recommender.recommend_healthier_alternatives(product_code, num_recommendations=5)
    
    # Format the product data
    product_data = {
        'code': product.get('code', 'N/A'),
        'product_name': product.get('product_name', 'N/A'),
        'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
        'nutriscore_color': get_nutriscore_color(product.get('nutriscore_grade', 'unknown')),
        'categories': product.get('categories_en', ''),
        'ingredients': product.get('ingredients_text_en', 'No ingredients information available')
    }
    
    return render_template('product.html', 
                          product=product_data, 
                          similar_products=similar_products,
                          healthier_products=healthier_products)

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
        return jsonify({"error": f"No products found matching '{query}'"}), 404
    
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

@app.errorhandler(404)
def page_not_found(e):
    """Custom 404 page."""
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    """Custom 500 page."""
    return render_template('error.html', error="Server error. Please try again later."), 500

if __name__ == '__main__':
    # Initialize necessary directories for the app
    app_dir = os.path.dirname(__file__)
    
    # Create template directory if it doesn't exist
    templates_dir = os.path.join(app_dir, 'templates')
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        print(f"Created templates directory: {templates_dir}")
    
    # Create static directory and subdirectories if they don't exist
    static_dir = os.path.join(app_dir, 'static')
    if not os.path.exists(static_dir):
        os.makedirs(static_dir)
        print(f"Created static directory: {static_dir}")
        
    css_dir = os.path.join(static_dir, 'css')
    if not os.path.exists(css_dir):
        os.makedirs(css_dir)
        print(f"Created CSS directory: {css_dir}")
        
    js_dir = os.path.join(static_dir, 'js')
    if not os.path.exists(js_dir):
        os.makedirs(js_dir)
        print(f"Created JS directory: {js_dir}")
        
    img_dir = os.path.join(static_dir, 'img')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        print(f"Created images directory: {img_dir}")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(app_dir), 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory: {data_dir}")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)