# Flask web application with comprehensive food recommendation features using MongoDB
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
import json
import traceback
from datetime import datetime

# Add the src directory to the path to be able to import modules from there
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.mongo_recommender import MongoFoodRecommender
from src.utils import get_nutriscore_color
from app.config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for API endpoints

# Load configuration
env = os.environ.get('FLASK_ENV', 'development')
app.config.from_object(config[env])

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

# Initialize the MongoDB recommender
try:
    # Initialize MongoDB recommender with configuration
    mongo_uri = app.config['MONGO_URI']
    db_name = app.config['MONGO_DB_NAME']
    
    recommender = MongoFoodRecommender(mongo_uri, db_name)
    app.config['RECOMMENDER_LOADED'] = True
    app.config['DATABASE_STATS'] = recommender.get_database_stats()
    logger.info(f"MongoDB recommender system loaded successfully")
    logger.info(f"Database stats: {app.config['DATABASE_STATS']}")
except Exception as e:
    app.config['RECOMMENDER_LOADED'] = False
    app.config['ERROR_MESSAGE'] = str(e)
    logger.error(f"Failed to load MongoDB recommender: {e}")
    logger.error("The app will run in limited mode. Please ensure MongoDB is running and data is migrated.")
    recommender = None

@app.route('/')
def index():
    """Home page with search form."""
    try:
        # Get basic status information
        recommender_status = "Active" if app.config.get('RECOMMENDER_LOADED', False) else "Inactive"
        
        # Get data info safely
        database_stats = app.config.get('DATABASE_STATS', {})
        error_message = app.config.get('ERROR_MESSAGE', None)
        
        data_info = {
            'database_stats': database_stats,
            'error': error_message,
            'using_mongodb': True
        }
        
        # Log the current status for debugging
        logger.info(f"Index route - Recommender status: {recommender_status}")
        logger.info(f"Index route - Database stats: {database_stats}")
        if error_message:
            logger.warning(f"Index route - Error message: {error_message}")
        
        return render_template('index.html', 
                             recommender_status=recommender_status, 
                             data_info=data_info)
                             
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return a safe error state
        return render_template('index.html', 
                             recommender_status="Error", 
                             data_info={
                                 'error': f"Application error: {str(e)}", 
                                 'using_mongodb': True,
                                 'database_stats': {}
                             })

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Search for a product by name."""
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        nutriscore = request.form.get('nutriscore', '').strip()
        category = request.form.get('category', '').strip()
        brands = request.form.get('brands', '').strip()
    else:
        query = request.args.get('query', '').strip()
        nutriscore = request.args.get('nutriscore', '').strip()
        category = request.args.get('category', '').strip()
        brands = request.args.get('brands', '').strip()
    
    if not query:
        return render_template('search.html', error="Please enter a search term")
    
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('search.html', 
                               error="Recommender system not loaded. Please ensure MongoDB is running and data is migrated.")
    
    # Build filters for quick search
    filters = {}
    if nutriscore:
        filters['nutriscore_grade'] = nutriscore
    if category:
        filters['main_category'] = category
    if brands:
        filters['brands'] = {'$regex': brands, '$options': 'i'}
    
    # Search for the product using MongoDB
    try:
        products = recommender.search_products(query, limit=20, filters=filters if filters else None)
        
        if not products:
            return render_template('search.html', 
                                query=query, 
                                error=f"No products found matching '{query}'" + 
                                      (f" in category '{category}'" if category else "") +
                                      (f" with nutri-score '{nutriscore.upper()}'" if nutriscore else ""))
        
        return render_template('search.html', query=query, products=products)
    except Exception as e:
        logger.error(f"Error searching for products: {e}")
        return render_template('search.html', 
                               query=query, 
                               error=f"Error processing your search: {str(e)}")

@app.route('/product/<product_code>')
def product(product_code):
    """Show product details and recommendations."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('error.html', 
                               error="Recommender system not loaded. Please ensure MongoDB is running and data is migrated.")
    
    # Get product details from MongoDB
    product = recommender.get_product_details(product_code)
    
    if product is None:
        return render_template('error.html', error=f"Product not found: {product_code}")
    
    # Get recommendations
    try:
        similar_products = recommender.get_recommendations(product_code, num_recommendations=5, recommendation_type='similarity')
        category_products = recommender.get_recommendations(product_code, num_recommendations=5, recommendation_type='category')
        nutritional_products = recommender.get_recommendations(product_code, num_recommendations=5, recommendation_type='nutritional')
        
        # Format the product data
        product_data = {
            'code': product.get('code', 'N/A'),
            'product_name': product.get('product_name', 'N/A'),
            'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
            'nutriscore_color': get_nutriscore_color(product.get('nutriscore_grade', 'unknown')),
            'categories': product.get('categories', ''),
            'ingredients_text': product.get('ingredients_text', 'N/A'),
            'brands': product.get('brands', 'N/A'),
            'energy_100g': product.get('energy_100g'),
            'fat_100g': product.get('fat_100g'),
            'carbohydrates_100g': product.get('carbohydrates_100g'),
            'proteins_100g': product.get('proteins_100g'),
            'salt_100g': product.get('salt_100g')        }
        
        return render_template('product.html', 
                              product=product_data, 
                              similar_products=similar_products,
                              category_products=category_products,
                              nutritional_products=nutritional_products)
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return render_template('error.html', 
                               error=f"Error generating recommendations: {str(e)}")

@app.route('/api/product/<product_code>')
def api_product_details(product_code):
    """API endpoint to get product details as JSON for popup display."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return jsonify({'error': 'Recommender system not loaded'}), 500
    
    logger.info(f"API request for product code: {product_code}")
    
    product = None
    
    try:
        # Try multiple approaches to find the product
        
        # 1. Try exact string match on code field
        product = recommender.product_model.collection.find_one({"code": product_code})
        
        # 2. If not found, try converting to float and search
        if product is None:
            try:
                float_code = float(product_code)
                product = recommender.product_model.collection.find_one({"code": float_code})
                logger.info(f"Found product using float conversion: {float_code}")
            except ValueError:
                pass
        
        # 3. Try as MongoDB ObjectId
        if product is None:
            try:
                from bson import ObjectId
                if ObjectId.is_valid(product_code):
                    product = recommender.product_model.collection.find_one({"_id": ObjectId(product_code)})
                    logger.info(f"Found product using ObjectId: {product_code}")
            except Exception:
                pass
        
        # 4. Try string _id (in case it's stored as string)
        if product is None:
            product = recommender.product_model.collection.find_one({"_id": product_code})
        
        # 5. Try a broader search with multiple fields
        if product is None:
            logger.info(f"Trying broader search for: {product_code}")
            product = recommender.product_model.collection.find_one({
                "$or": [
                    {"code": product_code},
                    {"_id": product_code},
                    {"barcode": product_code},
                    {"id": product_code}                ]
            })
        
    except Exception as e:
        logger.error(f"Error searching for product {product_code}: {e}")
        return jsonify({'error': f'Error searching for product: {str(e)}'}), 500
    
    if product is None:
        logger.warning(f"Product not found anywhere: {product_code}")
        return jsonify({'error': f'Product not found: {product_code}'}), 404
    
    try:
        # Format the product data for JSON response
        product_data = {
            'code': str(product.get('code', product.get('_id', 'N/A'))),
            'product_name': product.get('product_name', 'N/A'),
            'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
            'nutriscore_color': get_nutriscore_color(product.get('nutriscore_grade', 'unknown')),
            'categories': product.get('categories', ''),
            'ingredients_text': product.get('ingredients_text', 'N/A'),
            'brands': product.get('brands', 'N/A'),
            'image_url': product.get('image_url', ''),
            'energy_100g': product.get('energy_100g'),
            'fat_100g': product.get('fat_100g'),
            'carbohydrates_100g': product.get('carbohydrates_100g'),
            'proteins_100g': product.get('proteins_100g'),
            'salt_100g': product.get('salt_100g'),
            'sugars_100g': product.get('sugars_100g'),
            'fiber_100g': product.get('fiber_100g'),
            'sodium_100g': product.get('sodium_100g')        }
        
        logger.info(f"Successfully retrieved product: {product_data['product_name']}")
        return jsonify(product_data)
        
    except Exception as e:
        logger.error(f"Error formatting product details: {e}")
        return jsonify({'error': f'Error formatting product details: {str(e)}'}), 500

@app.route('/advanced_search', methods=['GET', 'POST'])
def advanced_search():
    """Advanced search with filtering options."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('advanced_search.html', 
                               error="Recommender system not loaded. Please ensure MongoDB is running and data is migrated.",
                               allergens=COMMON_ALLERGENS,
                               countries=AVAILABLE_COUNTRIES)
    
    # Check if we have any search parameters
    has_params = bool(request.args.get('query') or 
                     request.args.get('country') or 
                     request.args.get('category') or
                     request.args.get('nutriscore') or
                     request.args.get('brands') or
                     request.args.get('max_energy') or
                     request.args.get('min_proteins') or
                     request.args.get('max_sugars') or
                     request.args.get('max_fat') or
                     request.args.get('max_salt'))
    
    if not has_params:
        return render_template('advanced_search.html', 
                               allergens=COMMON_ALLERGENS,
                               countries=AVAILABLE_COUNTRIES)
    
    # Process search filters from GET parameters
    query = request.args.get('query', '').strip()
    country = request.args.get('country', '')
    nutriscore = request.args.get('nutriscore', '')
    category = request.args.get('category', '')
    brands = request.args.get('brands', '').strip()
    max_results = int(request.args.get('limit', 20))
    
    # Nutritional filters
    max_energy = request.args.get('max_energy', '')
    min_proteins = request.args.get('min_proteins', '')
    max_sugars = request.args.get('max_sugars', '')
    max_fat = request.args.get('max_fat', '')
    max_salt = request.args.get('max_salt', '')
      # Build MongoDB query filters
    filters = {}
    if country:
        filters['countries_tags'] = {'$regex': country, '$options': 'i'}
    if nutriscore:
        filters['nutriscore_grade'] = nutriscore.lower()
    if category:
        filters['main_category'] = {'$regex': category, '$options': 'i'}
    if brands:
        filters['brands'] = {'$regex': brands, '$options': 'i'}
    
    # Add nutritional filters
    if max_energy:
        try:
            filters['energy_100g'] = {'$lte': float(max_energy)}
        except ValueError:
            pass
    if min_proteins:
        try:
            filters['proteins_100g'] = {'$gte': float(min_proteins)}
        except ValueError:
            pass
    if max_sugars:
        try:
            filters['sugars_100g'] = {'$lte': float(max_sugars)}
        except ValueError:
            pass
    if max_fat:
        try:
            filters['fat_100g'] = {'$lte': float(max_fat)}
        except ValueError:
            pass
    if max_salt:
        try:
            filters['salt_100g'] = {'$lte': float(max_salt)}
        except ValueError:
            pass
    
    try:
        if query:
            products = recommender.search_products(query, limit=max_results, filters=filters)
        else:
            # Get products with filters only
            products = list(recommender.product_model.collection.find(filters).limit(max_results))
        
        if not products:
            return render_template('advanced_search.html', 
                                   query=query,
                                   country=country,
                                   nutriscore=nutriscore,
                                   category=category,
                                   brands=brands,
                                   allergens=COMMON_ALLERGENS,
                                   countries=AVAILABLE_COUNTRIES,
                                   error="No products found matching your criteria")
        
        return render_template('advanced_search.html', 
                               query=query,
                               country=country,
                               nutriscore=nutriscore,
                               category=category,
                               brands=brands,
                               products=products,
                               allergens=COMMON_ALLERGENS,
                               countries=AVAILABLE_COUNTRIES)
    except Exception as e:
        logger.error(f"Error in advanced search: {e}")
        return render_template('advanced_search.html', 
                               query=query,
                               country=country,
                               nutriscore=nutriscore,
                               category=category,
                               brands=brands,
                               allergens=COMMON_ALLERGENS,
                               countries=AVAILABLE_COUNTRIES,
                               error=f"Error processing your search: {str(e)}")

@app.route('/category')
def category_search():
    """Search for products by category."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('error.html', 
                               error="Recommender system not loaded. Please ensure MongoDB is running and data is migrated.")
    
    category = request.args.get('category', '').strip()
    nutriscore = request.args.get('nutriscore', None)
    
    if not category:
        # Get available categories
        try:
            categories = recommender.get_categories()
            return render_template('category.html', categories=categories[:20])  # Show top 20
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return render_template('category.html', error="Error loading categories")
    
    # Get products by category
    try:
        products = recommender.get_products_by_category(category, limit=20)
        
        # Filter by nutriscore if specified
        if nutriscore:
            products = [p for p in products if p.get('nutriscore_grade') == nutriscore]
        
        if not products:
            return render_template('category.html', 
                                  category=category,
                                  error=f"No products found in category '{category}'")
        
        return render_template('category.html', 
                              category=category,
                              nutriscore=nutriscore,
                              products=products)
    except Exception as e:
        logger.error(f"Error in category search: {e}")
        return render_template('category.html',
                               category=category,
                               error=f"Error searching category: {str(e)}")

@app.route('/nutrition_comparison')
def nutrition_comparison():
    """Compare nutrition facts between products."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('nutrition_comparison.html', 
                               error="Recommender system not loaded.")
    
    product_codes = request.args.getlist('products')
    
    # If no products specified, show empty form
    if len(product_codes) == 0:
        return render_template('nutrition_comparison.html')
    
    # If less than 2 products, show error
    if len(product_codes) < 2:
        return render_template('nutrition_comparison.html', 
                               error="Please select at least 2 products to compare")
    
    try:
        # Get nutritional analysis for the products
        analysis = recommender.get_nutritional_analysis(product_codes)
        
        # Get individual product details
        products = []
        for code in product_codes:
            product = recommender.get_product_details(code)
            if product:
                products.append(product)
        
        return render_template('nutrition_comparison.html', 
                               analysis=analysis,
                               products=products,
                               product_codes=product_codes)
    except Exception as e:
        logger.error(f"Error in nutrition comparison: {e}")
        return render_template('nutrition_comparison.html', 
                               error=f"Error comparing products: {str(e)}")

@app.route('/nutrition-comparison')
def nutrition_comparison_dash():
    """Alternative URL pattern for nutrition comparison (with dash)."""
    return nutrition_comparison()

@app.route('/recipe_recommendations', methods=['GET', 'POST'])
def recipe_recommendations():
    """Recipe recommendations based on ingredients and dietary preferences using ML."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('recipe_recommendations.html', 
                               error="Recommender system not loaded. Please ensure MongoDB is running and data is migrated.")
    
    if request.method == 'GET':
        return render_template('recipe_recommendations.html')
      # Process recipe form data
    try:
        # Extract form data
        ingredients_text = request.form.get('ingredients', '').strip()
        recipe_description = request.form.get('recipe_description', '').strip()
        country = request.form.get('country', '').strip()
        nutriscore_filter_text = request.form.get('nutriscore_filter', '').strip()
        dietary_restrictions = request.form.getlist('dietary_restrictions')
        exclude_allergens = request.form.getlist('exclude_allergens')
        num_recommendations = int(request.form.get('num_recommendations', 15))
        
        # Debug logging
        logger.info(f"DEBUG: Form data received - ingredients: '{ingredients_text}', country: '{country}', nutriscore: '{nutriscore_filter_text}'")
        logger.info(f"DEBUG: All form keys: {list(request.form.keys())}")
        
        # Validate required fields
        if not ingredients_text:
            return render_template('recipe_recommendations.html',
                                   error="Please provide recipe ingredients.")
        
        if not country:
            return render_template('recipe_recommendations.html',
                                   error="Please select a country/region.")
        
        # Process ingredients (split by comma and clean)
        ingredients = [ing.strip() for ing in ingredients_text.split(',') if ing.strip()]
        
        # Process nutriscore filter
        nutriscore_filter = None
        if nutriscore_filter_text:
            nutriscore_filter = [grade.strip() for grade in nutriscore_filter_text.split(',') if grade.strip()]
        
        # Use the advanced ML-based recipe recommendation system
        recommendations = recommender.recommend_for_recipe(
            country=country,
            ingredients=ingredients,
            recipe_description=recipe_description if recipe_description else None,
            nutriscore_filter=nutriscore_filter,
            exclude_allergens=exclude_allergens if exclude_allergens else None,
            dietary_restrictions=dietary_restrictions if dietary_restrictions else None,
            ecoscore_filter=None,  # Can be added later if needed
            packaging_preference=None,  # Can be added later if needed
            num_recommendations=num_recommendations
        )
        
        if not recommendations:
            error_msg = f"No products found matching your criteria for {country}. "
            error_msg += "Try selecting a different country, adjusting your ingredients, or relaxing your filters."
            return render_template('recipe_recommendations.html',
                                   error=error_msg)
        
        logger.info(f"Found {len(recommendations)} recipe recommendations for ingredients: {ingredients}")
        
        return render_template('recipe_recommendations.html',
                               recommendations=recommendations,
                               ingredients_count=len(ingredients))
                               
    except Exception as e:
        logger.error(f"Error in recipe recommendations: {e}")
        return render_template('recipe_recommendations.html',
                               error=f"Error processing your request: {str(e)}")

@app.route('/sustainability')
def sustainability_insights():
    """Sustainability and eco-score insights."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return render_template('sustainability.html', 
                               error="Recommender system not loaded.")
    
    try:
        # Get database statistics for sustainability insights
        stats = recommender.get_database_stats()
        
        return render_template('sustainability.html', stats=stats)
    except Exception as e:
        logger.error(f"Error getting sustainability insights: {e}")
        return render_template('sustainability.html', 
                               error=f"Error loading sustainability data: {str(e)}")

@app.route('/about')
def about():
    """About page with system information."""
    system_info = {
        'version': '2.0.0',
        'database': 'MongoDB',
        'last_updated': datetime.now().strftime('%Y-%m-%d'),
        'features': [
            'MongoDB-based data storage',
            'Advanced search and filtering',
            'Multiple recommendation algorithms',
            'Nutritional analysis and comparison',
            'Real-time product search',
            'Category-based browsing'
        ]
    }
    
    if app.config.get('RECOMMENDER_LOADED', False):
        system_info['database_stats'] = app.config.get('DATABASE_STATS', {})
    
    return render_template('about.html', system_info=system_info)

# API Endpoints
@app.route('/api/search')
def api_search():
    """API endpoint for searching products."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return jsonify({"error": "Recommender system not loaded"}), 503
    
    query = request.args.get('query', '').strip()
    limit = int(request.args.get('limit', 10))
    
    if not query:
        return jsonify({"error": "No search query provided"}), 400
    
    try:
        products = recommender.search_products(query, limit=limit)
        
        # Format products for API response
        api_products = []
        for product in products:
            api_products.append({
                'code': product.get('code', 'N/A'),
                'product_name': product.get('product_name', 'N/A'),
                'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
                'categories': product.get('categories', ''),
                'brands': product.get('brands', 'N/A')
            })
        
        return jsonify({
            "query": query,
            "total_results": len(api_products),
            "products": api_products
        })
    except Exception as e:
        logger.error(f"API search error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/recommend/<product_code>')
def api_recommend(product_code):
    """API endpoint for getting recommendations."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return jsonify({"error": "Recommender system not loaded"}), 503
    
    recommendation_type = request.args.get('type', 'similarity')
    limit = int(request.args.get('limit', 5))
    
    try:
        recommendations = recommender.get_recommendations(
            product_code, 
            num_recommendations=limit,
            recommendation_type=recommendation_type
        )
        
        # Format recommendations for API response
        api_recommendations = []
        for product in recommendations:
            api_recommendations.append({
                'code': product.get('code', 'N/A'),
                'product_name': product.get('product_name', 'N/A'),
                'nutriscore_grade': product.get('nutriscore_grade', 'unknown'),
                'categories': product.get('categories', ''),
                'brands': product.get('brands', 'N/A')
            })
        
        return jsonify({
            "product_code": product_code,
            "recommendation_type": recommendation_type,
            "total_results": len(api_recommendations),
            "recommendations": api_recommendations
        })
    except Exception as e:
        logger.error(f"API recommendation error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for database statistics."""
    if not app.config.get('RECOMMENDER_LOADED', False):
        return jsonify({"error": "Recommender system not loaded"}), 503
    
    try:
        stats = recommender.get_database_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"API stats error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/sustainability/filter', methods=['GET', 'POST'])
def sustainability_filter():
    """API endpoint for sustainability-focused product filtering."""
    try:
        logger.info("=== SUSTAINABILITY FILTER START ===")
        
        # Step 1: Check recommender availability
        if not app.config.get('RECOMMENDER_LOADED', False):
            logger.error("Recommender system not loaded")
            return jsonify({'error': 'Recommender system not loaded'}), 503
        
        if not recommender:
            logger.error("Recommender object is None")
            return jsonify({'error': 'Recommender system not available'}), 503
        
        logger.info("Recommender system check passed")
        
        # Step 2: Get and validate request data
        try:
            if request.method == 'POST':
                data = request.get_json() or {}
                logger.info(f"POST data received: {data}")
            else:
                data = request.args.to_dict()
                logger.info(f"GET data received: {data}")
        except Exception as e:
            logger.error(f"Error getting request data: {e}")
            return jsonify({'error': 'Invalid request data'}), 400
          # Step 3: Extract filter parameters
        eco_score_min = data.get('eco_score_min', '')
        packaging_preference = data.get('packaging_preference', '')
        origin_country = data.get('origin_country', '')
        category = data.get('category', '')
        limit = int(data.get('limit', 50))  # Default to 50 products
        
        logger.info(f"Filter parameters - eco_score: {eco_score_min}, packaging: {packaging_preference}, country: {origin_country}, category: {category}, limit: {limit}")
        
        # Step 4: Build MongoDB query
        try:
            query = {}
            
            # Filter by Nutri-Score (using as eco-score proxy)
            if eco_score_min:
                # Map eco-score to nutri-score for filtering
                score_mapping = {'A': ['a'], 'B': ['a', 'b'], 'C': ['a', 'b', 'c'], 'D': ['a', 'b', 'c', 'd']}
                if eco_score_min in score_mapping:
                    query['nutriscore_grade'] = {'$in': score_mapping[eco_score_min]}
            
            # Filter by category
            if category:
                query['categories'] = {'$regex': category, '$options': 'i'}
              # Filter by origin country
            if origin_country:
                # Handle different country format inputs
                country_search = origin_country.lower().replace('en:', '')
                country_tag = f'en:{country_search}'
                query['countries_tags'] = {'$regex': f'{country_tag}|{country_search}', '$options': 'i'}
            
            logger.info(f"Basic query built: {query}")
            
            # Build sustainability query with proper logic
            if query:  # If we have specific filters, use them
                sustainability_query = {
                    '$and': [
                        query,
                        {
                            '$or': [
                                {'nutriscore_grade': {'$in': ['a', 'b']}},  # Better nutritional quality
                                {'categories': {'$regex': 'organic|plant-based|eco|bio', '$options': 'i'}},
                                {'brands': {'$regex': 'organic|eco|bio|natural', '$options': 'i'}}
                            ]
                        }
                    ]
                }
            else:  # If no specific filters, just use sustainability criteria
                sustainability_query = {
                    '$or': [
                        {'nutriscore_grade': {'$in': ['a', 'b']}},  # Better nutritional quality
                        {'categories': {'$regex': 'organic|plant-based|eco|bio', '$options': 'i'}},
                        {'brands': {'$regex': 'organic|eco|bio|natural', '$options': 'i'}}
                    ]
                }
            
            logger.info(f"Full sustainability query: {sustainability_query}")
            
        except Exception as e:
            logger.error(f"Error building query: {e}")
            return jsonify({'error': f'Error building query: {str(e)}'}), 500
        # Step 5: Execute database query
        try:
            logger.info("Attempting to access product collection...")
            
            # Check if recommender.product_model exists
            if not hasattr(recommender, 'product_model'):
                logger.error("recommender.product_model not found")
                return jsonify({'error': 'Product model not available'}), 500
            
            logger.info("Product model found")
            
            # Check if collection exists
            if not hasattr(recommender.product_model, 'collection'):
                logger.error("recommender.product_model.collection not found")
                return jsonify({'error': 'Collection not available'}), 500
            
            logger.info("Collection found, executing query...")            # Execute query with sustainability focus
            products = list(recommender.product_model.collection.find(
                sustainability_query,
                {
                    'product_name': 1, 'brands': 1, 'categories': 1, 'countries_tags': 1,
                    'nutriscore_grade': 1, 'energy_100g': 1, 'code': 1, '_id': 1,
                    'image_url': 1, 'fat_100g': 1, 'carbohydrates_100g': 1, 'proteins_100g': 1,
                    'salt_100g': 1, 'sugars_100g': 1, 'fiber_100g': 1
                }
            ).limit(limit))
            
            logger.info(f"Query executed successfully. Found {len(products)} products")
            
        except Exception as e:
            logger.error(f"Database query error: {e}")
            logger.error(f"Error type: {type(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Database query failed: {str(e)}'}), 500          # Step 6: Process products and calculate sustainability scores
        try:
            logger.info("Processing sustainability scores...")
              # Calculate sustainability scores for each product
            for product in products:
                # Convert ObjectId to string for JSON serialization
                if '_id' in product:
                    product['_id'] = str(product['_id'])
                
                sustainability_score = 0
                
                # Base score from Nutri-Score (30% of total score)
                nutri_grade = product.get('nutriscore_grade', '').lower()
                grade_scores = {'a': 30, 'b': 24, 'c': 18, 'd': 12, 'e': 6}
                base_score = grade_scores.get(nutri_grade, 0)
                sustainability_score += base_score
                
                # Organic/eco keywords contribution (25% of total score)
                categories = product.get('categories', '').lower()
                brands = product.get('brands', '').lower()
                product_name = product.get('product_name', '').lower()
                text_to_check = categories + ' ' + brands + ' ' + product_name
                
                organic_keywords = ['organic', 'bio', 'natural']
                eco_keywords = ['eco', 'sustainable', 'green', 'plant-based', 'vegan']
                
                organic_score = 0
                if any(keyword in text_to_check for keyword in organic_keywords):
                    organic_score += 15
                if any(keyword in text_to_check for keyword in eco_keywords):
                    organic_score += 10
                sustainability_score += min(organic_score, 25)  # Cap at 25 points                # Local/regional contribution (15% of total score)
                countries_tags = product.get('countries_tags', '')
                if isinstance(countries_tags, list):
                    countries_tags = ','.join(countries_tags).lower()
                else:
                    countries_tags = str(countries_tags).lower()
                
                local_score = 0
                
                if origin_country:
                    # Handle different country format inputs
                    country_search = origin_country.lower().replace('en:', '')
                    country_tag = f'en:{country_search}'
                    
                    # Check for exact match in countries_tags
                    if country_tag in countries_tags or country_search in countries_tags:
                        local_score = 15
                    # Check for partial matches (e.g., "france" in "en:france")
                    elif any(country_search in tag for tag in countries_tags.split(',')):
                        local_score = 15
                elif countries_tags:
                    # Default bonus for European countries when no specific country filter
                    eu_countries = ['en:france', 'en:germany', 'en:spain', 'en:italy', 'en:belgium', 'en:netherlands', 'en:portugal']
                    if any(eu_country in countries_tags for eu_country in eu_countries):
                        local_score = 10
                
                sustainability_score += local_score
                
                # Low processing level bonus (15% of total score)
                if any(keyword in text_to_check for keyword in ['fresh', 'raw', 'whole', 'unprocessed']):
                    sustainability_score += 15
                elif nutri_grade in ['a', 'b']:  # Good nutriscore often indicates less processing
                    sustainability_score += 8
                
                # Packaging consideration (10% of total score)
                packaging_score = 0
                if packaging_preference:
                    if packaging_preference == 'minimal' and any(keyword in text_to_check for keyword in ['bulk', 'loose', 'minimal']):
                        packaging_score = 10
                    elif packaging_preference == 'recyclable' and any(keyword in text_to_check for keyword in ['recyclable', 'cardboard', 'glass']):
                        packaging_score = 10
                    elif packaging_preference == 'biodegradable' and any(keyword in text_to_check for keyword in ['biodegradable', 'compostable']):
                        packaging_score = 10
                else:
                    # Default small bonus for likely sustainable packaging
                    if any(keyword in text_to_check for keyword in ['glass', 'cardboard', 'paper']):
                        packaging_score = 5
                sustainability_score += packaging_score
                
                # Nutritional quality bonus (5% of total score)
                if nutri_grade == 'a' and sustainability_score > 50:
                    sustainability_score += 5  # Extra bonus for top-tier products                # Add sustainability metrics
                product['sustainability_score'] = min(sustainability_score, 100)
                product['eco_score'] = nutri_grade.upper() if nutri_grade else 'Unknown'
                product['is_organic'] = any(keyword in text_to_check for keyword in organic_keywords)
                product['is_local'] = bool(local_score > 0)
                
                # Add for frontend compatibility
                countries_raw = product.get('countries_tags', '')
                if isinstance(countries_raw, list):
                    product['countries'] = ','.join(countries_raw)
                else:
                    product['countries'] = str(countries_raw)
            
            # Sort by sustainability score
            products.sort(key=lambda x: x.get('sustainability_score', 0), reverse=True)
            
            logger.info(f"Successfully processed {len(products)} products")
            
        except Exception as e:
            logger.error(f"Error processing sustainability scores: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Error processing results: {str(e)}'}), 500
        # Step 7: Return successful response
        logger.info(f"=== SUSTAINABILITY FILTER SUCCESS: {len(products)} products ===")
        
        return jsonify({
            'products': products,
            'total_count': len(products),
            'filters_applied': {
                'eco_score_min': eco_score_min,
                'packaging_preference': packaging_preference,
                'origin_country': origin_country,
                'category': category
            }
        })
        
    except Exception as e:
        logger.error(f"=== SUSTAINABILITY FILTER ERROR: {e} ===")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint for monitoring."""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': '2.0.0',
            'services': {
                'recommender': 'available' if app.config.get('RECOMMENDER_LOADED', False) else 'unavailable',
                'database': 'mongodb'
            }
        }
        
        if app.config.get('RECOMMENDER_LOADED', False):
            health_status['database_stats'] = app.config.get('DATABASE_STATS', {})
        
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

# Cleanup on app shutdown
import atexit

def cleanup_on_exit():
    """Clean up resources when app shuts down."""
    if recommender and hasattr(recommender, 'close'):
        try:
            recommender.close()
            logger.info("MongoDB connections closed on app shutdown")
        except Exception as e:
            logger.error(f"Error closing recommender on shutdown: {e}")

# Register cleanup to run when the app exits
atexit.register(cleanup_on_exit)

if __name__ == '__main__':
    # Run the application
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
