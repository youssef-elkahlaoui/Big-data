# Configuration settings for the Flask application
import os
from datetime import timedelta

class Config:
    """Base configuration class."""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # MongoDB Configuration
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://localhost:27017/'
    MONGO_DB_NAME = os.environ.get('MONGO_DB_NAME') or 'food_recommendation_db'
    
    # Collection Names
    PRODUCTS_COLLECTION = 'products'
    CATEGORIES_COLLECTION = 'categories'
    BRANDS_COLLECTION = 'brands'
    COUNTRIES_COLLECTION = 'countries'
    
    # Application Settings
    MAX_SEARCH_RESULTS = 50
    DEFAULT_RECOMMENDATION_COUNT = 10
    
    # Caching Configuration
    CACHE_TIMEOUT = timedelta(hours=1)
    
    # File Upload Settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    MONGO_URI = 'mongodb://localhost:27017/'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb://localhost:27017/'

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    MONGO_DB_NAME = 'food_recommendation_test_db'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
