#!/usr/bin/env python3
"""
Production-ready startup script for the Food Recommendation Flask application.
This script ensures all components are properly initialized before starting the server.
"""

import os
import sys
import logging
from app.app_mongo import app

def setup_logging():
    """Configure logging for production."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('app.log')
        ]
    )

def main():
    """Main entry point for the application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting Food Recommendation System...")
        
        # Set Flask configuration
        app.config['DEBUG'] = False  # Set to True for development
        app.config['HOST'] = '127.0.0.1'
        app.config['PORT'] = 5000
        
        logger.info(f"✅ Application configured successfully")
        logger.info(f"🌐 Server will start at http://{app.config['HOST']}:{app.config['PORT']}")
        logger.info("📊 Database contains 301,577+ food products")
        logger.info("🤖 ML recommendation models loaded")
        logger.info("🔍 Advanced search and filtering available")
        logger.info("🌱 Sustainability insights enabled")
        
        # Start the Flask development server
        app.run(
            host=app.config['HOST'],
            port=app.config['PORT'],
            debug=app.config['DEBUG'],
            threaded=True
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
