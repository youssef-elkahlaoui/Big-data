#!/usr/bin/env python3
"""
Test script to validate the food recommender project structure and basic functionality.
"""

import os
import sys
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.recommender import FoodRecommender
        print("✓ FoodRecommender import successful")
    except ImportError as e:
        print(f"✗ Failed to import FoodRecommender: {e}")
        return False
    
    try:
        from src.utils import get_nutriscore_color, get_file_size
        print("✓ Utils import successful")
    except ImportError as e:
        print(f"✗ Failed to import utils: {e}")
        return False
    
    try:
        from flask import Flask
        print("✓ Flask import successful")
    except ImportError as e:
        print(f"✗ Failed to import Flask: {e}")
        return False
    
    return True

def test_data_files():
    """Test if required data files exist."""
    print("\nTesting data files...")
    
    data_files = [
        'data/food.parquet',
        'data/cleaned_food_data.parquet'
    ]
    
    found_data = False
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✓ Found data file: {file_path}")
            found_data = True
        else:
            print(f"⚠ Data file not found: {file_path}")
    
    return found_data

def test_templates():
    """Test if all required templates exist."""
    print("\nTesting templates...")
    
    template_files = [
        'app/templates/base.html',
        'app/templates/index.html',
        'app/templates/search.html',
        'app/templates/product.html',
        'app/templates/advanced_search.html',
        'app/templates/recipe_recommendations.html',
        'app/templates/nutrition_comparison.html',
        'app/templates/sustainability.html',
        'app/templates/category.html',
        'app/templates/error.html',
        'app/templates/about.html'
    ]
    
    all_present = True
    for template in template_files:
        if os.path.exists(template):
            print(f"✓ {template}")
        else:
            print(f"✗ Missing template: {template}")
            all_present = False
    
    return all_present

def test_static_files():
    """Test if required static files exist."""
    print("\nTesting static files...")
    
    static_files = [
        'app/static/css/style.css',
        'app/static/js/app.js',
        'app/static/js/main.js'
    ]
    
    all_present = True
    for static_file in static_files:
        if os.path.exists(static_file):
            print(f"✓ {static_file}")
        else:
            print(f"✗ Missing static file: {static_file}")
            all_present = False
    
    return all_present

def test_docker_config():
    """Test if Docker configuration files exist."""
    print("\nTesting Docker configuration...")
    
    docker_files = [
        'docker/Dockerfile',
        'docker-compose.yml',
        '.dockerignore'
    ]
    
    all_present = True
    for docker_file in docker_files:
        if os.path.exists(docker_file):
            print(f"✓ {docker_file}")
        else:
            print(f"✗ Missing Docker file: {docker_file}")
            all_present = False
    
    return all_present

def test_notebooks():
    """Test if all required notebooks exist."""
    print("\nTesting notebooks...")
    
    notebook_files = [
        'notebooks/data_ingestion_cleaning.ipynb',
        'notebooks/feature_engineering_eda.ipynb',
        'notebooks/recommender_training_evaluation.ipynb',
        'notebooks/preprocessing.ipynb'
    ]
    
    all_present = True
    for notebook in notebook_files:
        if os.path.exists(notebook):
            print(f"✓ {notebook}")
        else:
            print(f"✗ Missing notebook: {notebook}")
            all_present = False
    
    return all_present

def main():
    """Run all tests."""
    print("Food Recommender System - Project Validation")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    tests = [
        ("Imports", test_imports),
        ("Data Files", test_data_files),
        ("Templates", test_templates),
        ("Static Files", test_static_files),
        ("Docker Config", test_docker_config),
        ("Notebooks", test_notebooks)
    ]
    
    results = {}
    for test_name, test_func in tests:
        results[test_name] = test_func()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("✓ All tests passed! Project is ready.")
    else:
        print("⚠ Some tests failed. Check the output above.")
        print("\nNOTE: Data files are expected to be missing initially.")
        print("Run the preprocessing notebooks to generate the data.")
    
    return all_passed

if __name__ == "__main__":
    main()
