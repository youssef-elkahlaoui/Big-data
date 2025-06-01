# 🍽️ Food Recommendation System

**A modern Flask web application for food product recommendations using the Open Food Facts dataset.**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Project Structure](#project-structure)
6. [Technical Details](#technical-details)
7. [API Endpoints](#api-endpoints)
8. [Development](#development)

---

## 🎯 Overview

This Flask web application provides intelligent food product recommendations using machine learning and the Open Food Facts dataset. It offers content-based filtering, nutritional analysis, and health-focused product suggestions through a modern, responsive web interface.

### Key Capabilities

- **Product Search**: Find food products by name, brand, or category
- **Smart Recommendations**: ML-powered similar product suggestions
- **Nutritional Analysis**: Compare nutritional profiles and health scores
- **Health Alternatives**: Discover healthier product alternatives
- **Category Browsing**: Explore products by food categories
- **Sustainability Insights**: Environmental impact information

---

## ✨ Features

### 🔍 Search & Discovery

- **Advanced Search**: Multi-criteria product filtering
- **Category Navigation**: Browse by food categories
- **Real-time Results**: Fast, responsive search experience

### 🤖 AI-Powered Recommendations

- **Content-Based Filtering**: Recommendations based on ingredients and nutritional profiles
- **Health-Focused Suggestions**: Prioritize products with better nutritional scores
- **Similarity Matching**: Find products with similar characteristics

### 📊 Nutritional Intelligence

- **Nutri-Score Integration**: Color-coded nutritional ratings (A-E)
- **Detailed Nutrition Facts**: Comprehensive nutritional breakdowns
- **Health Comparisons**: Side-by-side nutritional comparisons

### 🌱 Sustainability Features

- **Environmental Impact**: Eco-score and sustainability metrics
- **Carbon Footprint**: Environmental impact assessments

---

## 🏗️ Architecture

### Technology Stack

- **Backend**: Flask (Python 3.8+)
- **Database**: MongoDB for product data storage
- **ML Engine**: Scikit-learn for recommendations
- **Frontend**: Modern HTML5, CSS3, JavaScript
- **Styling**: Tailwind CSS for responsive design
- **Data Processing**: Pandas, NumPy for data manipulation

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │────│  Flask Backend  │────│   MongoDB DB    │
│   (Frontend)    │    │   (API Layer)   │    │  (Data Store)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  ML Recommender │
                       │    (Engine)     │
                       └─────────────────┘
```

### Data Flow

1. **Data Ingestion**: Open Food Facts dataset → MongoDB
2. **Feature Engineering**: Product features → ML-ready format
3. **Model Training**: Similarity matrices and recommendation models
4. **Real-time Inference**: User queries → ML predictions → Recommendations

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- MongoDB (optional - can run with CSV data)
- 4GB+ RAM recommended

### Installation & Run

#### Method 1: One-Click Start (Recommended)

```bash
# Double-click this file in Windows Explorer
start_webapp.bat
```

#### Method 2: PowerShell

```powershell
.\start_flask.ps1
```

#### Method 3: Python Script

```bash
python direct_start.py
```

#### Method 4: Manual Flask Start

```bash
cd app
python app_mongo.py
```

### Access the Application

After starting, open your browser and navigate to:
**http://localhost:5000**

---

## 📁 Project Structure

```
Big data/
├── 📄 Documentation
│   └── README.md                     # This file
├── 🚀 Startup Scripts
│   ├── start_webapp.bat              # Windows batch starter
│   ├── direct_start.py               # Python starter
│   └── start_flask.ps1               # PowerShell starter
├── ⚙️ Configuration
│   ├── requirements.txt              # Python dependencies
│   ├── docker-compose.yml            # Docker configuration
│   └── .dockerignore                 # Docker ignore rules
├── 🏗️ Application Core
│   └── app/
│       ├── app_mongo.py              # Main Flask application
│       ├── app.py                    # Alternative Flask app
│       ├── config.py                 # Configuration settings
│       ├── models.py                 # Database models
│       ├── static/                   # Frontend assets
│       └── templates/                # HTML templates
├── 📊 Data Storage
│   └── data/
│       ├── cleaned_food_data_filtered.csv      # Processed dataset
│       ├── engineered_features_filtered.csv    # ML features
│       └── feature_metadata_filtered.json      # Feature metadata
├── 🤖 Machine Learning
│   └── models/
│       ├── feature_matrix.npy        # Similarity matrix
│       ├── model_metadata.json       # Model configuration
│       ├── scaler.pkl                # Feature scaler
│       └── tfidf_vectorizer.pkl      # Text vectorizer
├── 🔧 Source Code
│   └── src/
│       ├── mongo_recommender.py      # MongoDB-based recommender
│       ├── recommender.py            # Core recommendation engine
│       ├── preprocessing.py          # Data preprocessing
│       └── utils.py                  # Utility functions
└── 📜 Utilities
    └── scripts/
        ├── migrate_to_mongodb.py     # Data migration script
        ├── setup_mongodb.ps1         # MongoDB setup
        └── test_mongodb_integration.py # Database testing
```

---

## 🔧 Technical Details

### Dependencies

```python
# Core Framework
Flask>=2.3.0
Flask-CORS>=4.0.0

# Database
pymongo>=4.5.0

# Machine Learning
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0

# Additional
requests>=2.31.0
python-dotenv>=1.0.0
```

### Performance Features

- **Caching**: In-memory caching for frequent queries
- **Lazy Loading**: On-demand model loading
- **Optimized Queries**: Efficient database operations
- **Responsive Design**: Mobile-first interface

---

## 🌐 API Endpoints

### Web Interface

- `GET /` - Homepage with search interface
- `GET /search` - Product search page
- `GET /advanced_search` - Advanced filtering options
- `GET /product/<id>` - Individual product details
- `GET /category/<name>` - Category browsing

### API Routes

- `GET /api/search` - Product search API
- `GET /api/product/<id>` - Product data API
- `GET /api/recommendations/<id>` - Product recommendations
- `GET /api/stats` - System statistics
- `GET /health` - Health check endpoint

---

## 🛠️ Development

### Local Development Setup

1. **Clone/Download** the project
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Start development server**: `python app/app_mongo.py`
4. **Access at**: http://localhost:5000

### Database Options

- **CSV Mode**: Uses local CSV files (default)
- **MongoDB**: Full database integration
- **Docker**: Containerized MongoDB setup

### Docker Deployment

```bash
# Build and run with Docker
docker-compose up --build
```

---

## 📈 Status

**Current Version**: 1.0  
**Last Updated**: May 31, 2025

### Features Complete

- ✅ Modern web interface
- ✅ Machine learning recommendations
- ✅ MongoDB integration
- ✅ Docker support
- ✅ API endpoints
- ✅ Responsive design
- ✅ Health scoring
- ✅ Category browsing

---