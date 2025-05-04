# Food Recommender System

A recommender system for food products based on the Open Food Facts dataset. This project uses big data tools to provide content-based recommendations and healthier alternatives for food products.

## Overview

This application uses the Open Food Facts dataset to:

1. Find similar products based on ingredients and categories
2. Recommend healthier alternatives with better nutritional profiles
3. Browse products by category and nutritional scores

## Project Structure

```
.
├── app/                    # Web application
│   ├── app.py              # Flask application
│   ├── static/             # Static assets (CSS, JS)
│   └── templates/          # HTML templates
├── data/                   # Data storage
│   └── food.parquet        # Downloaded dataset
├── docker/                 # Docker configuration
│   └── Dockerfile          # Docker image definition
├── notebooks/              # Jupyter notebooks
│   └── preprocessing.ipynb # Data preprocessing notebook
└── src/                    # Source code
    ├── ingestion.py        # Data download and ingestion
    ├── preprocessing.py    # Data cleaning and preparation
    ├── recommender.py      # Recommendation algorithms
    └── utils.py            # Utility functions
```

## Getting Started

### Prerequisites

- Python 3.8+
- Pandas, NumPy, scikit-learn
- Flask
- NLTK
- Docker (optional)

### Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd food-recommender
   ```

2. Create a virtual environment and install dependencies:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Download and preprocess the dataset:

   ```
   python src/ingestion.py    # This will download the dataset
   python src/preprocessing.py  # This will clean and prepare the data
   ```

4. Start the web application:

   ```
   python app/app.py
   ```

5. Visit `http://localhost:5000` in your browser.

### Using Docker

Alternatively, you can run the application using Docker:

1. Build the Docker image:

   ```
   docker build -t food-recommender -f docker/Dockerfile .
   ```

2. Run the container:

   ```
   docker run -p 5000:5000 food-recommender
   ```

3. Visit `http://localhost:5000` in your browser.

## Features

### Content-Based Recommendations

Recommendations are based on similarities in ingredient profiles, product categories, and nutritional information.

### Healthier Alternatives

For any product, find healthier alternatives with better Nutriscore ratings but similar ingredient profiles.

### Category-Based Exploration

Browse products by category and filter by nutritional quality (Nutriscore).

### API Access

Access recommendation capabilities programmatically through a REST API:

- `/api/search?query=<product_name>`: Search for products
- `/api/recommend?product_code=<code>&type=similar`: Get similar products
- `/api/recommend?product_code=<code>&type=healthier`: Get healthier alternatives

## Nutriscore Explanation

Nutriscore is a nutritional rating system:

- **A** (Green): Excellent nutritional quality
- **B** (Light Green): Good nutritional quality
- **C** (Yellow): Average nutritional quality
- **D** (Orange): Poor nutritional quality
- **E** (Red): Very poor nutritional quality

## Data Source

This project uses the [Open Food Facts](https://world.openfoodfacts.org/) dataset, a free, open, collaborative database of food products from around the world.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
