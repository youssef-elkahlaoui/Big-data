FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p data

# Run the preprocessing script when building the image (optional)
# If you don't want to preprocess during build, this can be done at runtime
# RUN python src/preprocessing.py

# Expose port
EXPOSE 5000

# Command to run the application
CMD ["python", "app/app.py"]