# MongoDB Installation and Setup Script for Windows
# Simplified version for the Food Recommendation System

Write-Host "Food Recommendation System - MongoDB Setup" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

# Check if MongoDB is already installed
$mongoPath = Get-Command mongod -ErrorAction SilentlyContinue

if ($mongoPath) {
    Write-Host "MongoDB is already installed at: $($mongoPath.Source)" -ForegroundColor Green
} else {
    Write-Host "MongoDB not found. Installing via winget..." -ForegroundColor Yellow
    
    try {
        winget install MongoDB.Server
        Write-Host "MongoDB installation completed" -ForegroundColor Green
        
        # Add MongoDB to PATH if needed
        $mongoPath = "C:\Program Files\MongoDB\Server\7.0\bin"
        if (Test-Path $mongoPath) {
            $env:PATH += ";$mongoPath"
            Write-Host "Added MongoDB to PATH for this session" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "Failed to install MongoDB via winget" -ForegroundColor Red
        Write-Host "Please install MongoDB manually from: https://www.mongodb.com/try/download/community" -ForegroundColor Yellow
        exit 1
    }
}

# Check Python dependencies
Write-Host "Checking Python dependencies..." -ForegroundColor Yellow

$requirements = @("pymongo", "pandas", "numpy", "scikit-learn", "flask", "tqdm")
$missingDeps = @()

foreach ($req in $requirements) {
    $result = python -c "import $req; print('OK')" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $req is installed" -ForegroundColor Green
    } else {
        Write-Host "✗ $req is missing" -ForegroundColor Red
        $missingDeps += $req
    }
}

if ($missingDeps.Count -gt 0) {
    Write-Host "Installing missing Python dependencies..." -ForegroundColor Yellow
    python -m pip install ($missingDeps -join " ")
    Write-Host "Python dependencies installation completed" -ForegroundColor Green
}

# Create MongoDB data directory
$dataDir = "$env:USERPROFILE\mongodb-data"
if (!(Test-Path $dataDir)) {
    New-Item -ItemType Directory -Path $dataDir -Force
    Write-Host "Created MongoDB data directory: $dataDir" -ForegroundColor Cyan
}

# Start MongoDB
Write-Host "Starting MongoDB..." -ForegroundColor Yellow
try {
    Start-Process -FilePath "mongod" -ArgumentList "--dbpath", $dataDir -WindowStyle Minimized
    Write-Host "MongoDB started successfully" -ForegroundColor Green
    Write-Host "Data directory: $dataDir" -ForegroundColor Cyan
    
    # Wait for MongoDB to start
    Start-Sleep -Seconds 5
    
} catch {
    Write-Host "Failed to start MongoDB: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please start MongoDB manually" -ForegroundColor Yellow
}

# Verify connection
Write-Host "Verifying MongoDB connection..." -ForegroundColor Yellow
$pythonScript = @"
import pymongo
try:
    client = pymongo.MongoClient('mongodb://localhost:27017/', serverSelectionTimeoutMS=5000)
    client.admin.command('ismaster')
    print('SUCCESS: Connected to MongoDB')
except Exception as e:
    print(f'ERROR: {e}')
"@

$result = python -c $pythonScript
Write-Host $result

Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Green
Write-Host "1. Run data migration: python scripts\migrate_to_mongodb.py" -ForegroundColor Cyan
Write-Host "2. Start Flask app: python app\app_mongo.py" -ForegroundColor Cyan
Write-Host ""
Write-Host "Setup completed!" -ForegroundColor Green
