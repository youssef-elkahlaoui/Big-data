# MongoDB Installation and Setup Script for Windows
# This script helps set up MongoDB for the Food Recommendation System

Write-Host "Food Recommendation System - MongoDB Setup" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green

# Check if MongoDB is already installed
$mongoPath = Get-Command mongod -ErrorAction SilentlyContinue

if ($mongoPath) {
    Write-Host "✓ MongoDB is already installed at: $($mongoPath.Source)" -ForegroundColor Green
    
    # Test MongoDB connection
    Write-Host "`nTesting MongoDB connection..." -ForegroundColor Yellow
    try {
        $mongoTest = Start-Process -FilePath "mongo" -ArgumentList "--eval", "db.stats()" -NoNewWindow -Wait -PassThru
        if ($mongoTest.ExitCode -eq 0) {
            Write-Host "✓ MongoDB is running and accessible" -ForegroundColor Green
        } else {
            Write-Host "⚠ MongoDB is installed but not running" -ForegroundColor Yellow
            Write-Host "Please start MongoDB service before running the migration" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠ Could not test MongoDB connection: $($_.Exception.Message)" -ForegroundColor Yellow
    }
} else {
    Write-Host "MongoDB not found. Please install MongoDB first." -ForegroundColor Red
    Write-Host "`nInstallation options:" -ForegroundColor Yellow
    Write-Host "1. Download from: https://www.mongodb.com/try/download/community" -ForegroundColor Cyan
    Write-Host "2. Use Chocolatey: choco install mongodb" -ForegroundColor Cyan
    Write-Host "3. Use winget: winget install MongoDB.Server" -ForegroundColor Cyan
    
    $install = Read-Host "`nWould you like to try installing via winget? (y/n)"
    if ($install -eq "y" -or $install -eq "Y") {
        Write-Host "Installing MongoDB via winget..." -ForegroundColor Yellow
        try {
            winget install MongoDB.Server
            Write-Host "✓ MongoDB installation completed" -ForegroundColor Green
        } catch {
            Write-Host "✗ Failed to install MongoDB via winget: $($_.Exception.Message)" -ForegroundColor Red
            Write-Host "Please install MongoDB manually" -ForegroundColor Yellow
            exit 1
        }
    } else {
        Write-Host "Please install MongoDB manually and run this script again" -ForegroundColor Yellow
        exit 1
    }
}

# Check Python dependencies
Write-Host "`nChecking Python dependencies..." -ForegroundColor Yellow

$requirements = @("pymongo", "pandas", "numpy", "scikit-learn", "flask", "tqdm")
$missingDeps = @()

foreach ($req in $requirements) {
    try {
        python -c "import $req" 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ $req is installed" -ForegroundColor Green
        } else {
            Write-Host "✗ $req is missing" -ForegroundColor Red
            $missingDeps += $req
        }
    } catch {
        Write-Host "✗ $req is missing" -ForegroundColor Red
        $missingDeps += $req
    }
}

if ($missingDeps.Count -gt 0) {
    Write-Host "`nInstalling missing Python dependencies..." -ForegroundColor Yellow
    try {
        python -m pip install ($missingDeps -join " ")
        Write-Host "✓ Python dependencies installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "✗ Failed to install Python dependencies: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Please run: python -m pip install -r requirements.txt" -ForegroundColor Yellow
    }
}

# Check if MongoDB service is running
Write-Host "`nChecking MongoDB service..." -ForegroundColor Yellow

$mongoService = Get-Service -Name "MongoDB" -ErrorAction SilentlyContinue
if ($mongoService) {
    if ($mongoService.Status -eq "Running") {
        Write-Host "✓ MongoDB service is running" -ForegroundColor Green
    } else {
        Write-Host "⚠ MongoDB service is stopped. Starting service..." -ForegroundColor Yellow
        try {
            Start-Service -Name "MongoDB"
            Write-Host "✓ MongoDB service started successfully" -ForegroundColor Green
        } catch {
            Write-Host "✗ Failed to start MongoDB service: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
} else {
    Write-Host "⚠ MongoDB service not found. MongoDB might be running manually." -ForegroundColor Yellow
    
    # Try to start MongoDB manually for development
    Write-Host "Starting MongoDB manually for development..." -ForegroundColor Yellow
    try {
        # Create data directory if it doesn't exist
        $dataDir = "$env:USERPROFILE\mongodb-data"
        if (!(Test-Path $dataDir)) {
            New-Item -ItemType Directory -Path $dataDir -Force
            Write-Host "Created MongoDB data directory: $dataDir" -ForegroundColor Cyan
        }
        
        # Start MongoDB in background
        $mongoProcess = Start-Process -FilePath "mongod" -ArgumentList "--dbpath", $dataDir -WindowStyle Hidden -PassThru
        Write-Host "✓ MongoDB started manually (PID: $($mongoProcess.Id))" -ForegroundColor Green
        Write-Host "MongoDB data directory: $dataDir" -ForegroundColor Cyan
        
        # Wait a moment for MongoDB to start
        Start-Sleep -Seconds 3
        
    } catch {
        Write-Host "✗ Failed to start MongoDB manually: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Please start MongoDB manually or install as a service" -ForegroundColor Yellow
    }
}

# Verify MongoDB connection
Write-Host "`nVerifying MongoDB connection..." -ForegroundColor Yellow
try {
    python -c "
import pymongo
import sys
try:
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    client.admin.command('ismaster')
    print('✓ Successfully connected to MongoDB')
    sys.exit(0)
except Exception as e:
    print(f'✗ Failed to connect to MongoDB: {e}')
    sys.exit(1)
"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ MongoDB connection verified" -ForegroundColor Green
    } else {
        Write-Host "✗ MongoDB connection failed" -ForegroundColor Red
    }
} catch {
    Write-Host "✗ Could not verify MongoDB connection" -ForegroundColor Red
}

# Display next steps
Write-Host "`n" -ForegroundColor White
Write-Host "Next Steps:" -ForegroundColor Green
Write-Host "==========" -ForegroundColor Green
Write-Host "1. Ensure MongoDB is running" -ForegroundColor Cyan
Write-Host "2. Run the data migration script:" -ForegroundColor Cyan
Write-Host "   python scripts\migrate_to_mongodb.py --csv-path data\cleaned_food_data_filtered.csv" -ForegroundColor White
Write-Host "3. Start the Flask application:" -ForegroundColor Cyan
Write-Host "   python app\app_mongo.py" -ForegroundColor White
Write-Host ""
Write-Host "MongoDB Management:" -ForegroundColor Yellow
Write-Host "- Connect to MongoDB shell: mongo" -ForegroundColor White
Write-Host "- View databases: show dbs" -ForegroundColor White
Write-Host "- Use food database: use food_recommendation_db" -ForegroundColor White
Write-Host "- View collections: show collections" -ForegroundColor White

Write-Host "`nSetup completed! 🎉" -ForegroundColor Green
