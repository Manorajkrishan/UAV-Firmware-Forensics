# Backend Startup Guide

## Fixed Issues

### 1. SQLAlchemy Deprecation Warning
✅ Fixed: Changed from `sqlalchemy.ext.declarative.declarative_base()` to `sqlalchemy.orm.declarative_base()`

### 2. Database Connection Error
✅ Fixed: Database initialization is now lazy and happens in the startup event, not at import time. The app will start even if PostgreSQL/MongoDB aren't running.

## Running the Backend

### Option 1: Using uvicorn (Recommended)
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: Using Python directly
```bash
cd backend
python main.py
```

### Option 3: Using Docker Compose (Full Stack)
```bash
# From project root
docker-compose up -d
```

## Database Setup

### If you want to use the database features:

1. **Start PostgreSQL** (if using Docker):
   ```bash
   docker-compose up -d postgres
   ```

2. **Or use local PostgreSQL**:
   - Make sure PostgreSQL is installed and running
   - Create database: `createdb drone_forensics`
   - Update `DATABASE_URL` in `.env` file if needed

3. **Start MongoDB** (optional, for logging):
   ```bash
   docker-compose up -d mongodb
   ```

### The app will work without databases!
- The app will start successfully even if databases aren't running
- Database-dependent endpoints will return 503 errors with helpful messages
- ML model analysis will still work (models don't require database)

## Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/drone_forensics
MONGODB_URL=mongodb://localhost:27017

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Optional: Custom paths
MODELS_DIR=models
DATA_DIR=data
RESULTS_DIR=results
EVIDENCE_DIR=evidence
```

## Expected Startup Output

When the app starts successfully, you should see:

```
==================================================
Starting Drone Firmware Tampering Detection API
==================================================
✓ PostgreSQL connection established
✓ Database tables created/verified
✓ MongoDB connection established

Loading ML models...
Loaded scaler from models/feature_scaler.pkl
Loaded feature selector from models/feature_selector.pkl
...
Successfully loaded X models/objects: ['scaler', 'selector', ...]

✓ Application started successfully!
==================================================
```

If databases aren't running, you'll see warnings but the app will still start:

```
⚠ Warning: PostgreSQL connection failed: ...
  The application will start but database features will be unavailable.
  Make sure PostgreSQL is running or set DATABASE_URL environment variable.
  To start PostgreSQL with Docker: docker-compose up -d postgres
```

## Testing the API

Once started, visit:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## Troubleshooting

### "Connection refused" error
- This is now handled gracefully - the app will start anyway
- To use database features, start PostgreSQL/MongoDB first

### "Models not found" error
- Make sure the `models/` directory exists with trained models
- Required files: `feature_scaler.pkl`, `feature_selector.pkl`, `label_encoders.pkl`
- At least one ML model file should be present

### Port already in use
- Change the port: `uvicorn main:app --port 8001`
- Or stop the process using port 8000


