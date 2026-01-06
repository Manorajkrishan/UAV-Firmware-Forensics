# Database Connection Guide

## How to Check Database Connection

### Option 1: Check via Backend Logs

When you start the backend, look for these messages:

**✅ Connected:**
```
✓ PostgreSQL connection established
✓ Database tables created/verified
```

**❌ Not Connected:**
```
⚠ Warning: PostgreSQL connection failed: ...
The application will start but database features will be unavailable.
```

### Option 2: Check via Health Endpoint

Visit: http://localhost:8000/health

Look for database status in the response.

### Option 3: Test Connection Manually

**Using Python:**
```python
import psycopg2
try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="drone_forensics",
        user="postgres",
        password="password"
    )
    print("✓ Database connected!")
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

**Using Command Line:**
```bash
psql -h localhost -p 5432 -U postgres -d drone_forensics
```

## How to Start PostgreSQL

### Option 1: Using Docker (Recommended)

```bash
# Start PostgreSQL
docker-compose up -d postgres

# Check if running
docker ps | grep postgres
```

### Option 2: Install Locally

1. **Download PostgreSQL:**
   - Visit: https://www.postgresql.org/download/windows/
   - Download and install

2. **Start PostgreSQL Service:**
   ```bash
   # Windows (as Administrator)
   net start postgresql-x64-XX
   ```

3. **Create Database:**
   ```bash
   createdb -U postgres drone_forensics
   ```

### Option 3: Use Cloud Database

Update `.env` file:
```
DATABASE_URL=postgresql://user:password@host:5432/drone_forensics
```

## Current Status

**Your system works WITHOUT database!**

- ✅ Upload works (saves to file storage)
- ✅ Analyze works (saves to file storage)
- ✅ History works (reads from file storage)
- ✅ Dashboard works (reads from file storage)

**Database is OPTIONAL** - only needed for:
- Better performance with large datasets
- Advanced querying
- Multi-user scenarios

## File Storage Location

When database is not available, data is stored in:
- **Files:** `backend/evidence/` or `evidence/`
- **Metadata:** `backend/evidence/uploads_metadata.json`

## Summary

**You don't need PostgreSQL to use the system!**

The system automatically:
1. Tries database first (if available)
2. Falls back to file storage (if database unavailable)
3. Works perfectly either way

**To enable database (optional):**
1. Start PostgreSQL: `docker-compose up -d postgres`
2. Restart backend
3. System will automatically use database


