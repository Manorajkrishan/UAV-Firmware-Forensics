# 🚀 Quick Start - Backend Server

## ❌ Problem
**Connection Refused** - Backend is not running!

## ✅ Solution - Start Backend

### Option 1: Double-Click Script (Easiest)
1. Double-click `start_backend.bat` in the project root
2. Wait for "Uvicorn running on http://0.0.0.0:8000"
3. Keep this window open!

### Option 2: Manual Start
```bash
cd backend
python main.py
```

### Option 3: Using Uvicorn
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## ✅ Verify It's Running

Open browser: http://localhost:8000/health

Should see:
```json
{
  "status": "healthy",
  "models_loaded": true,
  ...
}
```

## 🎯 Then Start Frontend

In a **NEW terminal**:
```bash
cd frontend
npm run dev
```

## ⚠️ Important

- **Keep backend running** - Don't close the terminal!
- Backend must run on port 8000
- Frontend runs on port 3000
- Both must be running at the same time!

## 🎉 Once Both Are Running

- ✅ Backend: http://localhost:8000
- ✅ Frontend: http://localhost:3000
- ✅ Upload works!
- ✅ Training works!
- ✅ Analysis works!

## 🐛 Troubleshooting

### Port 8000 Already in Use
```bash
# Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Python Not Found
- Install Python 3.12+
- Or use full path: `C:\Python312\python.exe main.py`

### Models Not Loading
- Check `E:\Freed\models\` exists
- Verify model files are there

