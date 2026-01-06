# How to Use - Drone Firmware Tampering Detection System

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

#### Backend (Python)
```bash
cd backend
pip install -r requirements.txt
```

#### Frontend (Web App)
```bash
cd frontend
npm install
```

#### Desktop App
```bash
cd desktop
npm install
```

### Step 2: Start the System

#### Option A: Desktop App (Easiest)
```bash
cd desktop
npm start
# Or on Windows: start.bat
```
The desktop app automatically starts the backend server.

#### Option B: Web App
```bash
# Terminal 1: Start Backend
cd backend
python main.py

# Terminal 2: Start Frontend
cd frontend
npm run dev
```

### Step 3: Use the Application

1. **Upload Firmware File**
   - Click "Upload" or drag-and-drop
   - Supported formats: CSV, BIN, HEX, ELF
   - Maximum size: 100MB

2. **Select ML Model**
   - Choose from: Ensemble (recommended), Random Forest, LSTM, Isolation Forest, Autoencoder
   - Ensemble model has highest accuracy (95%)

3. **Analyze**
   - Click "Analyze Firmware"
   - Wait for analysis (30-60 seconds)

4. **View Results**
   - See tampering status (Clean/Tampered)
   - View probability score
   - Check detailed forensic analysis
   - Explore visualizations (charts, graphs, timeline)
   - Read recommendations

## 📊 Understanding Results

### Tampering Status
- **Clean**: Firmware appears untampered (green)
- **Tampered**: Malicious modifications detected (red)
- **Suspicious**: Uncertain, needs review (yellow)

### Probability Score
- **0-30%**: Likely clean
- **30-70%**: Suspicious, needs review
- **70-100%**: Likely tampered

### Forensic Analysis Includes
- SHA-256 hash verification
- Version anomaly detection
- Boot sequence analysis
- Sensor spoofing detection
- Timeline analysis
- Feature contribution

## 🎯 Common Use Cases

### 1. Verify Firmware Integrity
Upload firmware file → Select Ensemble model → Analyze → Check status

### 2. Compare Models
Upload same file → Try different models → Compare results

### 3. View History
Go to History tab → Click any analysis → View detailed report

### 4. Dashboard Statistics
Go to Dashboard → See overall statistics and charts

## 🔧 API Usage

### Upload File
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@firmware.csv"
```

### Analyze
```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"firmware_id": "your-id", "model_preference": "ensemble"}'
```

### Get Results
```bash
curl http://localhost:8000/api/analyses/your-firmware-id
```

## ⚙️ Configuration

### Backend Settings
Edit `backend/.env`:
```env
MAX_FILE_SIZE=104857600      # 100MB
RATE_LIMIT_PER_MINUTE=100    # Requests per minute
ALLOWED_ORIGINS=http://localhost:3000
```

### Desktop App
No configuration needed - works out of the box!

## 🐛 Troubleshooting

**Backend not starting?**
- Check Python 3.8+ installed
- Install dependencies: `pip install -r backend/requirements.txt`
- Check port 8000 is free

**Frontend connection error?**
- Ensure backend is running
- Check `http://localhost:8000/health`

**Models not loading?**
- Ensure model files in `models/` directory
- Check backend logs

**File upload fails?**
- Check file size (max 100MB)
- Verify file format (CSV, BIN, HEX, ELF)
- Check backend logs

## 📚 More Information

- **API Docs**: http://localhost:8000/docs
- **README.md**: Complete project documentation
- **Backend Logs**: `backend/logs/app.log`

---

**Need Help?** Check the logs or API documentation for detailed error messages.

