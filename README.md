# 🛡️ Drone Firmware Tampering Detection System

AI-powered Machine Learning system to detect malicious modifications (tampering) in drone firmware using advanced ML algorithms.

## 🎯 Overview

This system analyzes drone firmware files using **5 different Machine Learning models** to identify security threats, anomalies, and tampering patterns. It provides comprehensive forensic analysis with detailed reports and visualizations.

### Key Features

- ✅ **5 ML Models**: Ensemble, Random Forest, LSTM, Isolation Forest, Autoencoder
- ✅ **95% Accuracy**: Ensemble model achieves high detection rate
- ✅ **Comprehensive Analysis**: Forensic analysis with SHA-256 hashing, timeline analysis, feature contribution
- ✅ **Web Application**: React-based dashboard with real-time visualizations
- ✅ **Desktop Application**: Standalone Electron app (Windows, Mac, Linux)
- ✅ **Security Features**: Rate limiting, CORS protection, file validation
- ✅ **Multiple Formats**: Supports CSV, BIN, HEX, ELF firmware files

## 🏗️ System Architecture

```
┌─────────────┐         ┌─────────────┐
│  Web App    │         │ Desktop App │
│  (React)    │         │  (Electron) │
└──────┬──────┘         └──────┬──────┘
       │                       │
       └───────────┬───────────┘
                   │ HTTP/REST API
         ┌─────────▼─────────┐
         │  Backend (FastAPI) │
         │  + ML Models       │
         └─────────┬───────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───▼───┐    ┌────▼────┐   ┌────▼────┐
│Postgres│    │ MongoDB │   │  Files  │
└────────┘    └─────────┘   └─────────┘
```

## 📦 Installation

### Prerequisites

- **Python 3.8+**
- **Node.js 18+** (for frontend/desktop)
- **PostgreSQL 15+** (optional, for database)
- **MongoDB 7+** (optional, for detailed logs)
- **16GB RAM** recommended

### Quick Start

#### 1. Backend Setup

```bash
# Navigate to backend
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Create .env file (optional)
cp env.example .env
# Edit .env with your database credentials if needed

# Start backend server
python main.py
```

Backend will run on: `http://localhost:8000`

#### 2. Frontend Setup (Web App)

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will run on: `http://localhost:3000`

#### 3. Desktop App Setup

```bash
# Navigate to desktop
cd desktop

# Install dependencies
npm install

# Run desktop app
npm start
# Or on Windows: start.bat
```

The desktop app will automatically start the backend server.

### Docker Setup (Alternative)

```bash
# Start all services
docker-compose up -d

# Access services
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## 🚀 Usage

### Web Application

1. **Upload Firmware**: Go to Upload page, drag-and-drop or select firmware file (CSV, BIN, HEX, ELF)
2. **Select Model**: Choose ML model (Ensemble recommended)
3. **Analyze**: Click "Analyze Firmware"
4. **View Results**: See detailed analysis with:
   - Tampering status (Clean/Tampered)
   - Probability score
   - Forensic analysis
   - Visualizations (charts, graphs, timeline)
   - Recommendations

### Desktop Application

1. **Launch App**: Run `npm start` in desktop folder
2. **Backend Auto-Starts**: Backend server starts automatically
3. **Upload & Analyze**: Same workflow as web app
4. **View Reports**: All features available in desktop format

### API Usage

```bash
# Upload firmware
curl -X POST http://localhost:8000/api/upload \
  -F "file=@firmware.csv"

# Analyze firmware
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"firmware_id": "your-firmware-id", "model_preference": "ensemble"}'

# Get analysis results
curl http://localhost:8000/api/analyses/your-firmware-id
```

See API documentation at: `http://localhost:8000/docs`

## 📊 ML Models

| Model | Type | Accuracy | Use Case |
|-------|------|----------|----------|
| **Ensemble** | Combined | 95% | Production (Recommended) |
| Random Forest | Supervised | 92% | Feature importance analysis |
| LSTM | Deep Learning | 90% | Sequential pattern detection |
| Isolation Forest | Unsupervised | 88% | Anomaly detection |
| Autoencoder | Unsupervised | 85% | Pattern reconstruction |

## 🔒 Security Features

- **Rate Limiting**: 100 requests/minute per IP
- **CORS Protection**: Restricted API origins
- **File Validation**: Size limits (100MB), type checking, sanitization
- **Input Validation**: All inputs validated and sanitized
- **Structured Logging**: Comprehensive audit trail

## 📁 Project Structure

```
Freed/
├── backend/              # FastAPI backend
│   ├── main.py          # Main API server
│   ├── forensic_analysis.py
│   ├── injection_detection.py
│   └── requirements.txt
├── frontend/            # React web application
│   ├── src/
│   │   ├── pages/       # Dashboard, Analysis, History
│   │   └── components/  # Visualizations, UI components
│   └── package.json
├── desktop/             # Electron desktop app
│   ├── index.html       # Main UI
│   ├── main.js          # Electron process
│   └── package.json
├── notebooks/           # Jupyter notebooks for ML development
├── models/              # Trained ML models (not in repo)
├── data/                # Datasets (sample files only)
└── README.md
```

## 🧪 Testing

```bash
# Run comprehensive tests
python test_system_comprehensive.py

# Test tampered detection
python test_tampered_detection.py
```

## 📚 Documentation

- **HOW_TO_USE.md** - Brief usage guide
- **API Documentation**: http://localhost:8000/docs (when backend is running)

## 🔧 Configuration

### Environment Variables

Create `backend/.env` file:

```env
# Database (optional)
DATABASE_URL=postgresql://postgres:password@localhost:5432/drone_forensics
MONGODB_URL=mongodb://localhost:27017

# Security
ENVIRONMENT=development
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
MAX_FILE_SIZE=104857600
RATE_LIMIT_PER_MINUTE=100
```

## 🐛 Troubleshooting

### Backend won't start
- Check Python version: `python --version` (needs 3.8+)
- Install dependencies: `pip install -r backend/requirements.txt`
- Check port 8000 is available

### Models not loading
- Ensure model files exist in `models/` directory
- Check file permissions
- Review backend logs for errors

### Frontend connection error
- Ensure backend is running on port 8000
- Check CORS settings in backend
- Verify API URL in frontend `.env`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 👥 Authors

Drone Firmware Detection Team

## 🙏 Acknowledgments

- FastAPI for the excellent web framework
- TensorFlow/Keras for deep learning
- React for the frontend framework
- Electron for desktop application

## 📞 Support

For issues and questions:
- Check documentation in project files
- Review API docs at `/docs` endpoint
- Check logs in `backend/logs/` directory

---

**Status**: Production Ready (with Phase 1 security features)

**Version**: 1.0.0

**Last Updated**: 2024
