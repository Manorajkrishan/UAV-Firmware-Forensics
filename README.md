# ML-Based Drone Firmware Tampering Detection System

## Overview
This project implements a machine learning-based system to detect tampering activities in drone firmware using firmware logs, control signals, and system performance behaviors.

## Project Structure
```
Freed/
├── notebooks/              # Jupyter notebooks for ML model development
│   ├── 01_data_exploration_preprocessing.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_lstm_model.ipynb
│   ├── 04_autoencoder_model.ipynb
│   ├── 05_isolation_forest_ensemble.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_visualization_reporting.ipynb
├── backend/               # FastAPI backend application
│   ├── main.py            # Main API server
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile         # Docker configuration
├── frontend/              # React frontend application
│   ├── src/
│   │   ├── pages/         # Page components
│   │   ├── components/    # Reusable components
│   │   └── services/      # API services
│   ├── package.json       # Node.js dependencies
│   └── Dockerfile         # Docker configuration
├── data/                  # Datasets
│   ├── clean_drone_firmware_dataset.csv
│   └── tampered_drone_firmware_dataset.csv
├── models/                # Trained ML models
├── results/               # Analysis results
├── evidence/              # Uploaded firmware files
├── database/              # Database initialization scripts
├── docker-compose.yml     # Docker Compose configuration
├── requirements.txt       # Python dependencies (for notebooks)
└── README.md
```

## Installation

### Option 1: Full Stack Application (Recommended)

See [SETUP.md](SETUP.md) for detailed setup instructions.

**Quick Start with Docker:**
```bash
docker-compose up -d
```

Access:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Option 2: ML Notebooks Only

1. Install Python 3.8+ and required packages:
```bash
pip install -r requirements.txt
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## Usage
Execute notebooks in sequential order:
1. **01_data_exploration_preprocessing.ipynb** - Load and explore datasets
2. **02_feature_engineering.ipynb** - Extract and engineer features
3. **03_lstm_model.ipynb** - Train LSTM for time-series detection
4. **04_autoencoder_model.ipynb** - Train Autoencoder for anomaly detection
5. **05_isolation_forest_ensemble.ipynb** - Train Isolation Forest and ensemble models
6. **06_model_evaluation.ipynb** - Evaluate and compare all models
7. **07_visualization_reporting.ipynb** - Generate forensic reports and visualizations

## Features

### ML Models
- Multiple ML models (LSTM, Autoencoder, Isolation Forest, Random Forest, XGBoost, Ensemble)
- Comprehensive feature engineering
- Forensic-ready reporting
- Visualization dashboards
- Model comparison and evaluation

### Web Application
- **Dashboard**: Real-time statistics and visualizations
- **Upload Interface**: Drag-and-drop firmware CSV upload
- **Analysis Results**: Detailed tampering detection reports
- **History**: Complete analysis history with search
- **API**: RESTful API for integration

### Backend Services
- FastAPI backend with automatic API documentation
- PostgreSQL for structured data storage
- MongoDB for detailed analysis logs
- Secure evidence storage
- Multiple ML model support

## System Requirements

### For ML Notebooks
- Python 3.8+
- 16GB RAM recommended
- GPU (6GB VRAM) recommended for deep learning models

### For Full Stack Application
- Python 3.8+
- Node.js 18+
- PostgreSQL 15+
- MongoDB 7+
- Docker (optional, for containerized setup)
- 16GB RAM recommended

