# 🎓 Training System - Complete Guide

## ✅ What Was Added

A complete training system that allows you to:
1. **Upload a new dataset**
2. **Train all ML models** on that dataset
3. **Automatically analyze** the dataset
4. **Get results** with updated model accuracies

## 🚀 How It Works

### Backend (`backend/training.py`)
- Complete training pipeline for all models:
  - Random Forest
  - Isolation Forest
  - LSTM
  - Autoencoder
  - Ensemble
  - XGBoost (if available)

### Backend Endpoints (`backend/main.py`)

#### 1. `/api/train` - Train models on existing dataset
```json
POST /api/train
{
  "dataset_path": "path/to/dataset.csv",
  "test_size": 0.2
}
```

#### 2. `/api/upload-and-train` - Upload, train, and analyze
```json
POST /api/upload-and-train?train=true
FormData: file (CSV)
```

### Frontend (`frontend/src/pages/Upload.jsx`)
- **Checkbox**: "Train models on this dataset"
- When checked:
  - Uploads dataset
  - Trains all models
  - Shows training results
  - Automatically analyzes
  - Navigates to results

## 📋 Dataset Requirements

Your CSV dataset must have:
- **Required columns**: `entropy_score`, `is_signed`, `boot_time_ms`, `emulated_syscalls`
- **Target column**: One of `clean_label`, `is_tampered`, `label`, or `target`
  - Should be binary: 1 = clean/normal, 0 = tampered
- **Optional columns**: Any additional features (will be used if available)

## 🎯 Usage

### Option 1: Upload & Train (Recommended)
1. Go to Upload page
2. Check "Train models on this dataset"
3. Select your CSV file
4. Click "Upload, Train & Analyze"
5. Wait for training to complete
6. View results with updated accuracies

### Option 2: Train on Existing Dataset
```bash
# Use API directly
curl -X POST http://localhost:8000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/your_dataset.csv",
    "test_size": 0.2
  }'
```

## 📊 Training Process

1. **Load Dataset** - Reads CSV file
2. **Preprocess** - Feature engineering, encoding, scaling
3. **Split Data** - Train/test split (default 80/20)
4. **Train Models**:
   - Random Forest (100 trees)
   - Isolation Forest (anomaly detection)
   - LSTM (50 epochs)
   - Autoencoder (50 epochs)
   - Ensemble (voting classifier)
5. **Evaluate** - Calculate accuracies
6. **Save Models** - Save to `models/` directory
7. **Reload** - Load new models into memory
8. **Update Accuracies** - Update accuracy dictionary

## ✅ What Happens After Training

1. **Models are saved** to `models/` directory
2. **Models are reloaded** automatically
3. **Accuracies are updated** in the system
4. **Analysis uses new models** for predictions
5. **Results show new accuracies**

## 🎨 Frontend Features

- **Training checkbox** - Enable/disable training
- **Progress indicator** - Shows "Training Models..." during training
- **Results display** - Shows accuracy for each model after training
- **Automatic navigation** - Goes to analysis page after completion

## 📝 Example Dataset Format

```csv
entropy_score,is_signed,boot_time_ms,emulated_syscalls,hardcoded_ip_count,clean_label
7.2,1,1200,500,0,1
8.5,0,6000,1500,3,0
6.8,1,800,300,0,1
...
```

## ⚠️ Important Notes

1. **Training takes time** - LSTM and Autoencoder can take several minutes
2. **Dataset size** - Larger datasets = longer training time
3. **Memory usage** - Training uses more memory than inference
4. **Model overwrite** - New models replace old ones
5. **Backup recommended** - Save old models if needed

## 🐛 Troubleshooting

### "No target column found"
- Ensure your CSV has one of: `clean_label`, `is_tampered`, `label`, or `target`

### "Training failed"
- Check dataset format
- Ensure all required columns are present
- Check backend logs for details

### "Models not loading after training"
- Check `models/` directory permissions
- Verify models were saved successfully
- Restart backend if needed

## 🎉 Benefits

- **Continuous learning** - Retrain on new data
- **Improved accuracy** - Models adapt to new patterns
- **Easy workflow** - One-click training and analysis
- **Automatic updates** - Models and accuracies update automatically

## 📚 Next Steps

1. Upload your dataset
2. Check "Train models on this dataset"
3. Click "Upload, Train & Analyze"
4. View results with updated accuracies!

Your system now supports full training pipeline! 🚀

