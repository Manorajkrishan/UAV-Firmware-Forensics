# Quick Start Guide

## ML-Based Drone Firmware Tampering Detection System

### Prerequisites
- Python 3.8 or higher
- 16GB RAM (recommended)
- GPU with 6GB VRAM (optional, for faster training)

### Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Verify installation:**
```bash
python --version  # Should be 3.8+
```

### Running the Notebooks

1. **Start Jupyter Notebook:**
```bash
jupyter notebook
```

2. **Execute notebooks in order:**
   - `01_data_exploration_preprocessing.ipynb` - Explore and preprocess data
   - `02_feature_engineering.ipynb` - Extract and engineer features
   - `03_lstm_model.ipynb` - Train LSTM model
   - `04_autoencoder_model.ipynb` - Train Autoencoder model
   - `05_isolation_forest_ensemble.ipynb` - Train ensemble models
   - `06_model_evaluation.ipynb` - Evaluate all models
   - `07_visualization_reporting.ipynb` - Generate reports and visualizations

### Expected Outputs

After running all notebooks, you should have:

**Models saved in `models/`:**
- `lstm_model_final.h5`
- `autoencoder_model_final.h5`
- `isolation_forest_model.pkl`
- `random_forest_model.pkl`
- `ensemble_model.pkl`
- `xgboost_model.pkl` (if xgboost installed)

**Results saved in `results/`:**
- `forensic_report.txt` - Comprehensive forensic report
- `forensic_summary.json` - JSON summary for API integration
- `comprehensive_model_evaluation.csv` - All model metrics
- Various visualization PNG files

**Processed data in `data/`:**
- `combined_preprocessed_dataset.csv`
- `X_train.csv`, `X_test.csv`
- `y_train.csv`, `y_test.csv`

### Troubleshooting

**Issue: XGBoost not available**
- Solution: Install with `pip install xgboost` or skip XGBoost models

**Issue: TensorFlow/GPU errors**
- Solution: Install CPU-only version: `pip install tensorflow-cpu`

**Issue: Memory errors**
- Solution: Reduce batch size in model training notebooks
- Use smaller feature sets in feature engineering notebook

### Next Steps

1. Review the forensic report in `results/forensic_report.txt`
2. Examine model performance in `results/comprehensive_model_evaluation.csv`
3. Use saved models for inference on new firmware samples
4. Integrate with your forensic toolkit using the JSON summary

### Support

For issues or questions, refer to the main README.md file.





