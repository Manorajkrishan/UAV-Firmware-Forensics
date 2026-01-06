# MITRE Dataset Support Guide

## Problem
Your dataset (`drone_firmware_full_mitre_dataset.csv`) uses a different column structure than what the system expects.

### Your Dataset Columns (MITRE Format):
- `firmware_id`, `firmware_file_format`, `firmware_hash_sha256`
- `firmware_version`, `version_anomaly`, `boot_sequence_irregularities`
- `integrity_check_missing_or_altered`, `firmware_modification_type`
- `synthetic_attack_scenario`, `mitre_attack_technique`
- `anomaly_score`, `tampering_probability_percent`
- `classification`, `severity_level`, `malicious_activity_time_window`

### System Expected Columns:
- `entropy_score`, `is_signed`, `boot_time_ms`, `emulated_syscalls`
- `hardcoded_ip_count`, `hardcoded_url_count`, `crypto_function_count`
- etc.

## Solution
I've created a **MITRE Dataset Converter** that automatically:
1. Detects MITRE-format datasets
2. Converts them to system-compatible format
3. Maps forensic indicators to ML features

## How It Works

### Automatic Conversion
The system now automatically detects and converts MITRE datasets:

```python
# When you upload the CSV, the system:
1. Detects MITRE format (checks for MITRE-specific columns)
2. Converts columns:
   - anomaly_score → entropy_score (scaled)
   - integrity_check_missing_or_altered → is_signed (inverted)
   - boot_sequence_irregularities → boot_time_ms (derived)
   - version_anomaly + other indicators → emulated_syscalls
   - synthetic_attack_scenario → hardcoded_ip_count, hardcoded_url_count
3. Preserves MITRE columns for forensic analysis
4. Creates clean_label for training (from classification)
```

### Column Mapping

| MITRE Column | System Column | Conversion Logic |
|-------------|--------------|------------------|
| `anomaly_score` | `entropy_score` | Scaled to 0-8 range |
| `integrity_check_missing_or_altered` | `is_signed` | Inverted (0=unsigned, 1=signed) |
| `boot_sequence_irregularities` | `boot_time_ms` | Normal: 2000ms, Irregular: 5000-10000ms |
| `version_anomaly` + indicators | `emulated_syscalls` | Derived from multiple indicators |
| `synthetic_attack_scenario` | `hardcoded_ip_count` | Counts IP-related attacks |
| `synthetic_attack_scenario` | `hardcoded_url_count` | Counts URL-related attacks |
| `firmware_modification_type` | `crypto_function_count` | Counts crypto-related modifications |

## How to Use

### Option 1: Upload via Web Interface
1. Go to Upload page
2. Select your `drone_firmware_full_mitre_dataset.csv` file
3. System automatically detects and converts it
4. Proceed with analysis

### Option 2: Use for Training
1. Upload the dataset
2. Check "Train models on this dataset"
3. System will:
   - Convert MITRE format
   - Use `classification` column as target (Untampered = clean, others = tampered)
   - Train all ML models
   - Analyze the data

## Benefits

### 1. Preserves MITRE Data
- All original MITRE columns are preserved
- Can be used for enhanced forensic analysis
- MITRE attack techniques are available

### 2. Automatic Feature Generation
- ML features are automatically derived
- No manual conversion needed
- Maintains data relationships

### 3. Enhanced Analysis
- System can use both:
  - Converted ML features (for prediction)
  - Original MITRE columns (for forensic analysis)
- Best of both worlds!

## Example Conversion

### Input (MITRE Format):
```csv
firmware_id,version_anomaly,boot_sequence_irregularities,integrity_check_missing_or_altered,anomaly_score,classification
CLEAN_1,0,0,0,0.012,Untampered
TAMPERED_1,1,1,1,0.85,Tampered
```

### Output (System Format):
```csv
firmware_id,entropy_score,is_signed,boot_time_ms,emulated_syscalls,classification,clean_label
CLEAN_1,0.096,1,2000,100,Untampered,1
TAMPERED_1,6.8,0,7000,1200,Tampered,0
```

## Troubleshooting

### Issue: "Missing required columns"
**Solution**: Ensure `mitre_dataset_converter.py` is in the `backend/` directory

### Issue: Conversion not working
**Check**:
1. File is actually MITRE format (has MITRE-specific columns)
2. Backend logs show "Detected MITRE dataset format"
3. Converted CSV is created in `evidence/` directory

### Issue: Training fails
**Solution**: 
- Ensure `classification` column exists
- Values should be: "Untampered", "Tampered", or "Suspicious"
- System will create `clean_label` automatically

## Advanced Usage

### Using MITRE Columns in Analysis
The system preserves MITRE columns, so you can:
- Use `mitre_attack_technique` for detailed attack classification
- Use `severity_level` for risk assessment
- Use `malicious_activity_time_window` for timeline analysis

### Custom Conversion
If you need custom mapping, edit `backend/mitre_dataset_converter.py`:
```python
def convert_mitre_dataset_to_system_format(df):
    # Add your custom mappings here
    ...
```

## Files Created

- `backend/mitre_dataset_converter.py` - Conversion logic
- Updated `backend/main.py` - Auto-detection on upload
- Updated `backend/firmware_parser.py` - CSV validation with MITRE support

## Next Steps

1. **Upload your dataset**: The system will auto-convert it
2. **Train models**: Use the converted dataset for training
3. **Analyze**: All analyses will work with converted features
4. **View results**: MITRE columns are preserved for forensic analysis

---

**Status**: ✅ MITRE dataset support fully implemented!

Your dataset should now work perfectly with the system! 🎉

