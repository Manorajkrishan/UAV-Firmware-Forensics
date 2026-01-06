# 📋 Dataset Requirements for Analysis

## Simple Guide: What Data to Upload

### ✅ **For Analysis Only** (No Training Required)

If you just want to **analyze** firmware for tampering, your CSV needs these **required columns**:

| Column Name | What It Means | Example Value |
|------------|---------------|---------------|
| `entropy_score` | How random/complex the code is (0-8). Higher = more obfuscated | `7.2` |
| `is_signed` | Is firmware digitally signed? (1 = yes, 0 = no) | `1` |
| `boot_time_ms` | How long firmware takes to boot (milliseconds) | `1200` |
| `emulated_syscalls` | Number of system calls made | `500` |

### 📊 **Optional Columns** (Recommended for Better Analysis)

These help detect more types of tampering:

| Column Name | What It Means | Example Value |
|------------|---------------|---------------|
| `hardcoded_ip_count` | Number of IP addresses found in code | `2` |
| `hardcoded_url_count` | Number of URLs found in code | `1` |
| `crypto_function_count` | Number of cryptographic functions | `25` |
| `string_count` | Number of readable strings in firmware | `5000` |
| `file_size_bytes` | Size of firmware file in bytes | `11228452` |
| `num_executables` | Number of executable files detected | `7` |
| `num_scripts` | Number of script files detected | `1` |

### ✅ **For Training Models** (Training Required)

If you want to **train** models on your dataset, you need:

1. **All required columns above** ✅
2. **Plus a target column** (one of these):
   - `clean_label` (1 = clean/normal, 0 = tampered) ← **Most Common**
   - `is_tampered` (1 = tampered, 0 = clean)
   - `label` (binary: 0 or 1)
   - `target` (binary: 0 or 1)

### 📝 **Example Dataset Format**

#### **For Analysis Only:**
```csv
entropy_score,is_signed,boot_time_ms,emulated_syscalls,hardcoded_ip_count,hardcoded_url_count
7.2,1,1200,500,0,0
8.5,0,6000,1500,3,2
6.8,1,800,300,0,0
```

#### **For Training:**
```csv
entropy_score,is_signed,boot_time_ms,emulated_syscalls,hardcoded_ip_count,hardcoded_url_count,clean_label
7.2,1,1200,500,0,0,1
8.5,0,6000,1500,3,2,0
6.8,1,800,300,0,0,1
```

**Where:**
- `clean_label = 1` → Normal/Clean firmware ✅
- `clean_label = 0` → Tampered firmware ⚠️

### 🎯 **Quick Summary**

**Minimum for Analysis:**
- ✅ `entropy_score`
- ✅ `is_signed`
- ✅ `boot_time_ms`
- ✅ `emulated_syscalls`

**For Training:**
- ✅ All above columns
- ✅ Plus: `clean_label` (or `is_tampered`, `label`, `target`)

### 📤 **How to Upload**

1. **Create CSV file** with required columns
2. **Go to Upload page** in the app
3. **Drag & drop** your CSV file
4. **Check "Train models"** if you have target column
5. **Click Upload**

### 🔍 **What Each Column Detects**

| Column | Detects |
|--------|---------|
| `entropy_score` | Code obfuscation, encryption |
| `is_signed` | Signature tampering |
| `boot_time_ms` | Performance anomalies |
| `emulated_syscalls` | Sensor spoofing, system manipulation |
| `hardcoded_ip_count` | Backdoors, command injection |
| `hardcoded_url_count` | Data exfiltration |
| `crypto_function_count` | Cryptographic manipulation |

### ✅ **That's It!**

Just make sure your CSV has:
- **Required columns** for analysis
- **Target column** if training

The system will automatically:
- ✅ Parse your data
- ✅ Extract features
- ✅ Analyze for tampering
- ✅ Train models (if target column provided)

