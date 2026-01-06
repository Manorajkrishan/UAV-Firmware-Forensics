"""
MITRE Dataset Converter
Converts MITRE-style forensic dataset to system-compatible format
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

def convert_mitre_dataset_to_system_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert MITRE dataset format to system-compatible format
    
    MITRE Format:
    - firmware_id, firmware_file_format, firmware_hash_sha256, firmware_version
    - version_anomaly, boot_sequence_irregularities, integrity_check_missing_or_altered
    - firmware_modification_type, synthetic_attack_scenario, mitre_attack_technique
    - anomaly_score, tampering_probability_percent, classification, severity_level
    - malicious_activity_time_window
    
    System Format:
    - entropy_score, is_signed, boot_time_ms, emulated_syscalls
    - hardcoded_ip_count, hardcoded_url_count, crypto_function_count
    - etc.
    """
    converted_df = df.copy()
    
    # Map existing columns that match
    column_mapping = {
        'firmware_hash_sha256': 'sha256_hash',
        'firmware_version': 'firmware_version',
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in converted_df.columns and new_col not in converted_df.columns:
            converted_df[new_col] = converted_df[old_col]
    
    # Generate required ML features from MITRE forensic indicators
    
    # 1. entropy_score: Derive from anomaly_score and tampering indicators
    if 'entropy_score' not in converted_df.columns:
        if 'anomaly_score' in converted_df.columns:
            # Use anomaly_score as base, scale to 0-8 range
            # Handle NaN and convert to numeric
            converted_df['entropy_score'] = pd.to_numeric(converted_df['anomaly_score'], errors='coerce').fillna(0) * 8
            converted_df['entropy_score'] = converted_df['entropy_score'].clip(0, 8)
        else:
            # Default based on tampering indicators
            entropy_base = pd.Series([4.0] * len(converted_df), index=converted_df.index)  # Normal entropy
            if 'version_anomaly' in converted_df.columns:
                version_vals = pd.to_numeric(converted_df['version_anomaly'], errors='coerce').fillna(0)
                entropy_base += version_vals * 2.0
            if 'boot_sequence_irregularities' in converted_df.columns:
                boot_vals = pd.to_numeric(converted_df['boot_sequence_irregularities'], errors='coerce').fillna(0)
                entropy_base += boot_vals * 1.5
            converted_df['entropy_score'] = entropy_base.clip(0, 8)
    
    # 2. is_signed: Derive from integrity_check_missing_or_altered
    if 'is_signed' not in converted_df.columns:
        if 'integrity_check_missing_or_altered' in converted_df.columns:
            # Invert: 0 = missing/altered = unsigned, 1 = present = signed
            integrity_vals = pd.to_numeric(converted_df['integrity_check_missing_or_altered'], errors='coerce').fillna(0)
            converted_df['is_signed'] = (1 - integrity_vals).astype(int)
        else:
            # Default: assume signed if no tampering indicators
            converted_df['is_signed'] = 1
    
    # 3. boot_time_ms: Derive from boot_sequence_irregularities
    if 'boot_time_ms' not in converted_df.columns:
        if 'boot_sequence_irregularities' in converted_df.columns:
            # Normal boot: 1000-3000ms, irregular: 5000-10000ms
            base_boot = 2000  # Normal boot time
            irregularity_vals = pd.to_numeric(converted_df['boot_sequence_irregularities'], errors='coerce').fillna(0)
            irregularity_penalty = irregularity_vals * 5000
            converted_df['boot_time_ms'] = (base_boot + irregularity_penalty).clip(500, 15000)
        else:
            converted_df['boot_time_ms'] = 2000  # Default normal boot time
    
    # 4. emulated_syscalls: Derive from tampering indicators
    if 'emulated_syscalls' not in converted_df.columns:
        base_syscalls = pd.Series([100] * len(converted_df), index=converted_df.index)  # Normal syscall count
        if 'version_anomaly' in converted_df.columns:
            version_vals = pd.to_numeric(converted_df['version_anomaly'], errors='coerce').fillna(0)
            base_syscalls += version_vals * 500
        if 'boot_sequence_irregularities' in converted_df.columns:
            boot_vals = pd.to_numeric(converted_df['boot_sequence_irregularities'], errors='coerce').fillna(0)
            base_syscalls += boot_vals * 300
        if 'integrity_check_missing_or_altered' in converted_df.columns:
            integrity_vals = pd.to_numeric(converted_df['integrity_check_missing_or_altered'], errors='coerce').fillna(0)
            base_syscalls += integrity_vals * 400
        converted_df['emulated_syscalls'] = base_syscalls.clip(0, 5000).astype(int)
    
    # 5. hardcoded_ip_count: Derive from attack scenarios
    if 'hardcoded_ip_count' not in converted_df.columns:
        if 'synthetic_attack_scenario' in converted_df.columns:
            # Count IP-related attacks
            ip_attacks = converted_df['synthetic_attack_scenario'].astype(str).str.contains(
                'backdoor|command|injection|exfiltration', case=False, na=False
            )
            converted_df['hardcoded_ip_count'] = ip_attacks.astype(int) * 3
        else:
            converted_df['hardcoded_ip_count'] = 0
    
    # 6. hardcoded_url_count: Similar to IP count
    if 'hardcoded_url_count' not in converted_df.columns:
        if 'synthetic_attack_scenario' in converted_df.columns:
            url_attacks = converted_df['synthetic_attack_scenario'].astype(str).str.contains(
                'exfiltration|phishing|c2', case=False, na=False
            )
            converted_df['hardcoded_url_count'] = url_attacks.astype(int) * 2
        else:
            converted_df['hardcoded_url_count'] = 0
    
    # 7. crypto_function_count: Derive from modification type
    if 'crypto_function_count' not in converted_df.columns:
        if 'firmware_modification_type' in converted_df.columns:
            crypto_mods = converted_df['firmware_modification_type'].astype(str).str.contains(
                'encryption|crypto|signature', case=False, na=False
            )
            converted_df['crypto_function_count'] = crypto_mods.astype(int) * 50
        else:
            converted_df['crypto_function_count'] = 10  # Normal crypto usage
    
    # 8. Additional features for ML models
    if 'num_executables' not in converted_df.columns:
        converted_df['num_executables'] = 5  # Default
    if 'num_scripts' not in converted_df.columns:
        converted_df['num_scripts'] = 2  # Default
    if 'string_count' not in converted_df.columns:
        converted_df['string_count'] = 100  # Default
    if 'file_size_bytes' not in converted_df.columns:
        converted_df['file_size_bytes'] = 1024 * 1024  # Default 1MB
    
    # Preserve MITRE-specific columns for forensic analysis
    mitre_columns = [
        'version_anomaly',
        'boot_sequence_irregularities',
        'integrity_check_missing_or_altered',
        'firmware_modification_type',
        'synthetic_attack_scenario',
        'mitre_attack_technique',
        'classification',
        'severity_level',
        'malicious_activity_time_window'
    ]
    
    # Keep original MITRE columns if they exist
    for col in mitre_columns:
        if col in df.columns and col not in converted_df.columns:
            converted_df[col] = df[col]
    
    # Create target column for training (if classification exists)
    if 'classification' in converted_df.columns and 'clean_label' not in converted_df.columns:
        # Map classification to clean_label: Untampered = 1 (clean), others = 0 (tampered)
        converted_df['clean_label'] = (converted_df['classification'] == 'Untampered').astype(int)
    
    return converted_df

def detect_mitre_dataset(df: pd.DataFrame) -> bool:
    """Check if dataset is in MITRE format"""
    mitre_indicators = [
        'firmware_hash_sha256',
        'version_anomaly',
        'boot_sequence_irregularities',
        'integrity_check_missing_or_altered',
        'mitre_attack_technique',
        'classification',
        'severity_level'
    ]
    
    # Check if at least 4 MITRE indicators are present
    present_indicators = sum(1 for col in mitre_indicators if col in df.columns)
    return present_indicators >= 4

def process_mitre_dataset(file_path: Path) -> pd.DataFrame:
    """
    Load and convert MITRE dataset to system format
    """
    df = pd.read_csv(file_path)
    
    if not detect_mitre_dataset(df):
        raise ValueError("Dataset does not appear to be in MITRE format")
    
    converted_df = convert_mitre_dataset_to_system_format(df)
    
    return converted_df

