"""
Enhanced Forensic Analysis Module
Calculates comprehensive forensic features for firmware tampering detection
"""
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

def calculate_sha256_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of firmware file"""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        return file_hash
    except Exception as e:
        print(f"Warning: Failed to calculate hash: {e}")
        return "unknown"

def detect_version_anomalies(df: pd.DataFrame) -> Dict:
    """Detect firmware version anomalies (modifications, downgrades)"""
    anomalies = {
        'has_version_mod': False,
        'has_downgrade': False,
        'version_anomaly_score': 0.0,
        'version_details': []
    }
    
    # Check for version strings in data
    version_patterns = [
        r'v?(\d+)\.(\d+)\.(\d+)',  # Semantic versioning
        r'(\d+)\.(\d+)',  # Major.minor
        r'v(\d+)',  # Simple version
    ]
    
    version_values = []
    for col in df.columns:
        if 'version' in col.lower() or 'ver' in col.lower():
            for val in df[col].dropna():
                if isinstance(val, (int, float)):
                    version_values.append(float(val))
                elif isinstance(val, str):
                    for pattern in version_patterns:
                        match = re.search(pattern, val)
                        if match:
                            version_values.append(float(match.group(1)))
                            break
    
    if version_values:
        # Check for unexpected patterns
        if len(set(version_values)) > 3:  # Multiple versions suggest modification
            anomalies['has_version_mod'] = True
            anomalies['version_anomaly_score'] += 0.3
        
        # Check for downgrades (decreasing version numbers)
        sorted_versions = sorted(version_values, reverse=True)
        if len(sorted_versions) > 1 and sorted_versions[0] > sorted_versions[-1] * 1.5:
            anomalies['has_downgrade'] = True
            anomalies['version_anomaly_score'] += 0.5
    
    anomalies['version_anomaly_score'] = min(anomalies['version_anomaly_score'], 1.0)
    return anomalies

def analyze_boot_sequence(df: pd.DataFrame) -> Dict:
    """Analyze boot sequence for irregularities"""
    boot_analysis = {
        'boot_irregularities': [],
        'boot_anomaly_score': 0.0,
        'boot_time_anomaly': False,
        'boot_sequence_anomaly': False
    }
    
    # Check boot time
    if 'boot_time_ms' in df.columns:
        boot_times = df['boot_time_ms'].dropna()
        if len(boot_times) > 0:
            avg_boot_time = boot_times.mean()
            max_boot_time = boot_times.max()
            
            # Normal boot time should be < 5000ms
            if avg_boot_time > 5000:
                boot_analysis['boot_time_anomaly'] = True
                boot_analysis['boot_irregularities'].append(f"Extended boot time: {avg_boot_time:.0f}ms")
                boot_analysis['boot_anomaly_score'] += 0.4
            
            # Check for boot time variance (suggests inconsistent behavior)
            if len(boot_times) > 1:
                boot_variance = boot_times.std() / (boot_times.mean() + 1e-6)
                if boot_variance > 0.5:  # High variance
                    boot_analysis['boot_sequence_anomaly'] = True
                    boot_analysis['boot_irregularities'].append(f"High boot time variance: {boot_variance:.2f}")
                    boot_analysis['boot_anomaly_score'] += 0.3
    
    # Check for missing boot-related features
    boot_related_cols = ['boot_time_ms', 'is_signed', 'entropy_score']
    missing_boot_features = [col for col in boot_related_cols if col not in df.columns]
    if missing_boot_features:
        boot_analysis['boot_irregularities'].append(f"Missing boot features: {', '.join(missing_boot_features)}")
        boot_analysis['boot_anomaly_score'] += 0.2
    
    boot_analysis['boot_anomaly_score'] = min(boot_analysis['boot_anomaly_score'], 1.0)
    return boot_analysis

def check_integrity_checks(df: pd.DataFrame) -> Dict:
    """Check for missing or altered integrity checks"""
    integrity = {
        'missing_integrity_checks': False,
        'altered_integrity_checks': False,
        'integrity_anomaly_score': 0.0,
        'integrity_details': []
    }
    
    # Check signature status
    if 'is_signed' in df.columns:
        signed_values = df['is_signed'].dropna()
        if len(signed_values) > 0:
            signed_ratio = signed_values.mean()
            if signed_ratio < 0.5:  # Less than 50% signed
                integrity['missing_integrity_checks'] = True
                integrity['integrity_details'].append(f"Low signature coverage: {signed_ratio*100:.1f}%")
                integrity['integrity_anomaly_score'] += 0.5
            elif signed_ratio < 0.8:  # Partial signing
                integrity['altered_integrity_checks'] = True
                integrity['integrity_details'].append(f"Partial signature coverage: {signed_ratio*100:.1f}%")
                integrity['integrity_anomaly_score'] += 0.3
    
    # Check for hash mismatches (if hash column exists)
    if 'sha256_hash' in df.columns:
        hashes = df['sha256_hash'].dropna().unique()
        if len(hashes) > 1:  # Multiple different hashes suggest modification
            integrity['altered_integrity_checks'] = True
            integrity['integrity_details'].append(f"Multiple hash values detected: {len(hashes)} unique hashes")
            integrity['integrity_anomaly_score'] += 0.4
    
    # Check entropy (high entropy can indicate obfuscation/encryption)
    if 'entropy_score' in df.columns:
        entropy_values = df['entropy_score'].dropna()
        if len(entropy_values) > 0:
            high_entropy = (entropy_values > 7.5).sum()
            if high_entropy > len(entropy_values) * 0.3:  # >30% high entropy
                integrity['integrity_details'].append(f"High entropy detected: {high_entropy}/{len(entropy_values)} samples")
                integrity['integrity_anomaly_score'] += 0.2
    
    integrity['integrity_anomaly_score'] = min(integrity['integrity_anomaly_score'], 1.0)
    return integrity

def calculate_severity_level(tampering_prob: float, anomaly_score: float, 
                           forensic_features: Dict) -> Tuple[str, float]:
    """
    Calculate severity level based on multiple factors
    Returns: (severity_level, severity_score)
    Severity levels: Low, Medium, High, Critical
    """
    severity_score = 0.0
    
    # Base severity from tampering probability
    severity_score += tampering_prob * 0.4
    
    # Anomaly score contribution
    severity_score += anomaly_score * 0.3
    
    # Forensic features contribution
    if forensic_features.get('version_anomaly_score', 0) > 0.5:
        severity_score += 0.1
    if forensic_features.get('boot_anomaly_score', 0) > 0.5:
        severity_score += 0.1
    if forensic_features.get('integrity_anomaly_score', 0) > 0.5:
        severity_score += 0.1
    
    severity_score = min(severity_score, 1.0)
    
    # Determine severity level
    if severity_score >= 0.8:
        level = "Critical"
    elif severity_score >= 0.6:
        level = "High"
    elif severity_score >= 0.4:
        level = "Medium"
    else:
        level = "Low"
    
    return level, severity_score

def classify_tampering_status(tampering_prob: float, anomaly_score: float) -> str:
    """
    Classify firmware status: Untampered, Suspicious, Tampered
    """
    # Combine probability and anomaly score
    combined_score = (tampering_prob * 0.6) + (anomaly_score * 0.4)
    
    if combined_score >= 0.7:
        return "Tampered"
    elif combined_score >= 0.4:
        return "Suspicious"
    else:
        return "Untampered"

def calculate_feature_contributions(df: pd.DataFrame, tampering_prob: float) -> Dict:
    """Calculate which features contributed most to tampering detection"""
    contributions = {}
    
    # Feature importance weights (based on typical ML feature importance)
    feature_weights = {
        'entropy_score': 0.25,
        'is_signed': 0.20,
        'boot_time_ms': 0.15,
        'hardcoded_ip_count': 0.10,
        'hardcoded_url_count': 0.10,
        'emulated_syscalls': 0.10,
        'crypto_function_count': 0.10
    }
    
    total_contribution = 0.0
    for feature, weight in feature_weights.items():
        if feature in df.columns:
            feature_values = df[feature].dropna()
            if len(feature_values) > 0:
                # Normalize feature value (0-1 scale)
                if feature == 'is_signed':
                    normalized = 1 - feature_values.mean()  # Inverted (unsigned = suspicious)
                elif feature == 'boot_time_ms':
                    normalized = min(feature_values.mean() / 10000, 1.0)  # Normalize to 10s max
                elif feature in ['hardcoded_ip_count', 'hardcoded_url_count', 'emulated_syscalls', 'crypto_function_count']:
                    normalized = min(feature_values.sum() / 100, 1.0)  # Normalize to 100 max
                else:
                    # For entropy and others, normalize to 0-1
                    normalized = min(feature_values.mean() / 8.0, 1.0)
                
                contribution = normalized * weight * tampering_prob
                contributions[feature] = {
                    'value': float(feature_values.mean()) if feature != 'is_signed' else float(feature_values.mean()),
                    'contribution': contribution,
                    'percentage': 0.0  # Will be calculated after
                }
                total_contribution += contribution
    
    # Normalize contributions to percentages
    if total_contribution > 0:
        for feature in contributions:
            contributions[feature]['percentage'] = (contributions[feature]['contribution'] / total_contribution) * 100
    
    return contributions

def generate_timeline_data(df: pd.DataFrame, tampering_prob: float) -> List[Dict]:
    """Generate timeline data for visualization"""
    timeline = []
    
    # If we have time-based columns, use them
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    
    if time_cols and len(df) > 1:
        # Use first time column
        time_col = time_cols[0]
        for idx, row in df.iterrows():
            timestamp = row.get(time_col, idx)
            anomaly_score = row.get('entropy_score', 0) / 8.0 if 'entropy_score' in row else tampering_prob
            
            timeline.append({
                'time': str(timestamp),
                'anomaly_score': float(anomaly_score),
                'tampering_probability': float(tampering_prob),
                'is_anomaly': anomaly_score > 0.5
            })
    else:
        # Generate synthetic timeline based on row index
        for idx in range(len(df)):
            # Simulate anomaly detection over time
            base_anomaly = tampering_prob
            # Add some variation
            variation = np.sin(idx * 0.1) * 0.2
            anomaly_score = max(0, min(1, base_anomaly + variation))
            
            timeline.append({
                'time': f"Sample {idx + 1}",
                'anomaly_score': float(anomaly_score),
                'tampering_probability': float(tampering_prob),
                'is_anomaly': anomaly_score > 0.5
            })
    
    return timeline

def analyze_sensor_behavior(df: pd.DataFrame) -> Dict:
    """Analyze sensor behavior patterns (GPS, Altitude, IMU, Battery)"""
    sensor_analysis = {
        'gps': {'anomalies': [], 'score': 0.0},
        'altitude': {'anomalies': [], 'score': 0.0},
        'imu': {'anomalies': [], 'score': 0.0},
        'battery': {'anomalies': [], 'score': 0.0}
    }
    
    # Check for GPS-related columns
    gps_cols = [col for col in df.columns if 'gps' in col.lower() or 'latitude' in col.lower() or 'longitude' in col.lower()]
    if gps_cols:
        for col in gps_cols:
            values = df[col].dropna()
            if len(values) > 1:
                # Check for sudden jumps (spoofing indicator)
                diff = values.diff().abs()
                if diff.max() > values.std() * 3:
                    sensor_analysis['gps']['anomalies'].append(f"GPS jump detected in {col}")
                    sensor_analysis['gps']['score'] += 0.3
    
    # Check for altitude columns
    alt_cols = [col for col in df.columns if 'altitude' in col.lower() or 'alt' in col.lower()]
    if alt_cols:
        for col in alt_cols:
            values = df[col].dropna()
            if len(values) > 1:
                # Check for unrealistic altitude changes
                diff = values.diff().abs()
                if diff.max() > 100:  # More than 100 units change
                    sensor_analysis['altitude']['anomalies'].append(f"Rapid altitude change in {col}")
                    sensor_analysis['altitude']['score'] += 0.3
    
    # Check for IMU/gyro columns
    imu_cols = [col for col in df.columns if 'imu' in col.lower() or 'gyro' in col.lower() or 'accel' in col.lower()]
    if imu_cols:
        for col in imu_cols:
            values = df[col].dropna()
            if len(values) > 1:
                # Check for constant values (sensor freeze)
                if values.nunique() < 3:
                    sensor_analysis['imu']['anomalies'].append(f"IMU sensor freeze in {col}")
                    sensor_analysis['imu']['score'] += 0.4
    
    # Check for battery columns
    battery_cols = [col for col in df.columns if 'battery' in col.lower() or 'power' in col.lower()]
    if battery_cols:
        for col in battery_cols:
            values = df[col].dropna()
            if len(values) > 1:
                # Check for unrealistic battery behavior
                if values.min() < 0 or values.max() > 100:
                    sensor_analysis['battery']['anomalies'].append(f"Invalid battery values in {col}")
                    sensor_analysis['battery']['score'] += 0.3
    
    # Normalize scores
    for sensor in sensor_analysis:
        sensor_analysis[sensor]['score'] = min(sensor_analysis[sensor]['score'], 1.0)
    
    return sensor_analysis

def perform_comprehensive_forensic_analysis(
    df: pd.DataFrame, 
    evidence_path: Path,
    tampering_prob: float,
    anomaly_score: float
) -> Dict:
    """
    Perform comprehensive forensic analysis
    Returns all forensic features and classifications
    """
    # Calculate SHA-256 hash
    file_hash = calculate_sha256_hash(evidence_path)
    
    # Detect version anomalies
    version_anomalies = detect_version_anomalies(df)
    
    # Analyze boot sequence
    boot_analysis = analyze_boot_sequence(df)
    
    # Check integrity
    integrity_checks = check_integrity_checks(df)
    
    # Calculate severity
    forensic_features = {
        'version_anomaly_score': version_anomalies['version_anomaly_score'],
        'boot_anomaly_score': boot_analysis['boot_anomaly_score'],
        'integrity_anomaly_score': integrity_checks['integrity_anomaly_score']
    }
    severity_level, severity_score = calculate_severity_level(
        tampering_prob, anomaly_score, forensic_features
    )
    
    # Classify tampering status
    tampering_status = classify_tampering_status(tampering_prob, anomaly_score)
    
    # Calculate feature contributions
    feature_contributions = calculate_feature_contributions(df, tampering_prob)
    
    # Generate timeline
    timeline_data = generate_timeline_data(df, tampering_prob)
    
    # Analyze sensor behavior
    sensor_behavior = analyze_sensor_behavior(df)
    
    # Estimate time window
    time_window = datetime.utcnow().isoformat()
    
    return {
        'sha256_hash': file_hash,
        'version_anomalies': version_anomalies,
        'boot_analysis': boot_analysis,
        'integrity_checks': integrity_checks,
        'severity_level': severity_level,
        'severity_score': severity_score,
        'tampering_status': tampering_status,
        'tampering_probability': tampering_prob,
        'anomaly_score': anomaly_score,
        'feature_contributions': feature_contributions,
        'timeline_data': timeline_data,
        'sensor_behavior': sensor_behavior,
        'time_window': time_window,
        'forensic_features': forensic_features
    }

