"""
Firmware Injection Detection Module
Detects and classifies different types of firmware injection
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Injection type mappings from MITRE attack techniques
INJECTION_TYPE_MAPPINGS = {
    'Code Injection': [
        'T1055', 'T1059', 'T1106',  # Process Injection, Command Injection, Execution
        'code_injection', 'function_injection', 'module_injection'
    ],
    'Bootloader Injection': [
        'T1542', 'T1543',  # Pre-OS Boot, Create or Modify System Process
        'bootloader', 'boot_sequence', 'initialization'
    ],
    'Configuration Injection': [
        'T1112', 'T1562',  # Modify Registry, Disable Security Tools
        'config', 'parameter', 'settings', 'geofencing', 'altitude_limit'
    ],
    'Backdoor Injection': [
        'T1133', 'T1071', 'T1095',  # External Remote Services, Application Layer Protocol
        'backdoor', 'trigger', 'hidden', 'latent'
    ],
    'Sensor Spoofing': [
        'T1565', 'T1498',  # Data Manipulation, Network Denial of Service
        'gps_spoofing', 'sensor', 'altitude', 'battery', 'imu'
    ]
}

def detect_injection_type_from_mitre(df: pd.DataFrame, row_idx: int = 0) -> Dict:
    """
    Detect injection type from MITRE dataset columns
    Returns injection type, confidence, and evidence
    """
    injection_analysis = {
        'injection_detected': False,
        'injection_type': None,
        'injection_types': [],
        'confidence': 0.0,
        'evidence': [],
        'activation_timeline': None
    }
    
    # Get row data (use first row if single-row dataset)
    if len(df) > row_idx:
        row = df.iloc[row_idx]
    else:
        row = df.iloc[0]
    
    # Check MITRE attack technique
    mitre_technique = str(row.get('mitre_attack_technique', '')).upper()
    modification_type = str(row.get('firmware_modification_type', '')).lower()
    attack_scenario = str(row.get('synthetic_attack_scenario', '')).lower()
    classification = str(row.get('classification', '')).lower()
    
    detected_types = []
    evidence_list = []
    confidence_score = 0.0
    
    # 1. Check MITRE attack technique
    if mitre_technique and mitre_technique != 'NONE' and mitre_technique != 'NAN':
        for inj_type, techniques in INJECTION_TYPE_MAPPINGS.items():
            if any(tech in mitre_technique for tech in techniques):
                detected_types.append(inj_type)
                evidence_list.append(f"MITRE technique {mitre_technique} indicates {inj_type}")
                confidence_score += 0.3
    
    # 2. Check firmware modification type
    if modification_type and modification_type != 'none' and modification_type != 'nan':
        if 'code' in modification_type or 'function' in modification_type:
            if 'Code Injection' not in detected_types:
                detected_types.append('Code Injection')
            evidence_list.append(f"Modification type: {modification_type}")
            confidence_score += 0.25
        elif 'boot' in modification_type or 'loader' in modification_type:
            if 'Bootloader Injection' not in detected_types:
                detected_types.append('Bootloader Injection')
            evidence_list.append(f"Modification type: {modification_type}")
            confidence_score += 0.25
        elif 'config' in modification_type or 'parameter' in modification_type:
            if 'Configuration Injection' not in detected_types:
                detected_types.append('Configuration Injection')
            evidence_list.append(f"Modification type: {modification_type}")
            confidence_score += 0.25
        elif 'backdoor' in modification_type:
            if 'Backdoor Injection' not in detected_types:
                detected_types.append('Backdoor Injection')
            evidence_list.append(f"Modification type: {modification_type}")
            confidence_score += 0.3
    
    # 3. Check attack scenario
    if attack_scenario and attack_scenario != 'none' and attack_scenario != 'nan':
        if 'gps' in attack_scenario or 'sensor' in attack_scenario:
            if 'Sensor Spoofing' not in detected_types:
                detected_types.append('Sensor Spoofing')
            evidence_list.append(f"Attack scenario: {attack_scenario}")
            confidence_score += 0.2
        elif 'backdoor' in attack_scenario or 'trigger' in attack_scenario:
            if 'Backdoor Injection' not in detected_types:
                detected_types.append('Backdoor Injection')
            evidence_list.append(f"Attack scenario: {attack_scenario}")
            confidence_score += 0.2
    
    # 4. Check classification
    if classification:
        if 'tampered' in classification or 'suspicious' in classification or 'infected' in classification or 'malicious' in classification:
            injection_analysis['injection_detected'] = True
            confidence_score += 0.2
            evidence_list.append(f"Classification: {classification}")
        elif 'untampered' in classification or 'clean' in classification:
            # Still check for injection, but lower confidence
            confidence_score *= 0.5
    
    # 5. Check forensic indicators
    version_anomaly = row.get('version_anomaly', 0)
    boot_irregularities = row.get('boot_sequence_irregularities', 0)
    integrity_issues = row.get('integrity_check_missing_or_altered', 0)
    
    if version_anomaly > 0:
        evidence_list.append("Version anomaly detected")
        confidence_score += 0.1
    if boot_irregularities > 0:
        if 'Bootloader Injection' not in detected_types:
            detected_types.append('Bootloader Injection')
        evidence_list.append("Boot sequence irregularities detected")
        confidence_score += 0.15
    if integrity_issues > 0:
        evidence_list.append("Integrity checks missing or altered")
        confidence_score += 0.15
    
    # Determine primary injection type
    if detected_types:
        # Use most common or highest confidence type
        injection_analysis['injection_type'] = detected_types[0]
        injection_analysis['injection_types'] = list(set(detected_types))
        injection_analysis['injection_detected'] = True
    elif confidence_score > 0.3:
        # Generic injection if confidence is high but no specific type
        injection_analysis['injection_type'] = "General Injection"
        injection_analysis['injection_detected'] = True
    
    # Cap confidence at 1.0
    injection_analysis['confidence'] = min(confidence_score, 1.0)
    injection_analysis['evidence'] = evidence_list
    
    # Get activation timeline
    time_window = row.get('malicious_activity_time_window')
    if time_window and str(time_window) != 'None' and str(time_window) != 'nan':
        injection_analysis['activation_timeline'] = str(time_window)
    else:
        injection_analysis['activation_timeline'] = f"Detected at {datetime.utcnow().isoformat()}"
    
    return injection_analysis

def detect_execution_flow_anomalies(df: pd.DataFrame) -> Dict:
    """
    Detect execution flow anomalies that indicate code injection
    """
    anomalies = {
        'unexpected_calls': [],
        'rare_sequences': [],
        'unauthorized_routines': [],
        'anomaly_score': 0.0
    }
    
    # Check for execution flow indicators
    if 'emulated_syscalls' in df.columns:
        syscall_values = df['emulated_syscalls'].dropna()
        if len(syscall_values) > 0:
            avg_syscalls = syscall_values.mean()
            # High syscall count suggests injected routines
            if avg_syscalls > 500:
                anomalies['unexpected_calls'].append(f"High syscall count: {avg_syscalls:.0f}")
                anomalies['anomaly_score'] += 0.3
    
    # Check entropy for obfuscated/injected code
    if 'entropy_score' in df.columns:
        entropy_values = df['entropy_score'].dropna()
        if len(entropy_values) > 0:
            avg_entropy = entropy_values.mean()
            # High entropy suggests obfuscated/injected code
            if avg_entropy > 7.0:
                anomalies['unauthorized_routines'].append(f"High entropy suggests obfuscated code: {avg_entropy:.2f}")
                anomalies['anomaly_score'] += 0.25
    
    # Check boot time for bootloader injection
    if 'boot_time_ms' in df.columns:
        boot_times = df['boot_time_ms'].dropna()
        if len(boot_times) > 0:
            avg_boot = boot_times.mean()
            # Extended boot time suggests bootloader injection
            if avg_boot > 5000:
                anomalies['rare_sequences'].append(f"Extended boot time: {avg_boot:.0f}ms")
                anomalies['anomaly_score'] += 0.2
    
    anomalies['anomaly_score'] = min(anomalies['anomaly_score'], 1.0)
    return anomalies

def detect_sensor_spoofing(df: pd.DataFrame) -> Dict:
    """
    Detect sensor spoofing/data injection
    """
    spoofing_analysis = {
        'gps_spoofing': False,
        'altitude_spoofing': False,
        'battery_spoofing': False,
        'imu_spoofing': False,
        'evidence': [],
        'confidence': 0.0
    }
    
    # Check for sensor-related columns
    sensor_cols = [col for col in df.columns if any(s in col.lower() for s in ['gps', 'altitude', 'battery', 'imu', 'gyro', 'accel'])]
    
    for col in sensor_cols:
        values = df[col].dropna()
        if len(values) > 1:
            # Check for unrealistic values
            if 'gps' in col.lower() or 'latitude' in col.lower() or 'longitude' in col.lower():
                # GPS should be within valid range
                if values.min() < -180 or values.max() > 180:
                    spoofing_analysis['gps_spoofing'] = True
                    spoofing_analysis['evidence'].append(f"Invalid GPS values in {col}")
                    spoofing_analysis['confidence'] += 0.3
                # Check for sudden jumps
                diff = values.diff().abs()
                if diff.max() > 10:  # More than 10 degrees jump
                    spoofing_analysis['gps_spoofing'] = True
                    spoofing_analysis['evidence'].append(f"GPS jump detected in {col}")
                    spoofing_analysis['confidence'] += 0.2
            
            if 'altitude' in col.lower() or 'alt' in col.lower():
                # Check for unrealistic altitude changes
                diff = values.diff().abs()
                if diff.max() > 100:  # More than 100 units change
                    spoofing_analysis['altitude_spoofing'] = True
                    spoofing_analysis['evidence'].append(f"Rapid altitude change in {col}")
                    spoofing_analysis['confidence'] += 0.2
            
            if 'battery' in col.lower():
                # Battery should be 0-100%
                if values.min() < 0 or values.max() > 100:
                    spoofing_analysis['battery_spoofing'] = True
                    spoofing_analysis['evidence'].append(f"Invalid battery values in {col}")
                    spoofing_analysis['confidence'] += 0.2
    
    spoofing_analysis['confidence'] = min(spoofing_analysis['confidence'], 1.0)
    return spoofing_analysis

def detect_safety_system_bypass(df: pd.DataFrame) -> Dict:
    """
    Detect disabled failsafes or safety system overrides
    """
    bypass_analysis = {
        'failsafe_disabled': False,
        'geofencing_disabled': False,
        'emergency_bypass': False,
        'evidence': [],
        'confidence': 0.0
    }
    
    # Check for configuration injection indicators
    if 'hardcoded_ip_count' in df.columns:
        ip_count = df['hardcoded_ip_count'].sum()
        if ip_count > 0:
            bypass_analysis['evidence'].append(f"Hardcoded IPs found: {ip_count}")
            bypass_analysis['confidence'] += 0.2
    
    # Check integrity checks
    if 'is_signed' in df.columns:
        signed_ratio = df['is_signed'].mean()
        if signed_ratio < 0.5:
            bypass_analysis['failsafe_disabled'] = True
            bypass_analysis['evidence'].append(f"Low signature coverage: {signed_ratio*100:.1f}%")
            bypass_analysis['confidence'] += 0.3
    
    # Check boot sequence
    if 'boot_sequence_irregularities' in df.columns:
        irregularities = df['boot_sequence_irregularities'].sum()
        if irregularities > 0:
            bypass_analysis['emergency_bypass'] = True
            bypass_analysis['evidence'].append(f"Boot sequence irregularities: {irregularities}")
            bypass_analysis['confidence'] += 0.25
    
    bypass_analysis['confidence'] = min(bypass_analysis['confidence'], 1.0)
    return bypass_analysis

def comprehensive_injection_analysis(df: pd.DataFrame, tampering_prob: float, anomaly_score: float) -> Dict:
    """
    Comprehensive injection detection analysis
    Combines all injection detection methods
    """
    # Detect injection type from MITRE data
    injection_type_analysis = detect_injection_type_from_mitre(df)
    
    # Detect execution flow anomalies
    execution_anomalies = detect_execution_flow_anomalies(df)
    
    # Detect sensor spoofing
    sensor_spoofing = detect_sensor_spoofing(df)
    
    # Detect safety system bypass
    safety_bypass = detect_safety_system_bypass(df)
    
    # Combine results
    combined_confidence = (
        injection_type_analysis['confidence'] * 0.4 +
        execution_anomalies['anomaly_score'] * 0.3 +
        sensor_spoofing['confidence'] * 0.2 +
        safety_bypass['confidence'] * 0.1
    )
    
    # Use MITRE classification if available
    mitre_classification = None
    if 'classification' in df.columns:
        classifications = df['classification'].dropna().unique()
        if len(classifications) > 0:
            mitre_classification = str(classifications[0])
    
    # Determine final injection status
    if injection_type_analysis['injection_detected'] or combined_confidence > 0.5:
        injection_status = "Injected"
    elif combined_confidence > 0.3:
        injection_status = "Suspicious"
    else:
        injection_status = "Not Injected"
    
    return {
        'injection_detected': injection_type_analysis['injection_detected'] or combined_confidence > 0.5,
        'injection_status': injection_status,
        'injection_type': injection_type_analysis.get('injection_type', 'Unknown'),
        'injection_types': injection_type_analysis.get('injection_types', []),
        'confidence': min(combined_confidence, 1.0),
        'mitre_classification': mitre_classification,
        'execution_flow_anomalies': execution_anomalies,
        'sensor_spoofing': sensor_spoofing,
        'safety_system_bypass': safety_bypass,
        'evidence': injection_type_analysis.get('evidence', []),
        'activation_timeline': injection_type_analysis.get('activation_timeline'),
        'detailed_analysis': {
            'injection_type_analysis': injection_type_analysis,
            'execution_anomalies': execution_anomalies,
            'sensor_spoofing': sensor_spoofing,
            'safety_bypass': safety_bypass
        }
    }

