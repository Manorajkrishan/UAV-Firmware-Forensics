"""
Firmware Parser - Converts binary firmware files (.bin, .hex, .elf) to CSV format
Extracts features for ML analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import struct
import re
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

def calculate_entropy(data: bytes) -> float:
    """Calculate Shannon entropy of binary data"""
    if not data:
        return 0.0
    
    # Count byte frequencies
    byte_counts = {}
    for byte in data:
        byte_counts[byte] = byte_counts.get(byte, 0) + 1
    
    # Calculate entropy
    entropy = 0.0
    data_len = len(data)
    for count in byte_counts.values():
        probability = count / data_len
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy

def extract_strings(data: bytes, min_length: int = 4) -> List[str]:
    """Extract printable strings from binary data"""
    strings = []
    current_string = b''
    
    for byte in data:
        if 32 <= byte <= 126:  # Printable ASCII
            current_string += bytes([byte])
        else:
            if len(current_string) >= min_length:
                try:
                    strings.append(current_string.decode('utf-8', errors='ignore'))
                except:
                    pass
            current_string = b''
    
    # Add last string if exists
    if len(current_string) >= min_length:
        try:
            strings.append(current_string.decode('utf-8', errors='ignore'))
        except:
            pass
    
    return strings

def detect_ip_addresses(strings: List[str]) -> int:
    """Count IP addresses in strings"""
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    count = 0
    for s in strings:
        if re.search(ip_pattern, s):
            count += 1
    return count

def detect_urls(strings: List[str]) -> int:
    """Count URLs in strings"""
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    count = 0
    for s in strings:
        if re.search(url_pattern, s, re.IGNORECASE):
            count += 1
    return count

def detect_crypto_functions(strings: List[str]) -> int:
    """Count cryptographic function names"""
    crypto_keywords = [
        'aes', 'des', 'rsa', 'sha', 'md5', 'cipher', 'encrypt', 'decrypt',
        'hash', 'hmac', 'ssl', 'tls', 'crypto', 'signature', 'certificate'
    ]
    count = 0
    for s in strings:
        s_lower = s.lower()
        for keyword in crypto_keywords:
            if keyword in s_lower:
                count += 1
                break
    return count

def detect_sections(data: bytes) -> Dict:
    """Detect firmware sections (code, data, etc.)"""
    sections = {
        'code_section_size': 0,
        'data_section_size': 0,
        'bss_section_size': 0,
        'num_sections': 0
    }
    
    # Simple heuristic: high entropy = code, low entropy = data
    chunk_size = 1024
    code_size = 0
    data_size = 0
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        entropy = calculate_entropy(chunk)
        
        if entropy > 6.0:  # High entropy = likely code
            code_size += len(chunk)
        else:  # Low entropy = likely data
            data_size += len(chunk)
    
    sections['code_section_size'] = code_size
    sections['data_section_size'] = data_size
    sections['num_sections'] = max(1, (code_size + data_size) // chunk_size)
    
    return sections

def check_signature(data: bytes) -> bool:
    """Check if firmware appears to be signed"""
    # Look for common signature patterns
    signature_patterns = [
        b'-----BEGIN',  # PEM signature
        b'-----END',
        b'\x30\x82',  # ASN.1/DER signature
        b'PKCS',  # PKCS signature
        b'RSA',  # RSA signature marker
    ]
    
    for pattern in signature_patterns:
        if pattern in data:
            return True
    
    return False

def detect_executables(data: bytes) -> int:
    """Detect executable file markers"""
    executable_markers = [
        b'MZ',  # PE/Windows
        b'\x7fELF',  # ELF/Linux
        b'\xfe\xed\xfa',  # Mach-O
        b'#!/',  # Script shebang
    ]
    
    count = 0
    for marker in executable_markers:
        if marker in data:
            count += 1
    
    return count

def detect_scripts(data: bytes) -> int:
    """Detect script files"""
    script_markers = [
        b'#!/bin/sh',
        b'#!/bin/bash',
        b'#!/usr/bin/python',
        b'#!/usr/bin/env',
        b'<script',
        b'<?php',
    ]
    
    count = 0
    for marker in script_markers:
        if marker in data:
            count += 1
    
    return count

def parse_bin_firmware(file_path: Path) -> Dict:
    """Parse binary firmware file and extract features"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
    except Exception as e:
        raise ValueError(f"Failed to read file: {str(e)}")
    
    if not data:
        raise ValueError("File is empty")
    
    # Calculate basic features
    file_size = len(data)
    entropy = calculate_entropy(data)
    
    # Extract strings
    strings = extract_strings(data)
    string_count = len(strings)
    
    # Detect patterns
    ip_count = detect_ip_addresses(strings)
    url_count = detect_urls(strings)
    crypto_count = detect_crypto_functions(strings)
    
    # Detect sections
    sections = detect_sections(data)
    
    # Check signature
    is_signed = check_signature(data)
    
    # Detect executables and scripts
    num_executables = detect_executables(data)
    num_scripts = detect_scripts(data)
    
    # Calculate section entropies
    chunk_size = min(4096, file_size // 10) if file_size > 0 else 1024
    entropies = []
    for i in range(0, file_size, chunk_size):
        chunk = data[i:i+chunk_size]
        if chunk:
            entropies.append(calculate_entropy(chunk))
    
    avg_entropy = np.mean(entropies) if entropies else entropy
    max_entropy = max(entropies) if entropies else entropy
    min_entropy = min(entropies) if entropies else entropy
    
    # Calculate file hash
    file_hash = hashlib.sha256(data).hexdigest()
    
    # Build feature dictionary
    features = {
        'file_size_bytes': file_size,
        'entropy_score': entropy,
        'avg_section_entropy': avg_entropy,
        'max_section_entropy': max_entropy,
        'min_section_entropy': min_entropy,
        'is_signed': 1 if is_signed else 0,
        'signature_type': 'signed' if is_signed else 'unsigned',
        'string_count': string_count,
        'hardcoded_ip_count': ip_count,
        'hardcoded_url_count': url_count,
        'crypto_function_count': crypto_count,
        'num_executables': num_executables,
        'num_scripts': num_scripts,
        'code_section_size': sections['code_section_size'],
        'data_section_size': sections['data_section_size'],
        'num_sections': sections['num_sections'],
        'boot_time_ms': 0,  # Default, can be estimated
        'emulated_syscalls': 0,  # Default, can be estimated
        'sha256_hash': file_hash,
    }
    
    return features

def parse_hex_firmware(file_path: Path) -> Dict:
    """Parse Intel HEX format firmware"""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        raise ValueError(f"Failed to read HEX file: {str(e)}")
    
    # Convert HEX to binary
    binary_data = b''
    for line in lines:
        line = line.strip()
        if line.startswith(':'):
            # Intel HEX format
            try:
                byte_count = int(line[1:3], 16)
                data = bytes.fromhex(line[9:9+byte_count*2])
                binary_data += data
            except:
                continue
    
    if not binary_data:
        raise ValueError("No valid HEX data found")
    
    # Save as temporary binary and parse
    temp_bin = file_path.parent / f"{file_path.stem}_temp.bin"
    try:
        with open(temp_bin, 'wb') as f:
            f.write(binary_data)
        features = parse_bin_firmware(temp_bin)
    finally:
        if temp_bin.exists():
            temp_bin.unlink()
    
    return features

def parse_elf_firmware(file_path: Path) -> Dict:
    """Parse ELF format firmware"""
    # For ELF, we can extract more detailed section information
    try:
        with open(file_path, 'rb') as f:
            data = f.read()
    except Exception as e:
        raise ValueError(f"Failed to read ELF file: {str(e)}")
    
    # Check ELF magic
    if not data.startswith(b'\x7fELF'):
        raise ValueError("Not a valid ELF file")
    
    # Parse ELF header (simplified)
    # Use binary parsing as fallback
    features = parse_bin_firmware(file_path)
    
    # ELF-specific: mark as signed if has signature section
    if b'.gnu.signature' in data or b'.sig' in data:
        features['is_signed'] = 1
        features['signature_type'] = 'ELF_signed'
    
    return features

def parse_firmware_file(file_path: Path) -> pd.DataFrame:
    """
    Parse firmware file (any format) and convert to CSV format
    
    Supports:
    - .bin (binary firmware)
    - .hex (Intel HEX)
    - .elf (ELF executable)
    - .csv (already in CSV format, just validate)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extension = file_path.suffix.lower()
    
    # If already CSV, validate and return
    if extension == '.csv':
        try:
            df = pd.read_csv(file_path)
            
            # Check if this is a MITRE dataset format
            try:
                from mitre_dataset_converter import detect_mitre_dataset, convert_mitre_dataset_to_system_format
                if detect_mitre_dataset(df):
                    print("✓ Detected MITRE dataset format, converting...")
                    df = convert_mitre_dataset_to_system_format(df)
                    print(f"✓ Converted MITRE dataset: {len(df)} rows")
            except ImportError:
                pass  # Converter not available
            except Exception as e:
                print(f"⚠ MITRE conversion warning: {e}")
            
            # Validate required columns
            required = ['entropy_score', 'is_signed', 'boot_time_ms', 'emulated_syscalls']
            missing = [col for col in required if col not in df.columns]
            if missing:
                raise ValueError(
                    f"CSV missing required columns: {missing}. "
                    f"Available columns: {list(df.columns)}. "
                    f"If this is a MITRE dataset, ensure mitre_dataset_converter.py is available."
                )
            return df
        except Exception as e:
            raise ValueError(f"Invalid CSV file: {str(e)}")
    
    # Parse binary formats
    features = None
    
    if extension == '.bin':
        features = parse_bin_firmware(file_path)
    elif extension == '.hex':
        features = parse_hex_firmware(file_path)
    elif extension == '.elf':
        features = parse_elf_firmware(file_path)
    else:
        # Try as binary
        try:
            features = parse_bin_firmware(file_path)
        except Exception as e:
            raise ValueError(f"Unsupported file format: {extension}. Supported: .bin, .hex, .elf, .csv")
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Add default values for missing columns
    defaults = {
        'drone_vendor': 'unknown',
        'drone_model': 'unknown',
        'firmware_version': '1.0.0',
        'file_format': extension[1:] if extension else 'bin',
        'cpu_architecture': 'unknown',
        'os_type': 'embedded',
        'bootloader_present': 1,
        'filesystem_detected': 0,
        'encryption_used': 0,
        'compression_used': 0,
    }
    
    for key, value in defaults.items():
        if key not in df.columns:
            df[key] = value
    
    # Ensure boot_time_ms and emulated_syscalls have values
    if 'boot_time_ms' not in df.columns or df['boot_time_ms'].isna().any():
        df['boot_time_ms'] = df.get('boot_time_ms', 1000)  # Default 1 second
    if 'emulated_syscalls' not in df.columns or df['emulated_syscalls'].isna().any():
        df['emulated_syscalls'] = df.get('emulated_syscalls', 0)
    
    return df

def convert_firmware_to_csv(input_file: Path, output_file: Path) -> Path:
    """
    Convert firmware file to CSV format
    
    Args:
        input_file: Path to firmware file (.bin, .hex, .elf, etc.)
        output_file: Path to save CSV file
    
    Returns:
        Path to created CSV file
    """
    df = parse_firmware_file(input_file)
    
    # Save to CSV
    output_file = Path(output_file)
    df.to_csv(output_file, index=False)
    
    return output_file

