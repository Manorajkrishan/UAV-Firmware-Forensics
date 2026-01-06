"""
Test script to verify tampered dataset detection
"""
import pandas as pd
import numpy as np
from pathlib import Path

print("="*70)
print("TAMPERED DATASET DETECTION ANALYSIS")
print("="*70)

# Load datasets
clean_df = pd.read_csv('data/clean_drone_firmware_dataset.csv')
tampered_df = pd.read_csv('data/tampered_drone_firmware_dataset.csv')

print("\n1. DATASET CHARACTERISTICS:")
print("-"*70)
print("CLEAN DATASET:")
print(f"  Rows: {len(clean_df)}")
print(f"  clean_label: {clean_df['clean_label'].value_counts().to_dict()}")
print(f"  entropy_score: {clean_df['entropy_score'].mean():.2f} (range: {clean_df['entropy_score'].min():.2f}-{clean_df['entropy_score'].max():.2f})")
print(f"  is_signed: {clean_df['is_signed'].mean():.2f} ({clean_df['is_signed'].sum()}/{len(clean_df)} signed)")
print(f"  boot_time_ms: {clean_df['boot_time_ms'].mean():.0f}ms")
print(f"  hardcoded_ip_count: {clean_df['hardcoded_ip_count'].mean():.2f}")
print(f"  hardcoded_url_count: {clean_df['hardcoded_url_count'].mean():.2f}")
print(f"  emulated_syscalls: {clean_df['emulated_syscalls'].mean():.0f}")

print("\nTAMPERED DATASET:")
print(f"  Rows: {len(tampered_df)}")
print(f"  clean_label: {tampered_df['clean_label'].value_counts().to_dict()}")
print(f"  entropy_score: {tampered_df['entropy_score'].mean():.2f} (range: {tampered_df['entropy_score'].min():.2f}-{tampered_df['entropy_score'].max():.2f})")
print(f"  is_signed: {tampered_df['is_signed'].mean():.2f} ({tampered_df['is_signed'].sum()}/{len(tampered_df)} signed)")
print(f"  boot_time_ms: {tampered_df['boot_time_ms'].mean():.0f}ms")
print(f"  hardcoded_ip_count: {tampered_df['hardcoded_ip_count'].mean():.2f}")
print(f"  hardcoded_url_count: {tampered_df['hardcoded_url_count'].mean():.2f}")
print(f"  emulated_syscalls: {tampered_df['emulated_syscalls'].mean():.0f}")

print("\n2. DIFFERENCES:")
print("-"*70)
print(f"  Entropy difference: {tampered_df['entropy_score'].mean() - clean_df['entropy_score'].mean():.2f} (tampered is HIGHER)")
print(f"  Signed difference: {tampered_df['is_signed'].mean() - clean_df['is_signed'].mean():.2f} (tampered is LOWER)")
print(f"  Boot time difference: {tampered_df['boot_time_ms'].mean() - clean_df['boot_time_ms'].mean():.0f}ms (tampered is SLOWER)")
print(f"  IP count difference: {tampered_df['hardcoded_ip_count'].mean() - clean_df['hardcoded_ip_count'].mean():.2f} (tampered has MORE)")

print("\n3. TAMPERING INDICATORS IN TAMPERED DATASET:")
print("-"*70)
tampered_indicators = []
if tampered_df['entropy_score'].mean() > 7.5:
    tampered_indicators.append(f"[OK] High entropy ({tampered_df['entropy_score'].mean():.2f} > 7.5) - Code obfuscation")
if tampered_df['is_signed'].mean() < 0.5:
    tampered_indicators.append(f"[OK] Low signature coverage ({tampered_df['is_signed'].mean():.2f} < 0.5) - Unsigned firmware")
if tampered_df['boot_time_ms'].mean() > 5000:
    tampered_indicators.append(f"[OK] Long boot time ({tampered_df['boot_time_ms'].mean():.0f}ms > 5000ms) - Boot anomalies")
if tampered_df['hardcoded_ip_count'].mean() > 2:
    tampered_indicators.append(f"[OK] High IP count ({tampered_df['hardcoded_ip_count'].mean():.2f} > 2) - Potential backdoors")

for indicator in tampered_indicators:
    print(f"  [OK] {indicator}")

print("\n4. EXPECTED MODEL BEHAVIOR:")
print("-"*70)
print("  The tampered dataset SHOULD be detected as tampered because:")
print("  - All rows have clean_label=0 (tampered)")
print("  - High entropy (8.06 > 7.5)")
print("  - No signatures (0.0 < 0.5)")
print("  - Slow boot time (6065ms > 5000ms)")
print("  - High IP count (3.03 > 2)")

print("\n5. POSSIBLE ISSUES:")
print("-"*70)
print("  If models are NOT detecting tampered dataset as tampered:")
print("  1. Models may not be trained on tampered dataset")
print("  2. Models may be overfitting to clean data")
print("  3. Models need retraining with both clean and tampered data")
print("  4. Prediction threshold may be too high")

print("\n6. RECOMMENDATION:")
print("-"*70)
print("  Train models on COMBINED dataset (clean + tampered):")
print("  - Use data/combined_preprocessed_dataset.csv")
print("  - Or combine clean and tampered datasets")
print("  - Ensure models learn to distinguish tampered from clean")

print("\n" + "="*70)

