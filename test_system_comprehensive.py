
"""
Comprehensive System Test Suite
Tests the firmware tampering detection system with different datasets and use cases
"""
import requests
import json
import time
import pandas as pd
import uuid
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
BASE_URL = "http://localhost:8000"
TEST_RESULTS = []

def print_test_header(test_name: str):
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")

def print_result(success: bool, message: str):
    status = "[PASS]" if success else "[FAIL]"
    print(f"{status}: {message}")
    TEST_RESULTS.append({"test": message, "status": status, "success": success})
    return success

def test_backend_health():
    """Test 1: Backend is running"""
    print_test_header("Backend Health Check")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        success = response.status_code == 200
        return print_result(success, f"Backend is running (Status: {response.status_code})")
    except Exception as e:
        return print_result(False, f"Backend is not running: {str(e)}")

def test_upload_dataset(file_path: str, description: str) -> Tuple[bool, str]:
    """Test 2: Upload a dataset"""
    print_test_header(f"Upload Dataset: {description}")
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'text/csv')}
            response = requests.post(
                f"{BASE_URL}/api/upload",
                files=files,
                timeout=30
            )
        
        if response.status_code == 200:
            data = response.json()
            firmware_id = data.get('firmware_id')
            success = firmware_id is not None
            return print_result(success, f"Uploaded {description} (ID: {firmware_id})"), firmware_id
        else:
            return print_result(False, f"Upload failed: {response.status_code} - {response.text}"), None
    except Exception as e:
        return print_result(False, f"Upload error: {str(e)}"), None

def test_analyze_firmware(firmware_id: str, model_preference: str = "ensemble") -> bool:
    """Test 3: Analyze firmware"""
    print_test_header(f"Analyze Firmware: {firmware_id}")
    try:
        response = requests.post(
            f"{BASE_URL}/api/analyze",
            json={
                "firmware_id": firmware_id,
                "model_preference": model_preference
            },
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            is_tampered = data.get('is_tampered')
            probability = data.get('tampering_probability', 0)
            injection_analysis = data.get('injection_analysis', {})
            
            success = is_tampered is not None
            message = (
                f"Analysis completed - "
                f"Tampered: {is_tampered}, "
                f"Probability: {probability:.2%}, "
                f"Injection: {injection_analysis.get('injection_status', 'N/A')}"
            )
            return print_result(success, message)
        else:
            return print_result(False, f"Analysis failed: {response.status_code} - {response.text}")
    except Exception as e:
        return print_result(False, f"Analysis error: {str(e)}")

def test_get_analysis(firmware_id: str) -> bool:
    """Test 4: Get analysis results"""
    print_test_header(f"Get Analysis Results: {firmware_id}")
    try:
        response = requests.get(f"{BASE_URL}/api/analyses/{firmware_id}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            has_injection = 'injection_analysis' in data
            has_mitre = 'mitre_classification' in data
            has_severity = 'severity_level' in data
            
            success = response.status_code == 200
            message = (
                f"Retrieved analysis - "
                f"Injection: {has_injection}, "
                f"MITRE: {has_mitre}, "
                f"Severity: {has_severity}"
            )
            return print_result(success, message)
        else:
            return print_result(False, f"Get analysis failed: {response.status_code}")
    except Exception as e:
        return print_result(False, f"Get analysis error: {str(e)}")

def test_dashboard_stats() -> bool:
    """Test 5: Dashboard statistics"""
    print_test_header("Dashboard Statistics")
    try:
        response = requests.get(f"{BASE_URL}/api/dashboard/stats", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            total = data.get('total_analyses', 0)
            tampered = data.get('tampered_count', 0)
            clean = data.get('clean_count', 0)
            
            success = response.status_code == 200
            message = f"Dashboard stats - Total: {total}, Tampered: {tampered}, Clean: {clean}"
            return print_result(success, message)
        else:
            return print_result(False, f"Dashboard failed: {response.status_code}")
    except Exception as e:
        return print_result(False, f"Dashboard error: {str(e)}")

def test_upload_and_train(file_path: str, description: str) -> Tuple[bool, str]:
    """Test 6: Upload and train"""
    print_test_header(f"Upload and Train: {description}")
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (Path(file_path).name, f, 'text/csv')}
            response = requests.post(
                f"{BASE_URL}/api/upload-and-train?train=true",
                files=files,
                timeout=120  # Training takes longer
            )
        
        if response.status_code == 200:
            data = response.json()
            firmware_id = data.get('firmware_id')
            training = data.get('training', {})
            analysis = data.get('analysis')
            
            success = firmware_id is not None
            message = (
                f"Upload & Train completed - "
                f"ID: {firmware_id}, "
                f"Training: {bool(training)}, "
                f"Analysis: {bool(analysis)}"
            )
            return print_result(success, message), firmware_id
        else:
            return print_result(False, f"Upload & Train failed: {response.status_code} - {response.text}"), None
    except Exception as e:
        return print_result(False, f"Upload & Train error: {str(e)}"), None

def create_test_dataset(name: str, rows: int = 10, tampered_ratio: float = 0.5, 
                       clean_characteristics: bool = True, dataset_type: str = "standard") -> str:
    """Create a synthetic test dataset with configurable characteristics
    
    Args:
        name: Dataset name
        rows: Number of rows
        tampered_ratio: Ratio of tampered samples (0.0-1.0)
        clean_characteristics: Use clean firmware characteristics
        dataset_type: Type of dataset - "standard", "extreme", "minimal", "full_features"
    """
    import numpy as np
    
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Set random seed for reproducibility (vary by name for diversity)
    seed = hash(name) % 10000
    np.random.seed(seed)
    
    # Generate synthetic data with characteristics based on tampered ratio and type
    if dataset_type == "extreme":
        # Extreme values - edge cases
        entropy_range = (0.5, 8.5) if tampered_ratio > 0.5 else (2.0, 7.5)
        boot_time_range = (100, 15000)
        syscalls_range = (10, 3000)
        ip_count_range = (0, 20)
        url_count_range = (0, 15)
    elif dataset_type == "minimal":
        # Minimal features - only required columns
        entropy_range = (3.0, 7.0)
        boot_time_range = (500, 3000)
        syscalls_range = (50, 1000)
        ip_count_range = (0, 3)
        url_count_range = (0, 2)
    elif clean_characteristics and tampered_ratio < 0.3:
        # Clean firmware characteristics
        entropy_range = (3.0, 6.0)
        boot_time_range = (500, 2000)
        syscalls_range = (50, 500)
        ip_count_range = (0, 1)
        url_count_range = (0, 1)
    elif not clean_characteristics and tampered_ratio > 0.7:
        # Tampered firmware characteristics
        entropy_range = (6.5, 8.0)
        boot_time_range = (3000, 8000)
        syscalls_range = (800, 2000)
        ip_count_range = (2, 10)
        url_count_range = (1, 5)
    else:
        # Mixed characteristics
        entropy_range = (3.0, 8.0)
        boot_time_range = (500, 5000)
        syscalls_range = (50, 2000)
        ip_count_range = (0, 5)
        url_count_range = (0, 3)
    
    # Base required columns
    data = {
        'entropy_score': np.random.uniform(entropy_range[0], entropy_range[1], rows),
        'is_signed': np.random.choice([0, 1], rows, p=[tampered_ratio, 1-tampered_ratio]),
        'boot_time_ms': np.random.uniform(boot_time_range[0], boot_time_range[1], rows),
        'emulated_syscalls': np.random.randint(syscalls_range[0], syscalls_range[1], rows),
    }
    
    # Add optional columns based on dataset type
    if dataset_type != "minimal":
        data.update({
            'hardcoded_ip_count': np.random.randint(ip_count_range[0], ip_count_range[1], rows),
            'hardcoded_url_count': np.random.randint(url_count_range[0], url_count_range[1], rows),
            'crypto_function_count': np.random.randint(5, 50, rows),
        })
    
    if dataset_type == "full_features":
        data.update({
            'num_executables': np.random.randint(1, 20, rows),
            'num_scripts': np.random.randint(0, 10, rows),
            'file_size_bytes': np.random.randint(100000, 50000000, rows),
            'string_count': np.random.randint(100, 50000, rows),
            'avg_section_entropy': np.random.uniform(3.0, 8.0, rows),
            'max_section_entropy': np.random.uniform(4.0, 8.5, rows),
        })
    elif dataset_type != "minimal":
        data.update({
            'num_executables': np.random.randint(1, 10, rows),
            'num_scripts': np.random.randint(0, 5, rows),
            'file_size_bytes': np.random.randint(100000, 10000000, rows),
            'string_count': np.random.randint(100, 10000, rows),
        })
    
    # Add target column for training
    data['clean_label'] = np.random.choice([0, 1], rows, p=[tampered_ratio, 1-tampered_ratio])
    
    df = pd.DataFrame(data)
    file_path = test_dir / f"{name}.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)

def run_comprehensive_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("COMPREHENSIVE SYSTEM TEST SUITE")
    print("="*70)
    print(f"Testing against: {BASE_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test 1: Backend Health
    if not test_backend_health():
        print("\n[ERROR] Backend is not running. Please start the backend first.")
        print("   Run: cd backend && python main.py")
        return
    
    # Test 2: Test with multiple different datasets
    test_datasets = []
    dataset_configs = []
    
    # 2a: MITRE Dataset (if available) - check multiple possible locations
    mitre_paths = [
        r"d:\drone_firmware_full_mitre_dataset.csv",
        "data/drone_firmware_full_mitre_dataset.csv",
        "drone_firmware_full_mitre_dataset.csv"
    ]
    for mitre_path in mitre_paths:
        if Path(mitre_path).exists():
            dataset_configs.append((mitre_path, "MITRE Dataset (Full)"))
            break
    
    # 2b: All datasets from data directory
    data_dir = Path("data")
    if data_dir.exists():
        for csv_file in data_dir.glob("*.csv"):
            if csv_file.name not in ["X_train.csv", "X_test.csv", "X_final.csv", 
                                    "y_train.csv", "y_test.csv", "y_final.csv"]:
                dataset_name = csv_file.stem.replace("_", " ").title()
                dataset_configs.append((str(csv_file), f"Data Directory: {dataset_name}"))
    
    # 2c: All datasets from test_data directory (existing ones)
    test_data_dir = Path("test_data")
    if test_data_dir.exists():
        for csv_file in test_data_dir.glob("*.csv"):
            if not csv_file.name.startswith("train_"):  # Skip training temp files
                dataset_name = csv_file.stem.replace("_", " ").title()
                dataset_configs.append((str(csv_file), f"Test Data: {dataset_name}"))
    
    # 2d: Create comprehensive synthetic datasets with different characteristics
    print_test_header("Creating Comprehensive Synthetic Test Datasets")
    
    # Synthetic - Clean (mostly clean)
    clean_synthetic = create_test_dataset("synthetic_clean_v2", rows=50, tampered_ratio=0.1, 
                                         clean_characteristics=True, dataset_type="standard")
    dataset_configs.append((clean_synthetic, "Synthetic Clean Dataset (10% tampered, 50 rows)"))
    
    # Synthetic - Tampered (mostly tampered)
    tampered_synthetic = create_test_dataset("synthetic_tampered_v2", rows=50, tampered_ratio=0.9, 
                                            clean_characteristics=False, dataset_type="standard")
    dataset_configs.append((tampered_synthetic, "Synthetic Tampered Dataset (90% tampered, 50 rows)"))
    
    # Synthetic - Balanced
    balanced_synthetic = create_test_dataset("synthetic_balanced_v2", rows=100, tampered_ratio=0.5, 
                                             clean_characteristics=False, dataset_type="standard")
    dataset_configs.append((balanced_synthetic, "Synthetic Balanced Dataset (50% tampered, 100 rows)"))
    
    # Synthetic - Imbalanced (very few tampered)
    imbalanced_synthetic = create_test_dataset("synthetic_imbalanced_v2", rows=150, tampered_ratio=0.05, 
                                              clean_characteristics=True, dataset_type="standard")
    dataset_configs.append((imbalanced_synthetic, "Synthetic Imbalanced Dataset (5% tampered, 150 rows)"))
    
    # Synthetic - Small dataset (edge case)
    small_synthetic = create_test_dataset("synthetic_small_v2", rows=5, tampered_ratio=0.4, 
                                         clean_characteristics=False, dataset_type="minimal")
    dataset_configs.append((small_synthetic, "Synthetic Small Dataset (5 rows, minimal features)"))
    
    # Synthetic - Large dataset
    large_synthetic = create_test_dataset("synthetic_large_v2", rows=500, tampered_ratio=0.3, 
                                         clean_characteristics=False, dataset_type="full_features")
    dataset_configs.append((large_synthetic, "Synthetic Large Dataset (500 rows, full features)"))
    
    # Synthetic - Extreme values (edge cases)
    extreme_synthetic = create_test_dataset("synthetic_extreme_v2", rows=30, tampered_ratio=0.6, 
                                           clean_characteristics=False, dataset_type="extreme")
    dataset_configs.append((extreme_synthetic, "Synthetic Extreme Values Dataset (edge cases)"))
    
    # Synthetic - Minimal features only
    minimal_synthetic = create_test_dataset("synthetic_minimal_v2", rows=25, tampered_ratio=0.5, 
                                           clean_characteristics=False, dataset_type="minimal")
    dataset_configs.append((minimal_synthetic, "Synthetic Minimal Features Dataset (required columns only)"))
    
    # Synthetic - Full features
    full_features_synthetic = create_test_dataset("synthetic_full_features_v2", rows=75, tampered_ratio=0.4, 
                                                  clean_characteristics=False, dataset_type="full_features")
    dataset_configs.append((full_features_synthetic, "Synthetic Full Features Dataset (all columns)"))
    
    # Synthetic - Very high tampered ratio
    high_tampered_synthetic = create_test_dataset("synthetic_high_tampered_v2", rows=40, tampered_ratio=0.95, 
                                                  clean_characteristics=False, dataset_type="standard")
    dataset_configs.append((high_tampered_synthetic, "Synthetic High Tampered Dataset (95% tampered)"))
    
    # Synthetic - Very low tampered ratio
    low_tampered_synthetic = create_test_dataset("synthetic_low_tampered_v2", rows=40, tampered_ratio=0.02, 
                                                 clean_characteristics=True, dataset_type="standard")
    dataset_configs.append((low_tampered_synthetic, "Synthetic Low Tampered Dataset (2% tampered)"))
    
    # Synthetic - All clean
    all_clean_synthetic = create_test_dataset("synthetic_all_clean_v2", rows=30, tampered_ratio=0.0, 
                                             clean_characteristics=True, dataset_type="standard")
    dataset_configs.append((all_clean_synthetic, "Synthetic All Clean Dataset (0% tampered)"))
    
    # Synthetic - All tampered
    all_tampered_synthetic = create_test_dataset("synthetic_all_tampered_v2", rows=30, tampered_ratio=1.0, 
                                                 clean_characteristics=False, dataset_type="standard")
    dataset_configs.append((all_tampered_synthetic, "Synthetic All Tampered Dataset (100% tampered)"))
    
    # Test all datasets
    print_test_header(f"Testing {len(dataset_configs)} Different Datasets")
    successful_uploads = []
    
    for i, (dataset_path, description) in enumerate(dataset_configs, 1):
        print(f"\n[{i}/{len(dataset_configs)}] Testing: {description}")
        try:
            # Check if file exists
            if not Path(dataset_path).exists():
                print(f"  [SKIP] File not found: {dataset_path}")
                continue
            
            success, firmware_id = test_upload_dataset(dataset_path, description)
            if success and firmware_id:
                successful_uploads.append((description, firmware_id, dataset_path))
                test_datasets.append((description, firmware_id))
                
                # Analyze with ensemble model (default)
                test_analyze_firmware(firmware_id, model_preference="ensemble")
                
                # Get analysis results
                test_get_analysis(firmware_id)
                time.sleep(1)  # Rate limiting
            else:
                print(f"  [SKIP] Upload failed for {description}")
        except Exception as e:
            print(f"  [ERROR] Failed to test {description}: {str(e)}")
            print_result(False, f"Dataset test failed: {description} - {str(e)}")
    
    print(f"\n[INFO] Successfully tested {len(test_datasets)} out of {len(dataset_configs)} datasets")
    
    # Test 3: Upload and Train (test with datasets that have target column)
    print_test_header("Testing Upload and Train Functionality")
    training_datasets = [
        (balanced_synthetic, "Synthetic Balanced Dataset (100 rows)"),
        (clean_synthetic, "Synthetic Clean Dataset (50 rows)"),
        (large_synthetic, "Synthetic Large Dataset (500 rows)"),
        (full_features_synthetic, "Synthetic Full Features Dataset (75 rows)"),
    ]
    
    training_count = 0
    for dataset_path, description in training_datasets:
        if Path(dataset_path).exists():
            # Create unique filename to avoid duplicate upload error
            unique_name = f"train_{uuid.uuid4().hex[:8]}.csv"
            test_dir = Path("test_data")
            unique_path = test_dir / unique_name
            
            # Copy dataset with unique name
            try:
                shutil.copy(dataset_path, unique_path)
                
                success, firmware_id = test_upload_and_train(str(unique_path), f"{description} (Training)")
                if success and firmware_id:
                    training_count += 1
                    test_get_analysis(firmware_id)
                    time.sleep(2)  # Rate limiting for training
            except Exception as e:
                print(f"  [ERROR] Training test failed for {description}: {str(e)}")
                print_result(False, f"Training test failed: {description}")
            finally:
                # Clean up temp file
                if unique_path.exists():
                    try:
                        unique_path.unlink()
                    except:
                        pass
    
    print(f"\n[INFO] Successfully tested training with {training_count} datasets")
    
    # Test 4: Dashboard
    test_dashboard_stats()
    
    # Test 5: Test with different models on multiple datasets
    if test_datasets:
        print_test_header("Testing Different ML Models on Multiple Datasets")
        models_to_test = ["random_forest", "isolation_forest", "lstm", "autoencoder", "ensemble"]
        
        # Test each model on diverse datasets (mix of different types)
        datasets_to_test = test_datasets[:min(10, len(test_datasets))]  # Test on up to 10 diverse datasets
        
        for model in models_to_test:
            print(f"\n  Testing model: {model}")
            model_success_count = 0
            for dataset_name, firmware_id in datasets_to_test:
                try:
                    success = test_analyze_firmware(firmware_id, model_preference=model)
                    if success:
                        model_success_count += 1
                    time.sleep(0.5)  # Rate limiting
                except Exception as e:
                    print_result(False, f"Model {model} on {dataset_name} failed: {str(e)}")
            print(f"    Model {model}: {model_success_count}/{len(datasets_to_test)} successful")
    
    # Print Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total = len(TEST_RESULTS)
    passed = sum(1 for r in TEST_RESULTS if r['success'])
    failed = total - passed
    
    print(f"Total Tests: {total}")
    print(f"[PASS] Passed: {passed}")
    print(f"[FAIL] Failed: {failed}")
    print(f"Success Rate: {(passed/total*100):.1f}%")
    
    print("\nDetailed Results:")
    for result in TEST_RESULTS:
        print(f"  {result['status']}: {result['test']}")
    
    # Save results
    results_file = Path("test_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total": total,
            "passed": passed,
            "failed": failed,
            "results": TEST_RESULTS
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    try:
        run_comprehensive_tests()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Tests interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] Test suite error: {str(e)}")
        import traceback
        traceback.print_exc()

