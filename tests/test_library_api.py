"""
Quick test of the library on the creditcard dataset
"""
import sys
import os

# Add parent directory to path to import diagnoser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from diagnoser import run_diagnosis

# Test 1: Basic usage
print("=" * 70)
print("TEST 1: Basic run_diagnosis with tree classifier")
print("=" * 70)

try:
    result = run_diagnosis(
        dataset_path="../data/raw/creditcard.csv",
        model_name="tree_classifier",
        label_column="Class",
        output_dir="../test_outputs/basic",
        save_plot=True,
        save_report=True
    )
    
    print(f"[SUCCESS] Diagnosis completed successfully!")
    print(f"  Diagnosis Label: {result.diagnosis_label}")
    print(f"  Final Train Error: {result.final_train_error:.4f}")
    print(f"  Final Val Error: {result.final_val_error:.4f}")
    print(f"  Error Gap: {result.error_gap:.4f}")
    
    if result.plot_path:
        print(f"  Plot saved to: {result.plot_path}")
    if result.report_path:
        print(f"  Report saved to: {result.report_path}")
    
    # Test to_dict
    result_dict = result.to_dict()
    print(f"[SUCCESS] Result converted to dict with keys: {list(result_dict.keys())}")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST 2: Logistic regression with Windows path")
print("=" * 70)

try:
    result = run_diagnosis(
        dataset_path=r"..\data\raw\creditcard.csv",  # Windows path
        model_name="logistic_regression",  # Use logistic for classification
        label_column="Class",
        output_dir=r"..\test_outputs\windows_path",
        save_plot=False,
        save_report=True
    )
    
    print(f"[SUCCESS] Windows path test passed!")
    print(f"  Diagnosis: {result.diagnosis_label}")
    
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("All tests completed!")
print("=" * 70)
