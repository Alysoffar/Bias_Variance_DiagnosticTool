"""
Example: Basic Usage of the Diagnoser Library

This script demonstrates how an external project would use the diagnoser library
to run a full bias-variance diagnosis on their own dataset.

Install the library first:
    pip install -e .
    # or after proper packaging:
    pip install bias-variance-diagnoser
"""

import sys
import os

# Add parent directory to path to import diagnoser (for development)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from diagnoser import run_diagnosis

# Example 1: Simple usage with default settings
def example_basic():
    """Run diagnosis with minimal configuration."""
    print("=" * 70)
    print("Example 1: Basic Usage")
    print("=" * 70)
    
    result = run_diagnosis(
        dataset_path="../data/raw/creditcard.csv",
        model_name="tree_classifier",
        label_column="Class",
        output_dir="../example_outputs/basic"
    )
    
    print(f"\nDiagnosis Label: {result.diagnosis_label}")
    print(f"Final Train Error: {result.final_train_error:.4f}")
    print(f"Final Val Error: {result.final_val_error:.4f}")
    print(f"Error Gap: {result.error_gap:.4f}")
    print(f"\nRecommendations:\n{result.recommendations}")
    
    if result.data_quality_warnings:
        print("\nData Quality Warnings:")
        for warning in result.data_quality_warnings:
            print(f"  - {warning}")
    
    if result.plot_path:
        print(f"\nPlot saved to: {result.plot_path}")
    if result.report_path:
        print(f"Report saved to: {result.report_path}")


# Example 2: Using custom configuration
def example_custom_config():
    """Run diagnosis with custom configuration."""
    print("\n" + "=" * 70)
    print("Example 2: Custom Configuration")
    print("=" * 70)
    
    result = run_diagnosis(
        dataset_path="../data/raw/creditcard.csv",
        model_name="ridge_regression",
        label_column="Class",
        config_path="../configs/ridge.yaml",
        output_dir="../example_outputs/custom_config",
        save_model=True
    )
    
    print(f"\nDiagnosis: {result.diagnosis_label}")
    if result.model_path:
        print(f"Saved model to: {result.model_path}")


# Example 3: Using a pre-trained model
def example_pretrained_model():
    """Run diagnosis on a pre-trained model."""
    print("\n" + "=" * 70)
    print("Example 3: Pre-trained Model")
    print("=" * 70)
    
    # Assuming you have a pre-trained model
    result = run_diagnosis(
        dataset_path="../data/raw/creditcard.csv",
        model="../outputs/models/final_model.pkl",  # Path to .pkl file
        label_column="Class",
        output_dir="../example_outputs/pretrained"
    )
    
    print(f"\nDiagnosis: {result.diagnosis_label}")
    print("(Using pre-trained model - learning curve shows performance on different data sizes)")


# Example 4: Working with regression datasets
def example_regression():
    """Run diagnosis on a regression task."""
    print("\n" + "=" * 70)
    print("Example 4: Regression Task")
    print("=" * 70)
    
    result = run_diagnosis(
        dataset_path="../data/raw/your_regression_data.csv",
        model_name="ridge_regression",
        label_column="price",  # Your target column name
        output_dir="../example_outputs/regression"
    )
    
    print(f"\nDiagnosis: {result.diagnosis_label}")
    print(f"Final RMSE Train: {result.final_train_error:.4f}")
    print(f"Final RMSE Val: {result.final_val_error:.4f}")


# Example 5: Accessing detailed results as dictionary
def example_result_as_dict():
    """Access results as a dictionary for further processing."""
    print("\n" + "=" * 70)
    print("Example 5: Results as Dictionary")
    print("=" * 70)
    
    result = run_diagnosis(
        dataset_path="../data/raw/creditcard.csv",
        model_name="logistic_regression",
        label_column="Class",
        output_dir="../example_outputs/dict_output"
    )
    
    # Convert to dictionary
    result_dict = result.to_dict()
    
    import json
    print("\nResult as JSON (partial):")
    print(json.dumps({
        "diagnosis_label": result_dict["diagnosis_label"],
        "final_train_error": result_dict["final_train_error"],
        "final_val_error": result_dict["final_val_error"],
        "recommendations": result_dict["recommendations"]
    }, indent=2))


# Example 6: Comparison of multiple models
def example_model_comparison():
    """Compare diagnosis results across different models."""
    print("\n" + "=" * 70)
    print("Example 6: Model Comparison")
    print("=" * 70)
    
    models = [
        ("tree_classifier", "Decision Tree"),
        ("ridge_regression", "Ridge Regression"),
        ("logistic_regression", "Logistic Regression"),
    ]
    
    results = {}
    
    for model_name, display_name in models:
        result = run_diagnosis(
            dataset_path="../data/raw/creditcard.csv",
            model_name=model_name,
            label_column="Class",
            output_dir=f"../example_outputs/comparison/{model_name}"
        )
        results[display_name] = result
        print(f"{display_name}: {result.diagnosis_label} (gap: {result.error_gap:.4f})")
    
    # Find best model by gap
    best_model = min(results.items(), key=lambda x: x[1].error_gap)
    print(f"\nBest model (lowest error gap): {best_model[0]}")


if __name__ == "__main__":
    # Run examples (uncomment the ones you want to try)
    
    try:
        example_basic()
    except FileNotFoundError as e:
        print(f"⚠️  Skipping example_basic: {e}")
    except Exception as e:
        print(f"❌ Error in example_basic: {e}")
    
    # Uncomment other examples as needed
    # example_custom_config()
    # example_pretrained_model()
    # example_regression()
    # example_result_as_dict()
    # example_model_comparison()
