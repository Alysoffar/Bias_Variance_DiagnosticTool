# src/diagnoser/pipeline.py

import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_diabetes

try:
    from .curves.learning_curve import learning_curve
    from .data.loader import load_data, detect_task_type
    from .data.processing import clean_data, save_cleaned_data
    from .metrics.metrics import classification_error, regression_error, classification_metrics
    from .diagnosis.rules import diagnosis_rules
    from .diagnosis.recommendations import recommendation
    from .diagnosis.data_quality import check_data_quality
    from .reporting.plots import plot_learning_curve
    from .reporting.report import report_findings
    from .checks.sanity_checks import run_sanity_checks
except ImportError:
    from curves.learning_curve import learning_curve
    from data.loader import load_data, detect_task_type
    from data.processing import clean_data, save_cleaned_data
    from metrics.metrics import classification_error, regression_error, classification_metrics
    from diagnosis.rules import diagnosis_rules
    from diagnosis.recommendations import recommendation
    from diagnosis.data_quality import check_data_quality
    from reporting.plots import plot_learning_curve
    from reporting.report import report_findings
    from checks.sanity_checks import run_sanity_checks


def load_config(default_path: str, model_path: str) -> dict:
    with open(default_path, "r") as f:
        base = yaml.safe_load(f)
    with open(model_path, "r") as f:
        override = yaml.safe_load(f)

    # shallow merge is enough for your current structure
    base.update(override)
    return base


def load_dataset(config):
    """
    Load dataset from file or use built-in datasets.
    Auto-detects task type if not specified.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        tuple: (X, y, task_type)
    """
    data_path = config.get("data_path")
    
    # Use custom dataset if path is provided
    if data_path is not None:
        target_col = config.get("target_column", "target")
        print(f"Loading data from: {data_path}")
        X, y = load_data(data_path, target_col)
        
        # Auto-detect task type if not specified
        task_type = config.get("task_type")
        if task_type is None:
            task_type = detect_task_type(y)
            print(f"Auto-detected task type: {task_type}")
        
        # Clean data if enabled
        X, y = clean_data(X, y, config)
        
        # Save cleaned data if enabled
        if config.get("save_cleaned_data", False):
            output_path = config.get("cleaned_data_path", "data/processed/cleaned_data.csv")
            save_cleaned_data(X, y, output_path, target_col)
        
        # Convert to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        return X, y, task_type
    
    # Fall back to built-in datasets
    else:
        # When using built-in datasets, task_type must be specified or default to classification
        task_type = config.get("task_type")
        if task_type is None:
            task_type = "classification"  # Default to classification for built-in datasets
        
        print(f"Using built-in dataset for {task_type}")
        
        if task_type == "classification":
            X, y = load_breast_cancer(return_X_y=True)
        elif task_type == "regression":
            X, y = load_diabetes(return_X_y=True)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")
        
        return X, y, task_type


def run(config):
    # Load and prepare data
    X, Y, task_type = load_dataset(config)
    
    # Update config with detected task type
    config["task_type"] = task_type
    
    print(f"\nDataset shape: X={X.shape}, y={Y.shape}")
    print(f"Task type: {task_type}\n")
    
    # Check data quality and print warnings
    quality_warnings = check_data_quality(X, Y, task_type, config)
    for warning in quality_warnings:
        print(warning)
        print()

    if quality_warnings == []:

        X_train, X_val, Y_train, Y_val = train_test_split(
            X, Y, test_size=0.2, random_state=42)
        
        sizes, train_errs, val_errs, final_outputs = learning_curve(
            config, X_train, Y_train, X_val, Y_val
        )

        final_train_error = train_errs[-1]
        final_val_error = val_errs[-1]

        diagnosis = diagnosis_rules(final_train_error, final_val_error, config)

        final_metrics = None
        if task_type == "classification":
            final_metrics = {
                "train": classification_metrics(
                    final_outputs.get("y_train_true"),
                    final_outputs.get("y_train_pred"),
                    final_outputs.get("y_train_score")
                ),
                "val": classification_metrics(
                    final_outputs.get("y_val_true"),
                    final_outputs.get("y_val_pred"),
                    final_outputs.get("y_val_score")
                )
            }


            

        print("Learning Curve Results:")

        for size, train_err, val_err in zip(sizes, train_errs, val_errs):
            print(f"Training Size: {size:.2f}, Train Error: {train_err:.4f}, Val Error: {val_err:.4f}")

        print("Diagnosis:", diagnosis)

        print("Recommendation:", recommendation(diagnosis["label"]))

        plot_learning_curve(sizes, train_errs, val_errs, out_path=r"D:\\WORK\\projects\\Bias_Variance_DiagnosticTool\\outputs\\figures\\learning_curve.png")

        warnings = run_sanity_checks(config, X_train, Y_train, X_val, Y_val)
        for w in warnings:
            print(w)
        
        report_findings(
            diagnosis, train_errs, val_errs, sizes,
            recommendation(diagnosis["label"]),
            out_path=r"D:\\WORK\\projects\\Bias_Variance_DiagnosticTool\\outputs\\reports\\report.json",
            warnings=warnings,
            data_errors=quality_warnings,
            final_metrics=final_metrics
        )

    else:
        print("Data quality issues detected. Skipping learning curve and diagnosis.")

 


if __name__ == "__main__":
    # choose one model config at a time
    cfg = load_config("configs/default.yaml", "configs/logistic_reg.yaml")
    
    run(cfg)
