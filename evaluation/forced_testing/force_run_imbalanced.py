"""
Force-run diagnostic pipeline on imbalanced credit card dataset.
This bypasses the data quality skip and generates learning curves despite warnings.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from diagnoser.data.loader import load_data, detect_task_type
from diagnoser.data.processing import clean_data
from diagnoser.models.builder import build_model
from diagnoser.curves.learning_curve import learning_curve
from diagnoser.diagnosis.rules import diagnosis_rules
from diagnoser.diagnosis.recommendations import recommendation
from diagnoser.reporting.plots import plot_learning_curve
from diagnoser.reporting.report import report_findings
from diagnoser.checks.sanity_checks import run_sanity_checks
from diagnoser.utils.seed import set_seed
from diagnoser.metrics.metrics import classification_metrics
from sklearn.model_selection import train_test_split

# Configuration
DATA_PATH = "data/raw/creditcard.csv"
TARGET_COL = "Class"
DROP_COLS = ["Time"]
MODEL_TYPE = "logistic_regression"
OUTPUT_DIR = "outputs"

print("=" * 70)
print("FORCE-RUN: Diagnostic Pipeline with Imbalanced Dataset")
print("=" * 70)

# Set seed
set_seed(42)

# Load data
print(f"\nLoading data from {DATA_PATH}...")
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(DATA_PATH)
if DROP_COLS:
    df = df.drop(columns=DROP_COLS)

# Remove duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_rows - len(df)} duplicate rows")

# Separate features and target
Y = df[TARGET_COL].values
X = df.drop(columns=[TARGET_COL]).values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

task_type = detect_task_type(Y)
print(f"Task type: {task_type}")
print(f"Dataset shape: X={X.shape}, Y={Y.shape}")

# Show class distribution
unique, counts = np.unique(Y, return_counts=True)
class_dist = dict(zip(unique, counts))
minority_pct = min(counts) / len(Y) * 100
print(f"\nClass distribution: {class_dist}")
print(f"Minority class: {minority_pct:.2f}%")
print("\n[NOTE: Running despite severe imbalance to generate learning curves]")

# Split
print("\nSplitting data (stratified)...")
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

# Build model with class weighting
print(f"\nBuilding {MODEL_TYPE} with balanced class weighting...")
model_config = {
    'C': 1.0,
    'max_iter': 5000,
    'solver': 'lbfgs',
    'class_weight': 'balanced'  # Key parameter for imbalanced data
}
model = build_model(MODEL_TYPE, model_config, input_dim=X_train.shape[1])
print(f"Model: {model}")

# Generate learning curve
print("\nGenerating learning curve...")
sizes = [0.1, 0.3, 0.5, 0.7, 1.0]
config_for_curve = {
    'learning_curve': {'sizes': sizes},
    'model_type': MODEL_TYPE,
    'model': model_config,
    'random_seed': 42,
    'task_type': task_type
}
sizes_result, train_errs, val_errs, final_outputs = learning_curve(config_for_curve, X_train, Y_train, X_val, Y_val)

print("\nLearning Curve Results:")
for size, train_err, val_err in zip(sizes, train_errs, val_errs):
    print(f"  Size: {size:.2f}, Train Error: {train_err:.4f}, Val Error: {val_err:.4f}")

# Diagnosis
print("\nRunning diagnosis...")
diagnosis_config = {
    'diagnosis': {
        'gap_threshold': 0.1,
        'high_error_threshold': 0.3
    }
}
diagnosis = diagnosis_rules(train_errs[-1], val_errs[-1], diagnosis_config)
print(f"Diagnosis: {diagnosis}")
print(f"\nRecommendation:\n{recommendation(diagnosis['label'])}")

# Final metrics
print("\nComputing final metrics...")
model.fit(X_train, Y_train)
Y_pred = model.predict(X_val)
if hasattr(model, 'predict_proba'):
    Y_proba = model.predict_proba(X_val)[:, 1]
else:
    Y_proba = None

final_metrics = classification_metrics(Y_val, Y_pred, Y_proba)
print("Final Metrics:")
for metric, value in final_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Sanity checks
print("\nRunning sanity checks...")
config = {
    'model_type': MODEL_TYPE,
    'model': model_config,
    'random_seed': 42,
    'task_type': task_type
}
warnings = run_sanity_checks(config, X_train, Y_train, X_val, Y_val)
if warnings:
    print("Sanity Check Warnings:")
    for w in warnings:
        print(f"  {w}")
else:
    print("All sanity checks passed ✓")

# Plot learning curve
plot_path = os.path.join(OUTPUT_DIR, "figures", "learning_curve_forced.png")
os.makedirs(os.path.dirname(plot_path), exist_ok=True)
plot_learning_curve(sizes, train_errs, val_errs, out_path=plot_path)
print(f"\nLearning curve plot saved to: {plot_path}")

# Save report
report_path = os.path.join(OUTPUT_DIR, "reports", "report_forced.json")
os.makedirs(os.path.dirname(report_path), exist_ok=True)
report_findings(
    diagnosis, train_errs, val_errs, sizes,
    recommendation(diagnosis['label']),
    out_path=report_path,
    warnings=warnings,
    data_errors=[],  # No data errors in force mode
    final_metrics=final_metrics
)
print(f"Report saved to: {report_path}")

print("\n" + "=" * 70)
print("FORCE-RUN COMPLETE")
print("=" * 70)
print("\nNote: Learning curves generated despite severe class imbalance.")
print("Interpret results cautiously - accuracy-based diagnosis may be misleading.")
print("Consider using F1, Precision/Recall, or PR-AUC for evaluation.")
