# Using Bias-Variance Diagnoser as a Library

This guide explains how to use the **Bias-Variance Diagnoser** as a reusable Python library in your own projects.

## Installation

### From source (development)
```bash
git clone <repo-url>
cd bias-variance-diagnoser
pip install -e .
```

### Standard installation (once published)
```bash
pip install bias-variance-diagnoser
```

### With optional dependencies
```bash
# For neural network support
pip install bias-variance-diagnoser[nn]

# For Jupyter notebook support
pip install bias-variance-diagnoser[notebooks]

# Everything
pip install bias-variance-diagnoser[all]
```

## Quick Start

```python
from diagnoser import run_diagnosis

# Run diagnosis on your dataset
result = run_diagnosis(
    dataset_path="path/to/your/data.csv",
    model_name="ridge_regression",
    label_column="target_column",
    output_dir="results/"
)

# Access results
print(f"Diagnosis: {result.diagnosis_label}")
print(f"Recommendations: {result.recommendations}")
```

## API Reference

### `run_diagnosis()`

The main function to run the complete diagnosis pipeline.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset_path` | str | **required** | Path to CSV file with your data |
| `model` | str | None | Path to pre-trained model (.pkl or .h5) |
| `model_name` | str | None | Name of model to train: `tree_classifier`, `logistic_regression`, `linear_regression`, `ridge_regression`, `lasso_regression`, `nn_classifier` |
| `label_column` | str | "Class" | Name of target column in your dataset |
| `config_path` | str | None | Path to custom YAML config file |
| `output_dir` | str | None | Directory to save plots/reports/models. If None, uses temp directory |
| `save_model` | bool | False | Save the trained model after diagnosis |
| `save_plot` | bool | True | Save learning curve plot |
| `save_report` | bool | True | Save JSON report with findings |

#### Returns

**`DiagnosisResult`** dataclass with:

- **Learning Curve Data**:
  - `sizes` (list): Training set fractions used
  - `train_errors` (list): Training errors at each size
  - `val_errors` (list): Validation errors at each size

- **Final Metrics**:
  - `final_train_error` (float): Error on full training set
  - `final_val_error` (float): Error on validation set
  - `error_gap` (float): Difference (val - train)

- **Diagnosis**:
  - `diagnosis_label` (str): "high_bias", "high_variance", or "balanced"
  - `diagnosis_details` (dict): Complete diagnosis information
  - `recommendations` (str): Actionable recommendations

- **Quality & Warnings**:
  - `data_quality_warnings` (list): Issues detected in the data
  - `sanity_check_warnings` (list): Pipeline sanity check warnings

- **Artifacts**:
  - `plot_path` (str): Path to saved learning curve plot
  - `report_path` (str): Path to saved JSON report
  - `model_path` (str): Path to saved model (if requested)

#### Exceptions

- **`DatasetError`**: Dataset file not found or label column missing
- **`ModelError`**: Unknown model name or can't load pre-trained model
- **`ConfigError`**: Configuration file invalid or not found
- **`ValueError`**: Invalid parameter combination

### `DiagnosisResult`

The structured result returned by `run_diagnosis()`.

#### Methods

- `to_dict()`: Convert result to dictionary for serialization

#### Attributes

All dataclass fields listed above.

## Examples

### Example 1: Basic Classification

```python
from diagnoser import run_diagnosis

result = run_diagnosis(
    dataset_path="data/mydata.csv",
    model_name="tree_classifier",
    label_column="Class",
    output_dir="my_results/"
)

print(f"Diagnosis: {result.diagnosis_label}")
print(f"Train error: {result.final_train_error:.4f}")
print(f"Val error: {result.final_val_error:.4f}")
if result.plot_path:
    print(f"Plot saved to: {result.plot_path}")
```

### Example 2: Regression with Custom Config

```python
result = run_diagnosis(
    dataset_path="data/housing.csv",
    model_name="ridge_regression",
    label_column="price",
    config_path="my_config.yaml",
    output_dir="results/",
    save_model=True
)

if result.model_path:
    print(f"Saved model to: {result.model_path}")
```

### Example 3: Using Pre-trained Model

```python
result = run_diagnosis(
    dataset_path="new_data.csv",
    model="path/to/my_model.pkl",  # Use existing model
    label_column="target",
    output_dir="eval_results/"
)

# Evaluate pre-trained model on new data
print(f"Performance on new data: {result.diagnosis_label}")
```

### Example 4: Model Comparison

```python
models = ["tree_classifier", "ridge_regression", "logistic_regression"]
results = {}

for model_name in models:
    result = run_diagnosis(
        dataset_path="data.csv",
        model_name=model_name,
        label_column="target",
        output_dir=f"results/{model_name}"
    )
    results[model_name] = result

# Compare
for name, result in results.items():
    print(f"{name}: {result.error_gap:.4f} (diagnosis: {result.diagnosis_label})")
```

### Example 5: Processing Results as Dictionary

```python
result = run_diagnosis(
    dataset_path="data.csv",
    model_name="ridge_regression",
    label_column="target"
)

# Convert to dict
data = result.to_dict()

# Save to JSON
import json
with open("diagnosis_result.json", "w") as f:
    json.dump(data, f, indent=2)
```

## Supported Models

| Model Name | Type | Use Case |
|-----------|------|----------|
| `tree_classifier` | Classification | Non-linear boundaries, interpretability |
| `logistic_regression` | Classification | Linear boundaries, probability output |
| `ridge_regression` | Regression | Linear with L2 regularization |
| `lasso_regression` | Regression | Linear with L1 regularization (sparse) |
| `linear_regression` | Regression | Simple linear regression |
| `nn_classifier` | Classification | Deep learning (requires TensorFlow) |

## Configuration

If you provide a `config_path`, the YAML file should contain:

```yaml
# Learning curve settings
learning_curve:
  sizes: [0.1, 0.3, 0.5, 0.7, 1.0]

# Diagnosis thresholds
diagnosis:
  gap_threshold: 0.1
  high_error_threshold: 0.3
  imbalance_threshold: 5.0

# Data processing
scale_features: true
scaler_type: "standard"
encode_categorical: true
handle_missing: true

# Model hyperparameters (model-specific)
model:
  max_depth: 10
  min_samples_split: 5
```

## Dataset Format

Your CSV file should be a standard pandas-compatible CSV with:

- One row per sample
- One column per feature
- A target column (name specified by `label_column` parameter)

Example:
```
feature1,feature2,feature3,Class
1.2,3.4,5.6,0
2.3,4.5,6.7,1
...
```

## Diagnosis Labels

- **`high_bias`**: Model is too simple (underfitting). Recommendations: more features, more complex model, more training.
- **`high_variance`**: Model is overfitting. Recommendations: more data, regularization, simpler model.
- **`balanced`**: Model has good bias-variance tradeoff. Recommendations: model is well-tuned.

## Error Handling

```python
from diagnoser import run_diagnosis, DatasetError, ModelError, ConfigError

try:
    result = run_diagnosis(
        dataset_path="data.csv",
        model_name="ridge_regression",
        label_column="target"
    )
except DatasetError as e:
    print(f"Dataset error: {e}")
except ModelError as e:
    print(f"Model error: {e}")
except ConfigError as e:
    print(f"Config error: {e}")
```

## Working with Windows Paths

The library automatically handles Windows paths:

```python
result = run_diagnosis(
    dataset_path=r"C:\Users\Me\data\mydata.csv",
    model_name="ridge_regression",
    label_column="target",
    output_dir=r"C:\Users\Me\results\"
)
```

## Artifacts

When you specify `output_dir`, the library saves:

```
output_dir/
├── data/raw/
│   └── <copied dataset>
├── figures/
│   └── learning_curve.png
├── reports/
│   └── report.json
└── models/
    └── final_model.pkl (or .h5 for neural networks)
```

The JSON report includes:
- Diagnosis results
- Learning curve data
- Recommendations
- Data quality warnings
- Model metadata

## Advanced: Batch Processing

```python
import os
from diagnoser import run_diagnosis

data_dir = "datasets/"
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        result = run_diagnosis(
            dataset_path=os.path.join(data_dir, filename),
            model_name="ridge_regression",
            label_column="target",
            output_dir=f"results/{filename.split('.')[0]}"
        )
        print(f"{filename}: {result.diagnosis_label}")
```

## Tips

1. **Start with a simple model** (e.g., linear regression) to get a baseline diagnosis
2. **Use cross-validation configs** for more robust estimates
3. **Inspect warnings** - they often reveal data quality issues
4. **Try different models** to understand their bias-variance tradeoff
5. **Save outputs** for reproducibility and documentation

## Troubleshooting

### "Label column not found"
Check the exact column name in your CSV and match the `label_column` parameter.

### "Model file not found"
Provide absolute path or ensure relative path is correct from your working directory.

### "Unknown model name"
Use one of the supported models listed in the Model table above.

### Memory issues with large datasets
Try reducing learning curve sizes in your config:
```yaml
learning_curve:
  sizes: [0.3, 0.6, 1.0]  # Fewer sizes
```

## See Also

- `examples/library_usage.py`: Complete working examples
- `configs/`: YAML config files for different models
- Original repository: [link to repo]
