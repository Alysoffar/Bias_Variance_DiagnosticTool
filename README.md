# Bias-Variance Diagnostic Tool

A comprehensive Python toolkit for diagnosing and analyzing bias-variance tradeoffs in machine learning models through automated learning curve generation, model evaluation, and actionable recommendations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Command Line Interface](#command-line-interface)
  - [Python API](#python-api)
  - [Configuration Files](#configuration-files)
- [Supported Models](#supported-models)
- [Output and Reports](#output-and-reports)
- [Testing](#testing)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Overview

The Bias-Variance Diagnostic Tool helps machine learning practitioners identify whether their models are suffering from high bias (underfitting), high variance (overfitting), or achieving a balanced fit. By automatically generating learning curves and running diagnostic checks, this tool provides:

- **Automated Diagnosis**: Identify bias/variance issues through configurable thresholds
- **Learning Curves**: Generate training and validation error curves across different dataset sizes
- **Actionable Recommendations**: Get specific suggestions based on your model's performance
- **Sanity Checks**: Detect common issues like data leakage, label shuffle sensitivity, and training problems
- **Multi-Model Support**: Works with scikit-learn models and neural networks (TensorFlow/Keras)
- **Reproducibility**: Configurable random seeds and deterministic sampling

## Features

### Core Functionality

- **Learning Curve Generation**: Automatically train models on varying amounts of training data and measure performance
- **Bias-Variance Classification**: Categorize models into high bias, high variance, or balanced based on error patterns
- **Multiple Metrics**: Support for classification error, mean squared error, and custom metrics
- **Visualization**: Generate publication-ready learning curve plots
- **JSON Reports**: Export detailed diagnostic results for further analysis

### Data Processing Pipeline

- **Automatic Data Loading**: Support for CSV, JSON, and Excel formats with auto-format detection
- **Data Cleaning**: Remove duplicates, handle missing values with configurable imputation strategies
- **Categorical Encoding**: One-hot or label encoding for string/categorical columns
- **Feature Scaling**: StandardScaler and MinMaxScaler for feature normalization
- **Target Encoding**: Automatic label encoding for categorical target variables
- **Data Quality Checks**: Detects class imbalance and provides actionable warnings

### Diagnostic Checks

- **Data Leakage Detection**: Check for duplicate rows between train and validation sets
- **Label Shuffle Test**: Verify that model performance degrades when labels are randomized
- **Overfitting Verification**: Ensure models can fit small batches (sanity check for training pipeline)
- **Class Imbalance Detection**: Warns when minority class falls below configurable threshold

## Project Structure

```
bias-variance-diagnoser/
├── README.md                      # Project documentation
├── pyproject.toml                 # Project configuration and dependencies
├── .gitignore                     # Git ignore patterns
├── run.py                         # CLI entry point
│
├── configs/                       # Configuration files
│   ├── default.yaml               # Generic default settings
│   ├── example_creditcard.yaml    # Example project configuration
│   ├── tree.yaml                  # Decision tree configuration
│   ├── ridge.yaml                 # Ridge regression configuration
│   ├── logistic_reg.yaml          # Logistic regression configuration
│   ├── lasso.yaml                 # Lasso regression configuration
│   ├── nn.yaml                    # Neural network configuration
│   └── linear_reg.yaml            # Linear regression configuration
│
├── data/                          # Data storage
│   ├── raw/                       # Original datasets (CSV, JSON, Excel)
│   ├── processed/                 # Preprocessed datasets
│   └── README.md                  # Data documentation
│
├── notebooks/                     # Jupyter notebooks
│   ├── 01_quickstart_demo.ipynb   # Quick start guide
│   └── 02_error_analysis.ipynb    # Advanced analysis examples
│
├── src/                           # Source code
│   └── diagnoser/                 # Main package
│       ├── __init__.py
│       ├── diagnoser.py           # Command-line interface
│       ├── pipeline.py            # Main orchestration logic
│       │
│       ├── data/                  # Data loading & processing
│       │   ├── __init__.py
│       │   ├── loader.py          # Data file loaders (CSV/JSON/Excel)
│       │   └── processing.py      # Data cleaning and preprocessing
│       │
│       ├── models/                # Model builders
│       │   ├── __init__.py
│       │   ├── builder.py         # Model factory
│       │   ├── sklearn_models.py  # Scikit-learn implementations
│       │   └── nn_models.py       # Neural network implementations
│       │
│       ├── curves/                # Learning curve generation
│       │   ├── __init__.py
│       │   ├── learning_curve.py  # Main learning curve logic
│       │   └── sampling.py        # Data sampling utilities
│       │
│       ├── metrics/               # Performance metrics
│       │   ├── __init__.py
│       │   └── metrics.py         # Error metrics
│       │
│       ├── diagnosis/             # Diagnostic rules and checks
│       │   ├── __init__.py
│       │   ├── rules.py           # Bias/variance classification logic
│       │   ├── recommendations.py # Actionable suggestions
│       │   └── data_quality.py    # Data quality checks
│       │
│       ├── checks/                # Sanity checks
│       │   ├── __init__.py
│       │   └── sanity_checks.py   # Data quality and training checks
│       │
│       ├── reporting/             # Output generation
│       │   ├── __init__.py
│       │   ├── plots.py           # Visualization functions
│       │   └── report.py          # JSON report generation
│       │
│       └── utils/                 # Utility functions
│           ├── __init__.py
│           ├── io.py              # File I/O helpers
│           └── seed.py            # Random seed management
│
├── tests/                         # Test suite
│   ├── test_learning_curve.py     # Learning curve tests
│   ├── test_diagnosis_rules.py    # Diagnosis logic tests
│   └── test_metrics.py            # Metrics tests
│
├── outputs/                       # Generated outputs (auto-created)
│   ├── figures/                   # Saved plots (PNG/PDF)
│   ├── reports/                   # JSON diagnostic reports
│   ├── models/                    # Saved model artifacts
│   └── processed/                 # Processed data exports
│
└── scripts/                       # Helper scripts
    ├── run_demo.sh                # Quick demo runner
    └── download_data.sh           # Data download script
```

## Installation

### Requirements

- Python 3.9 or higher
- pip or conda package manager

### Install from Source

1. Clone the repository:

```bash
git clone https://github.com/yourusername/bias-variance-diagnoser.git
cd bias-variance-diagnoser
```

2. Install in development mode:

```bash
pip install -e .
```

3. (Optional) Install with neural network support:

```bash
pip install -e ".[nn]"
```

### Dependencies

**Core dependencies:**
- numpy >= 1.26
- pandas >= 2.0
- scikit-learn >= 1.3
- matplotlib >= 3.8
- seaborn >= 0.13
- pyyaml >= 6.0
- joblib >= 1.3
- openpyxl >= 3.1 (for Excel file support)

**Optional dependencies:**
- tensorflow >= 2.13 (for neural network models)
- jupyter >= 1.0 (for notebooks)

## Quick Start

### Using the Command-Line Interface (Recommended)

The easiest way to use the tool is through the CLI:

#### List available data files and models

```bash
python run.py --list
```

#### Quick start with minimal arguments

If you have one CSV file in `data/raw/`, the tool auto-detects it:

```bash
python run.py --target Class --model tree_classifier
```

#### Full control with all options

```bash
python run.py \
  --data data/raw/creditcard.csv \
  --target Class \
  --model logistic_regression \
  --drop-columns Time,Id \
  --scale-features \
  --encoding onehot \
  --handle-missing median \
  --output-dir results/my_analysis \
  --save-processed
```

#### Help and examples

```bash
python run.py -h
```

### Command-Line Options

**Required:**
- `--target COL`: Name of target column (REQUIRED)

**Data arguments:**
- `--data PATH`: Path to data file (auto-detects if omitted)
- `--drop-columns COL1,COL2`: Comma-separated columns to drop

**Model arguments:**
- `--model {tree_classifier|logistic_regression|ridge_regression|lasso_regression|linear_regression|nn_classifier}`: Model type (default: tree_classifier)
- `--task-type {classification|regression}`: Task type (auto-detects if omitted)

**Processing arguments:**
- `--scale-features`: Enable feature scaling (default: True)
- `--no-scale`: Disable feature scaling
- `--encoding {onehot|label}`: Categorical encoding (default: onehot)
- `--handle-missing {mean|median|mode}`: Missing value strategy (default: mean)

**Output arguments:**
- `--output-dir DIR`: Output directory (default: outputs/)
- `--save-processed`: Save cleaned data to CSV

**Utility arguments:**
- `--list`: Show available data files and models
- `-h, --help`: Show help message

## Usage

### Command Line Interface

See [Quick Start](#quick-start) section for detailed CLI examples and all available options.

**Key features:**
- **Auto-detection**: Automatically finds CSV/JSON files in `data/raw/`
- **Interactive selection**: Prompts user to choose if multiple files exist
- **Configuration summary**: Displays settings before execution
- **No config file needed**: All settings can be specified via arguments

### Python API

For programmatic usage, import the pipeline directly:

```python
from src.diagnoser.pipeline import load_config, run

# Load and customize config
config = load_config("configs/default.yaml", "configs/ridge.yaml")
config["data_path"] = "path/to/your/data.csv"
config["target_column"] = "target"

# Run the analysis
run(config)
```

### Configuration Files

Configuration files (`configs/*.yaml`) define model-specific parameters and are **optional** when using the CLI.

**Generic defaults** (`configs/default.yaml`):
- Task type, model type
- Data processing settings (scaling, encoding, imputation)
- Learning curve parameters
- Output paths and thresholds

**Project-specific configs** (e.g., `configs/example_creditcard.yaml`):
- Data file path and target column
- Columns to drop
- Model hyperparameters

Example of using both configs:

```python
config = load_config("configs/default.yaml", "configs/my_project.yaml")
```

## Data Requirements

### Input Formats

**Supported file types:**
- CSV (.csv)
- JSON (.json)
- Excel (.xlsx, .xls)

**Data structure:**
- Tabular format (rows = samples, columns = features + target)
- Target column can be numeric or categorical (auto-detected)
- Features can be numeric or categorical (automatically encoded)

### Data Processing Pipeline

Automatic preprocessing with configurable steps:

1. **Duplicate Removal**: Removes identical rows
2. **Missing Value Handling**: Imputation using mean/median/mode
3. **Categorical Encoding**: Converts strings to numeric (one-hot or label encoding)
4. **Feature Scaling**: Normalizes numeric features (StandardScaler or MinMaxScaler)
5. **Target Encoding**: Converts categorical targets to integers
6. **Data Quality Checks**: Detects class imbalance and provides warnings

All steps are configurable and can be disabled if needed.

## Supported Models

### Scikit-Learn Models

**Classification:**
- **tree_classifier**: Decision Tree Classifier
  - Config: `configs/tree.yaml`
  - Hyperparameters: max_depth, min_samples_split, min_samples_leaf

- **logistic_regression**: Logistic Regression
  - Config: `configs/logistic_reg.yaml`
  - Hyperparameters: C, max_iter, solver

**Regression:**
- **linear_regression**: Ordinary Least Squares
  - Config: `configs/linear_reg.yaml`

- **ridge_regression**: Ridge Regression (L2 regularization)
  - Config: `configs/ridge.yaml`
  - Hyperparameters: alpha

- **lasso_regression**: Lasso Regression (L1 regularization)
  - Config: `configs/lasso.yaml`
  - Hyperparameters: alpha, max_iter

### Neural Networks

- **nn_classifier**: Multi-layer Perceptron (requires TensorFlow)
  - Config: `configs/nn.yaml`
  - Hyperparameters: hidden_layers, activation, learning_rate, epochs, batch_size
  - Requires: `pip install -e ".[nn]"`

## Output and Reports

### Generated Files

The tool generates comprehensive outputs in the specified output directory:

1. **Learning Curve Plot** (`figures/learning_curve.png`)
   - X-axis: Training set size (percentage)
   - Y-axis: Error rate
   - Shows training and validation error curves
   - Indicates bias/variance characteristics

2. **Diagnostic Report** (`reports/report.json`)
   - Training and validation errors
   - Computed gap (variance indicator)
   - Diagnosis label (high_bias, high_variance, balanced)
   - Actionable recommendations
   - Sanity check warnings

3. **Processed Data** (`processed/cleaned_data.csv`) - optional
   - Cleaned and preprocessed features
   - Original target column
   - Enable with `--save-processed` flag

4. **Saved Models** (`models/`)
   - Final trained model (PKL for sklearn, H5 for neural networks)
   - Auto-saved when diagnosis is "balanced"

### Report Contents

Example JSON report:

```json
{
  "diagnosis": {
    "label": "balanced",
    "train_error": 0.0066,
    "val_error": 0.0614,
    "gap": 0.0548
  },
  "recommendation": "Your model is well-balanced. Continue monitoring performance.",
  "learning_curve": {
    "sizes": [0.1, 0.3, 0.5, 0.7, 1.0],
    "train_errors": [0.0, 0.0, 0.0088, 0.0094, 0.0066],
    "val_errors": [0.0439, 0.0614, 0.0789, 0.0702, 0.0614]
  },
  "warnings": []
}
```

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test Suites

```bash
# Test metrics
pytest tests/test_metrics.py -v

# Test diagnosis rules
pytest tests/test_diagnosis_rules.py -v

# Test learning curves
pytest tests/test_learning_curve.py -v
```

### Test Coverage

Generate coverage report:

```bash
pytest tests/ --cov=src/diagnoser --cov-report=html
```

## Development

### Setting Up Development Environment

1. Clone the repository
2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install in editable mode with dev dependencies:

```bash
pip install -e ".[dev]"
```

### Code Style

This project follows:
- PEP 8 style guidelines
- Type hints where applicable
- Docstrings for all public functions and classes

### Adding New Models

1. Implement builder function in `src/diagnoser/models/sklearn_models.py` or `nn_models.py`
2. Register in `src/diagnoser/models/builder.py`
3. Add configuration example in `configs/`
4. Add tests in `tests/`

### Adding New Metrics

1. Implement metric function in `src/diagnoser/metrics/metrics.py`
2. Update learning curve logic to support new metric
3. Add tests in `tests/test_metrics.py`

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Reporting Issues

Please use the GitHub issue tracker to report bugs or request features. Include:
- Description of the issue
- Steps to reproduce
- Expected vs. actual behavior
- Python version and OS
- Relevant configuration files

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Inspired by Andrew Ng's machine learning course material on bias-variance diagnostics
- Built with scikit-learn, TensorFlow, and other excellent open-source libraries
- Learning curve methodology based on established machine learning best practices

## Citation

If you use this tool in your research or projects, please cite:

```bibtex
@software{bias_variance_diagnoser,
  title={Bias-Variance Diagnostic Tool},
  author={Aly Soffar},
  year={2026},
  url={https://github.com/Alysoffar/bias-variance-diagnoser}
}
```

## Contact

For questions, suggestions, or collaboration opportunities:
- GitHub Issues: [https://github.com/Alysoffar/bias-variance-diagnoser/issues](https://github.com/yourusername/bias-variance-diagnoser/issues)
- Email: alysoffar06@gmail.com

## Roadmap

Planned features:
- Support for more model types (XGBoost, LightGBM, etc.)
- Cross-validation based learning curves
- Interactive visualizations with Plotly
- Automated hyperparameter tuning suggestions
- Multi-task learning support
- Web-based dashboard interface
