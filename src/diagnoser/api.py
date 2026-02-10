"""
Public API for the Bias-Variance Diagnostic Library

Main entry point for external projects to use the diagnosis pipeline.
"""

import os
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, Union
import yaml

from .data.loader import load_data, detect_task_type
from .data.processing import clean_data, save_cleaned_data
from .checks.sanity_checks import run_sanity_checks
from .models.builder import build_model
from .curves.learning_curve import learning_curve
from .diagnosis.rules import diagnosis_rules
from .diagnosis.recommendations import recommendation
from .diagnosis.data_quality import check_data_quality
from .reporting.plots import plot_learning_curve
from .reporting.report import report_findings
from .utils.seed import set_seed
from .metrics.metrics import classification_metrics

try:
    import joblib
    import pickle
except ImportError:
    raise ImportError("joblib is required for loading models. Install with: pip install joblib")


@dataclass
class DiagnosisResult:
    """Structured result from the diagnosis pipeline."""
    
    # Learning curve data
    sizes: list  # Training set fractions
    train_errors: list  # Training errors at each size
    val_errors: list  # Validation errors at each size
    
    # Final metrics
    final_train_error: float
    final_val_error: float
    error_gap: float
    
    # Diagnosis
    diagnosis_label: str  # "high_bias", "high_variance", or "balanced"
    diagnosis_details: Dict[str, Any]  # Full diagnosis dict
    recommendations: str  # Actionable recommendations

    
    # Data quality
    data_quality_warnings: list  # List of warning strings
    sanity_check_warnings: list  # List of sanity check warning strings

    # Final evaluation metrics (classification only)
    final_metrics: Optional[Dict[str, Any]] = None
    
    # Artifacts
    plot_path: Optional[str] = None  # Path to learning curve plot
    report_path: Optional[str] = None  # Path to JSON report
    model_path: Optional[str] = None  # Path to saved model (if requested)
    
    # Configuration used
    config: Optional[Dict[str, Any]] = None  # The config that was used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class LibraryError(Exception):
    """Base exception for library errors."""
    pass


class DatasetError(LibraryError):
    """Raised when dataset cannot be loaded or is invalid."""
    pass


class ModelError(LibraryError):
    """Raised when model cannot be loaded or built."""
    pass


class ConfigError(LibraryError):
    """Raised when configuration is invalid."""
    pass


def _load_or_build_model(
    model_input: Optional[Union[str, Any]],
    model_name: Optional[str],
    config: Dict[str, Any],
    input_dim: int,
    working_dir: str
) -> tuple:
    """
    Load a pre-trained model or build a new one.
    
    Args:
        model_input: Path to pre-trained model file (.pkl or Keras model dir/file)
        model_name: Name of model to build (e.g., "tree_classifier", "ridge_regression")
        config: Configuration dict
        input_dim: Number of input features
        working_dir: Working directory to save models
    
    Returns:
        tuple: (model, is_pretrained)
    
    Raises:
        ModelError: If model cannot be loaded or built
    """
    if model_input is not None:
        # Load pre-trained model from file
        if not isinstance(model_input, str):
            raise ModelError("model parameter must be a string path to a model file")
        
        if not os.path.exists(model_input):
            raise ModelError(f"Model file not found: {model_input}")
        
        try:
            if model_input.endswith('.pkl'):
                # Load scikit-learn model
                model = joblib.load(model_input)
                return model, True
            elif model_input.endswith('.h5') or os.path.isdir(model_input):
                # Load Keras model
                try:
                    from tensorflow import keras
                    if os.path.isdir(model_input):
                        model = keras.models.load_model(model_input)
                    else:
                        model = keras.models.load_model(model_input)
                    return model, True
                except ImportError:
                    raise ModelError("TensorFlow/Keras is required to load .h5 or Keras models. Install with: pip install tensorflow")
            else:
                raise ModelError(f"Unsupported model file format: {model_input}. Supported: .pkl, .h5, Keras model directory")
        except Exception as e:
            raise ModelError(f"Failed to load model from {model_input}: {str(e)}")
    
    elif model_name is not None:
        # Build model from name and config
        valid_models = [
            "tree_classifier",
            "logistic_regression",
            "linear_regression",
            "ridge_regression",
            "lasso_regression",
            "nn_classifier"
        ]
        
        if model_name not in valid_models:
            raise ModelError(
                f"Unknown model name: {model_name}. "
                f"Valid options: {', '.join(valid_models)}"
            )
        
        try:
            model = build_model(
                model_type=model_name,
                config=config.get("model", {}),
                input_dim=input_dim
            )
            return model, False
        except Exception as e:
            raise ModelError(f"Failed to build model '{model_name}': {str(e)}")
    
    else:
        raise ModelError("Either 'model' (path) or 'model_name' (string) must be provided")


def _load_config(config_path: Optional[str], model_name: Optional[str]) -> Dict[str, Any]:
    """
    Load configuration from YAML files.
    
    Args:
        config_path: Path to custom config YAML file
        model_name: Model name for selecting default config
    
    Returns:
        dict: Merged configuration
    
    Raises:
        ConfigError: If config file is invalid or not found
    """
    # Start with library default config
    library_dir = os.path.dirname(__file__)
    default_config_path = os.path.join(library_dir, '..', '..', 'configs', 'default.yaml')
    
    # Try to find default config
    if not os.path.exists(default_config_path):
        # Create a minimal default config if file not found
        base_config = {
            "learning_curve": {"sizes": [0.1, 0.3, 0.5, 0.7, 1.0]},
            "diagnosis": {
                "gap_threshold": 0.1,
                "high_error_threshold": 0.3,
                "imbalance_threshold": 5.0
            },
            "random_seed": 42,
            "scale_features": True,
            "scaler_type": "standard",
            "encode_categorical": True,
            "encoding_strategy": "onehot",
            "handle_missing": True,
            "missing_value_strategy": "mean",
            "remove_duplicates": True
        }
    else:
        try:
            with open(default_config_path, 'r') as f:
                base_config = yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigError(f"Failed to load default config: {str(e)}")
    
    # Load custom config if provided
    if config_path is not None:
        if not os.path.exists(config_path):
            raise ConfigError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f) or {}
            base_config.update(custom_config)
        except Exception as e:
            raise ConfigError(f"Failed to load config from {config_path}: {str(e)}")
    
    # Select model-specific config if using model name
    if model_name is not None and config_path is None:
        model_config_map = {
            "tree_classifier": "tree.yaml",
            "logistic_regression": "logistic_reg.yaml",
            "linear_regression": "linear_regression.yaml",  # Create if needed
            "ridge_regression": "ridge.yaml",
            "lasso_regression": "lasso.yaml",
            "nn_classifier": "nn.yaml"
        }
        
        if model_name in model_config_map:
            model_config_name = model_config_map[model_name]
            model_config_path = os.path.join(
                os.path.dirname(library_dir),
                '..',
                '..',
                'configs',
                model_config_name
            )
            
            if os.path.exists(model_config_path):
                try:
                    with open(model_config_path, 'r') as f:
                        model_config = yaml.safe_load(f) or {}
                    base_config.update(model_config)
                except Exception as e:
                    # Warn but don't fail if model config can't be loaded
                    print(f"Warning: Could not load model-specific config: {str(e)}")
    
    return base_config


def run_diagnosis(
    dataset_path: str,
    model: Optional[str] = None,
    model_name: Optional[str] = None,
    label_column: str = "Class",
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_model: bool = False,
    save_plot: bool = True,
    save_report: bool = True,
    **kwargs
) -> DiagnosisResult:
    """
    Run the full bias-variance diagnosis pipeline.
    
    This is the main public API for the library. It handles data loading,
    model selection, learning curve generation, diagnosis, and reporting.
    
    Args:
        dataset_path (str): Path to CSV file with training data.
                          Will be copied to a working directory.
        model (str, optional): Path to pre-trained model file (.pkl for scikit-learn
                             or .h5/.directory for Keras). Either 'model' or 'model_name'
                             must be provided.
        model_name (str, optional): Name of model to train. Options:
                                  - "tree_classifier"
                                  - "logistic_regression"
                                  - "linear_regression"
                                  - "ridge_regression"
                                  - "lasso_regression"
                                  - "nn_classifier"
        label_column (str): Name of target column in dataset. Default: "Class".
        config_path (str, optional): Path to YAML config file. If not provided,
                                    uses defaults (+ model-specific config if available).
        output_dir (str, optional): Directory to save artifacts (plots, reports, models).
                                   If None, uses temp directory (artifacts not persisted).
        save_model (bool): If True, save the trained/loaded model. Default: False.
        save_plot (bool): If True, save learning curve plot. Default: True.
        save_report (bool): If True, save JSON report. Default: True.
        **kwargs: Additional arguments (for future extensibility).
    
    Returns:
        DiagnosisResult: Structured result containing:
            - sizes, train_errors, val_errors: Learning curve data
            - final_train_error, final_val_error, error_gap: Final metrics
            - diagnosis_label, diagnosis_details: Diagnosis results
            - recommendations: Actionable recommendations
            - data_quality_warnings, sanity_check_warnings: Issues detected
            - plot_path, report_path, model_path: Paths to saved artifacts
    
    Raises:
        DatasetError: If dataset cannot be loaded or is invalid.
        ModelError: If model cannot be loaded or built.
        ConfigError: If configuration is invalid.
        ValueError: If required parameters are missing or invalid.
    
    Example:
        >>> from diagnoser import run_diagnosis
        >>> result = run_diagnosis(
        ...     dataset_path="data/mydata.csv",
        ...     model_name="ridge_regression",
        ...     label_column="target",
        ...     output_dir="results/"
        ... )
        >>> print(f"Diagnosis: {result.diagnosis_label}")
        >>> print(f"Recommendation: {result.recommendations}")
    """
    # Validate inputs
    if not isinstance(dataset_path, str) or not os.path.exists(dataset_path):
        raise DatasetError(f"Dataset file not found: {dataset_path}")
    
    if model is None and model_name is None:
        raise ModelError("Either 'model' (file path) or 'model_name' (string) must be provided")
    
    if model is not None and model_name is not None:
        raise ModelError("Cannot provide both 'model' and 'model_name'. Choose one.")
    
    # Load configuration
    config = _load_config(config_path, model_name)
    config["target_column"] = label_column
    config["model_type"] = model_name or "sklearn_model"
    
    # Set random seed
    seed = config.get("random_seed", 42)
    set_seed(seed)
    
    # Create working directory
    if output_dir is None:
        working_dir = tempfile.mkdtemp(prefix="diagnoser_")
        persist_artifacts = False
    else:
        working_dir = os.path.abspath(output_dir)
        os.makedirs(working_dir, exist_ok=True)
        persist_artifacts = True
    
    # Copy dataset to working directory
    work_data_dir = os.path.join(working_dir, "data", "raw")
    os.makedirs(work_data_dir, exist_ok=True)
    
    dataset_filename = os.path.basename(dataset_path)
    work_dataset_path = os.path.join(work_data_dir, dataset_filename)
    shutil.copy2(dataset_path, work_dataset_path)
    
    try:
        # Load and process data
        try:
            X, y = load_data(work_dataset_path, label_column)
        except KeyError:
            raise DatasetError(
                f"Label column '{label_column}' not found in dataset. "
                f"Available columns: {list(X.columns) if hasattr(X, 'columns') else 'unknown'}"
            )
        except Exception as e:
            raise DatasetError(f"Failed to load dataset: {str(e)}")
        
        # Detect task type
        task_type = detect_task_type(y)
        config["task_type"] = task_type
        
        # Clean data
        X, y = clean_data(X, y, config)
        
        # Check data quality
        data_quality_warnings = check_data_quality(X, y, task_type, config)
        
        # Convert to numpy arrays
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        
        # Split data (80/20)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        
        # Load or build model
        model_obj, is_pretrained = _load_or_build_model(
            model, model_name, config, X_train.shape[1], working_dir
        )
        
        # If we have data quality warnings, still try to run but include warnings
        if data_quality_warnings:
            print("[WARNING] Data quality issues detected:")
            for warning in data_quality_warnings:
                print(f"   - {warning}")
        
        # Generate learning curve
        sizes, train_errors, val_errors, final_outputs = learning_curve(
            config, X_train, y_train, X_val, y_val
        )
        
        # Get diagnosis
        final_train_error = train_errors[-1]
        final_val_error = val_errors[-1]
        diagnosis_dict = diagnosis_rules(final_train_error, final_val_error, config)
        diagnosis_label = diagnosis_dict["label"]
        recommendations_text = recommendation(diagnosis_label)
        
        # Run sanity checks
        sanity_warnings = run_sanity_checks(config, X_train, y_train, X_val, y_val)

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
        
        # Prepare artifact paths
        plot_path = None
        report_path = None
        model_path = None
        
        # Save artifacts if requested
        if save_plot or persist_artifacts:
            plot_dir = os.path.join(working_dir, "figures")
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, "learning_curve.png")
            plot_learning_curve(sizes, train_errors, val_errors, out_path=plot_path)
        
        if save_report or persist_artifacts:
            report_dir = os.path.join(working_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            report_path = os.path.join(report_dir, "report.json")
            report_findings(
                diagnosis_dict, train_errors, val_errors, sizes,
                recommendations_text,
                out_path=report_path,
                warnings=sanity_warnings,
                data_errors=data_quality_warnings,
                final_metrics=final_metrics
            )
        
        if save_model and not is_pretrained:
            model_dir = os.path.join(working_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            if hasattr(model_obj, 'save'):  # Keras model
                model_path = os.path.join(model_dir, "final_model.h5")
                model_obj.save(model_path)
            else:  # scikit-learn model
                model_path = os.path.join(model_dir, "final_model.pkl")
                joblib.dump(model_obj, model_path)
        
        # Create result
        result = DiagnosisResult(
            sizes=sizes,
            train_errors=train_errors,
            val_errors=val_errors,
            final_train_error=final_train_error,
            final_val_error=final_val_error,
            error_gap=final_val_error - final_train_error,
            diagnosis_label=diagnosis_label,
            diagnosis_details=diagnosis_dict,
            recommendations=recommendations_text,
            final_metrics=final_metrics,
            data_quality_warnings=data_quality_warnings,
            sanity_check_warnings=sanity_warnings,
            plot_path=plot_path,
            report_path=report_path,
            model_path=model_path,
            config=config
        )
        
        return result
    
    finally:
        # Clean up temp directory if not persisting
        if not persist_artifacts:
            try:
                shutil.rmtree(working_dir, ignore_errors=True)
            except:
                pass
