"""
Bias-Variance Diagnostic Library

A comprehensive toolkit for diagnosing bias-variance tradeoffs in machine learning models.

Main API:
    run_diagnosis: Execute the full diagnosis pipeline on a dataset.

Classes:
    DiagnosisResult: Structured result from the diagnosis pipeline.
    LibraryError, DatasetError, ModelError, ConfigError: Exception types.

Example:
    >>> from diagnoser import run_diagnosis
    >>> result = run_diagnosis(
    ...     dataset_path="data/mydata.csv",
    ...     model_name="ridge_regression",
    ...     label_column="target",
    ...     output_dir="results/"
    ... )
    >>> print(result.diagnosis_label)
    >>> print(result.recommendations)
"""

from .api import (
    run_diagnosis,
    DiagnosisResult,
    LibraryError,
    DatasetError,
    ModelError,
    ConfigError,
)

__version__ = "0.2.0"
__author__ = "Your Name"

__all__ = [
    "run_diagnosis",
    "DiagnosisResult",
    "LibraryError",
    "DatasetError",
    "ModelError",
    "ConfigError",
]
