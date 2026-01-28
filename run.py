#!/usr/bin/env python
"""
Bias-Variance Diagnostic Tool - Main Entry Point

This is the CLI entry point. For library usage, see example_library_usage.py

Usage:
    python run.py --target Class --model tree_classifier
    python run.py --data data/raw/mydata.csv --target target_col --model ridge_regression
    python run.py --list

For library usage:
    from diagnoser import run_diagnosis
    result = run_diagnosis(
        dataset_path="data/raw/data.csv",
        model_name="ridge_regression",
        label_column="Class",
        output_dir="results/"
    )
"""

import sys
import os

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from diagnoser.diagnoser import main

if __name__ == "__main__":
    main()
