#!/usr/bin/env python
"""
Bias-Variance Diagnostic Tool - Main Entry Point

Usage:
    python run.py --target Class --model tree_classifier
    python run.py --data data/raw/mydata.csv --target target_col --model ridge_regression
    python run.py --list
"""

import sys
import os

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'diagnoser'))

from diagnoser.diagnoser import main

if __name__ == "__main__":
    main()
