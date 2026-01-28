import argparse
import os
import sys
from pathlib import Path

try:
    from .pipeline import load_config, run
except ImportError:
    from pipeline import load_config, run


def find_data_files(data_dir="data/raw"):
    """
    Auto-detect CSV and JSON files in the data directory.
    
    Args:
        data_dir: Path to search for data files
    
    Returns:
        dict: {filename: full_path} of found files
    """
    data_files = {}
    
    if not os.path.exists(data_dir):
        return data_files
    
    for ext in ['*.csv', '*.json']:
        for file_path in Path(data_dir).glob(ext):
            data_files[file_path.name] = str(file_path)
    
    return data_files


def list_available_files():
    """List all available data files and models."""
    print("\n" + "=" * 70)
    print("AVAILABLE DATA FILES IN data/raw/")
    print("=" * 70)
    
    data_files = find_data_files()
    if data_files:
        for idx, (filename, filepath) in enumerate(data_files.items(), 1):
            print(f"  {idx}. {filename}")
    else:
        print("  No data files found. Add CSV or JSON files to data/raw/")
    
    print("\n" + "=" * 70)
    print("AVAILABLE MODELS")
    print("=" * 70)
    models = [
        "tree_classifier - Decision Tree",
        "logistic_regression - Logistic Regression",
        "linear_regression - Linear Regression",
        "ridge_regression - Ridge Regression",
        "lasso_regression - Lasso Regression",
        "nn_classifier - Neural Network"
    ]
    for model in models:
        print(f"  {model}")
    
    print()


def auto_select_data_file():
    """
    If only one data file exists, auto-select it.
    If multiple exist, ask user to choose.
    """
    data_files = find_data_files()
    
    if not data_files:
        print("❌ ERROR: No data files found in data/raw/")
        print("   Please add a CSV or JSON file to data/raw/")
        return None
    
    if len(data_files) == 1:
        filename, filepath = list(data_files.items())[0]
        print(f"✓ Auto-selected data file: {filename}")
        return filepath
    
    # Multiple files - ask user to choose
    print("\nMultiple data files found. Please choose:")
    for idx, (filename, filepath) in enumerate(data_files.items(), 1):
        print(f"  {idx}. {filename}")
    
    while True:
        try:
            choice = int(input("Enter number (1-{}): ".format(len(data_files))))
            if 1 <= choice <= len(data_files):
                filepath = list(data_files.values())[choice - 1]
                return filepath
        except ValueError:
            pass
        print("Invalid choice. Try again.")


def main():
    parser = argparse.ArgumentParser(
        description="Bias-Variance Diagnostic Tool - Analyze model performance across training set sizes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  # Quick start with auto-detected data file
  python cli.py --target Class --model tree_classifier
  
  # Specify data file explicitly
  python cli.py --data data/raw/mydata.csv --target target_col --model ridge_regression
  
  # List available data files and models
  python cli.py --list
  
  # Custom configuration
  python cli.py --data data/raw/data.csv --target Class \\
    --model logistic_regression --drop-columns Time,Id \\
    --scale-features --encoding onehot
        """
    )
    
    # Core arguments
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to data file (CSV/JSON). Auto-detects if not specified."
    )
    
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        required=True,
        help="Name of the target column (REQUIRED)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="tree_classifier",
        choices=["tree_classifier", "logistic_regression", "linear_regression", 
                 "ridge_regression", "lasso_regression", "nn_classifier"],
        help="Model to use (default: tree_classifier)"
    )
    
    # Data processing arguments
    parser.add_argument(
        "--drop-columns",
        type=str,
        default=None,
        help="Comma-separated list of columns to drop (e.g., 'Time,Id')"
    )
    
    parser.add_argument(
        "--scale-features",
        action="store_true",
        default=True,
        help="Scale features using StandardScaler (default: True)"
    )
    
    parser.add_argument(
        "--no-scale",
        action="store_false",
        dest="scale_features",
        help="Disable feature scaling"
    )
    
    parser.add_argument(
        "--encoding",
        type=str,
        default="onehot",
        choices=["onehot", "label"],
        help="Categorical encoding strategy (default: onehot)"
    )
    
    parser.add_argument(
        "--handle-missing",
        type=str,
        default="mean",
        choices=["mean", "median", "mode"],
        help="Strategy for handling missing values (default: mean)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output files (default: outputs)"
    )
    
    parser.add_argument(
        "--save-processed",
        action="store_true",
        help="Save cleaned/processed data to CSV"
    )
    
    # Utility arguments
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available data files and models, then exit"
    )
    
    parser.add_argument(
        "--task-type",
        type=str,
        choices=["classification", "regression"],
        default=None,
        help="Specify task type (auto-detect if not specified)"
    )
    
    args = parser.parse_args()
    
    # Handle --list option
    if args.list:
        list_available_files()
        return
    
    # Auto-detect data file if not specified
    if args.data is None:
        print(" Searching for data files in data/raw/...")
        args.data = auto_select_data_file()
        if args.data is None:
            sys.exit(1)
    
    # Validate data file exists
    if not os.path.exists(args.data):
        print(f" ERROR: Data file not found: {args.data}")
        sys.exit(1)
    
    # Load base config
    print("\n Loading configuration...")
    config = load_config("configs/default.yaml", "configs/default.yaml")
    
    # Override config with CLI arguments
    config["data_path"] = args.data
    config["target_column"] = args.target
    config["model_type"] = args.model
    config["scale_features"] = args.scale_features
    config["encoding_strategy"] = args.encoding
    config["missing_value_strategy"] = args.handle_missing
    config["save_cleaned_data"] = args.save_processed
    
    if args.task_type:
        config["task_type"] = args.task_type
    
    # Handle drop_columns
    if args.drop_columns:
        config["drop_columns"] = [col.strip() for col in args.drop_columns.split(",")]
    
    # Set output paths
    os.makedirs(args.output_dir, exist_ok=True)
    config["output_figure_path"] = os.path.join(args.output_dir, "figures", "learning_curve.png")
    config["output_report_path"] = os.path.join(args.output_dir, "reports", "report.json")
    
    if args.save_processed:
        config["cleaned_data_path"] = os.path.join(args.output_dir, "processed", "cleaned_data.csv")
    
    # Create subdirectories
    os.makedirs(os.path.dirname(config["output_figure_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["output_report_path"]), exist_ok=True)
    
    # Display configuration summary
    print("\n" + "=" * 70)
    print("CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"Data file:          {config['data_path']}")
    print(f"Target column:      {config['target_column']}")
    print(f"Model:              {config['model_type']}")
    print(f"Task type:          {config['task_type'] or 'Auto-detect'}")
    print(f"Scale features:     {config['scale_features']}")
    print(f"Encoding:           {config['encoding_strategy']}")
    print(f"Handle missing:     {config['missing_value_strategy']}")
    if config['drop_columns']:
        print(f"Drop columns:       {config['drop_columns']}")
    print(f"Output directory:   {args.output_dir}")
    print("=" * 70 + "\n")
    
    # Run the pipeline
    try:
        run(config)
        print("\n Analysis complete!")
        print(f" Results saved to: {args.output_dir}")
    except Exception as e:
        print(f"\n ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
