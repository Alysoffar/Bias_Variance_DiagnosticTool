import pandas as pd
import numpy as np


def detect_task_type(y):
    """
    Auto-detect task type (classification vs regression) from target variable.
    
    Args:
        y: Target variable (pandas Series or numpy array)
    
    Returns:
        str: "classification" or "regression"
    """
    if isinstance(y, pd.Series):
        y_array = y.values
    else:
        y_array = np.array(y)
    
    # Check if target is numeric
    if not np.issubdtype(y_array.dtype, np.number):
        return "classification"
    
    # Count unique values
    unique_values = len(np.unique(y_array))
    total_values = len(y_array)
    
    # If unique values are less than 5% of total or <= 20, likely classification
    if unique_values <= 20 or (unique_values / total_values) < 0.05:
        return "classification"
    else:
        return "regression"


def load_data(file_path: str, target_col: str, file_format: str = None):
    """
    Load data from file and separate features from target.
    Auto-detects file format if not specified.
    
    Args:
        file_path (str): Path to the data file
        target_col (str): Name of the target column
        file_format (str, optional): File format ('csv', 'json', 'excel')
    
    Returns:
        tuple: (X, y) features and target
    """
    if file_format is None:
        # Auto-detect from extension
        if file_path.endswith('.csv'):
            file_format = 'csv'
        elif file_path.endswith('.json'):
            file_format = 'json'
        elif file_path.endswith(('.xlsx', '.xls')):
            file_format = 'excel'
        else:
            raise ValueError(f"Could not auto-detect file format from {file_path}")
    
    if file_format == 'csv':
        return load_data_csv(file_path, target_col)
    elif file_format == 'json':
        return load_data_json(file_path, target_col)
    elif file_format == 'excel':
        return load_data_excel(file_path, target_col)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def load_data_csv(file_path: str, target_col: str):
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.
        target_col (str): The name of the target column.
    """
    df = pd.read_csv(file_path)
    
    # Separate target
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    return X, y
    


def load_data_json(file_path: str, target_col: str):
    """
    Load data from a JSON file into a pandas DataFrame.

    Args:
        file_path (str): The path to the JSON file.
        target_col (str): The name of the target column.
    """
    df = pd.read_json(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def load_data_excel(file_path: str, target_col: str):
    """
    Load data from an Excel file into a pandas DataFrame.

    Args:
        file_path (str): The path to the Excel file.
        target_col (str): The name of the target column.
    """
    df = pd.read_excel(file_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
