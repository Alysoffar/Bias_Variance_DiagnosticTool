import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def encode_target(Y):
    """
    Label encode the target variable if it's categorical (strings).
    For classification targets that are strings, convert to numeric labels.
    
    Args:
        Y (pd.Series or np.ndarray): Target variable
    
    Returns:
        tuple: (Y_encoded, label_map) where label_map maps original values to encoded integers
    """
    if isinstance(Y, np.ndarray):
        Y = pd.Series(Y)
    
    # Check if target is categorical (string/object)
    if Y.dtype == 'object':
        print(f"Detected categorical target with values: {Y.unique()}")
        
        # Map categorical values to integers
        unique_values = sorted(Y.unique())
        label_map = {val: idx for idx, val in enumerate(unique_values)}
        Y_encoded = Y.map(label_map)
        
        print(f"Applied label encoding to target: {label_map}")
        return Y_encoded, label_map
    else:
        # Target is already numeric
        return Y, None


def scale_features(X, config):
    """
    Scale numeric features using StandardScaler or MinMaxScaler.
    
    Args:
        X (pd.DataFrame): Features DataFrame
        config (dict): Configuration dictionary
    
    Returns:
        pd.DataFrame: Scaled features
    """
    scaler_type = config.get("scaler_type", "standard")  # "standard" or "minmax"
    
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    # Preserve column names and index
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    print(f"Applied {scaler_type} scaling to features.")
    return X_scaled


def encode_categorical(X, config):
    """
    Encode categorical columns using one-hot encoding.
    
    Args:
        X (pd.DataFrame): Features DataFrame
        config (dict): Configuration dictionary
    
    Returns:
        pd.DataFrame: DataFrame with categorical columns encoded
    """
    # Detect categorical columns (object/string dtype)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) == 0:
        print("No categorical columns detected.")
        return X
    
    print(f"Detected categorical columns: {categorical_cols}")
    
    encoding_strategy = config.get("encoding_strategy", "onehot")  # "onehot" or "label"
    
    if encoding_strategy == "onehot":
        # One-hot encoding
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        print(f"Applied one-hot encoding to {len(categorical_cols)} columns.")
        print(f"New shape after encoding: {X_encoded.shape}")
        return X_encoded
    
    elif encoding_strategy == "label":
        # Label encoding
        X_copy = X.copy()
        for col in categorical_cols:
            X_copy[col] = pd.factorize(X_copy[col])[0]
        print(f"Applied label encoding to {len(categorical_cols)} columns.")
        return X_copy
    
    else:
        raise ValueError(f"Unknown encoding strategy: {encoding_strategy}")


def save_cleaned_data(X, Y, output_path: str, target_col: str = "target"):
    """
    Save cleaned data to a CSV file.
    
    Args:
        X (pd.DataFrame or np.ndarray): Cleaned features
        Y (pd.Series or np.ndarray): Cleaned target
        output_path (str): Path where to save the file (e.g., 'data/processed/cleaned.csv')
        target_col (str): Name for the target column in the output file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to DataFrame if numpy array
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    if not isinstance(Y, pd.Series):
        Y = pd.Series(Y, name=target_col)
    
    # Combine X and Y
    df = X.copy()
    df[target_col] = Y.values
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to: {output_path}")
    print(f"Shape: {df.shape}")


def clean_data(X, Y, config):
    """
    Clean the input DataFrame by handling missing values, removing duplicates, encoding categorical variables, and encoding target.

    Args:
        X (pd.DataFrame): Features DataFrame
        Y (pd.Series): Target variable
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (X_cleaned, Y_cleaned)
    """
    
    # Drop specified columns if any
    drop_cols = config.get("drop_columns", [])
    if drop_cols:
        existing_cols = [col for col in drop_cols if col in X.columns]
        if existing_cols:
            X = X.drop(columns=existing_cols)
            print(f"Dropped columns: {existing_cols}")
    
    if config.get("remove_duplicates", True):
        initial_shape = X.shape
        X = X.drop_duplicates()
        Y = Y.loc[X.index]
        print(f"Removed {initial_shape[0] - X.shape[0]} duplicate rows.")
        

    if config.get("handle_missing", True):
        strategy = config.get("missing_value_strategy", "mean")
        for col in X.columns:
            if X[col].isnull().any():
                if strategy == "mean":
                    fill_value = X[col].mean()
                elif strategy == "median":
                    fill_value = X[col].median()
                elif strategy == "mode":
                    fill_value = X[col].mode()[0]
                else:
                    raise ValueError(f"Unknown missing value strategy: {strategy}")
                X[col] = X[col].fillna(fill_value)  # Fixed: avoid inplace on copy warning
                print(f"Filled missing values in column '{col}' using {strategy} strategy.")
    
    if config.get("encode_categorical", True):
        X = encode_categorical(X, config)
    
    # Encode target variable if it's categorical
    if config.get("encode_target", True):
        Y, label_map = encode_target(Y)
    
    # Scale features if enabled
    if config.get("scale_features", False):
        X = scale_features(X, config)
    
    return X, Y