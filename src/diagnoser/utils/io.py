from sklearn.model_selection import train_test_split
import pandas as pd


def load_and_split_data(file_path, test_size=0.2, random_state=42):
    """
    Load dataset from a CSV file and split it into training and validation sets.

    Parameters:
    - file_path: str, path to the CSV file.
    - test_size: float, proportion of the dataset to include in the validation split.
    - random_state: int, random seed for reproducibility.

    Returns:
    - X_train, Y_train: training features and labels.
    - X_val, Y_val: validation features and labels.
    """
    
    df = pd.read_csv(file_path)
    X = df.drop('target', axis=1)
    Y = df['target']
    
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    
    return X_train, Y_train, X_val, Y_val