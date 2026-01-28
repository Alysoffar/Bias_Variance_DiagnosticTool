import numpy as np


def check_class_imbalance(Y, task_type, threshold=5.0):
    """
    Check for class imbalance in classification tasks.
    
    Args:
        Y: Target variable (numpy array or pandas Series)
        task_type: "classification" or "regression"
        threshold: Minimum percentage for minority class (default 5%)
    
    Returns:
        dict: {
            'is_imbalanced': bool,
            'minority_pct': float,
            'class_distribution': dict,
            'warning_message': str or None
        }
    """
    result = {
        'is_imbalanced': False,
        'minority_pct': None,
        'class_distribution': None,
        'warning_message': None
    }
    
    if task_type != "classification":
        return result
    
    # Get class distribution
    unique, counts = np.unique(Y, return_counts=True)
    class_dist = dict(zip(unique, counts))
    minority_pct = min(counts) / len(Y) * 100
    
    result['minority_pct'] = minority_pct
    result['class_distribution'] = class_dist
    
    # Check if imbalanced
    if minority_pct < threshold:
        result['is_imbalanced'] = True
        
        warning_msg = [
            "=" * 70,
            "[WARNING] HIGHLY IMBALANCED DATASET DETECTED!",
            "=" * 70,
            f"Class distribution: {class_dist}",
            f"Minority class: {minority_pct:.2f}% of data",
            "",
            f"A 'dumb' model predicting only the majority class would achieve",
            f"~{100-minority_pct:.2f}% accuracy ({minority_pct:.2f}% error)!",
            "",
            "  Your tool uses ACCURACY which is misleading for imbalanced data.",
            "",
            "Recommended actions:",
            "  1. Use Precision/Recall/F1-Score for minority class evaluation",
            "  2. Balance dataset (SMOTE, undersampling, or class weights)",
            "  3. Use a different, balanced dataset for bias-variance diagnostics",
            "  4. Interpret results cautiously - 'balanced' diagnosis may be misleading",
            "=" * 70
        ]
        
        result['warning_message'] = "\n".join(warning_msg)
    
    return result


def check_data_quality(X, Y, task_type, config=None):
    """
    Run all data quality checks.
    
    Args:
        X: Feature matrix
        Y: Target variable
        task_type: "classification" or "regression"
        config: Configuration dict (optional, for thresholds)
    
    Returns:
        list: List of warning messages
    """
    warnings = []
    
    # Get imbalance threshold from config, default to 5%
    imbalance_threshold = 5.0
    if config:
        imbalance_threshold = config.get("diagnosis", {}).get("imbalance_threshold", 5.0)
    
    # Check class imbalance
    imbalance_result = check_class_imbalance(Y, task_type, threshold=imbalance_threshold)
    if imbalance_result['is_imbalanced']:
        warnings.append(imbalance_result['warning_message'])
    
    # Can add more checks here in the future:
    # - Feature variance check (constant features)
    # - Missing value percentage
    # - Outlier detection
    # - Feature correlation check
    
    return warnings
