import numpy as np


def diagnosis_rules(train_error, val_error, config):
    """Diagnose bias/variance from training and validation errors.
    
    Args:
        train_error: scalar or list of training errors
        val_error: scalar or list of validation errors
        config: config dict with diagnosis thresholds
    
    Returns:
        dict with label, train_error, val_error, and gap
    """
    # Handle list inputs by taking the final value
    if isinstance(train_error, (list, tuple)):
        if len(train_error) == 0:
            return {"label": "insufficient_data", "train_error": None, "val_error": None, "gap": None}
        train_err = train_error[-1]
    else:
        train_err = train_error
    
    if isinstance(val_error, (list, tuple)):
        if len(val_error) == 0:
            return {"label": "insufficient_data", "train_error": None, "val_error": None, "gap": None}
        val_err = val_error[-1]
    else:
        val_err = val_error
    
    gap = val_err - train_err
    
    # Use defaults if diagnosis config is missing
    gap_threshold = config.get("diagnosis", {}).get("gap_threshold", 0.1)
    high_error_threshold = config.get("diagnosis", {}).get("high_error_threshold", 0.3)

    if train_err > high_error_threshold and val_err > high_error_threshold:
        label = "high_bias"
    elif gap > gap_threshold:
        label = "high_variance"
    else:
        label = "balanced"
    
    return {
        "label": label,
        "train_error": float(train_err),
        "val_error": float(val_err),
        "gap": float(gap)
    }