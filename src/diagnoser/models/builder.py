from .sklearn_models import (
    build_tree_classifier,
    build_logistic_regression,
    build_linear_regression,
    build_ridge_regression,
    build_lasso_regression
)
from .nn_models import build_nn_classifier


def build_model(model_type, config, input_dim=None):

    if model_type == "tree_classifier":
        return build_tree_classifier(config)
    
    elif model_type == "logistic_regression":
        return build_logistic_regression(config)
    
    elif model_type == "linear_regression":
        return build_linear_regression(config)
    
    elif model_type == "ridge_regression":
        return build_ridge_regression(config)
    
    elif model_type == "lasso_regression":
        return build_lasso_regression(config)
    
    elif model_type == "nn_classifier":
        if input_dim is None:
            raise ValueError("input_dim must be provided for nn_classifier")
        return build_nn_classifier(config, input_dim)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")




