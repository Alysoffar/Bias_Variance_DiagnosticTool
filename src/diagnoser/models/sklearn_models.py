from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge


def build_tree_classifier(config):
    return DecisionTreeClassifier(
        max_depth=config.get("max_depth", None),
        min_samples_split=config.get("min_samples_split", 2),
        class_weight=config.get("class_weight", None),
        random_state=config.get("random_state", 42)
    )


def build_logistic_regression(config):
    return LogisticRegression(
        C=config.get("C", 1.0),
        max_iter=config.get("max_iter", 100),
        class_weight=config.get("class_weight", None),
        random_state=config.get("random_state", 42)
    )

def build_linear_regression(config):
    return LinearRegression()

def build_ridge_regression(config):
    return Ridge(
        alpha=config.get("alpha", 1.0)  
    )

def build_lasso_regression(config):
    return Lasso(
        alpha=config.get("alpha", 1.0)
    )