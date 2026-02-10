import numpy as np
try:
    from ..diagnosis.rules import diagnosis_rules
    from ..models.builder import build_model
    from ..metrics.metrics import classification_error, regression_error
    from .sampling import sampling
except ImportError:
    from diagnosis.rules import diagnosis_rules
    from models.builder import build_model
    from metrics.metrics import classification_error, regression_error
    from curves.sampling import sampling
import joblib
import os


def _get_class_scores(model, X, is_nn=False):
    if is_nn:
        return model.predict(X, verbose=0).reshape(-1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores.reshape(-1)

    return None


def learning_curve(config, X_train, y_train, X_val, y_val):

    size = config["learning_curve"]["sizes"]

    val_errors = []
    train_errors = []
    final_outputs = {}

    for frac in size:
        n = int(len(X_train) * frac)

        X_sub, y_sub = sampling(X_train, y_train, n)

        model = build_model(
            model_type=config["model_type"],
            config=config.get("model", {}),
            input_dim=X_train.shape[1],
        )
        
        if config["model_type"] == "nn_classifier":
            epochs = config["model"].get("epochs", 20)
            batch_size = config["model"].get("batch_size", 32)
            model.fit(X_sub, y_sub, epochs=epochs, batch_size=batch_size, verbose=0)

            y_train_score = _get_class_scores(model, X_sub, is_nn=True)
            y_val_score = _get_class_scores(model, X_val, is_nn=True)
            y_train_pred = (y_train_score > 0.5).astype(int).reshape(-1)
            y_val_pred = (y_val_score > 0.5).astype(int).reshape(-1)

        else:
            model.fit(X_sub, y_sub)
            y_train_pred = model.predict(X_sub)
            y_val_pred   = model.predict(X_val)
            y_train_score = _get_class_scores(model, X_sub)
            y_val_score = _get_class_scores(model, X_val)

        if config["task_type"] == "classification":
            train_errors.append(classification_error(y_sub, y_train_pred))
            val_errors.append(classification_error(y_val, y_val_pred))

        else:
            train_errors.append(regression_error(y_sub, y_train_pred))
            val_errors.append(regression_error(y_val, y_val_pred))

        if frac == size[-1]:
            final_outputs = {
                "model": model,
                "y_train_true": y_sub,
                "y_train_pred": y_train_pred,
                "y_train_score": y_train_score,
                "y_val_true": y_val,
                "y_val_pred": y_val_pred,
                "y_val_score": y_val_score
            }

    diagnosis = diagnosis_rules(train_errors[-1], val_errors[-1], config)
        
    if diagnosis["label"] == "balanced":
        print("Model training and evaluation pipeline appears to be functioning correctly.")
        
        if config["model_type"].startswith("nn"):
            save_keras_model(model, path=r"D:\\WORK\\projects\\Bias_Variance_DiagnosticTool\\outputs\\models\\final_model.h5")

        else:
            save_sklearn_model(model,path=r"D:\\WORK\\projects\\Bias_Variance_DiagnosticTool\\outputs\\models\\final_model.pkl")

    return size, train_errors, val_errors, final_outputs

def save_sklearn_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def save_keras_model(model, path):
    os.makedirs(path, exist_ok=True)
    model.save(path)
           
       