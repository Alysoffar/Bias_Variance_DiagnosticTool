import numpy as np
try:
    from ..models.builder import build_model
    from ..metrics.metrics import classification_error, regression_error
except ImportError:
    from models.builder import build_model
    from metrics.metrics import classification_error, regression_error


def check_split_overlap(X_train, X_val, max_check_rows=2000):
    """
    Detect exact duplicate rows between train and val (possible leakage).
    We only check up to max_check_rows for speed.
    """
    Xt = X_train[:max_check_rows]
    Xv = X_val[:max_check_rows]

    # hash rows by bytes
    train_hashes = set(map(bytes, np.ascontiguousarray(Xt).tobytes().split(b'\x00')[:0]))  # placeholder
    # simpler reliable hashing:
    train_hashes = set(hash(row.tobytes()) for row in np.ascontiguousarray(Xt))
    val_hashes = set(hash(row.tobytes()) for row in np.ascontiguousarray(Xv))

    overlap = train_hashes.intersection(val_hashes)
    if len(overlap) > 0:
        return f"WARNING: Train/val may overlap (found {len(overlap)} duplicate rows). Possible data leakage."
    return "No train/val overlap detected."


def label_shuffle_check(config, X_train, y_train, X_val, y_val):
    """
    Shuffle y_train and verify validation performance becomes bad.
    If it stays good, you likely have leakage or evaluation bug.
    """
    rng = np.random.default_rng(config.get("random_seed", 42))
    y_shuffled = np.array(y_train).copy()
    rng.shuffle(y_shuffled)

    model = build_model(
        model_type=config["model_type"],
        config=config.get("model", {}),
        input_dim=X_train.shape[1],
    )

    # train
    if config["model_type"] == "nn":
        epochs = config["model"].get("epochs", 10)
        batch_size = config["model"].get("batch_size", 32)
        model.fit(X_train, y_shuffled, epochs=epochs, batch_size=batch_size, verbose=0)

        if config["task_type"] == "classification":
            y_val_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int).reshape(-1)
        else:
            y_val_pred = model.predict(X_val, verbose=0).reshape(-1)
    else:
        model.fit(X_train, y_shuffled)
        y_val_pred = model.predict(X_val)

    # evaluate
    if config["task_type"] == "classification":
        err = classification_error(y_val, y_val_pred)
        # If error is still low, suspicious. Threshold depends on dataset; keep it simple:
        if err < 0.25:
            return "WARNING: Label-shuffle test still achieved low validation error. Possible leakage or evaluation bug."
    else:
        mse = regression_error(y_val, y_val_pred)
        # Hard to threshold MSE universally; just warn if it's extremely small
        if mse < 1e-6:
            return "WARNING: Label-shuffle test produced extremely low MSE. Possible leakage or bug."

    return "Label-shuffle test passed without warnings."


def overfit_tiny_batch_check(config, X_train, y_train, n_samples=30):
    """
    Train on a tiny set and see if training error gets very low.
    If not, something in training/pipeline might be wrong.
    """
    # For classification, check if we have at least 2 classes in the sample
    if config["task_type"] == "classification":
        # For imbalanced datasets, try to get samples from multiple classes
        unique_classes = np.unique(y_train)
        if len(unique_classes) > 1:
            # Stratified sampling to ensure multiple classes
            samples_per_class = max(2, n_samples // len(unique_classes))
            indices = []
            for cls in unique_classes:
                cls_indices = np.where(y_train == cls)[0]
                if len(cls_indices) >= samples_per_class:
                    indices.extend(cls_indices[:samples_per_class])
            
            if len(indices) < 10:  # Not enough samples with multiple classes
                return "SKIPPED: Dataset too imbalanced for tiny batch overfitting check."
            
            X_small = X_train[indices]
            y_small = y_train[indices]
        else:
            return "SKIPPED: Only one class in training data."
    else:
        X_small = X_train[:n_samples]
        y_small = y_train[:n_samples]

    model = build_model(
        model_type=config["model_type"],
        config=config.get("model", {}),
        input_dim=X_train.shape[1],
    )

    if config["model_type"] == "nn":
        epochs = max(30, config["model"].get("epochs", 20))
        batch_size = min(config["model"].get("batch_size", 32), n_samples)
        model.fit(X_small, y_small, epochs=epochs, batch_size=batch_size, verbose=0)

        if config["task_type"] == "classification":
            y_pred = (model.predict(X_small, verbose=0) > 0.5).astype(int).reshape(-1)
            err = classification_error(y_small, y_pred)
            if err > 0.05:
                return "WARNING: Model failed to overfit tiny batch. Check labels, preprocessing, or training loop."
        else:
            y_pred = model.predict(X_small, verbose=0).reshape(-1)
            mse = regression_error(y_small, y_pred)
            if mse > 1e-3:
                return "WARNING: Model failed to fit tiny regression batch. Check pipeline/training."
    else:
        model.fit(X_small, y_small)
        y_pred = model.predict(X_small)

        if config["task_type"] == "classification":
            err = classification_error(y_small, y_pred)
            if err > 0.05:
                return "WARNING: Model failed to overfit tiny batch. Check labels/features."
        else:
            mse = regression_error(y_small, y_pred)
            if mse > 1e-3:
                return "WARNING: Model failed to fit tiny regression batch. Check labels/features."

    return "No issues detected in tiny batch overfitting test."


def run_sanity_checks(config, X_train, y_train, X_val, y_val):
    warnings = []

    msg = check_split_overlap(X_train, X_val)
    if msg:
        warnings.append(msg)

    msg = label_shuffle_check(config, X_train, y_train, X_val, y_val)
    if msg:
        warnings.append(msg)

    msg = overfit_tiny_batch_check(config, X_train, y_train)
    if msg:
        warnings.append(msg)

    return warnings or []