"""
Run benchmark experiments across datasets and models.
Outputs LaTeX tables for performance and runtime.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diagnoser.data.loader import load_data, detect_task_type
from diagnoser.data.processing import clean_data
from diagnoser.metrics.metrics import classification_metrics, classification_error


def _get_scores(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores.reshape(-1)
    return None


def _evaluate_model(model, X_train, y_train, X_val, y_val):
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    start_inf = time.perf_counter()
    y_val_pred = model.predict(X_val)
    y_val_score = _get_scores(model, X_val)
    inf_time = time.perf_counter() - start_inf

    metrics = classification_metrics(y_val, y_val_pred, y_val_score)
    metrics["train_time_sec"] = train_time
    metrics["inference_time_per_sample_ms"] = (inf_time / max(len(X_val), 1)) * 1000.0

    # train error for diagnostics
    y_train_pred = model.predict(X_train)
    metrics["train_error"] = classification_error(y_train, y_train_pred)
    metrics["val_error"] = metrics.get("error")

    return metrics


def _diagselect(results, gap_threshold=0.1, high_error_threshold=0.3):
    # Choose best model among those diagnosed as balanced
    balanced = []
    for name, metrics in results.items():
        gap = (metrics.get("val_error") - metrics.get("train_error"))
        if metrics.get("train_error") is None or metrics.get("val_error") is None:
            continue
        if metrics.get("train_error") <= high_error_threshold and gap <= gap_threshold:
            balanced.append((name, metrics))

    if balanced:
        # pick best PR-AUC among balanced
        balanced.sort(key=lambda x: (x[1].get("pr_auc") or -1), reverse=True)
        return balanced[0][0]

    # fallback: best PR-AUC overall
    best = sorted(results.items(), key=lambda x: (x[1].get("pr_auc") or -1), reverse=True)
    return best[0][0]


def _to_dataframe(X_raw, y_raw):
    if isinstance(X_raw, np.ndarray):
        X_raw = pd.DataFrame(X_raw)
    if isinstance(y_raw, np.ndarray):
        y_raw = pd.Series(y_raw)
    return X_raw, y_raw


def _maybe_subsample(X, y, max_samples, seed):
    if max_samples is None or len(y) <= max_samples:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X,
        y,
        train_size=max_samples,
        random_state=seed,
        stratify=y
    )
    return X_sub, y_sub


def run_dataset(name, X_raw, y_raw, config, seeds):
    X_raw, y_raw = _to_dataframe(X_raw, y_raw)
    all_results = {}
    for seed in seeds:
        X, y = clean_data(X_raw.copy(), y_raw.copy(), config)
        if hasattr(X, "values"):
            X = X.values
        if hasattr(y, "values"):
            y = y.values

        X, y = _maybe_subsample(X, y, config.get("max_samples"), seed)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        n_train = len(y_train)
        rf_trees = 50 if n_train > 50000 else 100
        models = {
            "logreg": LogisticRegression(max_iter=500, random_state=seed),
            "svm_linear": LinearSVC(random_state=seed),
            "rf": RandomForestClassifier(n_estimators=rf_trees, random_state=seed),
            "dt": DecisionTreeClassifier(random_state=seed)
        }

        seed_results = {}
        for model_name, model in models.items():
            seed_results[model_name] = _evaluate_model(model, X_train, y_train, X_val, y_val)

        # proposed method: diagnosis-guided model selection
        selected = _diagselect(seed_results)
        seed_results["proposed_diagselect"] = seed_results[selected].copy()
        seed_results["proposed_diagselect"]["selected_model"] = selected

        all_results[seed] = seed_results

    # aggregate
    aggregate = {}
    for model_name in all_results[seeds[0]].keys():
        metrics_list = [all_results[s][model_name] for s in seeds]
        agg = {}
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate",
                    "train_time_sec", "inference_time_per_sample_ms"]:
            values = [m.get(key) for m in metrics_list]
            values = [v for v in values if v is not None]
            agg[key] = {
                "mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None
            }
        if model_name == "proposed_diagselect":
            agg["selected_model"] = metrics_list[0].get("selected_model")
        aggregate[model_name] = agg

    return aggregate


def to_latex_performance_table(results_by_dataset):
    header = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{llrrrrr}\n"
        "\\toprule\n"
        "Dataset & Model & Acc & F1 & ROC-AUC & PR-AUC & FPR \\\\ \n"
        "\\midrule\n"
    )
    rows = []
    for dataset_name, results in results_by_dataset.items():
        for model_name, metrics in results.items():
            rows.append(
                f"{dataset_name} & {model_name} & "
                f"{metrics['accuracy']['mean']:.4f}±{metrics['accuracy']['std']:.4f} & "
                f"{metrics['f1']['mean']:.4f}±{metrics['f1']['std']:.4f} & "
                f"{metrics['roc_auc']['mean']:.4f}±{metrics['roc_auc']['std']:.4f} & "
                f"{metrics['pr_auc']['mean']:.4f}±{metrics['pr_auc']['std']:.4f} & "
                f"{metrics['false_positive_rate']['mean']:.4f}±{metrics['false_positive_rate']['std']:.4f} \\\\"
            )
    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Benchmark performance (mean±std over seeds).}\\n"
        "\\label{tab:benchmark_performance}\\n"
        "\\end{table}\n"
    )
    return header + "\n".join(rows) + "\n" + footer


def to_latex_runtime_table(results_by_dataset):
    header = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{llrr}\n"
        "\\toprule\n"
        "Dataset & Model & Train (s) & Inference (ms/sample) \\\\ \n"
        "\\midrule\n"
    )
    rows = []
    for dataset_name, results in results_by_dataset.items():
        for model_name, metrics in results.items():
            rows.append(
                f"{dataset_name} & {model_name} & "
                f"{metrics['train_time_sec']['mean']:.4f}±{metrics['train_time_sec']['std']:.4f} & "
                f"{metrics['inference_time_per_sample_ms']['mean']:.4f}±{metrics['inference_time_per_sample_ms']['std']:.4f} \\\\"
            )
    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Runtime comparison (mean±std over seeds).}\\n"
        "\\label{tab:benchmark_runtime}\\n"
        "\\end{table}\n"
    )
    return header + "\n".join(rows) + "\n" + footer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seeds = [args.seed, args.seed + 10, args.seed + 20]

    results_by_dataset = {}

    # Dataset 1: Breast cancer (sklearn)
    X, y = load_breast_cancer(return_X_y=True)
    config = {
        "remove_duplicates": True,
        "handle_missing": True,
        "missing_value_strategy": "mean",
        "encode_categorical": True,
        "encoding_strategy": "onehot",
        "encode_target": True,
        "scale_features": True,
        "scaler_type": "standard",
        "drop_columns": []
    }
    results_by_dataset["breast_cancer"] = run_dataset("breast_cancer", X, y, config, seeds)

    # Dataset 2: Credit card fraud (if available)
    credit_path = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
    if credit_path.exists():
        X_cc, y_cc = load_data(str(credit_path), "Class")
        config_cc = config.copy()
        config_cc["drop_columns"] = ["Time"]
        config_cc["max_samples"] = 50000
        results_by_dataset["creditcard"] = run_dataset("creditcard", X_cc, y_cc, config_cc, seeds)

    # Optional Dataset 3: Adult income (OpenML)
    try:
        from sklearn.datasets import fetch_openml
        X_adult, y_adult = fetch_openml(data_id=1590, as_frame=True, return_X_y=True)
        config_adult = config.copy()
        results_by_dataset["adult"] = run_dataset("adult", X_adult, y_adult, config_adult, seeds)
    except Exception:
        pass

    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    tables_dir = reports_dir / "tables"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    with open(reports_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results_by_dataset, f, indent=4)

    perf_table = to_latex_performance_table(results_by_dataset)
    runtime_table = to_latex_runtime_table(results_by_dataset)

    with open(tables_dir / "benchmark_performance.tex", "w", encoding="utf-8") as f:
        f.write(perf_table)

    with open(tables_dir / "benchmark_runtime.tex", "w", encoding="utf-8") as f:
        f.write(runtime_table)

    print("Benchmark results saved to outputs/reports/")


if __name__ == "__main__":
    main()
