"""
Run single-factor sensitivity analysis and ablation studies.

Outputs:
  - outputs/reports/sensitivity_results.json
  - outputs/reports/ablation_results.json
  - outputs/reports/tables/sensitivity_table.tex
  - outputs/reports/tables/ablation_table.tex
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
from sklearn.model_selection import train_test_split

from diagnoser.pipeline import load_config
from diagnoser.data.loader import load_data, detect_task_type
from diagnoser.data.processing import clean_data
from diagnoser.models.builder import build_model
from diagnoser.metrics.metrics import classification_metrics, regression_error


def _get_class_scores(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.reshape(-1)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return scores.reshape(-1)
    return None


def _subset_training_data(X_train, y_train, fraction, seed, task_type):
    if fraction >= 1.0:
        return X_train, y_train
    stratify = y_train if task_type == "classification" else None
    X_sub, _, y_sub, _ = train_test_split(
        X_train,
        y_train,
        train_size=fraction,
        random_state=seed,
        stratify=stratify
    )
    return X_sub, y_sub


def run_single(config, X_raw, y_raw, task_type, seed=42):
    X, y = clean_data(X_raw.copy(), y_raw.copy(), config)
    if hasattr(X, "values"):
        X = X.values
    if hasattr(y, "values"):
        y = y.values
    stratify = y if task_type == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=stratify
    )

    fraction = config.get("training_subset_fraction", 1.0)
    X_sub, y_sub = _subset_training_data(X_train, y_train, fraction, seed, task_type)

    model = build_model(
        model_type=config["model_type"],
        config=config.get("model", {}),
        input_dim=X_sub.shape[1]
    )

    if config["model_type"] == "nn_classifier":
        raise ValueError("nn_classifier is not supported in this script.")

    model.fit(X_sub, y_sub)

    if task_type == "classification":
        y_val_pred = model.predict(X_val)
        y_val_score = _get_class_scores(model, X_val)
        return {"val": classification_metrics(y_val, y_val_pred, y_val_score)}

    y_val_pred = model.predict(X_val)
    return {"val": {"mse": regression_error(y_val, y_val_pred)}}


def _fmt(value):
    if value is None:
        return "N/A"
    try:
        return f"{value:.4f}"
    except (TypeError, ValueError):
        return "N/A"


def to_latex_table(results, caption, label):
    header = (
        "\\begin{table}[H]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\toprule\n"
        "Setting & AUC-ROC & PR-AUC & F1 & Precision & Recall & FPR \\\\ \n"
        "\\midrule\n"
    )

    rows = []
    for row in results:
        metrics = row["metrics"]["val"]
        rows.append(
            f"{row['name']} & "
            f"{_fmt(metrics.get('roc_auc'))} & "
            f"{_fmt(metrics.get('pr_auc'))} & "
            f"{_fmt(metrics.get('f1'))} & "
            f"{_fmt(metrics.get('precision'))} & "
            f"{_fmt(metrics.get('recall'))} & "
            f"{_fmt(metrics.get('false_positive_rate'))} \\\\"
        )

    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\end{table}\n"
    )

    return header + "\n".join(rows) + "\n" + footer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/raw/creditcard.csv")
    parser.add_argument("--target", type=str, default="Class")
    parser.add_argument("--config", type=str, default="configs/example_creditcard.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_config = load_config("configs/default.yaml", args.config)
    base_config["data_path"] = args.data
    base_config["target_column"] = args.target

    X_raw, y_raw = load_data(args.data, args.target)
    task_type = detect_task_type(y_raw)
    base_config["task_type"] = task_type

    baseline = copy.deepcopy(base_config)
    baseline.update({
        "model_type": "logistic_regression",
        "model": {"C": 1.0, "max_iter": 200, "class_weight": None},
        "scale_features": True,
        "training_subset_fraction": 1.0
    })

    sensitivity_experiments = [
        {"name": "baseline_logreg", "overrides": {}},
        {"name": "algo_tree", "overrides": {"model_type": "tree_classifier", "model": {"max_depth": None, "class_weight": None}}},
        {"name": "no_scaling", "overrides": {"scale_features": False}},
        {"name": "class_weight_balanced", "overrides": {"model": {"C": 1.0, "max_iter": 200, "class_weight": "balanced"}}},
        {"name": "train_frac_0.1", "overrides": {"training_subset_fraction": 0.1}},
        {"name": "train_frac_0.5", "overrides": {"training_subset_fraction": 0.5}},
        {"name": "reg_C_0.1", "overrides": {"model": {"C": 0.1, "max_iter": 200, "class_weight": None}}},
        {"name": "reg_C_10", "overrides": {"model": {"C": 10.0, "max_iter": 200, "class_weight": None}}},
        {"name": "tree_depth_3", "overrides": {"model_type": "tree_classifier", "model": {"max_depth": 3, "class_weight": None}}},
        {"name": "tree_depth_5", "overrides": {"model_type": "tree_classifier", "model": {"max_depth": 5, "class_weight": None}}}
    ]

    ablation_experiments = [
        {"name": "baseline_pipeline", "overrides": {}},
        {"name": "no_scaling", "overrides": {"scale_features": False}},
        {"name": "no_encoding", "overrides": {"encode_categorical": False}},
        {"name": "no_missing_imputation", "overrides": {"handle_missing": False}},
        {"name": "no_duplicate_removal", "overrides": {"remove_duplicates": False}},
        {"name": "no_class_weight", "overrides": {"model": {"C": 1.0, "max_iter": 200, "class_weight": None}}},
        {"name": "class_weight_balanced", "overrides": {"model": {"C": 1.0, "max_iter": 200, "class_weight": "balanced"}}}
    ]

    sensitivity_results = []
    for exp in sensitivity_experiments:
        cfg = copy.deepcopy(baseline)
        cfg.update(exp["overrides"])
        if "model" in exp["overrides"]:
            cfg["model"] = exp["overrides"]["model"]
        metrics = run_single(cfg, X_raw, y_raw, task_type, seed=args.seed)
        sensitivity_results.append({"name": exp["name"], "metrics": metrics})

    ablation_results = []
    for exp in ablation_experiments:
        cfg = copy.deepcopy(baseline)
        cfg.update(exp["overrides"])
        if "model" in exp["overrides"]:
            cfg["model"] = exp["overrides"]["model"]
        metrics = run_single(cfg, X_raw, y_raw, task_type, seed=args.seed)
        ablation_results.append({"name": exp["name"], "metrics": metrics})

    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    tables_dir = reports_dir / "tables"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    with open(reports_dir / "sensitivity_results.json", "w", encoding="utf-8") as f:
        json.dump(sensitivity_results, f, indent=4)

    with open(reports_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=4)

    sensitivity_table = to_latex_table(
        sensitivity_results,
        caption="Single-factor sensitivity analysis on the credit card fraud dataset.",
        label="tab:sensitivity"
    )
    ablation_table = to_latex_table(
        ablation_results,
        caption="Ablation study of pipeline components.",
        label="tab:ablation"
    )

    with open(tables_dir / "sensitivity_table.tex", "w", encoding="utf-8") as f:
        f.write(sensitivity_table)

    with open(tables_dir / "ablation_table.tex", "w", encoding="utf-8") as f:
        f.write(ablation_table)

    paper_tables_dir = PROJECT_ROOT / "paper" / "tables"
    os.makedirs(paper_tables_dir, exist_ok=True)
    with open(paper_tables_dir / "sensitivity_table.tex", "w", encoding="utf-8") as f:
        f.write(sensitivity_table)
    with open(paper_tables_dir / "ablation_table.tex", "w", encoding="utf-8") as f:
        f.write(ablation_table)

    print("Sensitivity and ablation results saved to outputs/reports/")


if __name__ == "__main__":
    main()
