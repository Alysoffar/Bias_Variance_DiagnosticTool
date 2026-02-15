import json
import os
import shutil
import sys
from pathlib import Path

# Ensure src is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from diagnoser.pipeline import load_config, run  # noqa: E402


OUTPUT_BASE = REPO_ROOT / "outputs" / "eval_builtin"
DEFAULT_FIG = REPO_ROOT / "outputs" / "figures" / "learning_curve.png"
DEFAULT_REPORT = REPO_ROOT / "outputs" / "reports" / "report.json"


def _copy_artifacts(dataset_name: str, model_name: str) -> dict:
    target_dir = OUTPUT_BASE / dataset_name / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    figure_path = target_dir / "learning_curve.png"
    report_path = target_dir / "report.json"

    if DEFAULT_FIG.exists():
        shutil.copy2(DEFAULT_FIG, figure_path)
    if DEFAULT_REPORT.exists():
        shutil.copy2(DEFAULT_REPORT, report_path)

    diagnosis = None
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)
        diagnosis = report.get("diagnosis", {}).get("label")

    return {
        "dataset": dataset_name,
        "model": model_name,
        "figure_path": str(figure_path),
        "report_path": str(report_path),
        "diagnosis": diagnosis,
    }


def _run_model(model_name: str, model_config_path: str, task_type: str) -> dict:
    config = load_config(str(REPO_ROOT / "configs" / "default.yaml"), model_config_path)
    config["model_type"] = model_name
    config["task_type"] = task_type
    config["data_path"] = None
    config["target_column"] = None

    run(config)

    dataset_name = "breast_cancer" if task_type == "classification" else "diabetes"
    return _copy_artifacts(dataset_name, model_name)


def main() -> None:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    results = []
    failures = []

    classification_models = [
        ("tree_classifier", str(REPO_ROOT / "configs" / "tree.yaml")),
        ("logistic_regression", str(REPO_ROOT / "configs" / "logistic_reg.yaml")),
        ("nn_classifier", str(REPO_ROOT / "configs" / "nn.yaml")),
    ]

    regression_models = [
        ("linear_regression", str(REPO_ROOT / "configs" / "default.yaml")),
        ("ridge_regression", str(REPO_ROOT / "configs" / "ridge.yaml")),
        ("lasso_regression", str(REPO_ROOT / "configs" / "lasso.yaml")),
    ]

    for model_name, model_config_path in classification_models:
        try:
            results.append(_run_model(model_name, model_config_path, "classification"))
        except Exception as exc:
            failures.append({"model": model_name, "error": str(exc)})

    for model_name, model_config_path in regression_models:
        try:
            results.append(_run_model(model_name, model_config_path, "regression"))
        except Exception as exc:
            failures.append({"model": model_name, "error": str(exc)})

    summary = {
        "results": results,
        "failures": failures,
    }

    summary_path = OUTPUT_BASE / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
