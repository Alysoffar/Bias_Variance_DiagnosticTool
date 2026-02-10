import os
import json
from datetime import datetime


def report_findings(
    diagnosis: str,
    train_errors: list,
    val_errors: list,
    sizes: list,
    recommendation: str,
    out_path: str = None,
    warnings: list = None,
    data_errors: list = None,
    final_metrics: dict = None
):
    report = {
        "timestamp": datetime.now().isoformat(),
        "diagnosis": diagnosis,
        "learning_curve": {
            "sizes": sizes,
            "train_errors": train_errors,
            "val_errors": val_errors
        },
        "final_metrics": final_metrics,
        "recommendation": recommendation,
        "warnings": warnings or None,
        "data_errors": data_errors or None
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=4)
    else:
        print(json.dumps(report, indent=4))
    