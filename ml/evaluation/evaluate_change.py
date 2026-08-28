from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from metrics import binary_segmentation_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate binary change masks stored as matching .npy files.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", default="evaluation-results/change-detection.json")
    args = parser.parse_args()
    prediction_dir, target_dir = Path(args.predictions), Path(args.targets)
    records = []
    for prediction_path in sorted(prediction_dir.glob("*.npy")):
        target_path = target_dir / prediction_path.name
        if target_path.exists():
            records.append(binary_segmentation_metrics(np.load(prediction_path), np.load(target_path)))
    if not records:
        raise SystemExit("No matching .npy prediction/target pairs found.")
    aggregate = {key: float(np.mean([record[key] for record in records])) for key in records[0]}
    result = {"task": "change_detection", "samples": len(records), "metrics": aggregate}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
