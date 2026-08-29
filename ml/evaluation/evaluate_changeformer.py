from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = PROJECT_ROOT / "apps" / "api"
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

from app.models.changeformer import ChangeFormerService
from app.models.manager import detect_device
from app.remote_sensing.preprocessing import load_visual
from metrics import binary_segmentation_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the configured ChangeFormer checkpoint on LEVIR-style pairs.")
    parser.add_argument("--dataset", default="models/changeformer/source/samples_LEVIR")
    parser.add_argument("--output", default="evaluation-results/change-detection.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    dataset = Path(args.dataset)
    names = [line.strip() for line in (dataset / "list" / "demo.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    service = ChangeFormerService((PROJECT_ROOT / "models").resolve(), detect_device(args.device))
    records = []
    try:
        for name in names:
            before = load_visual(dataset / "A" / name)
            after = load_visual(dataset / "B" / name)
            target = np.asarray(Image.open(dataset / "label" / name).convert("L")) >= 128
            probability = service.predict(before, after)
            records.append(binary_segmentation_metrics(probability >= args.threshold, target))
    finally:
        service.unload()
    aggregate = {key: float(np.mean([record[key] for record in records])) for key in records[0]}
    result = {
        "task_id": "change_detection",
        "dataset": "LEVIR-CD official ChangeFormer demo samples",
        "model": f"{service.model_name} {service.model_version}",
        "split": "official demo list",
        "sample_count": len(records),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": "models/changeformer/best_ckpt.pt",
        "parameters": {"threshold": args.threshold, "tile_size": service.input_size},
        "metrics": aggregate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
