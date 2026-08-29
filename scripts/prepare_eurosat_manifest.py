from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a balanced SatQuery training manifest from official EuroSAT RGB files.")
    parser.add_argument("--dataset", type=Path, default=Path("datasets/EuroSAT_RGB"))
    parser.add_argument("--output", type=Path, default=Path("datasets/EuroSAT_RGB/samples.jsonl"))
    parser.add_argument("--max-samples", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=26167)
    args = parser.parse_args()

    class_root = args.dataset / "EuroSAT_RGB"
    if not class_root.exists():
        class_root = args.dataset
    classes = sorted(path for path in class_root.iterdir() if path.is_dir())
    if not classes:
        raise SystemExit(f"No EuroSAT class directories found under {class_root}")
    rng = random.Random(args.seed)
    per_class = max(1, args.max_samples // len(classes))
    records = []
    for class_directory in classes:
        images = sorted(class_directory.glob("*.jpg"))
        rng.shuffle(images)
        for image in images[:per_class]:
            records.append({"image": str(image.resolve()), "labels": [class_directory.name], "dataset": "EuroSAT_RGB"})
    rng.shuffle(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.output), "samples": len(records), "classes": len(classes)}))


if __name__ == "__main__":
    main()
