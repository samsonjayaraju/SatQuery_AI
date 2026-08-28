from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BigEarthNetManifestDataset(Dataset):
    """Manifest-backed BigEarthNet subset; no dataset files are bundled or downloaded."""

    def __init__(self, manifest: Path, label_names: list[str], image_size: int = 224):
        self.root = manifest.parent
        self.label_names = label_names
        self.label_index = {label: index for index, label in enumerate(label_names)}
        self.image_size = image_size
        self.records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"No samples found in {manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        path = Path(record["image"])
        if not path.is_absolute():
            path = self.root / path
        image = Image.open(path).convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        pixels = (pixels - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array([0.26862954, 0.26130258, 0.27577711])
        tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).float()
        labels = torch.zeros(len(self.label_names), dtype=torch.float32)
        for label in record.get("labels", []):
            if label in self.label_index:
                labels[self.label_index[label]] = 1
        return tensor, labels


def collect_labels(manifest: Path) -> list[str]:
    labels: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            labels.update(json.loads(line).get("labels", []))
    return sorted(labels)
