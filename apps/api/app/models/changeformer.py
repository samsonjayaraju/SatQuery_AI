from __future__ import annotations

import gc
import importlib
import sys
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from app.remote_sensing.tiling import tile_windows


class ChangeFormerService:
    """Local adapter for the official ChangeFormer V6 LEVIR checkpoint."""

    model_name = "ChangeFormer V6 LEVIR"
    model_version = "official-v0.1.0"
    input_size = 256
    required_sources = (
        "ChangeFormer.py",
        "ChangeFormerBaseNetworks.py",
        "help_funcs.py",
        "pixel_shuffel_up.py",
    )

    def __init__(self, model_dir: Path, device: str, unload_after_request: bool = False):
        self.checkpoint = model_dir / "changeformer" / "best_ckpt.pt"
        self.source_root = model_dir / "changeformer" / "source"
        self.device = device
        self.unload_after_request = unload_after_request
        self.model = None
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        source_directory = self.source_root / "models"
        return self.checkpoint.exists() and all((source_directory / name).exists() for name in self.required_sources)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def _model_class(self):
        source = str(self.source_root.resolve())
        if source not in sys.path:
            sys.path.insert(0, source)
        existing = sys.modules.get("models")
        if existing is not None:
            raw_package_file = getattr(existing, "__file__", None)
            expected = (self.source_root / "models" / "__init__.py").resolve()
            if raw_package_file and Path(raw_package_file).resolve() != expected:
                package_file = Path(raw_package_file)
                raise RuntimeError(f"A conflicting top-level 'models' package is already loaded from {package_file}")
        module = importlib.import_module("models.ChangeFormer")
        return module.ChangeFormerV6

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            if not self.available:
                raise FileNotFoundError(
                    "ChangeFormer requires models/changeformer/best_ckpt.pt and the official source under "
                    "models/changeformer/source."
                )
            model_class = self._model_class()
            model = model_class(embed_dim=256)
            checkpoint = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["model_G_state_dict"], strict=True)
            self.model = model.requires_grad_(False).eval().to(self.device)

    def unload(self) -> None:
        with self._lock:
            self.model = None
            gc.collect()
            if self.device == "mps" and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

    def health(self) -> dict[str, object]:
        return {"ready": self.available, "loaded": self.loaded, "device": self.device}

    def metadata(self) -> dict[str, object]:
        return {
            "id": "changeformer",
            "name": self.model_name,
            "version": self.model_version,
            "tasks": ["change_detection"],
        }

    @staticmethod
    def _tensor(image: np.ndarray) -> torch.Tensor:
        resized = Image.fromarray(image.astype(np.uint8)).resize(
            (ChangeFormerService.input_size, ChangeFormerService.input_size),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return (tensor - 0.5) / 0.5

    def predict(self, before: np.ndarray, after: np.ndarray, overlap: int = 32) -> np.ndarray:
        if before.shape != after.shape:
            raise ValueError("ChangeFormer inputs must be aligned and have identical shapes")
        self.load()
        assert self.model is not None
        windows = tile_windows(before.shape[0], before.shape[1], self.input_size, overlap)
        total = np.zeros(before.shape[:2], dtype=np.float32)
        weights = np.zeros(before.shape[:2], dtype=np.float32)
        batch_size = 4 if self.device == "cuda" else 2 if self.device == "mps" else 1
        with self._lock, torch.inference_mode():
            for start in range(0, len(windows), batch_size):
                batch = windows[start : start + batch_size]
                before_tensor = torch.stack([self._tensor(before[item.rows, item.columns]) for item in batch]).to(self.device)
                after_tensor = torch.stack([self._tensor(after[item.rows, item.columns]) for item in batch]).to(self.device)
                outputs = self.model(before_tensor, after_tensor)
                logits = outputs[-1] if isinstance(outputs, (list, tuple)) else outputs
                probabilities = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                for item, probability in zip(batch, probabilities):
                    restored = np.asarray(
                        Image.fromarray(probability.astype(np.float32)).resize(
                            (item.width, item.height), Image.Resampling.BILINEAR
                        ),
                        dtype=np.float32,
                    )
                    total[item.rows, item.columns] += restored
                    weights[item.rows, item.columns] += 1.0
        return np.clip(total / np.maximum(weights, 1e-6), 0.0, 1.0)
