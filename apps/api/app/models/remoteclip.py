from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from app.remote_sensing.tiling import TileWindow, tile_windows


EUROSAT_TO_BROAD = {
    "AnnualCrop": "agriculture",
    "Forest": "vegetation",
    "HerbaceousVegetation": "vegetation",
    "Highway": "built_up",
    "Industrial": "built_up",
    "Pasture": "vegetation",
    "PermanentCrop": "agriculture",
    "Residential": "built_up",
    "River": "water",
    "SeaLake": "water",
}

BROAD_PROMPTS = {
    "water": "a satellite image patch dominated by a river, lake, reservoir, or open water",
    "vegetation": "a satellite image patch dominated by forest, grassland, pasture, or healthy vegetation",
    "built_up": "a satellite image patch dominated by buildings, roads, industrial areas, or dense settlement",
    "bare_land": "a satellite image patch dominated by exposed soil, rock, sand, or barren land",
    "agriculture": "a satellite image patch dominated by agricultural fields, annual crops, or permanent crops",
}

SCENE_CAPTIONS = (
    "A satellite image of dense urban and industrial development with roads and buildings.",
    "A satellite image of residential land mixed with roads and small vegetation patches.",
    "A satellite image of agricultural fields and cultivated cropland.",
    "A satellite image of forest, grassland, and other dense vegetation.",
    "A satellite image of a river, lake, reservoir, or coastal water body.",
    "A satellite image of bare land, exposed soil, rock, or sparse vegetation.",
    "A satellite image of mixed land cover containing built-up, vegetation, and water areas.",
)
SCENE_CAPTION_CLASSES = ("built_up", "built_up", "agriculture", "vegetation", "water", "bare_land", "mixed")


class AdapterHead(nn.Module):
    def __init__(self, feature_dim: int, bottleneck_dim: int, labels: list[str]):
        super().__init__()
        self.labels = labels
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, len(labels))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features + self.adapter(features))


@dataclass(frozen=True)
class LearnedSceneResult:
    answer: str
    score: float
    model_name: str
    label: str | None = None


class RemoteCLIPService:
    """Official RemoteCLIP RN50 inference plus an optional trained SatQuery adapter head."""

    model_name = "RemoteCLIP RN50"

    def __init__(self, model_dir: Path, device: str, unload_after_request: bool = False):
        self.checkpoint = model_dir / "remoteclip" / "RemoteCLIP-RN50.pt"
        self.adapter_checkpoint = model_dir / "satquery-adapter" / "best.pt"
        self.device = device
        self.unload_after_request = unload_after_request
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.adapter: AdapterHead | None = None

    @property
    def available(self) -> bool:
        return self.checkpoint.exists()

    @property
    def adapter_available(self) -> bool:
        return self.adapter_checkpoint.exists()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.loaded:
            return
        if not self.available:
            raise FileNotFoundError(f"RemoteCLIP checkpoint missing: {self.checkpoint}")
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms("RN50", pretrained=str(self.checkpoint))
        model.requires_grad_(False).eval().to(self.device)
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer("RN50")
        if self.adapter_available:
            checkpoint = torch.load(self.adapter_checkpoint, map_location="cpu", weights_only=False)
            labels = list(checkpoint["labels"])
            feature_dim = int(checkpoint.get("feature_dim", model.visual.output_dim))
            bottleneck_dim = int(checkpoint["config"]["bottleneck_dim"])
            adapter = AdapterHead(feature_dim, bottleneck_dim, labels)
            adapter.adapter.load_state_dict(checkpoint["adapter"])
            adapter.classifier.load_state_dict(checkpoint["classifier"])
            self.adapter = adapter.requires_grad_(False).eval().to(self.device)

    def unload(self) -> None:
        self.model = self.preprocess = self.tokenizer = self.adapter = None
        gc.collect()
        if self.device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def health(self) -> dict[str, object]:
        return {
            "ready": self.available,
            "loaded": self.loaded,
            "adapter_ready": self.adapter_available,
            "device": self.device,
        }

    def metadata(self) -> dict[str, object]:
        return {
            "id": "remoteclip_encoder",
            "name": self.model_name,
            "version": "official-2023",
            "tasks": ["embedding", "classification", "vqa", "caption", "grounding"],
        }

    @staticmethod
    def _pil(image: np.ndarray) -> Image.Image:
        return Image.fromarray(image.astype(np.uint8))

    def _image_features(
        self,
        images: list[np.ndarray],
        batch_size: int = 24,
        *,
        normalize: bool = True,
    ) -> torch.Tensor:
        self.load()
        assert self.model is not None and self.preprocess is not None
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                pixels = torch.stack([self.preprocess(self._pil(image)) for image in images[start : start + batch_size]]).to(self.device)
                features = self.model.encode_image(pixels)
                if normalize:
                    features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                outputs.append(features)
        return torch.cat(outputs)

    def _text_features(self, prompts: list[str]) -> torch.Tensor:
        self.load()
        assert self.model is not None and self.tokenizer is not None
        with torch.inference_mode():
            tokens = self.tokenizer(prompts).to(self.device)
            features = self.model.encode_text(tokens)
            return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    def classify(self, image: np.ndarray, prompts: list[str]) -> np.ndarray:
        image_features = self._image_features([image])
        text_features = self._text_features(prompts)
        return (100 * image_features @ text_features.T).softmax(dim=-1)[0].detach().cpu().numpy()

    @staticmethod
    def _patches(image: np.ndarray, size: int = 224, overlap: int = 64) -> tuple[list[TileWindow], list[np.ndarray]]:
        windows = tile_windows(image.shape[0], image.shape[1], min(size, max(image.shape[:2])), min(overlap, max(0, min(size, max(image.shape[:2])) - 1)))
        return windows, [image[window.rows, window.columns] for window in windows]

    @staticmethod
    def _stitch_scores(image: np.ndarray, windows: list[TileWindow], scores: np.ndarray) -> dict[str, np.ndarray]:
        totals = np.zeros((scores.shape[1], image.shape[0], image.shape[1]), dtype=np.float32)
        weights = np.zeros(image.shape[:2], dtype=np.float32)
        for window, score in zip(windows, scores):
            totals[:, window.rows, window.columns] += score[:, None, None]
            weights[window.rows, window.columns] += 1
        totals /= np.maximum(weights[None, ...], 1e-6)
        return {label: totals[index] for index, label in enumerate(BROAD_PROMPTS)}

    def landcover_probabilities(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], str]:
        windows, patches = self._patches(image)
        image_features = self._image_features(patches, normalize=self.adapter is None)
        if self.adapter is not None:
            with torch.inference_mode():
                class_scores = self.adapter(image_features).softmax(dim=-1).detach().cpu().numpy()
            broad = np.zeros((len(BROAD_PROMPTS), len(patches)), dtype=np.float32)
            broad_labels = list(BROAD_PROMPTS)
            for class_index, label in enumerate(self.adapter.labels):
                broad_label = EUROSAT_TO_BROAD.get(label)
                if broad_label:
                    broad[broad_labels.index(broad_label)] += class_scores[:, class_index]
            broad /= np.maximum(broad.sum(axis=0, keepdims=True), 1e-6)
            normalized_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            text_features = self._text_features(list(BROAD_PROMPTS.values()))
            zero_shot = (100 * normalized_features @ text_features.T).softmax(dim=-1).detach().cpu().numpy()
            # EuroSAT has no barren-land class. Retain the strong trained head for its
            # supported classes while blending RemoteCLIP zero-shot evidence so every
            # advertised broad class remains reachable.
            combined = 0.8 * broad.T + 0.2 * zero_shot
            combined /= np.maximum(combined.sum(axis=1, keepdims=True), 1e-6)
            return self._stitch_scores(image, windows, combined), "SatQuery EuroSAT Adapter + RemoteCLIP RN50"
        text_features = self._text_features(list(BROAD_PROMPTS.values()))
        scores = (100 * image_features @ text_features.T).softmax(dim=-1).detach().cpu().numpy()
        return self._stitch_scores(image, windows, scores), self.model_name

    def answer(self, image: np.ndarray, query: str, target: str | None, caption: bool = False) -> LearnedSceneResult:
        if caption or target is None:
            scores = self.classify(image, list(SCENE_CAPTIONS))
            index = int(scores.argmax())
            answer = SCENE_CAPTIONS[index]
            if not caption:
                answer = f"RemoteCLIP finds the closest learned scene description to be: {answer}"
            return LearnedSceneResult(answer, float(scores[index]), self.model_name, SCENE_CAPTION_CLASSES[index])
        readable = target.replace("_", " ")
        prompts = [
            f"A satellite image containing a clearly visible {readable} area.",
            f"A satellite image without a visible {readable} area.",
        ]
        scores = self.classify(image, prompts)
        present = bool(scores[0] >= scores[1])
        answer = f"{'Yes' if present else 'No'}—the learned RemoteCLIP comparison {'supports' if present else 'does not support'} visible {readable} in this scene ({scores[0]:.0%} positive evidence)."
        return LearnedSceneResult(answer, float(max(scores)), self.model_name, target)

    def ground(self, image: np.ndarray, query: str, target: str | None) -> tuple[np.ndarray, float]:
        description = (target or query).replace("_", " ")
        windows, patches = self._patches(image)
        image_features = self._image_features(patches)
        text_features = self._text_features([
            f"A satellite image patch containing {description}.",
            f"A satellite image patch without {description}.",
        ])
        scores = (100 * image_features @ text_features.T).softmax(dim=-1)[:, 0].detach().cpu().numpy()
        stitched = self._stitch_scores(image, windows, np.stack([scores] * len(BROAD_PROMPTS), axis=1))["water"]
        return stitched, float(scores.max())
