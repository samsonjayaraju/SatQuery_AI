from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.api.schemas.analysis import ModelStatus


@dataclass(frozen=True)
class ModelDefinition:
    id: str
    name: str
    version: str
    tasks: tuple[str, ...]
    modalities: tuple[str, ...]
    checkpoint: str | None
    implementation: str
    enabled: bool = True


DEFAULT_MODELS = (
    ModelDefinition("geochat_vqa", "GeoChat VQA", "interface-v1", ("vqa",), ("optical", "multispectral"), "geochat/model.safetensors", "adapter"),
    ModelDefinition("geochat_caption", "GeoChat Caption", "interface-v1", ("caption",), ("optical", "multispectral"), "geochat/model.safetensors", "adapter"),
    ModelDefinition("geochat_grounding", "GeoChat Grounding", "interface-v1", ("grounding",), ("optical",), "geochat/model.safetensors", "adapter"),
    ModelDefinition("remoteclip_encoder", "RemoteCLIP RN50", "official-2023", ("embedding", "classification", "vqa", "caption", "grounding"), ("optical", "multispectral"), "remoteclip/RemoteCLIP-RN50.pt", "openclip_adapter"),
    ModelDefinition("satquery_adapter", "SatQuery EuroSAT Adapter", "1.0", ("classification", "embedding"), ("optical", "multispectral"), "satquery-adapter/best.pt", "trained_adapter"),
    ModelDefinition("changeformer", "ChangeFormer V6 LEVIR", "official-v0.1.0", ("change_detection",), ("bi_temporal",), "changeformer/best_ckpt.pt", "official_adapter"),
    ModelDefinition("landcover_classifier", "Spectral Land-Cover Baseline", "1.0", ("land_cover",), ("optical", "multispectral"), None, "deterministic_baseline"),
    ModelDefinition("satfusion", "SatFusion", "baseline-v1", ("fusion",), ("optical", "sar"), None, "deterministic_baseline"),
    ModelDefinition("change_reasoner", "Change Reasoner", "1.0", ("change_reasoning",), ("bi_temporal",), None, "deterministic_baseline"),
)


class ModelRegistry:
    def __init__(self, model_dir: Path, device: str):
        self.model_dir = model_dir
        self.device = device
        self._models = {model.id: model for model in DEFAULT_MODELS}
        self._loaded: set[str] = set()

    def list(self) -> list[ModelStatus]:
        result = []
        for model in self._models.values():
            checkpoint_available = model.checkpoint is None or (self.model_dir / model.checkpoint).exists()
            status = "ready" if checkpoint_available and model.enabled else "checkpoint_missing"
            if not model.enabled:
                status = "disabled"
            result.append(
                ModelStatus(
                    id=model.id,
                    name=model.name,
                    version=model.version,
                    status=status,
                    loaded=model.id in self._loaded,
                    checkpoint_available=checkpoint_available,
                    device=self.device,
                    supported_tasks=list(model.tasks),
                    modalities=list(model.modalities),
                    implementation=model.implementation,
                )
            )
        return result

    def available_for(self, capability: str) -> list[ModelDefinition]:
        return [model for model in self._models.values() if capability in model.tasks and model.enabled]

    def mark_loaded(self, model_id: str, loaded: bool = True) -> None:
        if loaded:
            self._loaded.add(model_id)
        else:
            self._loaded.discard(model_id)
