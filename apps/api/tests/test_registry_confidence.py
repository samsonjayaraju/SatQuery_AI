from pathlib import Path

import numpy as np

from app.reasoning.confidence_engine import ConfidenceEngine
from app.registry.model_registry import ModelRegistry


def test_registry_does_not_claim_missing_checkpoint(tmp_path: Path):
    models = {model.id: model for model in ModelRegistry(tmp_path, "cpu").list()}
    assert models["geochat_vqa"].status == "disabled"
    assert models["landcover_classifier"].status == "ready"


def test_changeformer_requires_checkpoint_and_official_source(tmp_path: Path):
    checkpoint = tmp_path / "changeformer" / "best_ckpt.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    models = {model.id: model for model in ModelRegistry(tmp_path, "cpu").list()}
    assert models["changeformer"].status == "checkpoint_missing"
    source_directory = tmp_path / "changeformer" / "source" / "models"
    source_directory.mkdir(parents=True)
    for name in ("ChangeFormer.py", "ChangeFormerBaseNetworks.py", "help_funcs.py", "pixel_shuffel_up.py"):
        (source_directory / name).touch()
    models = {model.id: model for model in ModelRegistry(tmp_path, "cpu").list()}
    assert models["changeformer"].status == "ready"


def test_confidence_is_derived_from_components():
    probability = np.array([[0.9, 0.8], [0.1, 0.2]], dtype=np.float32)
    result = ConfidenceEngine().from_probability(probability, 0.5, agreement=0.75)
    assert result.type == "heuristic"
    assert 0 < result.overall < 1
    assert result.components["evidence_strength"] == 0.85
