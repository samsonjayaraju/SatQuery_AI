from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from dataset import BigEarthNetManifestDataset, collect_labels
from model import FrozenRemoteCLIPEncoder, FrozenSpectralEncoder, SatQueryRemoteSensingAdapter


@dataclass
class TrainingConfig:
    dataset_path: str
    manifest: str
    encoder_backend: str = "remoteclip"
    encoder_path: str = "models/remoteclip"
    encoder_model: str = "RN50"
    epochs: int = 8
    batch_size: int = 24
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    bottleneck_dim: int = 256
    image_size: int = 224
    validation_split: float = 0.15
    num_workers: int = 2
    seed: int = 26167
    output_dir: str = "ml/training/outputs/satquery-adapter"


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predicted = logits.sigmoid() >= 0.5
    truth = labels.bool()
    tp = (predicted & truth).sum(0).float()
    fp = (predicted & ~truth).sum(0).float()
    fn = (~predicted & truth).sum(0).float()
    return float((2 * tp / (2 * tp + fp + fn + 1e-8)).mean())


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels.argmax(dim=1)).float().mean())


def cache_features(encoder, loader, device):
    encoder = encoder.to(device).eval()
    features_all, labels_all = [], []
    with torch.inference_mode():
        for pixels, labels in loader:
            features_all.append(encoder(pixels.to(device)).detach().cpu())
            labels_all.append(labels)
    return TensorDataset(torch.cat(features_all), torch.cat(labels_all))


def run_epoch(model, loader, loss_fn, device, optimizer=None, features_cached=False):
    training = optimizer is not None
    model.train(training)
    losses, logits_all, labels_all = [], [], []
    for pixels, labels in loader:
        pixels, labels = pixels.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model.forward_features(pixels) if features_cached else model(pixels)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        logits_all.append(logits.detach().cpu())
        labels_all.append(labels.detach().cpu())
    logits = torch.cat(logits_all)
    labels = torch.cat(labels_all)
    return {
        "loss": float(np.mean(losses)),
        "macro_f1": macro_f1(logits, labels),
        "accuracy": classification_accuracy(logits, labels),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SatQuery Remote-Sensing Adapter on an open remote-sensing manifest.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--dataset-path")
    parser.add_argument("--manifest")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--evaluation-output", default="evaluation-results/domain-adapter.json")
    args = parser.parse_args()
    values = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key in ("dataset_path", "epochs", "batch_size", "learning_rate", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    if args.manifest is not None:
        values["manifest"] = args.manifest
    elif args.dataset_path is not None:
        values["manifest"] = str(Path(args.dataset_path) / "samples.jsonl")
    config = TrainingConfig(**values)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    device = resolve_device(args.device)
    manifest = Path(config.manifest)
    labels = collect_labels(manifest)
    dataset = BigEarthNetManifestDataset(manifest, labels, config.image_size)
    validation_size = max(1, int(len(dataset) * config.validation_split))
    train_set, validation_set = random_split(dataset, [len(dataset) - validation_size, validation_size], generator=torch.Generator().manual_seed(config.seed))
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    validation_loader = DataLoader(validation_set, batch_size=config.batch_size, num_workers=config.num_workers)
    if config.encoder_backend == "remoteclip":
        encoder = FrozenRemoteCLIPEncoder(config.encoder_path, config.encoder_model)
        feature_dim = encoder.output_dim
    elif config.encoder_backend == "spectral":
        encoder, feature_dim = FrozenSpectralEncoder(), FrozenSpectralEncoder.output_dim
        print("WARNING: spectral backend is for pipeline smoke tests, not final adaptation claims.")
    else:
        raise ValueError("encoder_backend must be remoteclip or spectral")
    model = SatQueryRemoteSensingAdapter(encoder, feature_dim, config.bottleneck_dim, len(labels)).to(device)
    print("Caching frozen RemoteCLIP features once for adapter training...")
    cached_train = cache_features(encoder, DataLoader(train_set, batch_size=config.batch_size, num_workers=config.num_workers), device)
    cached_validation = cache_features(encoder, DataLoader(validation_set, batch_size=config.batch_size, num_workers=config.num_workers), device)
    train_loader = DataLoader(cached_train, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(cached_validation, batch_size=config.batch_size)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    metrics, best_f1 = [], -1.0
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, loss_fn, device, optimizer, features_cached=True)
        validation_metrics = run_epoch(model, validation_loader, loss_fn, device, features_cached=True)
        record = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        metrics.append(record); print(json.dumps(record))
        checkpoint = {"adapter": model.adapter.state_dict(), "classifier": model.classifier.state_dict(), "labels": labels, "feature_dim": feature_dim, "config": asdict(config), "metrics": record}
        torch.save(checkpoint, output / "latest.pt")
        if validation_metrics["macro_f1"] > best_f1:
            best_f1 = validation_metrics["macro_f1"]
            torch.save(checkpoint, output / "best.pt")
    (output / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    best = max(metrics, key=lambda item: item["validation"]["macro_f1"])
    evaluation = {
        "task_id": "domain_adapter",
        "dataset": "EuroSAT RGB",
        "model": f"SatQuery adapter over RemoteCLIP {config.encoder_model}",
        "split": f"deterministic validation ({config.validation_split:.0%}, seed {config.seed})",
        "sample_count": validation_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(output / "best.pt"),
        "parameters": {
            "training_samples": len(dataset) - validation_size,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "bottleneck_dim": config.bottleneck_dim,
            "seed": config.seed,
        },
        "metrics": {
            "accuracy": best["validation"]["accuracy"],
            "macro_f1": best["validation"]["macro_f1"],
            "loss": best["validation"]["loss"],
        },
    }
    evaluation_path = Path(args.evaluation_output)
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
