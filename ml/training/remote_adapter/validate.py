from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import BigEarthNetManifestDataset
from model import FrozenRemoteCLIPEncoder, FrozenSpectralEncoder, SatQueryRemoteSensingAdapter
from train import macro_f1, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a SatQuery adapter checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--encoder-path", default="models/remoteclip")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="validation_metrics.json")
    args = parser.parse_args()
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config, labels = checkpoint["config"], checkpoint["labels"]
    if config["encoder_backend"] == "remoteclip":
        encoder = FrozenRemoteCLIPEncoder(args.encoder_path)
        feature_dim = encoder.output_dim
    else:
        encoder, feature_dim = FrozenSpectralEncoder(), FrozenSpectralEncoder.output_dim
    model = SatQueryRemoteSensingAdapter(encoder, feature_dim, config["bottleneck_dim"], len(labels))
    model.adapter.load_state_dict(checkpoint["adapter"])
    model.classifier.load_state_dict(checkpoint["classifier"])
    model.to(device).eval()
    dataset = BigEarthNetManifestDataset(Path(args.manifest), labels, config["image_size"])
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    logits_all, labels_all, losses = [], [], []
    loss_fn = nn.BCEWithLogitsLoss()
    with torch.inference_mode():
        for pixels, target in loader:
            logits = model(pixels.to(device))
            losses.append(float(loss_fn(logits, target.to(device)).cpu()))
            logits_all.append(logits.cpu()); labels_all.append(target)
    metrics = {
        "samples": len(dataset),
        "loss": sum(losses) / max(len(losses), 1),
        "macro_f1": macro_f1(torch.cat(logits_all), torch.cat(labels_all)),
        "checkpoint": str(Path(args.checkpoint)),
    }
    Path(args.output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
