from __future__ import annotations

import torch
from torch import nn


class FrozenSpectralEncoder(nn.Module):
    """Offline smoke-test encoder using fixed scene statistics, not a benchmark model."""

    output_dim = 12

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        means = pixels.mean(dim=(2, 3))
        stds = pixels.std(dim=(2, 3))
        maxima = pixels.amax(dim=(2, 3))
        minima = pixels.amin(dim=(2, 3))
        return torch.cat([means, stds, maxima, minima], dim=1)


class FrozenRemoteCLIPEncoder(nn.Module):
    def __init__(self, checkpoint: str):
        super().__init__()
        from transformers import CLIPVisionModel

        self.encoder = CLIPVisionModel.from_pretrained(checkpoint, local_files_only=True)
        self.output_dim = self.encoder.config.hidden_size
        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(pixel_values=pixels).pooler_output


class SatQueryRemoteSensingAdapter(nn.Module):
    """Trainable bottleneck projection and task head on a frozen image encoder."""

    def __init__(self, encoder: nn.Module, feature_dim: int, bottleneck_dim: int, num_labels: int):
        super().__init__()
        self.encoder = encoder
        self.adapter = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck_dim, feature_dim),
        )
        self.classifier = nn.Linear(feature_dim, num_labels)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        frozen_features = self.encoder(pixels)
        adapted = frozen_features + self.adapter(frozen_features)
        return self.classifier(adapted)
