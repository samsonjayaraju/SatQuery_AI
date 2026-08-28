# Local model checkpoints

Expected paths:

```text
models/geochat/model.safetensors
models/remoteclip/            # local CLIP-compatible config + weights
models/changeformer/model.pth
models/satfusion/model.pt
models/satquery-adapter/best.pt
```

Checkpoint files are ignored. The `/api/v1/models` route reports actual presence and never marks a missing learned model ready. Deterministic baselines require no checkpoint.
