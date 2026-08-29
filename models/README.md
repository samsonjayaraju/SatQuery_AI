# Local model checkpoints

Model files are external, ignored by Git and loaded only from this directory. The active learned stack expects:

```text
models/
├── remoteclip/
│   └── RemoteCLIP-RN50.pt
├── changeformer/
│   ├── best_ckpt.pt
│   └── source/models/ChangeFormer.py
└── satquery-adapter/
    └── best.pt
```

Run this from the repository root to install the official RemoteCLIP RN50 and ChangeFormer V6 LEVIR artifacts:

```bash
./scripts/setup_local_models.sh
```

The SatQuery adapter is produced locally by the documented EuroSAT training run and copied to `models/satquery-adapter/best.pt`. It is deliberately not downloaded as an opaque prebuilt claim.

After all three checkpoints are ready, set `MOCK_MODE=false` in `.env`. `/api/v1/models` validates both the ChangeFormer checkpoint and its official source before reporting it ready.

Optional GeoChat paths remain registered for future replacement:

```text
models/geochat/model.safetensors
```

GeoChat is not required by the current local build because RemoteCLIP supplies the lightweight remote-sensing VLM path. Review and comply with each upstream model and dataset license before redistribution.
