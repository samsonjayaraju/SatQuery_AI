# Dataset setup

Datasets are external, ignored and never downloaded automatically.

## EuroSAT RGB — completed adapter path

Download `EuroSAT_RGB.zip` from the official Zenodo EuroSAT record and extract it under `datasets/`. Create a balanced 5,000-sample manifest:

```bash
.venv/bin/python scripts/prepare_eurosat_manifest.py \
  --dataset datasets/EuroSAT_RGB \
  --max-samples 5000
```

Train the frozen-RemoteCLIP adapter and publish the best local checkpoint:

```bash
.venv/bin/python ml/training/remote_adapter/train.py \
  --dataset-path datasets/EuroSAT_RGB \
  --epochs 5 --batch-size 32 --device auto \
  --output-dir ml/training/outputs/satquery-adapter \
  --evaluation-output evaluation-results/domain-adapter.json
mkdir -p models/satquery-adapter
cp ml/training/outputs/satquery-adapter/best.pt models/satquery-adapter/best.pt
```

The script freezes RemoteCLIP RN50, caches its image features once, and trains only the residual adapter and 10-class head. The validation split is deterministic for seed `26167`.

## BigEarthNet

As an alternative or larger follow-up experiment, place a legal subset under `datasets/BigEarthNet-S2/` and create `samples.jsonl`:

```json
{"image":"patches/S2A_MSIL2A_example.png","labels":["Urban fabric","Arable land"]}
```

Each record points to a locally prepared RGB/false-color preview and its multi-label classes. Start with 5,000 records to validate the pipeline; use 20,000–50,000 for a meaningful prototype experiment. Preserve train/validation geographic separation to avoid leakage.

## VRSBench, RSVQA and CDVQA

- VRSBench: single-image VQA, captioning and grounding.
- RSVQA: remote-sensing VQA evaluation.
- CDVQA: change-focused VQA.
- LEVIR-CD / LEVIR-CC: optional change masks and captions.

Follow each dataset's license and official split. Do not copy source imagery into this repository.
