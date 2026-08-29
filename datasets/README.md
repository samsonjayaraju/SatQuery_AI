# External datasets

Actual imagery is ignored and must never be committed. The completed domain-adaptation path uses EuroSAT RGB:

```text
datasets/EuroSAT_RGB/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
├── SeaLake/
└── samples.jsonl
```

Download the official `EuroSAT_RGB.zip` from Zenodo record `7711810`, extract it under `datasets/`, then run:

```bash
.venv/bin/python scripts/prepare_eurosat_manifest.py --max-samples 5000
```

BigEarthNet, VRSBench, RSVQA, CDVQA and full LEVIR-CD remain supported external evaluation options through the manifest/evaluation contracts described in [docs/dataset-setup.md](../docs/dataset-setup.md). No large dataset is downloaded automatically.
