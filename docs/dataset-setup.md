# Dataset setup

Datasets are external, ignored and never downloaded automatically.

## BigEarthNet

Place a legal subset under `datasets/BigEarthNet-S2/` and create `samples.jsonl`:

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
