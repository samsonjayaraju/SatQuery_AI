# Evaluation

Measured results belong in `evaluation-results/*.json`; that directory starts empty and the UI says “Not evaluated yet.”

- VQA: accuracy and normalized exact match.
- Grounding: IoU, precision and recall.
- Change detection: IoU, F1, precision, recall and overall accuracy.
- Classification: accuracy and macro F1.
- Captioning: BLEU, ROUGE, METEOR and CIDEr where dataset tooling supports them.

Run `python ml/evaluation/evaluate_change.py --predictions <dir> --targets <dir>`. Files are paired by `.npy` filename. Evaluation JSON must include dataset/split identity, sample count, checkpoint identity, parameters and metrics before it is published in the interface.

Two measured artifacts are currently included:

- `domain-adapter.json`: best validation epoch from the deterministic 15% EuroSAT split (750 of 5,000 samples).
- `change-detection.json`: official ChangeFormer V6 checkpoint on the seven LEVIR-CD demo pairs from the upstream repository.

Reproduce the latter after model setup with `python ml/evaluation/evaluate_changeformer.py --device auto`. Other cards remain explicitly “Not evaluated yet” until legally obtained VRSBench, RSVQA or CDVQA results are produced.
