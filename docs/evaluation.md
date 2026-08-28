# Evaluation

Measured results belong in `evaluation-results/*.json`; that directory starts empty and the UI says “Not evaluated yet.”

- VQA: accuracy and normalized exact match.
- Grounding: IoU, precision and recall.
- Change detection: IoU, F1, precision, recall and overall accuracy.
- Classification: accuracy and macro F1.
- Captioning: BLEU, ROUGE, METEOR and CIDEr where dataset tooling supports them.

Run `python ml/evaluation/evaluate_change.py --predictions <dir> --targets <dir>`. Files are paired by `.npy` filename. Evaluation JSON must include dataset/split identity, sample count, checkpoint identity, parameters and metrics before it is published in the interface.
