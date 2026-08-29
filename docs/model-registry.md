# Model registry

| ID | Role | Default state |
| --- | --- | --- |
| `geochat_vqa` | Optional replacement remote-sensing VQA | disabled |
| `geochat_caption` | Optional replacement scene captioning | disabled |
| `geochat_grounding` | Optional replacement text-guided localization | disabled |
| `remoteclip_encoder` | RemoteCLIP image/text embedding, VQA, captioning and grounding | checkpoint detected at runtime |
| `satquery_adapter` | EuroSAT-adapted residual projection/head | checkpoint detected at runtime |
| `changeformer` | Bi-temporal change probability | checkpoint and source detected at runtime |
| `landcover_classifier` | Spectral deterministic baseline | ready |
| `satfusion` | Optical/SAR gated weighted-feature fusion | ready |
| `change_reasoner` | Structured change explanation | ready |

ChangeFormer is ready only when both `best_ckpt.pt` and the official source module are present. Routes never pretend a missing checkpoint executed. GeoChat stays optional because RemoteCLIP provides the active lightweight remote-sensing VLM implementation. Model services expose lazy load/unload, health and metadata boundaries, so replacements do not change the agent, API or frontend contracts.
