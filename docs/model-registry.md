# Model registry

| ID | Role | Default state |
| --- | --- | --- |
| `geochat_vqa` | Remote-sensing VQA | checkpoint missing |
| `geochat_caption` | Scene captioning | checkpoint missing |
| `geochat_grounding` | Text-guided localization | checkpoint missing |
| `remoteclip_encoder` | Image/text embedding | checkpoint missing |
| `satquery_adapter` | BigEarthNet-adapted projection/head | checkpoint missing |
| `changeformer` | Bi-temporal change probability | checkpoint missing |
| `landcover_classifier` | Spectral deterministic baseline | ready |
| `satfusion` | Optical/SAR weighted fusion baseline | ready |
| `change_reasoner` | Structured change explanation | ready |

Routes never pretend a missing checkpoint executed. Replace an adapter behind `BaseModelAdapter`; the agent, API and frontend contracts remain unchanged.
