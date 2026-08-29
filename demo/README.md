# Demo samples

Run `python scripts/generate_demo_data.py` to create synthetic, redistributable PNG inputs under `single`, `change` and `cross_modal`.

Use the samples in this order:

1. `single/agriculture-river.png` — “Describe this satellite image” and “Highlight the largest water body.”
2. `change/t1.png` + `change/t2.png` — “Has the built-up area increased?”
3. `cross_modal/optical.png` + `cross_modal/sar.png` — “Use both sensors to identify water-covered areas.”

With learned checkpoints and `MOCK_MODE=false`, the execution trace names RemoteCLIP, the SatQuery EuroSAT adapter, ChangeFormer and SatFusion. With `MOCK_MODE=true`, the same flow stays demoable and is explicitly labelled as a development result. You may replace these files with legally obtained benchmark samples; do not commit restricted datasets.
