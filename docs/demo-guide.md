# Demo guide

1. Generate samples with `python scripts/generate_demo_data.py`.
2. Start both local services and confirm the header reports `LOCAL`.
3. Single mode: upload `demo/single/agriculture-river.png`, ask for a description, then ask to highlight water.
4. Change mode: upload `demo/change/t1.png` and `t2.png`; ask “Has the built-up area increased?” Toggle split view and the change overlay.
5. Fusion mode: upload optical then SAR; ask for water-covered areas. Select Optical, SAR and Fused evidence cards.
6. Expand the trace to show automatic routing, tools, parameters and runtime.
7. Generate the local HTML report.
8. Open Models to distinguish ready baselines from missing checkpoints, and Benchmarks to show the no-fake-metrics policy.
