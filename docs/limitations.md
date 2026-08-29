# Limitations

- Deterministic color/backscatter baselines are development aids, not trained semantic segmentation models.
- Confidence remains an evidence score rather than calibrated probability; learned scores are capped until task-specific calibration is measured.
- Cloud, haze, seasonal variation, false-color composites and radiometric differences may bias optical evidence.
- Pixel-space resizing cannot replace authoritative co-registration; pair results depend on alignment.
- Environmental change uses hybrid appearance and semantic evidence; clouds, shadows and large radiometric differences can still appear as change when cloud masks or calibrated source products are unavailable.
- SAR interpretation depends on polarization, incidence angle, calibration, terrain and speckle preprocessing.
- PNG/JPEG inputs lack geospatial units; areas are reported as pixel percentages.
- Obvious multi-panel paper figures are rejected using high-precision layout signals, but unusual document layouts may still require manual source verification.
- Multispectral indices require correct band metadata/order. The current visual baseline does not claim NDVI/NDWI/NDBI when those bands are unavailable.
- Large rasters use overlapping full-resolution inference tiles, but model runtime still scales with scene area and available memory.
- The EuroSAT holdout result measures 10-class RGB scene classification, not pixel segmentation, VQA quality or performance under geographic domain shift.
- Single-image land-cover shares are normalized patch-class probabilities, not trained pixel-segmentation areas.
- The checked-in ChangeFormer result covers only seven official upstream demo pairs and must not be represented as a full LEVIR-CD benchmark.
- RemoteCLIP answers are retrieval/contrastive decisions rather than free-form generative reasoning; unusual questions may require GeoChat or another replaceable VLM.
- This is a research prototype, not a source for emergency response or irreversible land-use decisions.
