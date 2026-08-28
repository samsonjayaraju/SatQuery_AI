# Limitations

- Deterministic color/backscatter baselines are development aids, not trained semantic segmentation models.
- Heuristic confidence is evidence strength and agreement, not calibrated probability.
- Cloud, haze, seasonal variation, false-color composites and radiometric differences may bias optical evidence.
- Pixel-space resizing cannot replace authoritative co-registration; pair results depend on alignment.
- SAR interpretation depends on polarization, incidence angle, calibration, terrain and speckle preprocessing.
- PNG/JPEG inputs lack geospatial units; areas are reported as pixel percentages.
- Multispectral indices require correct band metadata/order. The current visual baseline does not claim NDVI/NDWI/NDBI when those bands are unavailable.
- Large rasters are previewed at bounded resolution by the current baseline; production tiling/stitching must be used for full-resolution inference.
- Domain adaptation is an executable pipeline, but no accuracy is claimed until a checkpoint and evaluation artifacts are produced locally.
- This is a research prototype, not a source for emergency response or irreversible land-use decisions.
