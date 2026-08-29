# Local setup

1. Install Node.js 20.9+, Python 3.9+, and Git. Rasterio wheels normally include GDAL; if installation fails, install GDAL through the platform package manager first.
2. Copy `.env.example` to `.env`. Relative data/model defaults resolve from the repository root.
3. Create `.venv`, activate it, and install `apps/api/requirements.txt`.
4. Run `npm install` at the repository root.
5. For learned inference, run `./scripts/setup_local_models.sh`, train/copy the SatQuery adapter, and set `MOCK_MODE=false`.
6. Start FastAPI from `apps/api` with `../../.venv/bin/uvicorn app.main:app --reload`.
7. Start Next.js from `apps/web` with `npm run dev`.
8. Verify `/api/v1/health` and `/api/v1/models`, then open `http://localhost:3000`.

`MOCK_MODE=true` is the fresh-clone default because large checkpoints are not bundled. Deterministic pixel analysis still runs and carries the Development Mock Result label. `MOCK_MODE=false` requires the selected learned specialist and exposes its real model/version in the execution trace.

For GeoTIFF failures, verify `python -c "import rasterio; print(rasterio.__gdal_version__)"`. For memory pressure, keep model unloading enabled, reduce tile size, and use CPU-safe checkpoints.
