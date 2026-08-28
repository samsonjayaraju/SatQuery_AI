# Local setup

1. Install Node.js 20.9+, Python 3.9+, and Git. Rasterio wheels normally include GDAL; if installation fails, install GDAL through the platform package manager first.
2. Copy `.env.example` to `.env`. Relative data/model defaults resolve from the repository root.
3. Create `.venv`, activate it, and install `apps/api/requirements.txt`.
4. Run `npm install` at the repository root.
5. Start FastAPI from `apps/api` with `../../.venv/bin/uvicorn app.main:app --reload`.
6. Start Next.js from `apps/web` with `npm run dev`.
7. Verify `/api/v1/health`, then open `http://localhost:3000`.

`MOCK_MODE=true` is the default because large checkpoints are not bundled. Deterministic pixel analysis still runs; outputs that depend on unavailable learned models carry the Development Mock Result label.

For GeoTIFF failures, verify `python -c "import rasterio; print(rasterio.__gdal_version__)"`. For memory pressure, keep model unloading enabled, reduce tile size, and use CPU-safe checkpoints.
