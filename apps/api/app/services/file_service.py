from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.core.config import Settings
from app.core.exceptions import SatQueryError
from app.remote_sensing.input_inspector import SUPPORTED_EXTENSIONS


def safe_filename(filename: str | None) -> str:
    value = Path(filename or "image").name
    value = re.sub(r"[^A-Za-z0-9._-]", "_", value)
    return value[:180]


async def save_uploads(files: list[UploadFile], settings: Settings) -> tuple[str, list[Path]]:
    request_id = str(uuid.uuid4())
    destination = settings.data_dir.resolve() / "uploads" / request_id
    destination.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    paths: list[Path] = []
    for index, upload in enumerate(files):
        filename = safe_filename(upload.filename)
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise SatQueryError(
                "UNSUPPORTED_FILE_TYPE",
                f"{filename} is unsupported. Accepted extensions: .tif, .tiff, .png, .jpg, .jpeg.",
                415,
            )
        path = destination / f"{index + 1}_{filename}"
        total = 0
        with path.open("wb") as stream:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    path.unlink(missing_ok=True)
                    raise SatQueryError("FILE_TOO_LARGE", f"{filename} exceeds the {settings.max_upload_size_mb} MB limit.", 413)
                stream.write(chunk)
        paths.append(path)
    return request_id, paths


def cleanup_uploads(paths: list[Path]) -> None:
    """Remove request-scoped source copies after durable thumbnails/results exist."""
    parents = {path.parent for path in paths}
    for path in paths:
        path.unlink(missing_ok=True)
    for parent in parents:
        try:
            parent.rmdir()
        except OSError:
            pass
