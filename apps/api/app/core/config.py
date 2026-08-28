from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    app_env: str = "development"
    backend_host: str = "127.0.0.1"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:3000"
    data_dir: Path = PROJECT_ROOT / "data"
    model_dir: Path = PROJECT_ROOT / "models"
    temp_dir: Path = PROJECT_ROOT / "data" / "temp"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    mock_mode: bool = True
    model_unload_after_request: bool = True
    max_upload_size_mb: int = 500
    tile_size: int = 512
    tile_overlap: int = 64
    change_threshold: float = 0.20

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    @model_validator(mode="after")
    def resolve_local_paths(self) -> "Settings":
        for field in ("data_dir", "model_dir", "temp_dir"):
            value = getattr(self, field)
            if not value.is_absolute():
                setattr(self, field, PROJECT_ROOT / value)
        return self

    def ensure_directories(self) -> None:
        for name in ("uploads", "outputs", "reports", "history", "temp"):
            (self.data_dir / name).resolve().mkdir(parents=True, exist_ok=True)
        self.temp_dir.resolve().mkdir(parents=True, exist_ok=True)
        self.model_dir.resolve().mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
