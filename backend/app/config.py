"""Application settings, read from environment variables (and an optional .env file).

Secrets are never given defaults: the server refuses to start without a strong SECRET_KEY.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
MIN_SECRET_KEY_LENGTH = 32

_DEV_CORS_ORIGINS = (
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8080,http://127.0.0.1:8080,"
    "http://localhost:5000,http://127.0.0.1:5000"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",  # .env also holds settings for later phases
        populate_by_name=True,
    )

    # --- security -------------------------------------------------------
    secret_key: str = Field(..., description="HMAC key for access tokens; at least 32 characters")
    access_token_expire_minutes: int = Field(60, ge=1, le=60 * 24 * 7)
    login_max_failures: int = Field(5, ge=1, le=100)
    login_lockout_seconds: int = Field(300, ge=1)

    # --- database -------------------------------------------------------
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'apk_guardian.db').as_posix()}"

    # --- uploads and analysis ------------------------------------------
    max_apk_size_mb: int = Field(100, ge=1, le=500)
    min_apk_size_bytes: int = Field(1024, ge=1)
    analysis_timeout_seconds: float = Field(120.0, gt=0, le=3600)
    upload_tmp_dir: Path = PROJECT_ROOT / "data" / "uploads_tmp"
    max_concurrent_scans: int = Field(2, ge=1, le=16)
    scan_queue_wait_seconds: float = Field(5.0, ge=0, le=120)

    # --- model ----------------------------------------------------------
    model_dir: Path = PROJECT_ROOT / "models"
    allow_synthetic_model: bool = False  # only for pipeline testing; see docs

    # --- apk library ---------------------------------------------------
    apk_library_dir: Path = PROJECT_ROOT / "data" / "apk_library"

    # --- optional LLM explanations --------------------------------------
    mistral_api_key: Optional[str] = None
    mistral_api_url: str = "https://api.mistral.ai/v1/chat/completions"
    mistral_model: str = "open-mistral-7b"
    enable_ai_explanations: bool = True

    # --- http -----------------------------------------------------------
    cors_origins: str = _DEV_CORS_ORIGINS
    json_body_limit_bytes: int = Field(64 * 1024, ge=1024)

    @field_validator("model_dir", "upload_tmp_dir", "apk_library_dir")
    @classmethod
    def _resolve_against_project_root(cls, value: Path) -> Path:
        """Relative folders in .env mean "relative to the project root", not the current directory."""
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("secret_key")
    @classmethod
    def _strong_secret(cls, value: str) -> str:
        if len(value) < MIN_SECRET_KEY_LENGTH or value.strip().lower() in {"change-me", "changeme"}:
            raise ValueError(
                f"SECRET_KEY must be a random value of at least {MIN_SECRET_KEY_LENGTH} characters. "
                "Generate one with: python -m app.cli generate-secret"
            )
        return value

    @property
    def max_apk_size_bytes(self) -> int:
        return self.max_apk_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip() and o.strip() != "*"]

    @property
    def resolved_database_url(self) -> str:
        """Make relative SQLite paths relative to the project folder, not the current directory."""
        prefix = "sqlite:///"
        url = self.database_url
        if url.startswith(prefix) and not url.startswith("sqlite:///:memory:"):
            raw = url[len(prefix):]
            path = Path(raw)
            if not path.is_absolute() and not (len(raw) > 1 and raw[1] == ":"):
                return prefix + (PROJECT_ROOT / path).resolve().as_posix()
        return url

    @property
    def sqlite_path(self) -> Optional[Path]:
        url = self.resolved_database_url
        if url.startswith("sqlite:///") and ":memory:" not in url:
            return Path(url[len("sqlite:///"):])
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # secret_key comes from the environment
