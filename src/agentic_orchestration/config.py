"""Application configuration. Reading the environment is explicit and local to the app."""

from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deployment_path: Path = Path("deployments/local.yaml")
    deployment_schema_path: Path = Path("schemas/deployment.schema.json")
    source_commit: str | None = None
    build_metadata_path: Path = Path("build-metadata.json")

    llm_service_url: str | None = None
    external_vendor: Literal["claude", "openai"] = "claude"
    external_model: str | None = None
    client_code: str | None = None
    gateway_token: SecretStr | None = Field(default=None, validation_alias="GATEWAY_TOKEN")
    token: SecretStr | None = None
    gateway_user: SecretStr | None = Field(default=None, validation_alias="GATEWAY_USER")
    external_user: SecretStr | None = None
    gateway_timeout_seconds: float = Field(default=180.0, gt=0)

    observability_flush_timeout_seconds: float = Field(default=2.0, gt=0)
    observability_max_pending_snapshots: int = Field(default=256, ge=1)

    # Lifecycle cleanup for pauses nobody came back to. Zero disables the periodic pass;
    # request-time closure still frees a session the moment anyone uses it again.
    pending_run_sweep_interval_seconds: float = Field(default=60.0, ge=0)
    pending_run_sweep_limit: int = Field(default=50, ge=1, le=500)

    mongodb_host: str | None = None
    mongodb_port: int | None = Field(default=None, ge=1, le=65535)
    mongodb_user: str | None = None
    mongodb_pwd: SecretStr | None = None
    mongodb_db: str | None = None
