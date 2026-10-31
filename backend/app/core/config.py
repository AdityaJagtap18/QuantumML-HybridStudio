from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HYBRIDSTUDIO_")

    app_name: str = "Hybrid Studio"
    api_prefix: str = "/api"
    cors_origins: list[str] = ["http://localhost:5175", "http://127.0.0.1:5175"]

    data_dir: Path = Path(__file__).resolve().parents[3] / "data"
    checkpoints_dir: Path = Path(__file__).resolve().parents[3] / "artifacts" / "checkpoints"
    database_url: str = "sqlite:///" + str(Path(__file__).resolve().parents[3] / "data" / "workbench.db")

    max_training_samples: int = 400


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return settings
