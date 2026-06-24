"""Configuration. Local-first: Ollama for embeddings and generation, no paid API."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ANCHORA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "anchora"
    ollama_base_url: str = "http://localhost:11434"
    gen_model: str = "qwen3:32b"
    embed_model: str = "nomic-embed-text"
    # "ollama" uses local models; "hash" is a deterministic offline fallback
    # (used by tests/CI so the suite runs without any model or network).
    embed_provider: str = "ollama"
    embed_dim: int = 256
    request_timeout: float = 60.0

    # Optional API-key gate for the serving endpoints. Empty = open (dev default).
    api_key: str = ""

    # Retrieval defaults.
    top_k: int = 4

    # Eval gate: CI fails if measured faithfulness drops below this.
    faithfulness_threshold: float = 0.70


settings = Settings()
