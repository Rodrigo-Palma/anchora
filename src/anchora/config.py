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

    # Retrieval defaults. Mode is one of "dense" | "bm25" | "hybrid"; hybrid
    # fuses dense cosine and BM25 rankings with Reciprocal Rank Fusion (RRF).
    # Frozen experiment replays (scripts/score_generations.py) pin their own
    # mode explicitly and are unaffected by this default.
    top_k: int = 4
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Eval gate: CI fails if measured faithfulness drops below this.
    faithfulness_threshold: float = 0.70


settings = Settings()
