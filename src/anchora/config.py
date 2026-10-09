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

    # API-key gate for the serving endpoints. Empty = open dev mode, which only
    # accepts requests from a loopback client; any other client gets 503 until a
    # key is set (fail closed when the server is reachable from the network).
    api_key: str = ""

    # The only directory tree POST /ingest may read. Empty = the bundled
    # data/corpus. A ``directory`` in the request must resolve inside it.
    corpus_root: str = ""
    ingest_max_files: int = 500
    ingest_max_file_bytes: int = 1_000_000

    # Retrieval defaults. Mode is one of "dense" | "bm25" | "hybrid"; hybrid
    # fuses dense cosine and BM25 rankings with Reciprocal Rank Fusion (RRF).
    # Frozen experiment replays (scripts/score_generations.py) pin their own
    # mode explicitly and are unaffected by this default.
    top_k: int = 4
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Out-of-domain floor (see docs/adr/0006). A question is in-domain only if it
    # overlaps the corpus on at least ``ood_min_overlap`` distinct (bridged)
    # tokens; this offline-reproducible signal closes the single incidental-token
    # collision (adversarial ``ood-008``). ``ood_similarity_threshold`` adds a
    # dense-cosine floor on top — meaningful on the real multilingual embedder,
    # so it defaults to 0.0 (disabled) because the offline ``hash`` cosine does
    # not separate in-domain from off-domain.
    ood_min_overlap: int = 2
    ood_similarity_threshold: float = 0.0

    # Eval gate: CI fails if mean proxy faithfulness drops below this. Checked
    # against two LLM judges and kept; see docs/adr/0007-faithfulness-threshold.md.
    faithfulness_threshold: float = 0.70


settings = Settings()
