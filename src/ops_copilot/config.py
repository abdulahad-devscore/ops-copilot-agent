from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model routing: a cheap model classifies, a strong model reasons and uses tools.
    cheap_model: str = "claude-haiku-5-5"
    strong_model: str = "claude-opus-5-5"
    strong_effort: str = "medium"

    # Postgres with pgvector: document chunks, LangGraph checkpoints, run log
    database_url: str = "postgresql://copilot:copilot@localhost:5434/copilot"
    vector_store: str = "pgvector"  # pgvector | memory

    # Retrieval
    corpus_dir: str = "corpus"
    embedder: str = "fastembed"  # fastembed (local ONNX) | hashing (tests only)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker: str = "fastembed"  # fastembed | cohere | none
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    cohere_api_key: str | None = None
    retrieve_top_k: int = 6

    # ERP MCP Bridge (Project 1)
    bridge_url: str = "http://localhost:8000"
    bridge_client_id: str = "ops-copilot"
    bridge_client_secret: str = "ops-copilot-demo-secret"
    # Where the bridge should POST decisions so paused runs resume themselves
    public_url: str = "http://localhost:8100"

    business_date: str = "2026-10-27"
    max_agent_steps: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
