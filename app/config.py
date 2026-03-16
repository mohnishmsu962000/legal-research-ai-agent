from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "Legal Research AI Agent"
    app_version: str = "0.1.0"
    debug: bool = False

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "legal-research"
    pinecone_dimension: int = 1536

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str

    # Redis
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "legal-research-ai-agent"

    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    retrieval_top_k: int = 10
    context_window_size: int = 8

    # Agent
    max_debate_rounds: int = 3
    confidence_threshold: float = 0.7

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()