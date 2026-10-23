from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Reads ../.env (repo root, used by docker-compose) and ./.env when run from backend/.
    model_config = SettingsConfigDict(env_file=("../.env", ".env"), extra="ignore")

    database_url: str = "mysql+aiomysql://edu:edu@localhost:3306/edu?charset=utf8mb4"
    # Managed MySQL/MariaDB in the cloud (e.g. Aiven) requires TLS; set DATABASE_SSL=true there.
    database_ssl: bool = False
    # PEM text of the provider's CA certificate (Aiven uses its own CA); empty = system CAs.
    database_ssl_ca: str = ""
    redis_url: str = "redis://localhost:6379/0"
    # Cloud Redis over TLS (e.g. Upstash) can take 1-3 s to connect from far away.
    redis_connect_timeout: float = 10.0

    jwt_secret: str = "change-me-in-production"
    jwt_expire_minutes: int = 60 * 24 * 7
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:8080"]

    storage_dir: str = "./data/uploads"
    max_upload_mb: int = 25
    # "queue" sends ingestion to the arq worker via Redis; "inline" runs it inside the API process.
    ingest_mode: str = "queue"

    max_questions_per_section: int = 30
    # Daily per-user caps on costly actions (reset at midnight UTC; 0 = unlimited). The per-chat
    # limit alone doesn't cap cost, since anyone can open more chats.
    daily_question_limit: int = 100
    daily_quiz_limit: int = 20
    daily_upload_limit: int = 30
    daily_chat_limit: int = 20
    rate_limit_per_minute: int = 20

    groq_api_key: str = ""
    # llama-3.3-70b-versatile / llama-3.1-8b-instant were retired on Groq's free/dev tiers (16 Aug 2026).
    # gpt-oss models reason before answering; reasoning_effort trades latency for answer quality.
    groq_answer_model: str = "openai/gpt-oss-120b"
    groq_fast_model: str = "openai/gpt-oss-20b"
    groq_answer_reasoning_effort: str = "medium"
    groq_fast_reasoning_effort: str = "low"

    pinecone_api_key: str = ""
    pinecone_index: str = "education-app"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Multilingual (50+ languages incl. Hindi/Hinglish). Changing the model re-embeds existing
    # documents automatically at startup, into a separate Pinecone namespace per model.
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384
    # Where the ONNX model is downloaded to (baked into the Docker image at /models).
    embedding_cache_dir: str = "./data/models"
    # Chunks embedded per ONNX run. 8 keeps peak memory ~250 MB (32 needs ~450 MB), which
    # matters on small hosts such as Render's free 512 MB instances.
    embedding_batch_size: int = 8

    # Chunks are measured in the embedding model's tokens when it can count them (the multilingual
    # model reads at most 128 tokens), otherwise in characters (chunk_size / chunk_overlap).
    chunk_tokens: int = 120
    chunk_overlap_tokens: int = 20
    chunk_size: int = 1000
    chunk_overlap: int = 150
    # Smaller (token-sized) chunks, so more of them are retrieved to give the LLM similar context.
    retrieval_top_k: int = 12
    retrieval_min_score: float = 0.25
    # Hybrid search: BM25 keyword search alongside vector search, merged with reciprocal rank fusion.
    hybrid_search: bool = True
    retrieval_candidates: int = 30
    rrf_k: int = 60
    # OCR for images and for PDF pages that have no text layer (scanned notes).
    ocr_languages: str = "eng+hin"
    ocr_max_pages: int = 40
    ocr_dpi: int = 200
    history_window: int = 8
    summary_min_new_messages: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
