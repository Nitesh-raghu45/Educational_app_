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

    # Comma-separated emails of admin accounts: they add books to the shared library (which every
    # user can ask about) and have no daily limits.
    admin_emails: str = ""

    storage_dir: str = "./data/uploads"
    max_upload_mb: int = 25
    # The database copy of an upload is only needed until it's processed (text and figures); then
    # it's deleted to save database space. True keeps every copy.
    keep_file_copies: bool = False
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
    # Signing up and in (0 = no limit). Accounts are free, so without these anyone could script
    # thousands of accounts and multiply every daily limit, or guess passwords endlessly.
    signup_limit_per_ip_per_hour: int = 5
    # App-wide, so it holds even when IPs are faked or rotated. Raise it for a class signing up at once.
    signup_limit_per_hour: int = 100
    login_limit_per_ip_per_15min: int = 30
    login_limit_per_email_per_15min: int = 10

    groq_api_key: str = ""
    # llama-3.3-70b-versatile / llama-3.1-8b-instant were retired on Groq's free/dev tiers (16 Aug 2026).
    # gpt-oss models reason before answering; reasoning_effort trades latency for answer quality.
    groq_answer_model: str = "openai/gpt-oss-120b"
    groq_fast_model: str = "openai/gpt-oss-20b"
    groq_answer_reasoning_effort: str = "medium"
    # Plain look-up questions ("what is X?") don't need medium reasoning, which only delays the
    # first word; questions that compare, explain why/how, solve or ask for visuals keep medium.
    groq_answer_simple_reasoning_effort: str = "low"
    # When the answer model is rate-limited or overloaded, answer with this one right away (it has
    # its own per-minute token budget) instead of waiting. Empty = wait for the answer model.
    groq_fallback_model: str = "openai/gpt-oss-20b"
    groq_fast_reasoning_effort: str = "low"

    pinecone_api_key: str = ""
    pinecone_index: str = "education-app"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Multilingual (50+ languages incl. Hindi/Hinglish). Changing the model re-embeds existing
    # documents automatically at startup, into a separate (shared) Pinecone namespace per model.
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
    # Questions about a whole chapter ("explain this chapter", "summarise it"): passages spread
    # through the chapter plus the best search matches, up to a prompt size that stays well under
    # Groq's free limit of 8,000 tokens per minute (counted in characters, ~4 per token; the
    # whole prompt: rules, earlier conversation and sources).
    overview_chunks: int = 40
    overview_search_hits: int = 8
    overview_prompt_chars: int = 22000
    # Room for a full chapter explanation plus the model's reasoning. Groq counts the tokens
    # actually used against the limit, not this cap.
    overview_max_tokens: int = 8192
    # Hybrid search: BM25 keyword search alongside vector search, merged with reciprocal rank fusion.
    hybrid_search: bool = True
    retrieval_candidates: int = 30
    rrf_k: int = 60
    # OCR for images and for PDF pages that have no text layer (scanned notes).
    ocr_languages: str = "eng+hin"
    ocr_max_pages: int = 40
    ocr_dpi: int = 200
    # Figures (diagrams, charts, photos) pulled out of uploads, described by a vision model so
    # they can be found by search and shown in answers.
    figures_enabled: bool = True
    groq_vision_model: str = "qwen/qwen3.8-27b"
    # Google Gemini takes over vision work when Groq's limits are reached (optional, free key
    # from Google AI Studio). Work no provider can take now waits and is retried every
    # VISION_RETRY_MINUTES.
    gemini_api_key: str = ""
    # Tried in order. On Google's free tier each model has its own daily limit, so several
    # models multiply the free capacity. All of these transcribe NCERT math pages well.
    gemini_vision_models: str = "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-flash-lite-latest"
    vision_retry_minutes: int = 10
    max_figures_per_document: int = 30
    daily_figure_limit: int = 150
    # Math-heavy PDF pages (fractions, powers, roots, symbols the text layer garbles) are read
    # again from a picture of the page by the vision model, which writes the math in LaTeX.
    math_vision_enabled: bool = True
    math_vision_max_pages: int = 40
    daily_math_page_limit: int = 300
    # Pictures from the web in answers: Wikipedia images (free) and, with a Serper.dev key,
    # Google Images (used only when the student asks for a picture). The search runs alongside
    # document retrieval and is dropped if it takes longer than web_image_timeout seconds.
    web_images_enabled: bool = True
    serper_api_key: str = ""
    web_image_timeout: float = 0.8
    # When the student asked for pictures, waiting a bit longer beats showing none.
    web_image_timeout_requested: float = 1.5
    web_image_cache_seconds: int = 60 * 60 * 24
    history_window: int = 8
    summary_min_new_messages: int = 4

    @property
    def gemini_model_list(self) -> list[str]:
        return list(dict.fromkeys(m.strip() for m in self.gemini_vision_models.split(",") if m.strip()))

    @property
    def admin_email_set(self) -> frozenset[str]:
        return frozenset(e.strip().lower() for e in self.admin_emails.split(",") if e.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
