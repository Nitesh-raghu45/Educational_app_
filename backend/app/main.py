import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import select

from .api import auth, documents, figures, quizzes, sections
from .config import get_settings
from .db import SessionLocal, create_tables, engine
from .errors import AppError
from .models import Document, DocumentEmbedding, FigureScan
from .ratelimit import get_redis
from .services import jobs, providers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def redis_status() -> str:
    client = get_redis()
    if client is None:
        return "disabled"
    try:
        await client.ping()
        return "connected"
    except Exception as exc:
        return f"unreachable ({type(exc).__name__})"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # MVP: create tables on boot. Switch to Alembic migrations before running several replicas.
    await create_tables()
    status = await redis_status()
    settings = get_settings()
    if status == "connected":
        logger.info("Redis connected: rate limiting on, ingestion mode=%s", settings.ingest_mode)
    elif status == "disabled":
        logger.info("Redis disabled (REDIS_URL empty): rate limiting off")
    else:
        logger.warning("Redis %s: check REDIS_URL. Rate limiting is skipped until it is reachable.", status)
    # Load the embedding model and open Pinecone now, in the background, instead of making the
    # first upload or question after a (re)start wait 10-20 s for them.
    jobs.spawn(warm_up())
    jobs.spawn(resume_unfinished_ingestion())
    jobs.spawn(reembed_outdated_documents())
    jobs.spawn(backfill_figures())
    yield
    await engine.dispose()


async def reembed_outdated_documents() -> None:
    """After the embedding model changes, re-embed every ready document that was embedded with
    another model, in the background, one at a time. Until then those documents still answer
    through keyword search."""
    model = get_settings().embedding_model
    try:
        async with SessionLocal() as session:
            current = select(DocumentEmbedding.document_id).where(DocumentEmbedding.model == model)
            ids = list(
                await session.scalars(
                    select(Document.id).where(Document.status == "ready", Document.id.not_in(current))
                )
            )
            if ids and model == providers.LEGACY_EMBEDDING_MODEL:
                # Documents from before model tracking were embedded with this model already.
                untracked = select(DocumentEmbedding.document_id)
                legacy = list(await session.scalars(select(Document.id).where(Document.id.in_(ids), Document.id.not_in(untracked))))
                session.add_all(DocumentEmbedding(document_id=i, model=model) for i in legacy)
                await session.commit()
                ids = [i for i in ids if i not in set(legacy)]
        for document_id in ids:
            jobs.enqueue_reembed(document_id)
        if ids:
            logger.info("Re-embedding %d document(s) with %s in the background", len(ids), model)
    except Exception:
        logger.warning("Could not check documents for re-embedding", exc_info=True)


async def backfill_figures() -> None:
    """Documents uploaded before figure extraction existed (or left mid-way by a restart) get
    their figures now, if the uploaded file is still on disk."""
    if not get_settings().figures_enabled:
        return
    try:
        async with SessionLocal() as session:
            docs = (
                await session.execute(
                    select(Document.id, Document.storage_path, FigureScan.status)
                    .outerjoin(FigureScan, FigureScan.document_id == Document.id)
                    .where(Document.status == "ready")
                )
            ).all()
        # Never scanned and the file is here, or interrupted mid-way (the job marks it failed
        # if the file is gone, so the app stops waiting for it).
        ids = [
            doc_id for doc_id, path, scan in docs
            if scan == "processing" or (scan is None and Path(path).exists())
        ]
        for document_id in ids:
            jobs.enqueue_figures(document_id)
        if ids:
            logger.info("Looking for figures in %d earlier document(s) in the background", len(ids))
    except Exception:
        logger.warning("Could not check documents for figures", exc_info=True)


async def resume_unfinished_ingestion() -> None:
    """Documents left pending/processing by a restart (e.g. Render spinning down mid-upload) would
    otherwise show "processing" forever. Process them again; if the file was lost with the
    host's disk, ingestion marks them failed so the student knows to upload again."""
    try:
        async with SessionLocal() as session:
            ids = list(await session.scalars(select(Document.id).where(Document.status.in_(["pending", "processing"]))))
        for document_id in ids:
            await jobs.enqueue_ingestion(document_id)
        if ids:
            logger.info("Resumed %d unfinished document(s)", len(ids))
    except Exception:
        logger.warning("Could not resume unfinished documents", exc_info=True)


async def warm_up() -> None:
    settings = get_settings()
    try:
        await asyncio.to_thread(providers.embedder().embed_query, "warm up")
        if settings.pinecone_api_key:
            await asyncio.to_thread(providers.vector_store().query, "warm-up", [0.0] * settings.embedding_dim, 1, ["-"])
        logger.info("Warm-up complete: embedding model loaded, vector store connected")
    except Exception:
        logger.warning("Warm-up failed; models will load on first use", exc_info=True)


class UnhandledErrorMiddleware:
    """Turn crashes into a JSON 500 *inside* CORSMiddleware.

    Starlette's default 500 response is produced outside the CORS layer, so browsers see only
    a blocked request ("upload failed") instead of an error they can read.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = False

        async def send_wrapper(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception("Unhandled error on %s %s", scope.get("method"), scope.get("path"))
            if started:
                raise
            error = AppError(500, "INTERNAL_ERROR", "Something went wrong on the server. Please try again.")
            await JSONResponse(status_code=500, content=error.to_body())(scope, receive, send)


app = FastAPI(title="Study Assistant API", version="1.0.0", lifespan=lifespan)
# Added first so it sits inside CORSMiddleware (Starlette wraps later middleware around earlier).
app.add_middleware(UnhandledErrorMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=exc.to_body())


for router in (auth.router, auth.usage_router, sections.router, documents.router, figures.router, quizzes.router):
    app.include_router(router, prefix="/api/v1")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "redis": await redis_status()}
