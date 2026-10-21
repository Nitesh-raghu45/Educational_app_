# Study Assistant: conversational RAG for students

A ChatGPT-style study app. Students upload their own documents, then ask questions in chat
**sections**. Answers stream in and come only from those documents, with citations. Each
section accepts **30 questions**. After that it becomes read-only and the student starts a new chat.

| Layer | Tech |
|---|---|
| Frontend | React 18 + TypeScript (Vite): sidebar, streaming chat, documents panel, quizzes |
| API | FastAPI (stateless, so you can run several replicas) |
| Database | **MariaDB**: users, sections, messages, quotas, documents, chunk text, quizzes |
| Vector DB | **Pinecone** serverless index (384-dim, cosine), one namespace per student |
| Embeddings | **all-MiniLM-L6-v2**, run locally on CPU with ONNX Runtime (`fastembed`, so no PyTorch) |
| LLM | **Groq**: `openai/gpt-oss-120b` for answers and quizzes, `openai/gpt-oss-20b` for query rewriting and summaries (reasoning hidden from students) |
| Queue / cache | Redis: arq ingestion queue and per-user rate limits |
| File storage | Private volume (`/data/uploads`), standing in for object storage |

## Quick start (Docker)

1. Copy `.env.example` to `.env`. Set `GROQ_API_KEY` and `PINECONE_API_KEY`.
2. Run `docker compose up --build`.
3. Open http://localhost:8080, create an account, start a chat and upload a PDF, DOCX, TXT, MD or image.

API docs are at http://localhost:8000/docs.

## Local development (without Docker for the app)

Start only MariaDB and Redis:

```bash
docker compose up -d mariadb redis
```

Backend (from `backend/`):

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload     # reads ../.env
arq app.worker.WorkerSettings     # second terminal: ingestion worker
```

To skip the worker, set `INGEST_MODE=inline` and ingestion runs inside the API process.
Image OCR needs the [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) binary on your PATH.
The Docker image already includes it.

Frontend (from `frontend/`). The dev server proxies `/api` to `localhost:8000`:

```bash
npm install
npm run dev                       # http://localhost:5173
```

Tests use SQLite plus fake embedder, Pinecone and Groq clients, so they need no keys or services:

```bash
cd backend && pytest
```

To run the same suite against MariaDB (recommended, since the quota relies on InnoDB row locking), set
`TEST_DATABASE_URL=mysql+aiomysql://edu:edu@localhost:3306/edu_test?charset=utf8mb4` after creating that
database. The suite drops and recreates its tables. `backend/schema.sql` is the MariaDB DDL for reference.

## How it works

### Sections and the 30-question limit
- A section is one sidebar chat. The backend generates its UUIDv4. Every query also filters by the
  authenticated `user_id`. Another user's section returns the same `404 SECTION_NOT_FOUND` as a missing one.
- A **question** is one accepted user message. Assistant replies don't count.
- The limit is enforced in MariaDB, not in React or Redis. In one transaction
  ([services/sections.py](backend/app/services/sections.py)):
  1. `UPDATE chat_sections SET question_count = question_count + 1 WHERE id=? AND user_id=? AND status='active' AND question_count < max_questions`.
     The InnoDB row lock serialises concurrent requests. MariaDB has no `UPDATE … RETURNING`, so the code checks `rowcount`.
  2. If the new count reaches the maximum, set `status='limit_reached'`.
  3. Insert the user message and a `pending` assistant message, then commit.
- If no row was updated, the API returns `409 SECTION_LIMIT_REACHED`, `SECTION_CLOSED` or `NO_READY_DOCUMENTS`.
- **Idempotency:** the client sends `client_message_id` with each submission. A unique
  `(section_id, idempotency_key)` index means a network retry replays the stored answer and doesn't use another slot.
- **Failed answers** stay in history with `status='failed'`. `POST /sections/{id}/retry/{message_id}`
  regenerates the answer without reserving a slot. Answers stuck in `pending` or `streaming` for more than
  2 minutes (for example after a server restart) can be retried the same way.

### Streaming (SSE)
`POST /api/v1/sections/{id}/messages` returns `text/event-stream` with these events:
`message_start` (saved message IDs and the updated quota) → `token`… → `citations` → `done`, or `error`.
A `: ping` heartbeat goes out every 15 s. Generation runs as a background task with its own DB session. If the
browser disconnects, the answer still finishes and is saved, and the UI reloads it. The frontend reads the
stream with `fetch()` because `EventSource` can't send a POST body or an `Authorization` header.

### Conversational RAG pipeline ([services/rag.py](backend/app/services/rag.py))
1. Load the last 8 completed messages and the section's rolling summary.
2. Rewrite follow-ups ("explain that again") into a standalone query with the fast Groq model.
3. **Hybrid search**, run in parallel:
   - Vector search: embed the query with MiniLM and query Pinecone in the student's namespace, filtered to `document_id ∈ section's ready documents`. Matches below `RETRIEVAL_MIN_SCORE` are dropped.
   - Keyword search: BM25 over the stored chunk text ([services/keyword.py](backend/app/services/keyword.py)), with stopwords removed and words stemmed. A chunk must share at least 2 query terms (1 for a one-word query). Term statistics are cached per document.
4. Merge both lists with reciprocal rank fusion (RRF) and keep the top `RETRIEVAL_TOP_K`. Load chunk text from the database, which re-checks ownership. Each citation records `matched_by`: `vector`, `keyword` or `both`.
5. **Evidence gate:** with no evidence, the app returns a fixed "couldn't find it in your documents" answer and never calls the LLM.
6. Otherwise, stream a grounded answer. The prompt requires `[n]` citations and treats history as context, not evidence.
7. Save the answer with its cited chunks. Fold older messages into the rolling summary, so all 30 turns keep context without sending every turn to the LLM.

### Document ingestion ([services/ingestion.py](backend/app/services/ingestion.py))
Upload → private storage → arq job (or inline) → parse (pypdf / python-docx / Tesseract OCR) →
**recursive chunking** → MiniLM embeddings → Pinecone upsert (vector id = chunk id) → chunk text into
MariaDB → `ready`. The UI shows upload progress and a processing spinner, and polls
`GET /documents/{id}/status`.

- **Recursive chunking:** text is split on the coarsest boundary that fits, in this order: paragraphs,
  lines, sentences, clauses, words, then characters. Neighbouring pieces are merged up to `CHUNK_SIZE`
  (1000 characters). So paragraphs stay whole when they fit, and neighbouring chunks share up to
  `CHUNK_OVERLAP` (150) characters of whole sentences. PDFs are chunked page by page, so citations keep
  their page numbers.
- **Reliability:** Pinecone calls have timeouts and retries. Documents left processing by a restart are
  picked up again at startup. If the file was lost (Render's disk is wiped on restart), the document is
  marked failed with "please upload again".

### Latency
- **Fewer round trips:** one question costs 12 SQL statements (pinned by
  `tests/test_latency_budget.py`). The generation task reuses what the request already loaded, the
  question and its pending answer go in one multi-row INSERT, and duplicate submissions are caught by
  the unique index instead of a lookup.
- **Redis:** the rate-limit check is a single pipelined round trip.
- **Query rewriting:** follow-ups are rewritten only when they refer back to the conversation ("what
  does *it* produce?", "and the stroma?"). Standalone questions skip that extra LLM call.
- **Warm start:** the embedding model and the Pinecone connection are warmed up when the API starts.

### Quizzes ([services/quiz.py](backend/app/services/quiz.py))
The app picks chunks: retrieved by topic, or a random spread when no topic is given. Groq JSON mode turns them into
multiple-choice questions. Each question is validated: 4 unique options, a valid answer index, and a
`source` that points at a real chunk. Invalid questions are dropped. Answers stay hidden until the student submits an attempt.
Attempts are scored and stored.

## API

| Method & path | Purpose |
|---|---|
| `POST /api/v1/auth/register`, `POST /auth/login`, `GET /auth/me` | JWT auth |
| `POST /api/v1/sections` | Create a section (UUID) |
| `GET /api/v1/sections` | Sidebar list, most recent first |
| `GET /PATCH /DELETE /api/v1/sections/{id}` | Metadata and quota, rename, delete |
| `GET /api/v1/sections/{id}/messages` | Conversation history |
| `POST /api/v1/sections/{id}/messages` | Ask and stream the answer (SSE) |
| `POST /api/v1/sections/{id}/retry/{message_id}` | Retry a failed answer without using a slot |
| `GET /POST /api/v1/sections/{id}/documents`, `DELETE …/{document_id}` | Section document scope |
| `POST /api/v1/sections/{id}/quizzes` | Generate a grounded quiz |
| `GET /api/v1/quizzes/{id}`, `POST /quizzes/{id}/attempts` | Take and score a quiz |
| `POST /api/v1/documents` (multipart `file`, optional `section_id`) | Upload and queue ingestion |
| `GET /api/v1/documents`, `GET /documents/{id}/status`, `DELETE /documents/{id}` | Library and ingestion status |

## Production next steps
- Replace `create_all` on startup with Alembic migrations before running several API replicas.
- Move uploads to S3/GCS/R2 with private buckets. `storage.py` is the only module that touches files.
- Add malware scanning and MIME sniffing on uploads, OCR for scanned PDFs (pdf2image + Poppler), and refresh tokens or SSO.

## Deploy to Render

[`render.yaml`](render.yaml) defines two services: `edu-api` (Docker, free plan) and `edu-web` (static site).

- **Database:** use a managed MySQL that doesn't expire (for example Aiven's free MySQL). Set `DATABASE_URL=mysql+aiomysql://…`, keep `DATABASE_SSL=true`, and paste the provider's CA certificate into `DATABASE_SSL_CA`. Render's free Postgres is deleted after 30 days, and this app's driver is MySQL/MariaDB.
- **Documents are processed inline** (`INGEST_MODE=inline`). Render's free plan has no background workers, and its disk is wiped on restart.
- **Memory:** `EMBEDDING_BATCH_SIZE=8` keeps peak memory around 315 MB, under the free plan's 512 MB.
- **URLs:** set `VITE_API_URL` on `edu-web` to the API URL, and `CORS_ORIGINS` on `edu-api` to `["https://<edu-web>.onrender.com"]`.
