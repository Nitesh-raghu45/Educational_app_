FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EMBEDDING_CACHE_DIR=/models \
    HF_HUB_DISABLE_XET=1

# Tesseract OCR reads photos of notes and scanned PDF pages, in English and Hindi.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the multilingual embedding model into the image, with its weights stored as ONNX external
# data so they are memory-mapped at run time (load peak ~60 MB instead of ~480 MB). `onnx` is only
# needed for that one-off conversion.
COPY app/services/embeddings.py /tmp/embeddings.py
RUN pip install onnx==1.17.0 \
    && python /tmp/embeddings.py /models \
    && pip uninstall -y onnx \
    && rm /tmp/embeddings.py

COPY app ./app

# The model is baked in above, so at run time never ask Hugging Face whether a newer copy exists:
# that check cost every cold start ~0.7 s and 3 network requests, and depended on huggingface.co
# being reachable. To switch EMBEDDING_MODEL to one that isn't baked in, set HF_HUB_OFFLINE=0.
ENV HF_HUB_OFFLINE=1

# The app runs as a non-root user: give it the upload folder and its default ./data location.
RUN useradd --create-home appuser && mkdir -p /data/uploads /app/data && chown -R appuser /data /models /app/data

ENV STORAGE_DIR=/data/uploads
USER appuser

EXPOSE 8000
# Render (and similar hosts) pass the port in $PORT; docker-compose uses the default 8000.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
