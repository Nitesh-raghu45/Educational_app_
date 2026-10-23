"""Local CPU embedding models (no PyTorch).

The default, paraphrase-multilingual-MiniLM-L12-v2, understands 50+ languages, including Hindi
and Hinglish, so students can ask in the language of their notes. It is loaded with a lean
ONNX Runtime setup instead of fastembed's: identical vectors (checked to cosine 1.0000), but the
whole API peaks at ~310 MB instead of ~650 MB, which fits a 512 MB host (Render free):
  - the compact SentencePiece vocabulary (~40 MB) instead of tokenizer.json (~250 MB in memory)
  - weights stored as ONNX external data, which ONNX Runtime memory-maps instead of copying
  - basic graph optimisation and no memory arena
"""

import os
import threading
from pathlib import Path

import numpy as np

MULTILINGUAL_MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Quantised ONNX export of the model, and the original repo for its SentencePiece vocabulary.
_XLMR_ONNX_REPO = "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
_XLMR_ONNX_FILE = "model_optimized.onnx"
_XLMR_VOCAB_FILE = "sentencepiece.bpe.model"

# XLM-RoBERTa special token ids (fairseq layout); SentencePiece ids are shifted by one.
_BOS, _PAD, _EOS, _UNK = 0, 1, 2, 3


def download_multilingual_model(cache_dir: str | None) -> tuple[Path, Path]:
    """Fetch (or reuse from cache) the ONNX model and SentencePiece vocabulary."""
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTPS downloads are more reliable
    from huggingface_hub import hf_hub_download

    model = hf_hub_download(_XLMR_ONNX_REPO, _XLMR_ONNX_FILE, cache_dir=cache_dir)
    vocab = hf_hub_download(MULTILINGUAL_MINILM, _XLMR_VOCAB_FILE, cache_dir=cache_dir)
    return Path(model), Path(vocab)


def _external_data_path(cache_dir: str | None) -> Path:
    return Path(cache_dir or ".").resolve() / "multilingual-minilm-l12-external" / "model.onnx"


def prepare_multilingual_model(cache_dir: str | None) -> tuple[Path, Path]:
    """Download the model and store its weights as ONNX "external data" (a separate file).

    ONNX Runtime memory-maps external weights instead of copying them: loading peaks at ~60 MB
    instead of ~480 MB (outputs are bit-identical), which matters on a 512 MB host. Needs the
    `onnx` package once; the Docker build does this, so the server itself doesn't need `onnx`.
    Returns (model path, vocabulary path); falls back to the single-file model if conversion
    isn't possible.
    """
    model, vocab = download_multilingual_model(cache_dir)
    external = _external_data_path(cache_dir)
    if external.exists():
        return external, vocab
    try:
        import onnx
    except ImportError:
        return model, vocab
    external.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        onnx.load(str(model)), str(external), save_as_external_data=True,
        all_tensors_to_one_file=True, location="model.onnx.data", size_threshold=1024,
    )
    return external, vocab


class MultilingualMiniLMEmbedder:
    """paraphrase-multilingual-MiniLM-L12-v2: 384-dim, mean pooling, L2-normalised, 128 tokens max."""

    max_tokens = 128

    def __init__(self, cache_dir: str | None = None, batch_size: int = 8):
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._session = None
        self._sp = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._session is None:
                import onnxruntime as ort
                import sentencepiece as spm

                model_path, vocab_path = prepare_multilingual_model(self._cache_dir)
                options = ort.SessionOptions()
                options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                options.enable_cpu_mem_arena = False
                options.enable_mem_pattern = False
                self._sp = spm.SentencePieceProcessor(model_file=str(vocab_path))
                self._session = ort.InferenceSession(
                    str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
                )
        return self._session, self._sp

    def _token_ids(self, text: str) -> list[int]:
        _, sp = self._load()
        ids = [i + 1 if i != 0 else _UNK for i in sp.encode(text)]
        return [_BOS] + ids[: self.max_tokens - 2] + [_EOS]

    def count_tokens(self, text: str) -> int:
        """Tokens the model would see for `text`, excluding the start/end markers."""
        _, sp = self._load()
        return len(sp.encode(text))

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        session, _ = self._load()
        encoded = [self._token_ids(t) for t in texts]
        width = max(len(ids) for ids in encoded)
        input_ids = np.full((len(texts), width), _PAD, dtype=np.int64)
        mask = np.zeros((len(texts), width), dtype=np.int64)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = ids
            mask[row, : len(ids)] = 1
        hidden = session.run(
            None, {"input_ids": input_ids, "attention_mask": mask, "token_type_ids": np.zeros_like(input_ids)}
        )[0]
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        return pooled / np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            vectors.extend(self._embed_batch(texts[start : start + self._batch_size]).tolist())
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder:
    """Any other fastembed model (e.g. the previous English-only all-MiniLM-L6-v2), L2-normalised."""

    def __init__(self, model_name: str, cache_dir: str | None = None, batch_size: int = 8):
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from fastembed import TextEmbedding

                self._model = TextEmbedding(self._model_name, cache_dir=self._cache_dir)
            return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = np.array(list(self._get_model().embed(texts, batch_size=self._batch_size)))
        return (vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embedder(model_name: str, cache_dir: str | None, batch_size: int):
    if model_name == MULTILINGUAL_MINILM:
        return MultilingualMiniLMEmbedder(cache_dir, batch_size)
    return FastEmbedEmbedder(model_name, cache_dir, batch_size)


if __name__ == "__main__":
    # Used by the Docker build: python embeddings.py /models
    import sys

    model_path, vocab_path = prepare_multilingual_model(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"Prepared {model_path} and {vocab_path}")
