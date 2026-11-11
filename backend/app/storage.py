"""Private file storage. Local disk (a shared volume in docker-compose) stands in for object storage."""

import asyncio
import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import UploadFile

from .config import get_settings
from .errors import AppError

logger = logging.getLogger(__name__)


def path_for(user_id: uuid.UUID, document_id: uuid.UUID, extension: str) -> Path:
    # Never build paths from the user's filename.
    return Path(get_settings().storage_dir).resolve() / user_id.hex / f"{document_id.hex}{extension}"


async def save_upload(upload: UploadFile, path: Path, max_bytes: int) -> tuple[int, str]:
    """Saves the file; returns its size and SHA-256 (computed while writing, so it costs no extra pass)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    digest = hashlib.sha256()
    try:
        with path.open("wb") as out:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > max_bytes:
                    raise AppError(413, "FILE_TOO_LARGE", f"Files must be at most {max_bytes // (1024 * 1024)} MB.")
                out.write(chunk)
                digest.update(chunk)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    if size == 0:
        path.unlink(missing_ok=True)
        raise AppError(400, "EMPTY_FILE", "The uploaded file is empty.")
    return size, digest.hexdigest()


def file_sha256(path: str) -> str | None:
    """SHA-256 of a stored file, or None if the file is gone."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


_RETRY_SECONDS = 5
_RETRY_ATTEMPTS = 120  # about 10 minutes
_pending: set[asyncio.Task] = set()  # keeps the retry tasks alive


def delete(path: str) -> None:
    """Removes a stored file. Never raises: on Windows a file that is still open (e.g. a book
    deleted while its processing job is reading it) can't be removed yet, so it is retried in
    the background until the job lets go of it."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as exc:
        try:
            task = asyncio.get_running_loop().create_task(_delete_later(path))
            _pending.add(task)
            task.add_done_callback(_pending.discard)
            logger.info("Stored file %s is in use (%s); removing it once it is released", path, exc)
        except RuntimeError:  # no event loop (a script): leave the file
            logger.warning("Couldn't remove stored file %s: %s", path, exc)


async def _delete_later(path: str) -> None:
    for _ in range(_RETRY_ATTEMPTS):
        await asyncio.sleep(_RETRY_SECONDS)
        try:
            Path(path).unlink(missing_ok=True)
            return
        except OSError:
            continue
    logger.warning("Gave up removing stored file %s (still in use)", path)
