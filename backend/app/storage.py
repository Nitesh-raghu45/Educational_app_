"""Private file storage. Local disk (a shared volume in docker-compose) stands in for object storage."""

import hashlib
import uuid
from pathlib import Path

from fastapi import UploadFile

from .config import get_settings
from .errors import AppError


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


def delete(path: str) -> None:
    Path(path).unlink(missing_ok=True)
