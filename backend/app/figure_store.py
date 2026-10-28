"""Where figure images live: Cloudflare R2 (S3-compatible) when configured, else local disk.

Images are always served through the API (which checks ownership), never by public URL, so the
bucket stays private and needs no CORS setup.
"""

from functools import lru_cache
from pathlib import Path
from typing import Protocol

from .config import Settings, get_settings


class FigureStore(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...
    def get(self, key: str) -> bytes | None: ...
    def delete(self, keys: list[str]) -> None: ...


class LocalFigureStore:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if self.root not in path.parents:  # keys are built by the app, but never trust a path
            raise ValueError(f"Invalid figure key: {key}")
        return path

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.exists() else None

    def delete(self, keys: list[str]) -> None:
        for key in keys:
            self._path(key).unlink(missing_ok=True)


class R2FigureStore:
    def __init__(self, settings: Settings):
        import boto3  # imported only when R2 is configured
        from botocore.config import Config

        self.bucket = settings.r2_bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=f"https://{settings.r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
            config=Config(connect_timeout=10, read_timeout=30, retries={"max_attempts": 3, "mode": "standard"}),
        )

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def delete(self, keys: list[str]) -> None:
        for start in range(0, len(keys), 1000):  # S3 deletes at most 1000 keys per call
            batch = keys[start : start + 1000]
            self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": [{"Key": k} for k in batch], "Quiet": True})


@lru_cache
def figure_store() -> FigureStore:
    settings = get_settings()
    if settings.r2_account_id and settings.r2_bucket:
        return R2FigureStore(settings)
    return LocalFigureStore(Path(settings.storage_dir) / "figures")


def figure_key(user_id, document_id, index: int) -> str:
    return f"user-{user_id.hex}/doc-{document_id.hex}/fig-{index}.webp"
