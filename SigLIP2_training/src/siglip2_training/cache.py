from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import requests
from PIL import Image

from .config import gib_to_bytes, mib_to_bytes, resolve_from


@dataclass(frozen=True)
class CacheLimits:
    maximum_bytes: int
    eviction_target_bytes: int
    minimum_free_bytes: int
    maximum_file_bytes: int
    maximum_pixels: int
    minimum_width: int
    minimum_height: int
    allowed_formats: tuple[str, ...]
    connect_timeout_seconds: float
    read_timeout_seconds: float
    maximum_attempts: int
    backoff_seconds: float
    chunk_size_bytes: int
    user_agent: str


@dataclass(frozen=True)
class CachedImage:
    key: str
    url: str
    path: Path
    size_bytes: int
    content_sha256: str
    width: int
    height: int
    image_format: str


def cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


class ImageCache:
    def __init__(self, root: Path, database: Path, limits: CacheLimits) -> None:
        self.root = Path(root)
        self.database = Path(database)
        self.limits = limits
        self.objects = self.root / "objects"
        self.temporary = self.root / "tmp"
        self.lock_path = self.root / "cache.lock"
        self._thread_lock = threading.RLock()
        self.objects.mkdir(parents=True, exist_ok=True)
        self.temporary.mkdir(parents=True, exist_ok=True)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.touch(exist_ok=True)
        self._initialize_database()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], project_root: str | Path) -> "ImageCache":
        cache = config["cache"]
        download = config["download"]
        validation = config["image_validation"]
        root = resolve_from(project_root, cache["root"])
        database = resolve_from(project_root, cache["database"])
        limits = CacheLimits(
            maximum_bytes=gib_to_bytes(cache["maximum_size_gib"]),
            eviction_target_bytes=gib_to_bytes(cache["eviction_target_gib"]),
            minimum_free_bytes=gib_to_bytes(cache["minimum_free_disk_gib"]),
            maximum_file_bytes=mib_to_bytes(download["maximum_file_mib"]),
            maximum_pixels=int(validation["maximum_pixels"]),
            minimum_width=int(validation["minimum_width"]),
            minimum_height=int(validation["minimum_height"]),
            allowed_formats=tuple(str(value).upper() for value in validation["allowed_formats"]),
            connect_timeout_seconds=float(download["connect_timeout_seconds"]),
            read_timeout_seconds=float(download["read_timeout_seconds"]),
            maximum_attempts=int(download["maximum_attempts"]),
            backoff_seconds=float(download["backoff_seconds"]),
            chunk_size_bytes=int(download["chunk_size_kib"]) * 1024,
            user_agent=str(download["user_agent"]),
        )
        if limits.eviction_target_bytes > limits.maximum_bytes:
            raise ValueError("Cache eviction target cannot exceed maximum size")
        return cls(root=root, database=database, limits=limits)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    relative_path TEXT,
                    size_bytes INTEGER NOT NULL DEFAULT 0,
                    content_sha256 TEXT,
                    width INTEGER,
                    height INTEGER,
                    image_format TEXT,
                    status TEXT NOT NULL,
                    failures INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    created_at REAL NOT NULL,
                    last_access REAL NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS cache_entries_access ON cache_entries(status, last_access)"
            )

    @contextlib.contextmanager
    def _exclusive_lock(self) -> Iterator[None]:
        with self._thread_lock:
            with self.lock_path.open("r+") as lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _row_to_image(self, row: sqlite3.Row) -> CachedImage | None:
        if row["status"] != "ready" or not row["relative_path"]:
            return None
        path = self.root / row["relative_path"]
        if not path.exists():
            return None
        return CachedImage(
            key=row["key"],
            url=row["url"],
            path=path,
            size_bytes=int(row["size_bytes"]),
            content_sha256=str(row["content_sha256"]),
            width=int(row["width"]),
            height=int(row["height"]),
            image_format=str(row["image_format"]),
        )

    def get(self, url: str) -> CachedImage | None:
        key = cache_key(url)
        now = time.time()
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM cache_entries WHERE key = ?", (key,)).fetchone()
            image = self._row_to_image(row) if row is not None else None
            if image is not None:
                connection.execute("UPDATE cache_entries SET last_access = ? WHERE key = ?", (now, key))
                return image
            if row is not None and row["status"] == "ready":
                connection.execute(
                    "UPDATE cache_entries SET status = 'missing', last_error = ? WHERE key = ?",
                    ("Cached file is missing", key),
                )
        return None

    def _validate_image(self, data: bytes) -> tuple[int, int, str]:
        if not data:
            raise ValueError("Downloaded image is empty")
        if len(data) > self.limits.maximum_file_bytes:
            raise ValueError(f"Image exceeds {self.limits.maximum_file_bytes} bytes")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                image_format = str(image.format or "").upper()
                width, height = image.size
                if image_format not in self.limits.allowed_formats:
                    raise ValueError(f"Unsupported image format: {image_format}")
                if width < self.limits.minimum_width or height < self.limits.minimum_height:
                    raise ValueError(f"Image is too small: {width}x{height}")
                if width * height > self.limits.maximum_pixels:
                    raise ValueError(f"Image has too many pixels: {width}x{height}")
                image.verify()
        return width, height, image_format

    def _ensure_free_disk(self) -> None:
        free = shutil.disk_usage(self.root).free
        if free < self.limits.minimum_free_bytes:
            raise OSError(
                f"Free disk safety floor reached: {free} available, "
                f"{self.limits.minimum_free_bytes} required"
            )

    def put_bytes(self, url: str, data: bytes) -> CachedImage:
        width, height, image_format = self._validate_image(data)
        self._ensure_free_disk()
        key = cache_key(url)
        content_hash = hashlib.sha256(data).hexdigest()
        relative_path = Path("objects") / f"{key}.img"
        destination = self.root / relative_path
        now = time.time()
        with self._exclusive_lock():
            with tempfile.NamedTemporaryFile(dir=self.temporary, suffix=".part", delete=False) as temporary:
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            try:
                os.replace(temporary_path, destination)
            finally:
                temporary_path.unlink(missing_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO cache_entries (
                        key, url, relative_path, size_bytes, content_sha256, width, height,
                        image_format, status, failures, last_error, created_at, last_access
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'ready', 0, NULL, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        url = excluded.url,
                        relative_path = excluded.relative_path,
                        size_bytes = excluded.size_bytes,
                        content_sha256 = excluded.content_sha256,
                        width = excluded.width,
                        height = excluded.height,
                        image_format = excluded.image_format,
                        status = 'ready',
                        failures = 0,
                        last_error = NULL,
                        last_access = excluded.last_access
                    """,
                    (
                        key,
                        url,
                        str(relative_path),
                        len(data),
                        content_hash,
                        width,
                        height,
                        image_format,
                        now,
                        now,
                    ),
                )
            self.evict_if_needed(protected_key=key)
        image = self.get(url)
        if image is None:
            raise RuntimeError("Image was evicted immediately; cache limit is smaller than the object")
        return image

    def _record_failure(self, url: str, error: Exception) -> None:
        key = cache_key(url)
        now = time.time()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO cache_entries (
                    key, url, status, failures, last_error, created_at, last_access
                ) VALUES (?, ?, 'failed', 1, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    status = 'failed',
                    failures = cache_entries.failures + 1,
                    last_error = excluded.last_error,
                    last_access = excluded.last_access
                """,
                (key, url, f"{type(error).__name__}: {error}"[:1000], now, now),
            )

    def _download_bytes(self, url: str, session: requests.Session) -> bytes:
        timeout = (self.limits.connect_timeout_seconds, self.limits.read_timeout_seconds)
        with session.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > self.limits.maximum_file_bytes:
                raise ValueError(f"Remote image exceeds {self.limits.maximum_file_bytes} bytes")
            data = bytearray()
            for chunk in response.iter_content(chunk_size=self.limits.chunk_size_bytes):
                if not chunk:
                    continue
                data.extend(chunk)
                if len(data) > self.limits.maximum_file_bytes:
                    raise ValueError(f"Remote image exceeds {self.limits.maximum_file_bytes} bytes")
            return bytes(data)

    def get_or_download(self, url: str, session: requests.Session | None = None) -> CachedImage:
        cached = self.get(url)
        if cached is not None:
            return cached
        owned_session = session is None
        active_session = session or requests.Session()
        active_session.headers.update({"User-Agent": self.limits.user_agent})
        try:
            last_error: Exception | None = None
            for attempt in range(1, self.limits.maximum_attempts + 1):
                try:
                    self._ensure_free_disk()
                    data = self._download_bytes(url, active_session)
                    return self.put_bytes(url, data)
                except (requests.RequestException, OSError, ValueError) as error:
                    last_error = error
                    self._record_failure(url, error)
                    if attempt < self.limits.maximum_attempts:
                        time.sleep(self.limits.backoff_seconds * attempt)
            assert last_error is not None
            raise last_error
        finally:
            if owned_session:
                active_session.close()

    def evict_if_needed(self, protected_key: str | None = None) -> int:
        removed = 0
        with self._connect() as connection:
            total = int(
                connection.execute(
                    "SELECT COALESCE(SUM(size_bytes), 0) FROM cache_entries WHERE status = 'ready'"
                ).fetchone()[0]
            )
            if total <= self.limits.maximum_bytes:
                return 0
            rows = connection.execute(
                "SELECT key, relative_path, size_bytes FROM cache_entries "
                "WHERE status = 'ready' ORDER BY last_access ASC"
            ).fetchall()
            for row in rows:
                if row["key"] == protected_key:
                    continue
                path = self.root / row["relative_path"]
                path.unlink(missing_ok=True)
                connection.execute(
                    "UPDATE cache_entries SET status = 'evicted', relative_path = NULL, size_bytes = 0 "
                    "WHERE key = ?",
                    (row["key"],),
                )
                total -= int(row["size_bytes"])
                removed += 1
                if total <= self.limits.eviction_target_bytes:
                    break
        return removed

    def status(self) -> dict[str, int]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total_entries,
                    SUM(CASE WHEN status = 'ready' THEN 1 ELSE 0 END) AS ready_entries,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_entries,
                    COALESCE(SUM(CASE WHEN status = 'ready' THEN size_bytes ELSE 0 END), 0) AS ready_bytes
                FROM cache_entries
                """
            ).fetchone()
        return {key: int(row[key] or 0) for key in row.keys()}
