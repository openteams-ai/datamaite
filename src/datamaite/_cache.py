"""SQLite-backed validation result cache."""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from datamaite._types import Finding, Severity
from datamaite._upath import to_dataset_path

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MB


@dataclass(frozen=True)
class FileFingerprint:
    """Identity of a file based on content hash, size, and mtime."""

    hash: str
    size: int
    mtime: float


def fingerprint_file(path: Path) -> FileFingerprint | None:
    """Compute a fingerprint for a file.

    Hashes only the first 1MB of content for speed (video files
    can be 500MB+). Returns None if the file doesn't exist or
    can't be read.
    """
    try:
        stat = path.stat()
    except Exception:
        # A fingerprint failure is a cache miss, not a crash: object-store
        # backends can raise non-OSError families (throttling/auth) on stat.
        return None

    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            h.update(f.read(_HASH_CHUNK_SIZE))
        return FileFingerprint(hash=h.hexdigest(), size=stat.st_size, mtime=stat.st_mtime)
    except Exception:
        # A fingerprint failure is a cache miss, not a crash: the read can
        # fail with non-OSError families on remote filesystems.
        return None


logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "1"


@dataclass
class CacheHit:
    """Cached validation results for a file pair."""

    findings: list[Finding]
    labels: Counter[str]


@dataclass
class CacheStats:
    """Cache utilization counters."""

    hits: int = 0
    misses: int = 0


class ValidationCache:
    """SQLite-backed validation result cache."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.stats = CacheStats()
        self._db: sqlite3.Connection | None = None
        self._pending_writes: int = 0
        if db_path is None:
            db_path = self.default_path()
        try:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(db_path), timeout=5)
            self._db.execute("PRAGMA journal_mode=WAL")
            self._init_schema()
        except (OSError, sqlite3.Error) as e:
            logger.warning("Cache unavailable (%s), proceeding without cache", e)
            self._db = None

        # Register a flush on interpreter shutdown so Ctrl-C / SIGTERM
        # don't lose up to 49 pending writes (batching commits every 50).
        # close() clears _db to None, so the atexit hook becomes a no-op
        # after an explicit close.
        if self._db is not None:
            atexit.register(self._atexit_flush)

    def _atexit_flush(self) -> None:
        """Best-effort flush at interpreter shutdown.

        Guarded so a crashed connection doesn't raise during exit.
        """
        try:
            self.flush()
        except sqlite3.Error as e:
            logger.debug("atexit flush skipped: %s", e)

    def _init_schema(self) -> None:
        if self._db is None:
            return
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS file_results (
                file_path    TEXT    PRIMARY KEY,
                ann_hash     TEXT    NOT NULL,
                ann_size     INTEGER NOT NULL,
                ann_mtime    REAL    NOT NULL,
                vid_hash     TEXT,
                vid_size     INTEGER,
                vid_mtime    REAL,
                check_video  INTEGER NOT NULL,
                findings     TEXT    NOT NULL,
                labels       TEXT    NOT NULL,
                validated_at TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cache_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        row = self._db.execute("SELECT value FROM cache_meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            self._db.execute(
                "INSERT INTO cache_meta (key, value) VALUES ('schema_version', ?)",
                (_SCHEMA_VERSION,),
            )
            self._db.commit()
        elif row[0] != _SCHEMA_VERSION:
            logger.info(
                "Cache schema version changed (%s -> %s), clearing",
                row[0],
                _SCHEMA_VERSION,
            )
            self._db.execute("DELETE FROM file_results")
            self._db.execute(
                "UPDATE cache_meta SET value = ? WHERE key = 'schema_version'",
                (_SCHEMA_VERSION,),
            )
            self._db.commit()

    def lookup(
        self,
        annotation_path: Path,
        video_path: Path | None,
        *,
        check_video: bool,
    ) -> CacheHit | None:
        """Look up cached validation results for an annotation file.

        Returns a ``CacheHit`` if the cached entry is still valid,
        or ``None`` on a miss.  A cached entry validated *with* video
        checks satisfies a lookup that does *not* request them, but
        not vice-versa.
        """
        if self._db is None:
            self.stats.misses += 1
            return None
        ann_fp = fingerprint_file(annotation_path)
        if ann_fp is None:
            self.stats.misses += 1
            return None
        vid_fp = fingerprint_file(video_path) if video_path else None
        try:
            row = self._db.execute(
                "SELECT ann_hash, ann_size, ann_mtime, vid_hash, vid_size, vid_mtime, "
                "check_video, findings, labels FROM file_results WHERE file_path = ?",
                (str(annotation_path),),
            ).fetchone()
        except sqlite3.Error:
            self.stats.misses += 1
            return None
        if row is None:
            self.stats.misses += 1
            return None
        (
            c_ann_hash,
            c_ann_size,
            c_ann_mtime,
            c_vid_hash,
            c_vid_size,
            c_vid_mtime,
            c_check_video,
            findings_json,
            labels_json,
        ) = row
        if ann_fp.hash != c_ann_hash or ann_fp.size != c_ann_size or ann_fp.mtime != c_ann_mtime:
            self.stats.misses += 1
            return None
        if vid_fp is not None and (
            vid_fp.hash != c_vid_hash or vid_fp.size != c_vid_size or vid_fp.mtime != c_vid_mtime
        ):
            self.stats.misses += 1
            return None
        if check_video and not c_check_video:
            self.stats.misses += 1
            return None
        # A corrupted row (bad write, schema drift, manual edit) would
        # otherwise crash the worker. Treat a deserialize failure as a
        # miss, evict the offending row so the next write replaces it.
        try:
            findings = _deserialize_findings(findings_json)
            labels = Counter(json.loads(labels_json))
        except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            logger.warning("Evicting unreadable cache row for %s: %s", annotation_path, exc)
            with contextlib.suppress(sqlite3.Error):
                self._db.execute("DELETE FROM file_results WHERE file_path = ?", (str(annotation_path),))
                self._db.commit()
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return CacheHit(findings=findings, labels=labels)

    def store(
        self,
        annotation_path: Path,
        video_path: Path | None,
        findings: list[Finding],
        labels: Counter[str],
        *,
        check_video: bool,
    ) -> None:
        """Store validation results in the cache."""
        if self._db is None:
            return
        ann_fp = fingerprint_file(annotation_path)
        if ann_fp is None:
            return
        vid_fp = fingerprint_file(video_path) if video_path else None
        findings_json = json.dumps([f.to_dict() for f in findings])
        labels_json = json.dumps(dict(labels))
        try:
            self._db.execute(
                "INSERT OR REPLACE INTO file_results "
                "(file_path, ann_hash, ann_size, ann_mtime, vid_hash, vid_size, vid_mtime, "
                "check_video, findings, labels, validated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (
                    str(annotation_path),
                    ann_fp.hash,
                    ann_fp.size,
                    ann_fp.mtime,
                    vid_fp.hash if vid_fp else None,
                    vid_fp.size if vid_fp else None,
                    vid_fp.mtime if vid_fp else None,
                    int(check_video),
                    findings_json,
                    labels_json,
                ),
            )
            self._pending_writes += 1
            if self._pending_writes >= 50:
                self._db.commit()
                self._pending_writes = 0
        except sqlite3.Error as e:
            logger.warning("Failed to write cache entry: %s", e)

    def flush(self) -> None:
        """Commit any pending writes to disk."""
        if self._db is not None and self._pending_writes > 0:
            try:
                self._db.commit()
                self._pending_writes = 0
            except sqlite3.Error as e:
                logger.warning("Failed to flush cache: %s", e)

    def clear(self) -> None:
        """Remove all cached validation results."""
        if self._db is None:
            return
        try:
            self._db.execute("DELETE FROM file_results")
            self._db.commit()
        except sqlite3.Error as e:
            logger.warning("Failed to clear cache: %s", e)

    def close(self) -> None:
        """Flush pending writes and close the database connection."""
        self.flush()
        if self._db is not None:
            self._db.close()
            self._db = None

    @staticmethod
    def default_path() -> Path:
        """Return the default cache database path."""
        return Path.home() / ".cache" / "datamaite" / "validation.db"


def _deserialize_findings(findings_json: str) -> list[Finding]:
    """Deserialize JSON string back to a list of Finding objects."""
    raw = json.loads(findings_json)
    return [
        Finding(
            severity=Severity(f["severity"]),
            path=to_dataset_path(f["path"]),
            check=f["check"],
            message=f["message"],
        )
        for f in raw
    ]
