"""Tests for validation result cache."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from datamaite._cache import FileFingerprint, ValidationCache, fingerprint_file
from datamaite._types import Finding, Severity


class TestFingerprint:
    def test_fingerprint_small_file(self, tmp_path: Path) -> None:
        f = tmp_path / "small.json"
        f.write_text('{"task_id": "t1"}')
        fp = fingerprint_file(f)
        assert isinstance(fp, FileFingerprint)
        assert fp.size == f.stat().st_size
        assert fp.mtime == f.stat().st_mtime
        assert len(fp.hash) == 64  # SHA256 hex

    def test_fingerprint_changes_on_content_change(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text('{"version": 1}')
        fp1 = fingerprint_file(f)
        f.write_text('{"version": 2}')
        fp2 = fingerprint_file(f)
        assert fp1.hash != fp2.hash

    def test_fingerprint_same_content_same_hash(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.json"
        f2 = tmp_path / "b.json"
        content = '{"same": "content"}'
        f1.write_text(content)
        f2.write_text(content)
        assert fingerprint_file(f1).hash == fingerprint_file(f2).hash

    def test_fingerprint_large_file_only_reads_first_mb(self, tmp_path: Path) -> None:
        f = tmp_path / "large.bin"
        f.write_bytes(b"\x00" * 1024 * 1024 + b"\x01" * 1024 * 1024)
        fp = fingerprint_file(f)
        f2 = tmp_path / "large2.bin"
        f2.write_bytes(b"\x00" * 1024 * 1024 + b"\x02" * 1024 * 1024)
        fp2 = fingerprint_file(f2)
        assert fp.hash == fp2.hash  # same first 1MB
        assert fp.size == fp2.size  # same size

    def test_fingerprint_nonexistent_file(self, tmp_path: Path) -> None:
        f = tmp_path / "missing.json"
        fp = fingerprint_file(f)
        assert fp is None


class TestValidationCache:
    def test_create_cache_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        cache = ValidationCache(db_path)
        assert db_path.exists()
        cache.close()

    def test_store_and_lookup_hit(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        cache = ValidationCache(db_path)
        ann = tmp_path / "ann.json"
        ann.write_text('{"task_id": "t1"}')
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake mp4 content")
        findings = [Finding(severity=Severity.WARNING, path=ann, check="annotation_missing_afr", message="Missing AFR")]
        labels = Counter({"boat": 5, "car": 2})
        cache.store(ann, video, findings, labels, check_video=True)
        hit = cache.lookup(ann, video, check_video=True)
        assert hit is not None
        assert len(hit.findings) == 1
        assert hit.findings[0].check == "annotation_missing_afr"
        assert hit.labels["boat"] == 5
        cache.close()

    def test_lookup_miss_on_content_change(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        cache = ValidationCache(db_path)
        ann = tmp_path / "ann.json"
        ann.write_text('{"task_id": "t1"}')
        cache.store(ann, None, [], Counter(), check_video=False)
        ann.write_text('{"task_id": "t2"}')
        hit = cache.lookup(ann, None, check_video=False)
        assert hit is None
        cache.close()

    def test_lookup_miss_when_video_check_requested_but_not_cached(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        cache = ValidationCache(db_path)
        ann = tmp_path / "ann.json"
        ann.write_text('{"task_id": "t1"}')
        cache.store(ann, None, [], Counter(), check_video=False)
        hit = cache.lookup(ann, None, check_video=True)
        assert hit is None
        cache.close()

    def test_lookup_hit_when_video_check_not_requested_but_cached_with(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        cache = ValidationCache(db_path)
        ann = tmp_path / "ann.json"
        ann.write_text('{"task_id": "t1"}')
        cache.store(ann, None, [], Counter(), check_video=True)
        hit = cache.lookup(ann, None, check_video=False)
        assert hit is not None
        cache.close()

    def test_clear_wipes_all_entries(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        cache = ValidationCache(db_path)
        ann = tmp_path / "ann.json"
        ann.write_text('{"task_id": "t1"}')
        cache.store(ann, None, [], Counter(), check_video=False)
        cache.clear()
        hit = cache.lookup(ann, None, check_video=False)
        assert hit is None
        cache.close()

    def test_stats_tracking(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        cache = ValidationCache(db_path)
        ann = tmp_path / "ann.json"
        ann.write_text('{"task_id": "t1"}')
        cache.store(ann, None, [], Counter(), check_video=False)
        cache.lookup(ann, None, check_video=False)  # hit
        ann.write_text('{"task_id": "t2"}')
        cache.lookup(ann, None, check_video=False)  # miss
        assert cache.stats.hits == 1
        assert cache.stats.misses == 1
        cache.close()

    def test_corrupted_db_degrades_gracefully(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        db_path.write_text("this is not a sqlite database")
        cache = ValidationCache(db_path)
        ann = tmp_path / "ann.json"
        ann.write_text('{"task_id": "t1"}')
        hit = cache.lookup(ann, None, check_video=False)
        assert hit is None
        cache.close()

    def test_malformed_row_is_treated_as_miss_and_evicted(self, tmp_path: Path) -> None:
        """A corrupt cache row (bad JSON) must not crash lookup; the row is evicted."""
        import sqlite3

        from datamaite._cache import ValidationCache, fingerprint_file

        db = tmp_path / "cache.db"
        ann = tmp_path / "ann.json"
        ann.write_text("{}")
        cache = ValidationCache(db_path=db)

        # Poison the row: findings column is not valid JSON.
        fp = fingerprint_file(ann)
        assert fp is not None
        con = sqlite3.connect(db)
        con.execute(
            """
            INSERT OR REPLACE INTO file_results
            (file_path, ann_hash, ann_size, ann_mtime, vid_hash, vid_size, vid_mtime,
             check_video, findings, labels, validated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (str(ann), fp.hash, fp.size, fp.mtime, None, None, None, 0, "not-json", "{}"),
        )
        con.commit()
        con.close()

        # Lookup returns miss, does not raise.
        hit = cache.lookup(ann, None, check_video=False)
        assert hit is None
        assert cache.stats.misses == 1

        # Row was evicted; a fresh lookup still returns miss (not a stale hit).
        hit2 = cache.lookup(ann, None, check_video=False)
        assert hit2 is None
        cache.close()

    def test_default_path(self) -> None:
        path = ValidationCache.default_path()
        assert "datamaite" in str(path)
        assert path.name == "validation.db"

    def test_flush_persists_pending_writes_without_close(self, tmp_path: Path) -> None:
        """Calling flush() must persist pending writes without requiring close().

        The cache batches commits every 50 writes. On SIGINT / SIGTERM
        the finally: close() may never run, so flush() must be safe to
        call independently and must make the pending writes queryable
        from another connection.
        """
        import sqlite3

        db_path = tmp_path / "test.db"
        cache = ValidationCache(db_path)
        # Write a handful of entries (fewer than the 50-write batch threshold)
        for i in range(3):
            ann = tmp_path / f"ann_{i}.json"
            ann.write_text(f'{{"task_id": "t{i}"}}')
            cache.store(ann, None, [], Counter(), check_video=False)
        assert cache._pending_writes == 3
        cache.flush()
        assert cache._pending_writes == 0

        # Read through an independent connection (same file), not through
        # the cache's own handle -- this proves the write hit disk.
        other = sqlite3.connect(str(db_path))
        rows = other.execute("SELECT COUNT(*) FROM file_results").fetchone()
        assert rows[0] == 3
        other.close()
        cache.close()

    def test_worker_crash_finding_not_cached(self, tmp_path: Path) -> None:
        """Worker-crash findings must not be persisted to the cache.

        Crash findings indicate transient failures (OOM, pickle error,
        killed worker). Caching them would permanently mask the bug on
        subsequent runs. ``_validate_pairs_cached`` is responsible for
        skipping ``cache.store()`` when the finding list contains a
        ``worker_crash`` entry.
        """
        from datamaite._formats.hmie import SnippetPair
        from datamaite.validation import _FindingAccumulator, _validate_pairs_cached

        # Build a real pair so fingerprinting succeeds
        ann = tmp_path / "ann.json"
        ann.write_text('{"task_id": "t1"}')
        pair = SnippetPair(annotation_path=ann, video_path=None)

        db_path = tmp_path / "test.db"
        cache = ValidationCache(db_path)

        # Monkey-patch the upstream worker helper to return a worker_crash
        # finding, so we test the cache-store skip path end-to-end.
        from datamaite import validation as validation_module

        def _fake_validate_and_yield(pairs, *, check_video, workers):  # type: ignore[no-untyped-def]
            for p in pairs:
                crash = Finding(
                    severity=Severity.ERROR,
                    path=p.annotation_path,
                    check="worker_crash",
                    message="simulated",
                )
                yield p, [crash], Counter()

        original = validation_module._validate_and_yield
        validation_module._validate_and_yield = _fake_validate_and_yield  # type: ignore[assignment]
        try:
            accumulator = _FindingAccumulator(None)
            aggregate_labels: Counter[str] = Counter()
            _validate_pairs_cached(
                [pair],
                cache,
                accumulator,
                aggregate_labels,
                check_video=False,
                workers=1,
            )
        finally:
            validation_module._validate_and_yield = original  # type: ignore[assignment]

        # The crash finding was surfaced to the caller ...
        assert any(f.check == "worker_crash" for f in accumulator.findings)
        # ... but the cache must have no row for that pair.
        cache.flush()
        assert cache._db is not None
        row = cache._db.execute(
            "SELECT COUNT(*) FROM file_results WHERE file_path = ?",
            (str(ann),),
        ).fetchone()
        assert row[0] == 0
        cache.close()
