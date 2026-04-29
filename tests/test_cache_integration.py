"""Integration tests for validation cache with the full pipeline."""

from __future__ import annotations

from pathlib import Path

from databridge._cache import ValidationCache
from databridge.validation import validate
from tests._hmie_factory import FullVideoSpec, SnippetSpec, make_hmie_dataset


class TestCacheIntegration:
    def test_second_run_uses_cache(self, tmp_path: Path) -> None:
        root = make_hmie_dataset(
            tmp_path / "hmie",
            [FullVideoSpec(name="v_000000", snippets=[SnippetSpec(name="v_000001")])],
        )
        db_path = tmp_path / "cache.db"
        cache = ValidationCache(db_path)

        result1 = validate(root, check_video_integrity=False, cache=cache, workers=1)
        first_misses = cache.stats.misses
        assert cache.stats.hits == 0
        assert first_misses > 0

        cache.stats.hits = 0
        cache.stats.misses = 0

        result2 = validate(root, check_video_integrity=False, cache=cache, workers=1)
        assert cache.stats.hits == first_misses
        assert cache.stats.misses == 0
        assert result1.passed == result2.passed
        assert len(result1.findings) == len(result2.findings)
        cache.close()

    def test_cache_invalidated_on_file_change(self, tmp_path: Path) -> None:
        root = make_hmie_dataset(
            tmp_path / "hmie",
            [FullVideoSpec(name="v_000000", snippets=[SnippetSpec(name="v_000001")])],
        )
        db_path = tmp_path / "cache.db"
        cache = ValidationCache(db_path)

        validate(root, check_video_integrity=False, cache=cache, workers=1)

        # Find and modify an annotation file in a labeler subdir
        for ann in root.rglob("*.json"):
            if ann.parent.name not in (root.name, "v_000000") and "CDAO" in ann.name:
                ann.write_text('{"task_id": "modified", "response": {"annotations": {}}}')
                break

        cache.stats.hits = 0
        cache.stats.misses = 0
        validate(root, check_video_integrity=False, cache=cache, workers=1)
        assert cache.stats.misses >= 1
        cache.close()

    def test_no_cache_works_like_before(self, tmp_path: Path) -> None:
        root = make_hmie_dataset(
            tmp_path / "hmie",
            [FullVideoSpec(name="v_000000", snippets=[SnippetSpec(name="v_000001")])],
        )
        result = validate(root, check_video_integrity=False, cache=None, workers=1)
        assert result.cache_hits == 0
        assert result.cache_misses == 0
