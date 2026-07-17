"""Destination policy (#55): write()/convert() mode="error"|"replace"|"append".

The policy is enforced centrally in ``writers.write()`` so every registered
writer inherits it; these tests drive it through the cheapest real writer
(MOTChallenge with an empty dataset writes nothing, but the policy check runs
first) plus loaded round-trip datasets where file content matters.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from datamaite import DatasetFormat, Task, convert, load_ic, load_od, write
from datamaite._formats.huggingface_video_classification.loader import (
    load_huggingface_video_classification,
)
from datamaite.image_classification import ImageClassificationDataset
from datamaite.loaders import load, load_mot
from datamaite.model import BoxTrackDataset, VideoClassificationDataset
from datamaite.object_detection import ObjectDetectionDataset
from datamaite.writers import WriterKey, available_writer_keys

from .test_coco_writer import _coco_root
from .test_coco_writer import _fingerprint as _od_fingerprint
from .test_conversion_matrix import (
    SOURCE_BUILDERS,
    WRITABLE_FORMATS,
    _neutral_fingerprint,
    _reload_root,
)
from .test_huggingface_video_classification_writer import (
    _classification_fingerprint,
    _write_video,
)
from .test_huggingface_video_classification_writer import (
    _dataset as _hfvc_dataset,
)
from .test_huggingface_video_classification_writer import (
    _sample as _hfvc_sample,
)
from .test_motchallenge_writer import write_mot_sequence
from .test_yolo_image_classification import _dataset as _yolo_ic_root
from .test_yolo_image_classification import _write_image as _write_ic_image
from .test_yolo_object_detection import _fingerprint as _yolo_od_fingerprint
from .test_yolo_object_detection import _od_dataset as _yolo_od_root

_EMPTY = BoxTrackDataset(sequences=(), categories={})


def _assert_dest_files_exactly(dest: Path, written: list[Path]) -> None:
    """Every file on disk under ``dest`` must have been written by the last write.

    The reload-and-fingerprint assertions alone cannot catch stale files for
    manifest-driven formats (MOT gt.txt, COCO instances.json, HF metadata.csv):
    the replacing write fully overwrites the manifest, so leftover frame/image
    files from the previous write would not change the reloaded fingerprint.
    This disk-level check is what actually guards the issue-#55 regression.
    """
    on_disk = {path for path in dest.rglob("*") if path.is_file()}
    assert on_disk == set(written)


class TestModeValidation:
    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="mode"):
            write(_EMPTY, tmp_path / "out", output_format="motchallenge", mode="clobber")

    def test_mode_is_case_insensitive(self, tmp_path: Path) -> None:
        write(_EMPTY, tmp_path / "out", output_format="motchallenge", mode="ERROR")


class TestWriteModeEnum:
    """Issue #55 Fix A4: `WriteMode` enum members and equivalent strings behave identically."""

    def test_enum_replace_matches_string_replace(self, tmp_path: Path) -> None:
        from datamaite import WriteMode

        enum_out = tmp_path / "enum_out"
        (enum_out / "old_dir").mkdir(parents=True)
        (enum_out / "old_dir" / "stale.txt").write_text("old", encoding="utf-8")
        write(_EMPTY, enum_out, output_format="motchallenge", mode=WriteMode.REPLACE)
        assert list(enum_out.iterdir()) == []

        string_out = tmp_path / "string_out"
        (string_out / "old_dir").mkdir(parents=True)
        (string_out / "old_dir" / "stale.txt").write_text("old", encoding="utf-8")
        write(_EMPTY, string_out, output_format="motchallenge", mode="replace")
        assert list(string_out.iterdir()) == []

    def test_enum_error_matches_string_error(self, tmp_path: Path) -> None:
        from datamaite import WriteMode

        out = tmp_path / "out"
        out.mkdir()
        (out / "stale.txt").write_text("old", encoding="utf-8")
        with pytest.raises(FileExistsError):
            write(_EMPTY, out, output_format="motchallenge", mode=WriteMode.ERROR)

    def test_invalid_string_still_raises_with_enum_available(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="mode"):
            write(_EMPTY, tmp_path / "out", output_format="motchallenge", mode="clobber")


class TestErrorMode:
    def test_missing_dest_is_created(self, tmp_path: Path) -> None:
        write(_EMPTY, tmp_path / "out", output_format="motchallenge")
        assert (tmp_path / "out").is_dir()

    def test_empty_existing_dest_passes(self, tmp_path: Path) -> None:
        (tmp_path / "out").mkdir()
        write(_EMPTY, tmp_path / "out", output_format="motchallenge")

    def test_non_empty_dest_raises_by_default(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "stale.txt").write_text("old", encoding="utf-8")
        with pytest.raises(FileExistsError, match="replace"):
            write(_EMPTY, out, output_format="motchallenge")
        # The foreign file was not touched.
        assert (out / "stale.txt").read_text(encoding="utf-8") == "old"

    def test_hidden_file_counts_as_non_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / ".DS_Store").write_bytes(b"")
        with pytest.raises(FileExistsError):
            write(_EMPTY, out, output_format="motchallenge")

    def test_file_dest_raises_not_a_directory(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.write_text("i am a file", encoding="utf-8")
        for mode in ("error", "replace", "append"):
            with pytest.raises(NotADirectoryError):
                write(_EMPTY, out, output_format="motchallenge", mode=mode)


class TestValidateOptionsBeforeReplace:
    """Issue #55 Fix A1: invalid writer options must not trigger a replace-mode wipe.

    ``write(..., mode="replace")`` used to clear ``dest`` before the writer's
    own option validation (``validate_class_map``, ``_validate_split``, etc.)
    ran, so a plain user mistake (a bad ``class_map`` or ``split``) destroyed
    a non-empty destination and only then raised. Validation must now happen
    before ``_prepare_destination`` runs.
    """

    def test_invalid_class_map_does_not_wipe_non_empty_dest(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "precious.txt").write_text("keep me", encoding="utf-8")
        with pytest.raises(ValueError, match="class ids must be"):
            write(_EMPTY, out, output_format="motchallenge", mode="replace", class_map={"x": 0})
        assert (out / "precious.txt").read_text(encoding="utf-8") == "keep me"

    def test_invalid_split_does_not_wipe_non_empty_dest(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "precious.txt").write_text("keep me", encoding="utf-8")
        with pytest.raises(ValueError, match="split"):
            write(_EMPTY, out, output_format="motchallenge", mode="replace", split="bogus")
        assert (out / "precious.txt").read_text(encoding="utf-8") == "keep me"


# One minimal, empty dataset per task -- writer option validation runs before
# any writer touches the sequences/samples, so an empty dataset is enough to
# reach it. Reused across every case in the cross-writer regression table.
_EMPTY_OD = ObjectDetectionDataset(samples=())
_EMPTY_IC = ImageClassificationDataset(samples=())
_EMPTY_VC = VideoClassificationDataset(samples=(), categories={})

# (task, format, variant, minimal dataset, one known-invalid option, error-match).
# Every registered writer that validates an option that can raise from write()
# gets a row here; TestOptionValidationBeforeReplaceAllWriters below both
# exercises each row and asserts (via the coverage test) that no registered
# writer is missing from this table.
_OPTION_VALIDATION_CASES = [
    (Task.MOT, DatasetFormat.MOTCHALLENGE, "default", _EMPTY, {"split": "bogus"}, "split"),
    (Task.MOT, DatasetFormat.VISDRONE_VIDEO, "default", _EMPTY, {"variant": "nope"}, "variant"),
    (Task.MOT, DatasetFormat.TAO, "default", _EMPTY, {"split": "bogus"}, "split"),
    (
        Task.OD,
        DatasetFormat.COCO,
        "default",
        _EMPTY_OD,
        {"annotation_file_name": "../evil.json"},
        "annotation_file_name",
    ),
    (Task.OD, DatasetFormat.YOLO, "default", _EMPTY_OD, {"precision": 0}, "precision"),
    (
        Task.VC,
        DatasetFormat.HUGGINGFACE_VIDEO_CLASSIFICATION,
        "default",
        _EMPTY_VC,
        {"metadata_format": "xml"},
        "metadata_format",
    ),
    (
        Task.IC,
        DatasetFormat.HUGGINGFACE_VISION,
        "default",
        _EMPTY_IC,
        {"default_split": "../evil"},
        "default_split",
    ),
    (
        Task.OD,
        DatasetFormat.HUGGINGFACE_VISION,
        "default",
        _EMPTY_OD,
        {"metadata_format": "xml"},
        "metadata_format",
    ),
]

# Writers that intentionally have NO option that can raise from write()
# (HMIE's only option `labeler` is a path segment it does not validate; the
# YOLO IC writer's `default_split` is validated per-sample and only skips
# offending samples). Listed so the coverage test below fails loudly if a
# future writer adds a raising option without a validate_options override + a
# row in _OPTION_VALIDATION_CASES.
_NO_OPTION_VALIDATION_KEYS = {
    WriterKey(task=Task.MOT, format=DatasetFormat.HMIE, variant="default"),
    WriterKey(task=Task.IC, format=DatasetFormat.YOLO, variant="default"),
}


class TestOptionValidationBeforeReplaceAllWriters:
    """Issue #55 Fix A1, generalized: NO writer may clear a replace-mode dest before option validation.

    This parametrized table is the real safety net against future drift: if any
    writer validates an option inside its ``write()`` body (after
    ``_prepare_destination`` has already cleared the dest under
    ``mode="replace"``) without a ``validate_options`` override, the matching
    row here fails -- the sentinel file would already be gone.
    """

    @pytest.mark.parametrize(
        "case",
        _OPTION_VALIDATION_CASES,
        ids=[f"{case[1].value}:{case[0].value}" for case in _OPTION_VALIDATION_CASES],
    )
    def test_invalid_option_does_not_wipe_non_empty_dest(self, case: tuple, tmp_path: Path) -> None:
        _task, fmt, variant, dataset, bad_option, match = case
        out = tmp_path / "out"
        out.mkdir()
        (out / "precious.txt").write_text("keep me", encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            write(dataset, out, output_format=fmt, output_variant=variant, mode="replace", **bad_option)
        assert (out / "precious.txt").read_text(encoding="utf-8") == "keep me"

    def test_every_registered_writer_is_accounted_for(self) -> None:
        covered = {WriterKey(task=case[0], format=case[1], variant=case[2]) for case in _OPTION_VALIDATION_CASES}
        accounted = covered | _NO_OPTION_VALIDATION_KEYS
        registered = set(available_writer_keys())
        missing = registered - accounted
        assert not missing, (
            "Registered writer(s) not accounted for in the #55 option-validation regression guard: "
            f"{sorted(str(k) for k in missing)}. If the writer validates an option that can raise from "
            "write(), add a validate_options override plus a row in _OPTION_VALIDATION_CASES; otherwise "
            "add its key to _NO_OPTION_VALIDATION_KEYS."
        )


class TestReplaceMode:
    def test_replace_clears_dest_contents(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        (out / "old_dir").mkdir(parents=True)
        (out / "old_dir" / "stale.txt").write_text("old", encoding="utf-8")
        (out / "stale_top.txt").write_text("old", encoding="utf-8")
        write(_EMPTY, out, output_format="motchallenge", mode="replace")
        assert out.is_dir()
        assert list(out.iterdir()) == []

    def test_replace_refuses_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "sentinel.txt").write_text("keep me", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match=r"[Rr]efus"):
            write(_EMPTY, tmp_path, output_format="motchallenge", mode="replace")
        assert (tmp_path / "sentinel.txt").exists()

    def test_replace_refuses_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        (fake_home / "sentinel.txt").write_text("keep me", encoding="utf-8")
        monkeypatch.setenv("HOME", str(fake_home))
        with pytest.raises(ValueError, match=r"[Rr]efus"):
            write(_EMPTY, fake_home, output_format="motchallenge", mode="replace")
        assert (fake_home / "sentinel.txt").exists()

    def test_replace_refuses_ancestor_of_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A dest that CONTAINS cwd (but isn't cwd itself) must also be refused (#55 Fix A2)."""
        (tmp_path / "sentinel.txt").write_text("keep me", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        monkeypatch.chdir(sub)
        with pytest.raises(ValueError, match=r"[Rr]efus"):
            write(_EMPTY, tmp_path, output_format="motchallenge", mode="replace")
        assert (tmp_path / "sentinel.txt").exists()
        assert sub.exists()

    @pytest.mark.skipif(
        not hasattr(Path, "symlink_to"),
        reason="platform has no symlink support",
    )
    def test_replace_refuses_symlinked_dest(self, tmp_path: Path) -> None:
        """A `dest` that is ITSELF a symlink to a directory must be refused, not followed (#55 Fix A2)."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        sentinel = real_dir / "sentinel.txt"
        sentinel.write_text("keep me", encoding="utf-8")

        link = tmp_path / "link"
        try:
            link.symlink_to(real_dir, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"cannot create symlinks on this platform/user: {exc}")

        with pytest.raises(ValueError, match=r"[Rr]efus"):
            write(_EMPTY, link, output_format="motchallenge", mode="replace")
        assert sentinel.read_text(encoding="utf-8") == "keep me"

    @pytest.mark.skipif(
        not hasattr(Path, "symlink_to"),
        reason="platform has no symlink support",
    )
    def test_replace_unlinks_symlinked_directory_without_recursing(self, tmp_path: Path) -> None:
        """A symlink *entry* inside dest must be unlinked, not walked into.

        `_prepare_destination` special-cases `entry.is_symlink()` so that a
        symlink pointing at a directory outside `dest` is removed as a single
        directory-entry `unlink()`, never as `shutil.rmtree(entry)` (which
        would follow the link and delete the target's actual contents). This
        pins that behavior down with a real symlink instead of just inspecting
        the guard.
        """
        target_dir = tmp_path / "outside_target"
        target_dir.mkdir()
        sentinel = target_dir / "sentinel.txt"
        sentinel.write_text("do not delete me", encoding="utf-8")

        out = tmp_path / "out"
        out.mkdir()
        link = out / "linked_dir"
        try:
            link.symlink_to(target_dir, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"cannot create symlinks on this platform/user: {exc}")

        write(_EMPTY, out, output_format="motchallenge", mode="replace")

        assert not link.exists()
        assert not link.is_symlink()
        assert target_dir.is_dir()
        assert sentinel.read_text(encoding="utf-8") == "do not delete me"


class TestAppendMode:
    def test_append_leaves_foreign_files(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "foreign.txt").write_text("keep", encoding="utf-8")
        write(_EMPTY, out, output_format="motchallenge", mode="append")
        assert (out / "foreign.txt").read_text(encoding="utf-8") == "keep"


class TestPolicyOrdering:
    def test_cross_task_type_error_beats_destination_check(self, tmp_path: Path) -> None:
        # A wrong-task write must raise TypeError without clearing dest.
        out = tmp_path / "out"
        out.mkdir()
        (out / "precious.txt").write_text("keep", encoding="utf-8")
        with pytest.raises(TypeError, match="consumes"):
            write(_EMPTY, out, output_format="coco", mode="replace")
        assert (out / "precious.txt").exists()


class TestConvertMode:
    def test_convert_defaults_to_error_on_non_empty_dest(self, tmp_path: Path) -> None:
        write_mot_sequence(tmp_path / "src", gt_rows=["1,1,10,20,30,40,1,1,1"])
        out = tmp_path / "out"
        out.mkdir()
        (out / "stale.txt").write_text("old", encoding="utf-8")
        with pytest.raises(FileExistsError, match="replace"):
            convert(tmp_path / "src", out, input_format="motchallenge", output_format="motchallenge")

    def test_convert_replace_passes_through(self, tmp_path: Path) -> None:
        write_mot_sequence(tmp_path / "src", gt_rows=["1,1,10,20,30,40,1,1,1"])
        out = tmp_path / "out"
        out.mkdir()
        (out / "stale.txt").write_text("old", encoding="utf-8")
        convert(tmp_path / "src", out, input_format="motchallenge", output_format="motchallenge", mode="replace")
        assert not (out / "stale.txt").exists()
        assert (out / "train" / "MOT17-02" / "gt" / "gt.txt").exists()

    def test_mode_rejected_as_writer_option(self, tmp_path: Path) -> None:
        write_mot_sequence(tmp_path / "src", gt_rows=["1,1,10,20,30,40,1,1,1"])
        with pytest.raises(ValueError, match="mode"):
            convert(
                tmp_path / "src",
                tmp_path / "out",
                input_format="motchallenge",
                output_format="motchallenge",
                write_options={"mode": "replace"},
            )

    def test_convert_checks_destination_before_loading_src(self, tmp_path: Path) -> None:
        """The destination guardrail must run before `src` is loaded (#55).

        `src` here does not exist at all, so if `convert` loaded before
        checking `dest`, this would raise a loader error (e.g.
        `FileNotFoundError`) instead of the destination `FileExistsError`.
        Asserting the destination error proves the ordering: dest is checked
        first, so a doomed load never even starts and a bad `src` can't burn
        time before the cheap guardrail check would have caught the mistake.
        """
        missing_src = tmp_path / "does-not-exist"
        out = tmp_path / "out"
        out.mkdir()
        (out / "stale.txt").write_text("old", encoding="utf-8")
        with pytest.raises(FileExistsError, match="replace"):
            convert(
                missing_src,
                out,
                input_format="motchallenge",
                output_format="motchallenge",
                mode="error",
            )
        # dest untouched -- the (non-existent) source was never loaded.
        assert (out / "stale.txt").read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize("output_format", WRITABLE_FORMATS, ids=lambda f: f.value)
class TestReplaceLeavesOnlyTheNewDataset:
    """Issue #55 contamination scenario for the box-track matrix formats.

    Write dataset A, write a smaller dataset B to the same destination with
    mode="replace", reload the destination, and assert only B is present.
    """

    def test_reload_sees_only_b(self, output_format, tmp_path: Path) -> None:
        big_src = SOURCE_BUILDERS[output_format](tmp_path / "a")
        big = load(big_src, dataset_format=output_format)
        write_mot_sequence(tmp_path / "b", gt_rows=["1,1,10,20,30,40,1,1,1"], frame_count=1)
        small = load(tmp_path / "b", dataset_format="motchallenge")
        assert _neutral_fingerprint(small) != _neutral_fingerprint(big)

        dest = tmp_path / "out"
        write(big, dest, output_format=output_format)
        written = write(small, dest, output_format=output_format, mode="replace", verbose=True)
        _assert_dest_files_exactly(dest, written)

        reloaded = load(_reload_root(dest, output_format), dataset_format=output_format)
        assert _neutral_fingerprint(reloaded) == _neutral_fingerprint(small)


class TestReplaceLeavesOnlyTheNewDatasetOtherTasks:
    """Same contamination scenario for the OD / IC / video-classification writers."""

    def test_coco(self, tmp_path: Path) -> None:
        big = load_od(_coco_root(tmp_path / "a"), dataset_format="coco")
        small_payload = {
            "images": [{"id": 1, "file_name": "solo.jpg", "width": 10, "height": 10}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5]}],
            "categories": [{"id": 1, "name": "cat"}],
        }
        small_root = _coco_root(tmp_path / "b", small_payload, with_images=False)
        small = load_od(small_root, dataset_format="coco")

        dest = tmp_path / "out"
        write(big, dest, output_format="coco")
        written = write(small, dest, output_format="coco", mode="replace", verbose=True)
        _assert_dest_files_exactly(dest, written)

        assert _od_fingerprint(load_od(dest, dataset_format="coco")) == _od_fingerprint(small)

    def test_yolo_object_detection(self, tmp_path: Path) -> None:
        _yolo_od_root(tmp_path / "a")
        big = load_od(tmp_path / "a", dataset_format="yolo")
        _yolo_od_root(tmp_path / "b")
        (tmp_path / "b" / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (tmp_path / "b" / "images" / "val" / "b.png").unlink()
        (tmp_path / "b" / "labels" / "val" / "b.txt").unlink()
        small = load_od(tmp_path / "b", dataset_format="yolo")

        dest = tmp_path / "out"
        write(big, dest, output_format="yolo")
        written = write(small, dest, output_format="yolo", mode="replace", verbose=True)
        _assert_dest_files_exactly(dest, written)

        assert _yolo_od_fingerprint(load_od(dest, dataset_format="yolo")) == _yolo_od_fingerprint(small)

    def test_yolo_image_classification(self, tmp_path: Path) -> None:
        _yolo_ic_root(tmp_path / "a")
        big = load_ic(tmp_path / "a", dataset_format="yolo")
        _write_ic_image(tmp_path / "b" / "train" / "cat" / "only.jpg", b"only")
        small = load_ic(tmp_path / "b", dataset_format="yolo")

        dest = tmp_path / "out"
        write(big, dest, output_format="yolo")
        written = write(small, dest, output_format="yolo", mode="replace", verbose=True)
        _assert_dest_files_exactly(dest, written)

        reloaded = load_ic(dest, dataset_format="yolo")
        assert reloaded.sample_count == small.sample_count == 1

    def test_huggingface_video_classification(self, tmp_path: Path) -> None:
        big = _hfvc_dataset(
            _hfvc_sample(0, _write_video(tmp_path / "cat.mp4", b"cat"), label="cat", split="train"),
            _hfvc_sample(1, _write_video(tmp_path / "dog.mp4", b"dog"), label="dog", split="train"),
        )
        small = _hfvc_dataset(
            _hfvc_sample(0, _write_video(tmp_path / "solo.mp4", b"solo"), label="cat", split="train"),
        )

        dest = tmp_path / "out"
        write(big, dest, output_format="huggingface_video_classification")
        written = write(small, dest, output_format="huggingface_video_classification", mode="replace", verbose=True)
        _assert_dest_files_exactly(dest, written)

        # The metadata.csv round trip folds `file_name`/`label` into each
        # sample's `metadata` dict (see test_huggingface_video_classification_writer
        # .py::test_round_trip_from_folder_layout), so the raw in-memory `small`
        # is not a valid baseline -- it would differ from *any* single write of
        # `small`, independent of the replace-mode contamination behavior under
        # test. Compare against `small` written fresh through the same round
        # trip instead.
        solo_only = tmp_path / "solo_only"
        write(small, solo_only, output_format="huggingface_video_classification")
        expected = load_huggingface_video_classification(solo_only)

        reloaded = load_huggingface_video_classification(dest)
        assert _classification_fingerprint(reloaded) == _classification_fingerprint(expected)


class TestConvertReplaceSourceOverlap:
    """Issue #55 follow-up: `convert(src, dest, mode="replace")` must not clear a dest
    that is (or contains) the source, or a symlink aliasing it -- the clear happens
    after the load but before the writer reads the source's lazy media, so an overlap
    silently destroys the source mid-conversion.
    """

    def test_convert_replace_refuses_dest_equal_to_src(self, tmp_path: Path) -> None:
        src = tmp_path / "ds"
        write_mot_sequence(src, gt_rows=["1,1,10,20,30,40,1,1,1"])
        with pytest.raises(ValueError, match=r"[Rr]efus"):
            convert(src, src, input_format="motchallenge", output_format="motchallenge", mode="replace")
        assert (src / "train" / "MOT17-02" / "gt" / "gt.txt").exists()

    def test_convert_replace_refuses_dest_ancestor_of_src(self, tmp_path: Path) -> None:
        parent = tmp_path / "parent"
        src = parent / "inner"
        write_mot_sequence(src, gt_rows=["1,1,10,20,30,40,1,1,1"])
        with pytest.raises(ValueError, match=r"[Rr]efus"):
            convert(src, parent, input_format="motchallenge", output_format="motchallenge", mode="replace")
        assert (src / "train" / "MOT17-02" / "gt" / "gt.txt").exists()

    @pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="platform has no symlink support")
    def test_convert_replace_refuses_symlink_alias_of_src(self, tmp_path: Path) -> None:
        src = tmp_path / "ds"
        write_mot_sequence(src, gt_rows=["1,1,10,20,30,40,1,1,1"])
        alias = tmp_path / "alias"
        try:
            alias.symlink_to(src, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"cannot create symlinks on this platform/user: {exc}")
        with pytest.raises(ValueError, match=r"[Rr]efus"):
            convert(src, alias, input_format="motchallenge", output_format="motchallenge", mode="replace")
        assert (src / "train" / "MOT17-02" / "gt" / "gt.txt").exists()


class TestReplaceEmptyDestinationGuards:
    """Issue #55 follow-up: an *empty* destination must still hit the replace safety
    checks -- the empty-directory shortcut previously returned before the symlink /
    protected-path guards, so an empty symlink / cwd / home slipped through.
    """

    @pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="platform has no symlink support")
    def test_replace_refuses_empty_symlinked_dest(self, tmp_path: Path) -> None:
        real_dir = tmp_path / "real"
        real_dir.mkdir()  # empty on purpose
        link = tmp_path / "link"
        try:
            link.symlink_to(real_dir, target_is_directory=True)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"cannot create symlinks on this platform/user: {exc}")
        with pytest.raises(ValueError, match=r"[Rr]efus"):
            write(_EMPTY, link, output_format="motchallenge", mode="replace")
        assert link.is_symlink()  # not followed / cleared

    def test_replace_refuses_empty_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        empty = tmp_path / "empty_cwd"
        empty.mkdir()
        monkeypatch.chdir(empty)
        with pytest.raises(ValueError, match=r"[Rr]efus"):
            write(_EMPTY, empty, output_format="motchallenge", mode="replace")

    def test_replace_refuses_empty_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        empty_home = tmp_path / "empty_home"
        empty_home.mkdir()
        monkeypatch.setenv("HOME", str(empty_home))
        with pytest.raises(ValueError, match=r"[Rr]efus"):
            write(_EMPTY, empty_home, output_format="motchallenge", mode="replace")


class TestPreflightRejectsDangerousOptions:
    """Issue #55 follow-up: options that previously passed preflight but crashed
    *after* the replace clear (COCO `.`/`..`, YOLO non-int/`<1` precision) must be
    rejected before any destination is touched, leaving a sentinel intact.
    """

    @pytest.mark.parametrize("bad_name", ["..", "."])
    def test_coco_dot_annotation_name_rejected_before_replace(self, bad_name: str, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "precious.txt").write_text("keep me", encoding="utf-8")
        with pytest.raises(ValueError, match="annotation_file_name"):
            write(_EMPTY_OD, out, output_format="coco", mode="replace", annotation_file_name=bad_name)
        assert (out / "precious.txt").read_text(encoding="utf-8") == "keep me"

    @pytest.mark.parametrize("bad_precision", [1.5, True, 0])
    def test_yolo_bad_precision_rejected_before_replace(self, bad_precision: object, tmp_path: Path) -> None:
        out = tmp_path / "out"
        out.mkdir()
        (out / "precious.txt").write_text("keep me", encoding="utf-8")
        with pytest.raises(ValueError, match="precision"):
            write(_EMPTY_OD, out, output_format="yolo", mode="replace", precision=bad_precision)
        assert (out / "precious.txt").read_text(encoding="utf-8") == "keep me"


class TestWriteReplaceSourceUnderDestination:
    """Issue #55 follow-up: the module-level `write(dataset, dest, mode="replace")`
    must not clear a `dest` that contains the loaded dataset's own lazy media --
    the writer reads that media after the clear, so it would be destroyed. The
    earlier overlap fix only covered `convert()`; this covers direct `write()`.
    """

    def test_write_replace_refuses_reloading_source_dir(self, tmp_path: Path) -> None:
        src = tmp_path / "ds"
        write_mot_sequence(src, gt_rows=["1,1,10,20,30,40,1,1,1"])
        dataset = load_mot(src, dataset_format="motchallenge")
        with pytest.raises(ValueError, match=r"[Rr]efus"):
            write(dataset, src, output_format="motchallenge", mode="replace")
        # source frames + annotations survive untouched
        assert (src / "train" / "MOT17-02" / "gt" / "gt.txt").exists()
        assert list((src / "train" / "MOT17-02" / "img1").glob("*.jpg"))

    def test_write_replace_allows_unrelated_dest(self, tmp_path: Path) -> None:
        src = tmp_path / "ds"
        write_mot_sequence(src, gt_rows=["1,1,10,20,30,40,1,1,1"])
        dataset = load_mot(src, dataset_format="motchallenge")
        out = tmp_path / "out"  # disjoint from src -> allowed
        write(dataset, out, output_format="motchallenge", mode="replace")
        assert (out / "train" / "MOT17-02" / "gt" / "gt.txt").exists()
