"""Guardrails that keep format modules backend-neutral."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_FORMAT_ROOT = _REPO_ROOT / "src/datamaite/_formats"
_FORBIDDEN_NATIVE_CALLS = {
    ("shutil", "copy"),
    ("shutil", "copy2"),
    ("shutil", "copyfile"),
    ("shutil", "rmtree"),
    ("cv2", "imread"),
    ("cv2", "imwrite"),
    ("cv2", "VideoCapture"),
    ("av", "open"),
    ("os", "path", "getsize"),
}
_PATH_CALL = ("pathlib", "Path")


def _dotted_name(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value)
        if prefix is not None:
            return (*prefix, node.attr)
    return None


def _import_aliases(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                bound = imported.asname or imported.name.split(".")[0]
                target = tuple(imported.name.split(".")) if imported.asname else (bound,)
                aliases[bound] = target
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            module = tuple(node.module.split("."))
            for imported in node.names:
                if imported.name == "*":
                    continue
                aliases[imported.asname or imported.name] = (*module, imported.name)
    return aliases


def _resolve_call(node: ast.AST, aliases: dict[str, tuple[str, ...]]) -> tuple[str, ...] | None:
    name = _dotted_name(node)
    if name is None:
        return None
    target = aliases.get(name[0])
    return (*target, *name[1:]) if target is not None else name


def _scan_source(source: str, *, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    aliases = _import_aliases(tree)
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call = _resolve_call(node.func, aliases)
        if call in _FORBIDDEN_NATIVE_CALLS:
            findings.append(f"{filename}:{node.lineno}: direct {'.'.join(call)} call")
        if call == _PATH_CALL and (
            not node.args
            or any(not isinstance(arg, ast.Constant) or not isinstance(arg.value, str) for arg in node.args)
        ):
            findings.append(f"{filename}:{node.lineno}: storage-sensitive pathlib.Path(...) coercion")
    return findings


def _format_paths() -> list[Path]:
    return sorted(_FORMAT_ROOT.rglob("*.py"))


def _scan_format_modules() -> tuple[list[Path], list[str]]:
    paths = _format_paths()
    findings: list[str] = []
    for path in paths:
        findings.extend(_scan_source(path.read_text(encoding="utf-8"), filename=str(path)))
    return paths, findings


def test_format_modules_do_not_own_storage_or_native_media_io() -> None:
    paths, findings = _scan_format_modules()
    assert paths, f"storage architecture scan found no Python files under {_FORMAT_ROOT}"
    assert not findings, "\n".join(findings)


def test_format_scan_is_independent_of_working_directory(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    paths, findings = _scan_format_modules()
    assert paths
    assert not findings, "\n".join(findings)


def test_guard_resolves_aliases_and_all_prohibited_constructs() -> None:
    source = """
import shutil as files
from cv2 import imread as read_image, imwrite, VideoCapture as Capture
import av as media
from os import path as osp
import pathlib as paths
from pathlib import Path as LocalPath
files.copy(source, dest)
files.copy2(source, dest)
files.copyfile(source, dest)
files.rmtree(dest)
read_image(source)
imwrite(dest, image)
Capture(video_path)
media.open(video_path)
osp.getsize(source)
paths.Path(root)
LocalPath(frame_file)
"""

    findings = _scan_source(source, filename="fixture.py")

    assert len(findings) == 11
    assert any("direct av.open call" in finding for finding in findings)
    assert any("direct os.path.getsize call" in finding for finding in findings)
    assert sum("storage-sensitive pathlib.Path" in finding for finding in findings) == 2


def test_guard_allows_literal_paths_and_pure_filename_manipulation() -> None:
    source = """
from pathlib import Path, PurePosixPath
relative = Path("labels") / "data.txt"
suffix = PurePosixPath(file_name).suffix
"""

    assert _scan_source(source, filename="fixture.py") == []
