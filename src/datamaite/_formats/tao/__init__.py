"""TAO (Tracking Any Object) format package."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datamaite._formats.tao.loader import TaoLoader, load_tao
    from datamaite._formats.tao.writer import TaoWriter

__all__ = ["TaoLoader", "TaoWriter", "load_tao"]


def __getattr__(name: str) -> object:
    """Lazily resolve TAO loader and writer exports."""
    if name in {"TaoLoader", "load_tao"}:
        from datamaite._formats.tao.loader import TaoLoader, load_tao

        exports = {"TaoLoader": TaoLoader, "load_tao": load_tao}
    elif name == "TaoWriter":
        from datamaite._formats.tao.writer import TaoWriter

        exports = {"TaoWriter": TaoWriter}
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = exports[name]
    globals()[name] = value
    return value
