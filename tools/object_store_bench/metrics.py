"""Small S3 API-call meter for the manual object-store benchmarks."""

from __future__ import annotations

import re
import threading
from collections import Counter
from contextlib import AbstractContextManager
from typing import Any

from aiobotocore.client import AioBaseClient


class S3CallMeter(AbstractContextManager["S3CallMeter"]):
    """Count S3 API calls and object-body bytes requested by s3fs.

    The meter hooks aiobotocore's request boundary so it sees requests made by
    both synchronous and asynchronous s3fs paths. Keep it developer-only and
    record dependency versions when comparing runs.
    """

    def __init__(self) -> None:
        self.operations: Counter[str] = Counter()
        self.download_bytes = 0
        self.upload_bytes = 0
        self.listed_objects = 0
        self.failed_requests = 0
        self._lock = threading.Lock()
        self._original: Any = None

    def __enter__(self) -> S3CallMeter:
        original = AioBaseClient._make_api_call
        self._original = original
        meter = self

        async def tracked(client: Any, operation_name: str, api_params: dict[str, Any]) -> Any:
            method = _snake_case(operation_name)
            upload_size = _body_size(api_params) if method in {"put_object", "upload_part"} else 0
            try:
                response = await original(client, operation_name, api_params)
            except Exception:
                meter._record(method, None, upload_size)
                raise
            meter._record(method, response, upload_size)
            return response

        AioBaseClient._make_api_call = tracked
        return self

    def __exit__(self, *exc: object) -> None:
        AioBaseClient._make_api_call = self._original

    def _record(self, method: str, response: dict[str, Any] | None, upload_size: int) -> None:
        with self._lock:
            self.operations[method] += 1
            if response is None:
                self.failed_requests += 1
            elif method == "get_object":
                self.download_bytes += int(response.get("ContentLength", 0) or 0)
            elif method in {"put_object", "upload_part"}:
                self.upload_bytes += upload_size
            elif method in {"list_objects", "list_objects_v2"}:
                self.listed_objects += int(response.get("KeyCount", len(response.get("Contents", ()))) or 0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "operations": dict(sorted(self.operations.items())),
            "download_bytes": self.download_bytes,
            "upload_bytes": self.upload_bytes,
            "listed_objects": self.listed_objects,
            "failed_requests": self.failed_requests,
        }


def _snake_case(value: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()


def _body_size(request: dict[str, Any]) -> int:
    explicit = request.get("ContentLength")
    if explicit is not None:
        return int(explicit)
    body = request.get("Body")
    if isinstance(body, (bytes, bytearray, memoryview)):
        return len(body)
    try:
        current = body.tell()
        body.seek(0, 2)
        size = body.tell() - current
        body.seek(current)
        return max(0, int(size))
    except (AttributeError, OSError, TypeError, ValueError):
        return 0
