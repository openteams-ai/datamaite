"""Minimal HTTP server with Range support that counts bytes served.

Used as the fixed point for the probe transport benchmark: every
approach in ``bench.py`` reads from this server, so bytes/requests
served are a durable, host-independent measure of what each transport
actually pulled over the wire.

Endpoints:
  GET /video.mp4   -- the file (206 partial content when Range given)
  GET /stats       -- {"bytes": N, "requests": M} since last reset
  GET /reset       -- zero the counters

Usage:
    python range_server.py <video_path> <port> [latency_ms]
"""

import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VIDEO = sys.argv[1]
PORT = int(sys.argv[2])
LATENCY_S = (float(sys.argv[3]) / 1000.0) if len(sys.argv) > 3 else 0.0

_lock = threading.Lock()
_stats = {"bytes": 0, "requests": 0}

_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _parse_range(rng_header: str | None, size: int) -> tuple[int, int, int]:
    """Return (start, end, status) for an optional 'Range: bytes=...' header."""
    start, end = 0, size - 1
    if not rng_header:
        return start, end, 200
    m = _RANGE_RE.match(rng_header)
    if not m:
        return start, end, 200
    if m.group(1):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else size - 1
    else:  # suffix range: bytes=-N
        start = size - int(m.group(2))
    end = min(end, size - 1)
    return start, end, 206


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silence
        pass

    def do_HEAD(self):
        if self.path.startswith("/video.mp4"):
            size = os.path.getsize(VIDEO)
            self.send_response(200)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "video/mp4")
            self.end_headers()
            # HEAD probes carry no body, but they are still a request against
            # the fixed point -- count them (bytes += 0) so /stats reflects
            # every hit a transport makes, not just the ones that read bytes.
            with _lock:
                _stats["requests"] += 1
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/stats":
            with _lock:
                body = json.dumps(_stats).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/reset":
            with _lock:
                _stats["bytes"] = 0
                _stats["requests"] = 0
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")
            return
        if not self.path.startswith("/video.mp4"):
            self.send_response(404)
            self.end_headers()
            return

        if LATENCY_S:
            time.sleep(LATENCY_S)
        size = os.path.getsize(VIDEO)
        start, end, status = _parse_range(self.headers.get("Range"), size)
        length = end - start + 1
        self.send_response(status)
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Type", "video/mp4")
        self.end_headers()
        sent = 0
        try:
            with open(VIDEO, "rb") as fh:
                fh.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = fh.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    sent += len(chunk)
                    remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client hung up early (ffmpeg does this constantly) -- count what was sent
        finally:
            with _lock:
                _stats["bytes"] += sent
                _stats["requests"] += 1


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
