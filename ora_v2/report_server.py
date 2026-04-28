"""ReportServer: minimal HTTP file server for research reports."""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

logger = logging.getLogger(__name__)

_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def save_report(run_id: str, goal: str, report: str, server_url: str) -> str:
    """Save report as HTML and return download URL."""
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in goal[:40]).strip() or "report"
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{safe_name}_{run_id[:8]}.html"
    filepath = _REPORTS_DIR / filename

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head><meta charset="utf-8"><title>ORA — {goal[:60]}</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 720px; margin: 2em auto; padding: 0 1em; line-height: 1.6; }}
pre {{ background: #f5f5f5; padding: 1em; border-radius: 6px; overflow-x: auto; }}
.meta {{ color: #666; font-size: 0.9em; }}
h1 {{ font-size: 1.4em; }}
</style></head>
<body>
<h1>ORA — исследование</h1>
<p class="meta">{ts} · {run_id[:8]}</p>
<pre>{report}</pre>
</body></html>"""

    filepath.write_text(html, encoding="utf-8")
    server_url = server_url.rstrip("/")
    url = f"{server_url}/{filename}"
    logger.info("Report saved: %s -> %s", filepath, url)
    return url


class _Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        super().__init__(*args, directory=str(_REPORTS_DIR), **kwargs)

    def log_message(self, fmt, *args):
        logger.debug("ReportServer: " + fmt, *args)


class ReportServer:
    """Lightweight HTTP server serving saved reports.
    Runs in a daemon thread. Uses Python stdlib — zero extra dependencies."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._host = host
        self._port = port
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.url = f"http://{_public_ip()}:{port}"

    def start(self) -> None:
        _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self._server = HTTPServer((self._host, self._port), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("ReportServer started on %s", self.url)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
            logger.info("ReportServer stopped")


def _public_ip() -> str:
    """Return best guess for public IP — configurable via env for portability."""
    from ora_v2.config import get_settings
    override = get_settings().report_server_host or ""
    if override:
        return override
    import subprocess, sys
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import urllib.request; print(urllib.request.urlopen('https://api.ipify.org', timeout=5).read().decode())"],
            capture_output=True, text=True, timeout=10,
        )
        ip = result.stdout.strip()
        if ip:
            return ip
    except Exception:
        pass
    return "localhost"
