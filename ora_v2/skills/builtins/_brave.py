"""Shared Brave Search API helper."""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

import certifi

logger = logging.getLogger(__name__)

_API_URL = "https://api.search.brave.com/res/v1/web/search"
_ssl_ctx = ssl.create_default_context(cafile=certifi.where())

# In-process cache: cache_key → (result, expires_at)
# TTL=600s — long enough to cover a full agent run (typically <15 min)
_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_TTL = 600


def _cache_key(query: str, count: int, freshness: str | None, site: str | None) -> str:
    raw = f"{query}|{count}|{freshness}|{site}"
    return hashlib.md5(raw.lower().encode()).hexdigest()


def _get_key() -> str:
    from ora_v2.config import get_settings
    key = get_settings().brave_api_key.strip()
    if not key:
        raise RuntimeError("BRAVE_API_KEY is not set")
    return key


def brave_search(
    query: str,
    *,
    count: int = 5,
    freshness: str | None = None,
    site: str | None = None,
    timeout: int = 30,
) -> str:
    key = _cache_key(query, count, freshness, site)
    cached, expires = _CACHE.get(key, ("", 0.0))
    if cached and time.monotonic() < expires:
        logger.debug("brave_search cache hit: %r", query[:60])
        return cached

    q = f"site:{site} {query}" if site else query
    params: dict[str, str | int] = {"q": q, "count": min(count, 20)}
    if freshness:
        params["freshness"] = freshness

    url = f"{_API_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": _get_key(),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx) as resp:
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            data = json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"Brave API error {e.code}: {body}") from e

    results = (data.get("web") or {}).get("results") or []
    if not results:
        return "No results found."

    lines = []
    for r in results:
        title = r.get("title", "").strip()
        url_r = r.get("url", "").strip()
        desc = (r.get("description") or "").strip()
        age = r.get("age", "")
        line = f"• {title}"
        if age:
            line += f" [{age}]"
        line += f"\n  {url_r}"
        if desc:
            line += f"\n  {desc}"
        lines.append(line)

    result = "\n\n".join(lines)
    _CACHE[key] = (result, time.monotonic() + _CACHE_TTL)
    return result
