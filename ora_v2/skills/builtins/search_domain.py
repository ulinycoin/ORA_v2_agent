"""Skill: domain-focused web search."""
from ora_v2.skills.builtins._brave import brave_search

DESCRIPTION = "Search within a specific domain or source type (reddit, github, docs, news, academic, or site.com). Use ONLY when the source is the primary constraint."
ARGS = {
    "query": "search query string",
    "domain": "domain: 'reddit', 'github', 'news', 'docs', 'academic', or a specific site like 'stackoverflow.com'",
}

_SITES = {"reddit": "reddit.com", "github": "github.com", "stackoverflow": "stackoverflow.com", "academic": "arxiv.org"}
_PREFIXES = {"docs": "official documentation", "news": "news"}


def run(query: str, domain: str) -> str:
    key = domain.lower().strip()
    site = _SITES.get(key, key if "." in key else None)
    prefix = _PREFIXES.get(key, "")
    full_query = f"{prefix} {query}".strip() if prefix else query
    return brave_search(full_query, count=5, site=site)[:2000]
