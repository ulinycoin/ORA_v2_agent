"""Skill: single-shot web search via Brave Search API."""
from ora_v2.skills.builtins._brave import brave_search

DESCRIPTION = "Search the web for a single query. Use for quick, self-contained facts. Do NOT use if you need multiple angles (use web_search_multi), recency (use search_recent), or domain filtering (use search_domain). Do NOT repeat a query you already searched."
ARGS = {"query": "search query string"}


def run(query: str) -> str:
    return brave_search(query, count=5)[:2000]
