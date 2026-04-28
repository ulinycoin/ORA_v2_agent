"""Skill: temporal web search — recent information within a time window."""
from ora_v2.skills.builtins._brave import brave_search

DESCRIPTION = "Search for information published within a specific time window (days param). Use ONLY when recency is the primary constraint — prices, API changes, news, releases. One call per topic — do NOT repeat with the same query."
ARGS = {
    "query": "search query string",
    "days": "integer — how many days back to look (e.g. 7, 30, 90)",
}

_FRESHNESS = {1: "pd", 7: "pw", 30: "pm", 365: "py"}


def _to_freshness(days: int) -> str:
    for threshold, code in sorted(_FRESHNESS.items()):
        if days <= threshold:
            return code
    return "py"


def run(query: str, days: int = 30) -> str:
    return brave_search(query, count=5, freshness=_to_freshness(int(days)))[:2000]
