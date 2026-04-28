"""Skill: run 2-5 independent search queries in parallel."""
from __future__ import annotations
import concurrent.futures
from ora_v2.skills.builtins._brave import brave_search

DESCRIPTION = "Run 2-5 independent search queries in parallel and merge results. Use ONLY when the task has clearly distinct sub-questions that don't depend on each other. Prefer this over calling web_search multiple times separately."
ARGS = {"queries": "list of search query strings (2-5 queries)"}


def _single(query: str) -> str:
    return f"[{query}]\n{brave_search(query, count=3)[:800]}"


def run(queries: list[str]) -> str:
    if not queries:
        return "ERROR: queries list is empty"
    queries = queries[:5]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as ex:
        results = list(ex.map(_single, queries))
    return "\n\n".join(results)[:4000]
