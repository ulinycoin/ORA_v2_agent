"""Skill: verify a claim against independent sources."""
from __future__ import annotations
from ora_v2.skills.builtins._brave import brave_search

DESCRIPTION = "Verify a specific claim or fact against independent sources. Use ONLY after you have a concrete fact to check. Do NOT use for general research."
ARGS = {"claim": "the specific claim or fact to verify (be precise, include numbers/dates if known)"}


def run(claim: str) -> str:
    search_results = brave_search(claim, count=5)
    prompt = (
        f"Verify the following claim using the search results below.\n\n"
        f"CLAIM: {claim}\n\n"
        f"SEARCH RESULTS:\n{search_results}\n\n"
        "Return a structured verdict:\n"
        "- VERDICT: confirmed / contradicted / uncertain\n"
        "- EVIDENCE: key supporting or contradicting facts with source names\n"
        "- CAVEATS: important nuances or conditions\n"
        "Be concise and factual."
    )
    try:
        import asyncio
        from ora_v2.llm.client import ModelTier, get_client
        client = get_client()
        try:
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                client.call(prompt, tier=ModelTier.FLASH, timeout=30, caller="skill:verify_claim")
            )
        except RuntimeError:
            result = asyncio.run(
                client.call(prompt, tier=ModelTier.FLASH, timeout=30, caller="skill:verify_claim")
            )
        return result[:1500]
    except Exception as e:
        return f"ERROR: {e}"
