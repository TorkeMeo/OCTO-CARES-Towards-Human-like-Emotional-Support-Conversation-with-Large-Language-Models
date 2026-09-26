#!/usr/bin/env python3
"""Run the official-protocol Qwen3 adaptation with a strict refiner guard.

The v1 adapter reproduced the upstream refiner's bracketed output template too
literally.  Qwen3 sometimes returned the placeholder text ``original
response`` instead of a message to the user.  This additive entry point keeps
the v1 control flow unchanged, makes the refiner format unambiguous, and
rejects placeholder-only responses so the existing pipeline safely falls back
to the real response selected by debate/voting.
"""

from __future__ import annotations

import re
import unicodedata

import generate_multiagentesc_official_no_rag_qwen3 as base


PROTOCOL_VERSION = "multiagentesc-official-protocol-no-rag-qwen3-v2-refiner-guard"
_ORIGINAL_PARSE_RESPONSE = base.parse_response


def normalize_placeholder_candidate(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    normalized = re.sub(r"[\[\]<>*_`\"'.:：/\\-]+", " ", normalized)
    return " ".join(normalized.split())


def is_placeholder_response(value: str) -> bool:
    normalized = normalize_placeholder_candidate(value)
    return normalized in {
        "response",
        "original response",
        "refined response",
        "original refined response",
        "actual response",
        "actual response text",
    }


def guarded_parse_response(
    text: str, fallback_strategy: str | None = None
) -> tuple[str | None, str]:
    strategy, response = _ORIGINAL_PARSE_RESPONSE(text, fallback_strategy)
    if is_placeholder_response(response):
        return strategy, ""
    return strategy, response


def guarded_refiner_prompt(context: str, strategy: str, response: str) -> str:
    return f"""You will be provided with a dialogue context between an 'Assistant' and a 'User'.

### Dialogue context
{context}

The following response was generated using the {strategy} strategy. Analyze whether it is consistent with the ongoing conversation, aligns with the strategy, and effectively helps alleviate the user's emotional stress.

### Response
[{strategy}] {response}

If the response meets the requirements, return that response verbatim; otherwise provide a refined response. The response must be fewer than 30 words.

On the first line, write exactly "Response: [{strategy}] " followed immediately by the actual message that should be sent to the User. Never output placeholder phrases such as "response", "original response", "refined response", "original/refined response", or "actual response text" in place of the message.
On the second line, write "Reasoning: " followed by the actual reason for the decision."""


# Patch only this process.  The audited v1 adapter remains unchanged.
base.PROTOCOL_VERSION = PROTOCOL_VERSION
base.parse_response = guarded_parse_response
base.refiner_prompt = guarded_refiner_prompt


if __name__ == "__main__":
    base.main()
