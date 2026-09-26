#!/usr/bin/env python3
"""MultiAgentESC adaptation for seeker-only background plus latest-turn priority."""

from __future__ import annotations

import generate_multiagentesc_official_no_rag_qwen3_refiner_guard as guard


base = guard.base
base.CONDITION = "reddit_multiagent"
base.PROTOCOL_VERSION = "escot-seeker-only-multiagentesc-reddit-latest-priority-v2"
base.COUNSELOR_SYSTEM = (
    "You reason about how an ordinary Reddit user should reply naturally to the latest "
    "Seeker message. Background explains the situation but is never the immediate turn."
)
base.AUDIT_SUMMARY_USED_IN_GENERATION_PROMPT = True
base.AUDIT_DIALOGUE_USED_IN_GENERATION_PROMPT = False
base.AUDIT_LAST_SEEKER_USED_IN_GENERATION_PROMPT = True
base.AUDIT_INPUT_CONTEXT = "seeker_only_summary_plus_latest_seeker"

STYLE = """Write one natural conversational Reddit-style paragraph. There is no fixed sentence count or word limit. Reply to the <Latest Seeker message> first; use <Seeker-only background> only to understand it. If the latest message thanks you, confirms a plan, or closes the conversation, respond naturally to that closure instead of reopening the earlier problem or repeating advice. Do not use headings, bullet points, numbered lists, or an essay-like structure. Output only the reply content in the required Response field."""


def mirror_context_turn_count(context: str) -> int:
    if not str(context or "").strip():
        raise ValueError("mirror target context is empty")
    return 1


def zero_shot_prompt(context: str) -> str:
    return f"""Write a direct Reddit-style reply to the latest Seeker message below.

### Seeker context
{context}

{STYLE}
Output exactly:
Response: [the actual reply]"""


def candidate_response_prompt(
    context: str, emotion: str, cause: str, intention: str, strategy: str
) -> str:
    return f"""You are given seeker-only background, the latest Seeker message, and analyses of the latest message.

### Seeker context
{context}

### Emotional state
{emotion}

### Situation or cause
{cause}

### Likely intention
{intention}

Write a direct Reddit-style reply using the {strategy} support strategy.
Strategy definition: {base.STRATEGIES[strategy]}

{STYLE}
Output exactly:
Response: [{strategy}] [the actual reply]"""


def refiner_prompt(context: str, strategy: str, response: str) -> str:
    return f"""Review a proposed reply for relevance, natural conversational style, latest-turn priority, and consistency with the {strategy} strategy.

### Seeker context
{context}

### Proposed reply
[{strategy}] {response}

If it already works, return it verbatim; otherwise refine it. {STYLE}
On the first line, write exactly "Response: [{strategy}] " followed immediately by the actual reply. Never output placeholder phrases.
On the second line, write "Reasoning: " followed by the reason."""


base.zero_shot_prompt = zero_shot_prompt
base.candidate_response_prompt = candidate_response_prompt
base.refiner_prompt = refiner_prompt
base.dialogue_turn_count = mirror_context_turn_count


if __name__ == "__main__":
    base.main()
