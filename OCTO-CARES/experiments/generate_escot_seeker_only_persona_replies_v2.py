#!/usr/bin/env python3
"""Generate the seeker-only Persona Reddit condition with latest-turn priority."""

from __future__ import annotations

from typing import Any

import generate_simple_persona_replies as base


base.PERSONA_PROTOCOL_VERSION = "escot-seeker-only-persona-local-qwen3-v2"
base.PROTOCOL_VERSION = "escot-seeker-only-persona-reply-latest-priority-v2"
base.PROMPT_VERSION = "escot-seeker-only-persona-reddit-latest-priority-v2"
base.ALL_CONDITIONS = ("simple_persona",)
base.AUDIT_GENERATION_CONTEXT_SOURCE = "seeker_only_summary_plus_latest_plus_persona"
base.AUDIT_SUMMARY_USED_IN_GENERATION_PROMPT = True
base.AUDIT_DIALOGUE_USED_IN_GENERATION_PROMPT = False
base.AUDIT_LAST_SEEKER_USED_IN_GENERATION_PROMPT = True


def make_prompt(
    row: dict[str, Any],
    persona_row: dict[str, Any] | None = None,
    condition: str = "simple_persona",
) -> str:
    if condition != "simple_persona" or not isinstance(persona_row, dict):
        raise ValueError("Seeker-only persona generation requires a persona row")
    summary = str(row.get("summary") or "").strip()
    latest = str(row.get("last_seeker") or "").strip()
    if not summary or not latest:
        raise ValueError(f"{row.get('query_id')}: missing seeker-only summary or latest turn")
    persona = persona_row["persona"]
    card = "\n".join(
        [
            "Socio-demographic description: "
            + str(persona.get("socio_demographic_description") or "unknown"),
            "Problem or situation: " + str(persona.get("problem") or "unknown"),
            f"Age: {persona.get('age', 'unknown')}",
            f"Gender: {persona.get('gender', 'unknown')}",
            f"Occupation: {persona.get('occupation', 'unknown')}",
        ]
    )
    return f"""You need to act as an ordinary Reddit user rather than an AI assistant. Reply directly to the Seeker's latest message. The seeker-only background and persona explain the situation but are not themselves the message to answer.

Write one natural conversational paragraph. There is no fixed sentence count or word limit. Match the conversational state of the latest message. If it thanks you, confirms a plan, or closes the conversation, respond naturally to that closure instead of reopening the earlier problem or repeating advice. Do not use headings, bullet points, numbered lists, or an essay-like structure. Output only the Reddit-style reply.

<Seeker persona>
{card}
</Seeker persona>

The persona describes the Seeker, not the commenter. Use it only as background. Do not recite it, mention its extraction, or invent unsupported details.

<Seeker-only background>
{summary}
</Seeker-only background>

<Latest Seeker message>
{latest}
</Latest Seeker message>

Reply to the latest Seeker message now. Output only the Reddit-style reply."""


base.make_prompt = make_prompt


if __name__ == "__main__":
    base.main()
