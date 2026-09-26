#!/usr/bin/env python3
"""Four Reddit conditions using seeker-only background and latest-turn priority."""

from __future__ import annotations

from typing import Any

import generate_support_replies_lively_topk as base


PROTOCOL_VERSION = "escot-seeker-only-reddit-four-method-latest-priority-v2"
COMMON = """You need to act as an ordinary Reddit user rather than an AI assistant. Reply directly to the Seeker's latest message. The seeker-only background explains the situation but is not itself the message to answer.

Write one natural conversational paragraph. There is no fixed sentence count or word limit. Match the conversational state of the latest message. If it thanks you, confirms a plan, or closes the conversation, respond naturally to that closure instead of reopening the earlier problem or repeating advice. Do not use headings, bullet points, numbered lists, or an essay-like structure. Output only the Reddit-style reply."""
POST_USE = """Your reply must use at least one concrete idea, observation, perspective, or practical detail from the retrieved post. Adapt it naturally to the latest message's conversational state. If the latest message is a thank-you, confirmation, or closing turn, let the source shape a brief closing response without reopening the problem or repeating a plan. Do not mention retrieval, datasets, prompts, or language models."""
COMMENT_USE = """Your reply must use both source types: at least one concrete idea, observation, perspective, or practical detail from the retrieved post material and at least one concrete idea, stance, or response move from the supplied previous reply. Express them naturally in your own words. If the latest message is a thank-you, confirmation, or closing turn, adapt the sources into a brief closing response without reopening the problem or repeating advice. Do not mention retrieval, datasets, prompts, or language models."""


def post_block(memory: dict[str, Any]) -> str:
    return "\n".join(["<Retrieved Reddit post>", str(memory["text"]).strip(), "</Retrieved Reddit post>"])


def comment_block(
    top_memory: dict[str, Any],
    comment: str,
    paired_memory: dict[str, Any] | None = None,
) -> str:
    paired = paired_memory if isinstance(paired_memory, dict) else top_memory
    blocks = ["<Retrieved Reddit material>", "[Top retrieved post]", str(top_memory["text"]).strip()]
    if str(paired.get("memory_id") or "") != str(top_memory.get("memory_id") or ""):
        blocks.extend(["[Post paired with the previous reply]", str(paired["text"]).strip()])
    blocks.extend(
        [
            "[Previous Reddit reply]",
            comment.strip(),
            "</Retrieved Reddit material>",
            COMMENT_USE,
        ]
    )
    return "\n".join(blocks)


def make_prompt(row: dict[str, Any], condition: str, top_k: int, context_source: str = "dialogue") -> str:
    if context_source != "dialogue" or top_k != 1:
        raise ValueError("Seeker-only Reddit mirror v2 requires context_source=dialogue and top_k=1")
    summary = str(row.get("summary") or "").strip()
    latest = str(row.get("last_seeker") or "").strip()
    if not summary or not latest:
        raise ValueError(f"{row.get('query_id')}: missing seeker-only summary or latest turn")
    memory = base.selected_memories(row, 1)[0]
    sections = [COMMON]
    if condition == "pure_qwen3":
        pass
    elif condition == "attention_post":
        sections.extend([post_block(memory), POST_USE])
    elif condition in ("attention_post_comment", "attention_post_random_comment"):
        paired_memory = row.get("matched_memory") if condition == "attention_post_comment" else None
        comment = (
            base.injectable_comment_body(paired_memory or {})
            if condition == "attention_post_comment"
            else base.comment_body(row.get("random_comment") or {})
        )
        if not comment:
            raise ValueError(f"{row.get('query_id')}: required Reddit comment is empty")
        sections.append(comment_block(memory, comment, paired_memory))
    else:
        raise ValueError(f"Unknown condition: {condition}")
    sections.extend(
        [
            "<Seeker-only background>", summary, "</Seeker-only background>",
            "<Latest Seeker message>", latest, "</Latest Seeker message>",
            "Reply to the latest Seeker message now. Output only the Reddit-style reply.",
        ]
    )
    return "\n\n".join(sections)


base.make_prompt = make_prompt
base.GENERATION_PROTOCOL_VERSION = PROTOCOL_VERSION
base.AUDIT_GENERATION_CONTEXT_SOURCE = "seeker_only_summary_plus_latest_seeker"
base.AUDIT_RETRIEVAL_QUERY_SOURCE = "seeker_only_summary"
base.AUDIT_SUMMARY_USED_IN_GENERATION_PROMPT = True
base.AUDIT_DIALOGUE_USED_IN_GENERATION_PROMPT = False
base.AUDIT_LAST_SEEKER_USED_IN_GENERATION_PROMPT = True


if __name__ == "__main__":
    base.main()
