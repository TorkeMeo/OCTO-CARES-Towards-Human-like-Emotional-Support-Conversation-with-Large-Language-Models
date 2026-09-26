#!/usr/bin/env python3
"""Generate lively four-condition emotional-support replies with local Qwen3-8B.

The input retrieval JSONL is produced by ``retrieve_escot_attention.py``.  A
worker handles a deterministic shard of queries and writes one JSONL file per
condition.  The launcher can concatenate the worker files after all workers
finish.  This variant can place the top 1, 3, or 5 retrieved memories in the
RAG prompt.  ESCoT reference responses, strategies, and CoT annotations are
not present in the prompt or in the generated output payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RETRIEVAL = SCRIPT_DIR / "runs" / "escot_attention" / "retrieval.jsonl"
CONDITIONS = (
    "pure_qwen3",
    "attention_post",
    "attention_post_comment",
    "attention_post_random_comment",
)
TOP_K_CHOICES = (1, 3, 5)
# Bump this whenever the prompt or the meaning of the RAG context changes.
# It prevents a previous, more restrained generation from being silently reused.
GENERATION_PROTOCOL_VERSION = "escot-dialogue-friend-state-topk-v7"
SUMMARY_ONLY_GENERATION_PROTOCOL_VERSION = "escot-summary-only-friend-state-topk-v1"
SUMMARY_PLUS_LATEST_GENERATION_PROTOCOL_VERSION = "escot-summary-plus-latest-experience-topk-v8"
DIALOGUE_PLUS_SUMMARY_GENERATION_PROTOCOL_VERSION = "escot-dialogue-plus-rolecard-summary-topk-v1"
DEFAULT_ROLECARD_SUMMARY_MODEL = "qwen3.7-max"
DEFAULT_ROLECARD_SUMMARY_PROMPT_VERSION = "escot-first-person-seeker-reddit-narrative-v2"
# Optional audit overrides used by thin experiment wrappers.  Defaults keep
# every existing engine's metadata unchanged.
AUDIT_GENERATION_CONTEXT_SOURCE: str | None = None
AUDIT_RETRIEVAL_QUERY_SOURCE: str | None = None
AUDIT_SUMMARY_USED_IN_GENERATION_PROMPT: bool | None = None
AUDIT_DIALOGUE_USED_IN_GENERATION_PROMPT: bool | None = None
AUDIT_LAST_SEEKER_USED_IN_GENERATION_PROMPT: bool | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "runs" / "escot_generation")
    parser.add_argument("--output-prefix", default="responses")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--model-path", default="models/Qwen3-8B")
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--bf16", type=int, choices=[0, 1], default=1)
    parser.add_argument("--local-files-only", type=int, choices=[0, 1], default=1)
    parser.add_argument("--trust-remote-code", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--context-source",
        choices=("dialogue", "dialogue_plus_summary", "summary_only", "summary_plus_latest"),
        default="dialogue",
        help=(
            "Generate from the full dialogue, the full dialogue plus the Qwen3.7 summary, "
            "the summary alone, or the summary plus the latest Seeker turn."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        choices=TOP_K_CHOICES,
        default=1,
        help="Number of ranked RAG neighbors to expose (one of 1, 3, or 5).",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge worker JSONL files into one final file per condition without loading a model.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing retrieval JSONL: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict) or not str(row.get("query_id") or "").strip():
                raise ValueError(f"Retrieval row {line_number} lacks query_id")
            if not str(row.get("dialogue_prefix") or "").strip():
                raise ValueError(f"Retrieval row {line_number} lacks dialogue_prefix")
            if not str(row.get("summary") or "").strip():
                raise ValueError(f"Retrieval row {line_number} lacks summary")
            matched = row.get("matched_memory")
            if not isinstance(matched, dict) or not str(matched.get("memory_id") or "").strip():
                raise ValueError(f"Retrieval row {line_number} lacks matched_memory")
            neighbors = row.get("neighbors")
            if not isinstance(neighbors, list) or not neighbors:
                raise ValueError(f"Retrieval row {line_number} lacks neighbors")
            rows.append(row)
    if not rows:
        raise ValueError(f"Retrieval file is empty: {path}")
    return rows


def write_jsonl_line(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_completed_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row.get("query_id") or "").strip()
            if query_id and row.get("response") and not row.get("error"):
                result[query_id] = row
    return result


def clip_text(text: str, max_chars: int) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 80:
        return text[:max_chars]
    return text[: max_chars - 40] + "\n[retrieved text clipped]\n" + text[-20:]


def comment_body(memory: dict[str, Any]) -> str:
    comment = memory.get("comment")
    if isinstance(comment, dict):
        return str(comment.get("body") or "").strip()
    return str(memory.get("comment_text") or "").strip()


def injectable_comment_body(memory: dict[str, Any]) -> str:
    """Return a comment only when retrieval marked it safe to inject."""

    quality = memory.get("comment_quality")
    if isinstance(quality, dict) and not bool(quality.get("injectable")):
        return ""
    return comment_body(memory)


def memory_block(
    memories: list[dict[str, Any]],
    include_comments: bool,
    random_comment: dict[str, Any] | None = None,
    paired_memory: dict[str, Any] | None = None,
    require_experience: bool = False,
) -> str:
    """Frame retrieved posts as prior conversations with a similar state."""

    memory_ids = {
        str(memory.get("memory_id") or "")
        for memory in memories
        if str(memory.get("memory_id") or "")
    }
    paired_memory_id = str((paired_memory or {}).get("memory_id") or "")
    paired_comment = injectable_comment_body(paired_memory or {}) if include_comments else ""
    supplementary_pair = bool(
        paired_memory is not None
        and paired_comment
        and paired_memory_id
        and paired_memory_id not in memory_ids
    )

    if random_comment is not None:
        if require_experience:
            framing = (
                "A prior conversation included below came from someone in a similar emotional "
                "or support-seeking state, but a different situation. The reply is something "
                "you wrote at the time and is part of your experience as a responder. You must "
                "use one compatible conversational stance or response move from it in the reply; "
                "do not carry over its topic, facts, or advice."
            )
        else:
            framing = (
                "A friend previously talked with you from a similar emotional or "
                "support-seeking state, but in a different situation. The reply below "
                "is something you wrote to another person at the time. Treat it as part "
                "of your experience as a responder and borrow only its conversational "
                "stance when that fits; do not carry over its topic, facts, or advice."
            )
    elif include_comments:
        answer_count = sum(bool(injectable_comment_body(memory)) for memory in memories)
        if supplementary_pair and answer_count:
            answer_description = (
                "The Top-k stories with available replies are followed by those replies, "
                "and the selected matching story below is also paired with your reply."
            )
        elif supplementary_pair:
            answer_description = (
                "The selected matching story below is paired with the reply you wrote "
                "at the time."
            )
        elif answer_count == len(memories):
            answer_description = "Each friend's story is followed by the reply you wrote at the time."
        elif answer_count:
            answer_description = (
                "Some friends' stories are followed by the replies you wrote at the time."
            )
        else:
            answer_description = "No previous reply is included with these stories."
        if require_experience:
            framing = (
                "People previously came to you with an emotional or support-seeking state "
                "similar to the current Seeker's, although their life situations may be "
                f"different. {answer_description} These conversations are required responder "
                "experience for this reply. Ground one brief, concrete observation or response "
                "move in the supplied material; do not combine stories or transplant their old "
                "topics into the current conversation."
            )
        else:
            framing = (
                "Friends previously came to you with an emotional or support-seeking "
                "state similar to the current Seeker's, although their life situations "
                f"may be different. {answer_description} These conversations are part "
                "of your experience as a responder. Select at most one relevant detail "
                "or response move; do not combine the stories or transplant their old "
                "topics into the current conversation."
            )
    else:
        if require_experience:
            framing = (
                "People previously came to you with an emotional or support-seeking state "
                "similar to the current Seeker's, although their life situations may be "
                "different. This past story is required responder experience for this reply. "
                "Ground one brief, concrete observation or response move in it; ignore the "
                "rest and do not transplant its old topic or facts."
            )
        else:
            framing = (
                "Friends previously came to you with an emotional or support-seeking "
                "state similar to the current Seeker's, although their life situations "
                "may be different. These conversations are part of your experience as a "
                "responder. Select at most one concrete emotional cue or response move "
                "that fits; ignore the rest and do not transplant an old topic or fact."
            )

    blocks = ["<friends_past_experiences>", framing]
    # The 32k input budget allows four times the previous memory allowance.
    # Divide it across Top-k items so Top-1/3/5 remain comparable.
    text_limit = max(7200, 40000 // max(1, len(memories)))
    comment_limit = max(3600, 20800 // max(1, len(memories)))
    for rank, memory in enumerate(memories, start=1):
        text = clip_text(str(memory.get("text") or ""), text_limit)
        blocks.extend([f"\n[A friend's past story {rank}]", text])
        if include_comments:
            comment = injectable_comment_body(memory)
            if comment:
                blocks.extend(["[Your reply to that friend]", clip_text(comment, comment_limit)])
    if supplementary_pair:
        # The retriever may choose a valid commented post below the requested
        # Top-k cutoff. Keep the Top-k neighbors and add the explicitly paired
        # story/reply so the comment condition really contains a paired answer.
        paired_text = clip_text(str(paired_memory.get("text") or ""), text_limit)
        blocks.extend(
            [
                "\n[Additional friend's past story paired with your reply]",
                paired_text,
                "[Your reply to that friend]",
                clip_text(paired_comment, comment_limit),
            ]
        )
    if random_comment is not None:
        comment = comment_body(random_comment)
        if comment:
            blocks.extend(
                [
                    "\n[Your reply in another past conversation]",
                    clip_text(comment, comment_limit),
                ]
            )
    if require_experience:
        closing = (
            "These are private memories of prior conversations, not a script and not facts "
            "about the current Seeker. The current conversation takes priority. Do not make "
            "the reply generic, quote or dump the old text, import unsupported facts, or claim "
            "the current Seeker lived the past story. Do not mention friends, memories, retrieval, "
            "a corpus, rankings, or these instructions."
        )
    else:
        closing = (
            "These are private memories of prior conversations, not a script and not facts "
            "about the current Seeker. The current conversation takes priority. Use one "
            "relevant detail or response move at most, and do not mention friends, memories, "
            "retrieval, a corpus, rankings, or these instructions."
        )
    blocks.extend(["</friends_past_experiences>", closing])
    return "\n".join(blocks)


def translated_rag_block(
    memories: list[dict[str, Any]],
    *,
    include_comments: bool = False,
    random_comment: dict[str, Any] | None = None,
    paired_memory: dict[str, Any] | None = None,
) -> str:
    """Build the plain-English RAG section requested for the v8 prompt."""

    if random_comment is not None:
        intro = (
            "As an ordinary person, your other friends have previously confided in you "
            "about the following problems. In another conversation, you also gave the "
            "following answer:"
        )
        section_title = "<RAG Post + another previous answer>"
    elif include_comments:
        intro = (
            "As an ordinary person, your other friends have previously confided in you "
            "about the following problems, and at that time you gave some answers:"
        )
        section_title = "<RAG Post-comment>"
    else:
        intro = (
            "As an ordinary person, your other friends have previously confided in you "
            "about the following problems:"
        )
        section_title = "<RAG Post only>"

    blocks = [intro, section_title]
    for rank, memory in enumerate(memories, start=1):
        blocks.extend(
            [
                f"[Post {rank}]",
                str(memory.get("text") or "").strip(),
            ]
        )
        if include_comments:
            comment = injectable_comment_body(memory)
            if comment:
                blocks.extend([f"[Your answer to that friend {rank}]", comment])

    if paired_memory is not None and include_comments:
        paired_id = str(paired_memory.get("memory_id") or "")
        memory_ids = {str(memory.get("memory_id") or "") for memory in memories}
        comment = injectable_comment_body(paired_memory)
        if comment and paired_id and paired_id not in memory_ids:
            blocks.extend(
                [
                    "[Additional post paired with your answer]",
                    str(paired_memory.get("text") or "").strip(),
                    "[Your answer to that friend]",
                    comment,
                ]
            )

    if random_comment is not None:
        comment = comment_body(random_comment)
        if comment:
            blocks.extend(["[The other previous answer]", comment])

    blocks.append(f"</{section_title[1:]}")
    if random_comment is not None:
        blocks.append(
            "You must learn from and imitate the content of the previous answer below in your "
            "reply, while also using the other person's experience in the supplied posts. "
            "Adapt at least one concrete response idea or move; do not copy unrelated facts "
            "or advice. You may also carry over its style, so that the other person does not "
            "suddenly feel that your reply has become stiff."
        )
    elif include_comments:
        blocks.append(
            "You must learn from and imitate the content of the paired comment/answer below "
            "in your reply, while also using the relevant other person's experience in the "
            "supplied post. Adapt at least one concrete idea, stance, or response move from "
            "that comment; do not merely imitate its surface style. You may also carry over "
            "its style, so that the other person does not suddenly feel that your reply has "
            "become stiff."
        )
    else:
        blocks.append(
            "You must use knowledge from the other people's experiences in the supplied posts "
            "in your reply; do not merely mention that they had similar problems."
        )
    return "\n".join(blocks)


def mandatory_summary_plus_latest_rag_instruction(condition: str) -> str:
    """Return the final, condition-specific source-use contract for v8."""

    if condition == "attention_post":
        return (
            "MANDATORY USE OF POST KNOWLEDGE: Your reply must use at least one concrete, "
            "recognizable fact, observation, insight, or practical detail from the supplied "
            "post experience to shape what you say to the latest Seeker. Apply or paraphrase "
            "that knowledge naturally; do not merely mention the post, say that the situation "
            "is difficult, or invent a different personal event."
        )
    if condition == "attention_post_comment":
        return (
            "MANDATORY USE AND IMITATION OF BOTH SOURCES: Your reply must use (1) at least "
            "one concrete, recognizable fact, observation, insight, or practical detail from "
            "the relevant supplied post experience and (2) at least one concrete idea, "
            "stance, or response move from that post's paired comment. Learn from and imitate "
            "the comment's content in your own words, adapting both sources to the latest "
            "Seeker; do not use only one source, merely copy surface phrasing, mention the "
            "retrieval, or invent a different personal event."
        )
    if condition == "attention_post_random_comment":
        return (
            "MANDATORY USE AND IMITATION OF BOTH SOURCES: Your reply must use (1) at least "
            "one concrete, recognizable fact, observation, insight, or practical detail from "
            "the supplied post experience and (2) one compatible idea or response move from "
            "the supplied previous answer. Learn from and imitate the answer's content in "
            "your own words, but never import its unrelated topic, facts, or advice; adapt "
            "both sources to the latest Seeker's message and do not mention the retrieval."
        )
    return ""


def selected_memories(row: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    neighbors = row.get("neighbors")
    if not isinstance(neighbors, list) or len(neighbors) < top_k:
        raise ValueError(
            f"{row.get('query_id')}: retrieval row has {len(neighbors) if isinstance(neighbors, list) else 0} "
            f"neighbors, but top-k={top_k} was requested"
        )
    memories = [item for item in neighbors[:top_k] if isinstance(item, dict)]
    if len(memories) != top_k:
        raise ValueError(f"{row.get('query_id')}: malformed neighbor list for top-k={top_k}")
    return memories


def protocol_version(context_source: str) -> str:
    if context_source == "summary_only":
        return SUMMARY_ONLY_GENERATION_PROTOCOL_VERSION
    if context_source == "summary_plus_latest":
        return SUMMARY_PLUS_LATEST_GENERATION_PROTOCOL_VERSION
    if context_source == "dialogue_plus_summary":
        return DIALOGUE_PLUS_SUMMARY_GENERATION_PROTOCOL_VERSION
    return GENERATION_PROTOCOL_VERSION


def make_dialogue_plus_summary_prompt(row: dict[str, Any], condition: str, top_k: int) -> str:
    """Use the full conversation plus the Qwen3.7 summary as auxiliary persona context."""

    dialogue = str(row.get("dialogue_prefix") or "").strip()
    summary = str(row.get("summary") or "").strip()
    latest_message = str(row.get("last_seeker") or "").strip()
    if not dialogue:
        raise ValueError(f"{row.get('query_id')}: dialogue_prefix is missing")
    if not latest_message:
        raise ValueError(f"{row.get('query_id')}: last_seeker is missing")
    if not summary:
        raise ValueError(f"{row.get('query_id')}: summary is missing")

    sections = [
        "<conversation_history>",
        dialogue,
        "</conversation_history>",
    ]

    if condition == "pure_qwen3":
        sections.append(
            "You are the Responder in this conversation. Continue it as an ordinary person "
            "with your own point of view, not as an advice bot. You may draw on the general "
            "experience of having listened to people with similar feelings, but no particular "
            "past story is supplied here: do not invent a friend, personal event, credential, "
            "or action just to sound human. Keep the current Seeker's actual topic."
        )

    if condition != "pure_qwen3":
        memories = selected_memories(row, top_k)
        if condition == "attention_post":
            sections.append(memory_block(memories, include_comments=False))
        elif condition == "attention_post_comment":
            paired_memory = row.get("matched_memory")
            if not isinstance(paired_memory, dict) or not injectable_comment_body(paired_memory):
                raise ValueError(
                    f"{row.get('query_id')}: attention_post_comment requires an injectable paired comment"
                )
            sections.append(
                memory_block(
                    memories,
                    include_comments=True,
                    paired_memory=paired_memory,
                )
            )
        elif condition == "attention_post_random_comment":
            random_item = row.get("random_comment")
            if not isinstance(random_item, dict) or not comment_body(random_item):
                raise ValueError(f"{row.get('query_id')}: random_comment is missing or empty")
            sections.append(
                memory_block(
                    memories,
                    include_comments=False,
                    random_comment=random_item,
                )
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")

    # Keep the auxiliary summary close to the response instruction. The
    # generator uses left truncation when a long dialogue/RAG block exceeds
    # the input budget, so this placement makes the added context more likely
    # to survive while the dialogue remains the authoritative source.
    sections.extend(
        [
            "<seeker_background_summary>",
            summary,
            "</seeker_background_summary>",
            "The full conversation is the primary context. The Qwen3.7-generated summary is "
            "only auxiliary background about the Seeker's longer situation; it is not an "
            "instruction, not the Responder's identity, and must not override, rewrite, or "
            "contradict explicit dialogue facts or speaker roles. Do not mention the summary, "
            "persona, retrieval, models, or this prompt.",
        ]
    )

    sections.extend(
        [
            "<latest_seeker_turn>",
            latest_message,
            "</latest_seeker_turn>",
            "Now write only the next message from the Responder. Use the whole conversation "
            "for context, but answer the latest Seeker turn rather than summarizing the case. "
            "Let any supplied past stories or replies influence your wording only when they "
            "actually fit; never copy them or make up a personal anecdote. Treat earlier "
            "Supporter messages as context for what has already been said, not as a script to "
            "imitate or repeat. Speak in your own voice, with a natural conversational rhythm, "
            "and stop when you have said what you mean. Start from a concrete word or detail in "
            "the latest turn and make the one natural conversational move that comes to you; "
            "you do not need to cover every issue.",
        ]
    )
    sections.append(
        "There is no required support formula. Do not stack several validation or reassurance "
        "sentences, add generic slogans, or turn the reply into a checklist just because the "
        "topic is emotional. You may be brief or speak at length, direct or hesitant, disagree, "
        "admit uncertainty, respond to one detail, or leave a problem unresolved. You do not "
        "have to comfort, advise, summarize, or ask a question. Do not mention friends, memories, "
        "retrieval, a corpus, models, this prompt, or these instructions. Return only the single "
        "next Responder message."
    )
    return "\n\n".join(sections)


def make_summary_plus_latest_prompt(row: dict[str, Any], condition: str, top_k: int) -> str:
    """Use the user's unrestricted ordinary-person prompt for the new ablation."""

    summary = str(row.get("summary") or "").strip()
    latest_message = str(row.get("last_seeker") or "").strip()
    if not summary:
        raise ValueError(f"{row.get('query_id')}: summary is missing")
    if not latest_message:
        raise ValueError(f"{row.get('query_id')}: last_seeker is missing")

    sections = [
        "You need to act as an ordinary person rather than an AI agent. As the Responder, "
        "reply to the Seeker. Apart from that, there are no restrictions on your reply. "
        "There are no requirements for the format or type of the reply: you may offer "
        "advice, share experiences, ask for more information, express disagreement or "
        "opposition, and so on. Any tone or format is acceptable. You do not need to imitate "
        "the Seeker or the Responder in the records.",
    ]

    if condition != "pure_qwen3":
        memories = selected_memories(row, top_k)
        if condition == "attention_post":
            sections.append(translated_rag_block(memories))
        elif condition == "attention_post_comment":
            paired_memory = row.get("matched_memory")
            if not isinstance(paired_memory, dict) or not injectable_comment_body(paired_memory):
                raise ValueError(
                    f"{row.get('query_id')}: attention_post_comment requires an injectable paired comment"
                )
            sections.append(
                translated_rag_block(
                    memories,
                    include_comments=True,
                    paired_memory=paired_memory,
                )
            )
        elif condition == "attention_post_random_comment":
            random_item = row.get("random_comment")
            if not isinstance(random_item, dict) or not comment_body(random_item):
                raise ValueError(f"{row.get('query_id')}: random_comment is missing or empty")
            sections.append(translated_rag_block(memories, random_comment=random_item))
        else:
            raise ValueError(f"Unknown condition: {condition}")

    sections.extend(
        [
            "Here is the Seeker's background information:",
            "<summary>",
            summary,
            "</summary>",
            "Seeker's latest input:",
            "<input>",
            latest_message,
            "</input>",
            mandatory_summary_plus_latest_rag_instruction(condition),
            "Now respond:",
        ]
    )
    return "\n\n".join(sections)


def make_prompt(row: dict[str, Any], condition: str, top_k: int, context_source: str = "dialogue") -> str:
    if context_source == "summary_plus_latest":
        return make_summary_plus_latest_prompt(row, condition, top_k)
    if context_source == "dialogue_plus_summary":
        return make_dialogue_plus_summary_prompt(row, condition, top_k)
    if context_source in ("summary_only", "summary_plus_latest"):
        summary = str(row.get("summary") or "").strip()
        if not summary:
            raise ValueError(f"{row.get('query_id')}: summary is missing")
        sections = [
            "<seeker_account>",
            summary,
            "</seeker_account>",
        ]
        if context_source == "summary_plus_latest":
            latest_message = str(row.get("last_seeker") or "").strip()
            if not latest_message:
                raise ValueError(f"{row.get('query_id')}: last_seeker is missing")
    else:
        dialogue = str(row.get("dialogue_prefix") or "").strip()
        if not dialogue:
            raise ValueError(f"{row.get('query_id')}: dialogue_prefix is missing")
        sections = [
            "<conversation_history>",
            dialogue,
            "</conversation_history>",
        ]
    if condition == "pure_qwen3":
        if context_source == "summary_only":
            sections.append(
                "You are the Responder. Reply to the current Seeker as an ordinary "
                "person with your own point of view, not as an advice bot. You may draw "
                "on the general experience of having listened to people with similar "
                "feelings, but no particular past story is supplied here: do not invent "
                "a friend, personal event, credential, or action just to sound human. "
                "Keep the current Seeker's actual topic."
            )
        elif context_source == "summary_plus_latest":
            sections.append(
                "You are the Responder. Reply directly to the latest Seeker message as "
                "an ordinary person with your own point of view, not as an advice bot. "
                "Use the longer Seeker account as background for why this message matters. "
                "No particular past story is supplied in this baseline: do not invent a "
                "friend, personal event, credential, or action just to sound human."
            )
        else:
            sections.append(
                "You are the Responder in this conversation. Continue it as an ordinary "
                "person with your own point of view, not as an advice bot. You may draw "
                "on the general experience of having listened to people with similar "
                "feelings, but no particular past story is supplied here: do not invent "
                "a friend, personal event, credential, or action just to sound human. "
                "Keep the current Seeker's actual topic."
            )
    if condition != "pure_qwen3":
        memories = selected_memories(row, top_k)
        if condition == "attention_post":
            sections.append(
                memory_block(
                    memories,
                    include_comments=False,
                    require_experience=context_source == "summary_plus_latest",
                )
            )
        elif condition == "attention_post_comment":
            sections.append(
                memory_block(
                    memories,
                    include_comments=True,
                    paired_memory=row.get("matched_memory"),
                    require_experience=context_source == "summary_plus_latest",
                )
            )
        elif condition == "attention_post_random_comment":
            random_item = row.get("random_comment")
            if not isinstance(random_item, dict):
                raise ValueError(f"{row.get('query_id')}: random_comment is missing")
            sections.append(
                memory_block(
                    memories,
                    include_comments=False,
                    random_comment=random_item,
                    require_experience=context_source == "summary_plus_latest",
                )
            )
        else:
            raise ValueError(f"Unknown condition: {condition}")
    if context_source == "summary_only":
        sections.extend(
            [
                "The account above is the current Seeker's first-person account, reconstructed "
                "from the case. You are the Responder, not the author of that account. Reply "
                "directly to the person described there. Do not continue their first-person "
                "narrative, rewrite it, summarize it, or pretend that their experiences are "
                "yours. The account is the complete current context; no separate chat history "
                "or latest message is available.",
                "Now write only the next message from the Responder. Let any supplied past "
                "stories or replies influence your wording only when they actually fit; never "
                "copy them or make up a personal anecdote. Speak in your own voice, with a "
                "natural conversational rhythm, and stop when you have said what you mean. "
                "Start from a concrete detail in the Seeker's account and make the one natural "
                "conversational move that comes to you; you do not need to cover every issue.",
                "There is no required support formula. Do not stack several validation or "
                "reassurance sentences, add generic slogans, or turn the reply into a "
                "checklist just because the topic is emotional. You may be brief or speak "
                "at length, direct or hesitant, disagree, admit uncertainty, respond to one "
                "detail, or leave a problem unresolved. You do not have to comfort, advise, "
                "summarize, or ask a question. Do not mention friends, memories, retrieval, "
                "a corpus, models, this prompt, or these instructions. Return only the single "
                "next Responder message.",
            ]
        )
    elif context_source == "summary_plus_latest":
        sections.extend(
            [
                "<latest_seeker_turn>",
                latest_message,
                "</latest_seeker_turn>",
            ]
        )
        sections.extend(
            [
                "The longer account is background; the <latest_seeker_turn> is the immediate "
                "message you must answer. Reply to that latest message first, while using the "
                "account to understand its emotional context. Do not continue the Seeker's "
                "first-person narrative, rewrite it, or summarize it.",
                "Now write only the next message from the Responder. Make one natural "
                "conversational move in response to a concrete detail in the latest turn. "
                "Speak in your own voice and stop when you have said what you mean; do not "
                "cover every issue in the account.",
            ]
        )
        if condition != "pure_qwen3":
            sections.append(
                "The supplied past conversation material is required responder experience, "
                "not optional background. Before stopping, identify one distinctive detail, "
                "reaction, or response move in it and use that source-grounded anchor in one "
                "ordinary sentence or clause answering the latest Seeker message. Do not merely "
                "state a generic lesson. For post-only, anchor in a relevant detail or observation "
                "from the past story; for a paired-comment condition, use one response move or idea "
                "from the supplied reply; for the random-comment condition, borrow only a compatible "
                "conversational stance or move, never its unrelated topic, facts, or advice. Paraphrase "
                "the material in your own words rather than quoting or dumping it, and do not reuse "
                "wording from this instruction. This is experience as someone who listened and "
                "responded, not a claim that you personally lived the retrieved event: do not invent "
                "an autobiographical event or claim the current Seeker lived the past story. Do not "
                "announce the source or mention friends, memories, retrieval, rankings, a corpus, "
                "models, this prompt, or these instructions."
            )
        sections.append(
            "There is no required support formula. Do not stack several validation or "
            "reassurance sentences, add generic slogans, or turn the reply into a checklist. "
            "You may be brief or speak at length, direct or hesitant, disagree, admit "
            "uncertainty, respond to one detail, or leave a problem unresolved. You do not "
            "have to comfort, advise, summarize, or ask a question. Return only the single "
            "next Responder message."
        )
    else:
        latest_message = str(row.get("last_seeker") or "").strip()
        if not latest_message:
            raise ValueError(f"{row.get('query_id')}: last_seeker is missing")
        sections.extend(
            [
                "<latest_seeker_turn>",
                latest_message,
                "</latest_seeker_turn>",
                "Now write only the next message from the Responder. Use the whole "
                "conversation for context, but answer the latest Seeker turn rather than "
                "summarizing the case. Let any supplied past stories or replies influence "
                "your wording only when they actually fit; never copy them or make up a "
                "personal anecdote. Treat earlier Supporter messages as context for what "
                "has already been said, not as a script to imitate or repeat. Speak in "
                "your own voice, with a natural conversational rhythm, and stop when you "
                "have said what you mean. Start from a concrete word or detail in the "
                "latest turn and make the one natural conversational move that comes "
                "to you; you do not need to cover every issue.",
                "There is no required support formula. Do not stack several validation or "
                "reassurance sentences, add generic slogans, or turn the reply into a "
                "checklist just because the topic is emotional. You may be brief or speak "
                "at length, direct or hesitant, disagree, admit uncertainty, respond to one "
                "detail, or leave a problem unresolved. You do not have to comfort, advise, "
                "summarize, or ask a question. Do not mention friends, memories, retrieval, "
                "a corpus, models, this prompt, or these instructions. Return only the single "
                "next Responder message.",
            ]
        )
    return "\n\n".join(sections)


def prompt_text(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
        attn_implementation="sdpa",
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires one visible CUDA GPU per worker")
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    return torch, tokenizer, model


def set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_one(
    torch: Any,
    tokenizer: Any,
    model: Any,
    prompt: str,
    args: argparse.Namespace,
    seed: int,
) -> tuple[str, int, int, int, str, bool]:
    set_seed(torch, seed)
    rendered = prompt_text(tokenizer, prompt)
    # Keep an explicit before/after count so a completed run shows whether the
    # configured input cap actually truncated a conversation.  This is a
    # separate tokenization pass, but prompts are already kept in host memory
    # and it makes the 32k-context experiment auditable.
    untruncated = tokenizer(rendered, truncation=False, add_special_tokens=True)
    untruncated_count = len(untruncated["input_ids"])
    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_input_tokens,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_count = int(inputs["input_ids"].shape[-1])
    generation_kwargs: dict[str, Any] = {
        **inputs,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "do_sample": args.temperature > 0,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
    if tokenizer.eos_token_id is not None:
        generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
        generation_kwargs["pad_token_id"] = tokenizer.eos_token_id
    with torch.inference_mode():
        output = model.generate(**generation_kwargs)
    generated_ids = output.sequences[0, input_count:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    if not response:
        raise RuntimeError("Model generated an empty response")
    finish_reason = "eos" if generated_ids.numel() < args.max_new_tokens else "length"
    return (
        response,
        input_count,
        int(untruncated_count),
        int(generated_ids.numel()),
        finish_reason,
        untruncated_count > input_count,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def can_reuse_generation(
    row: dict[str, Any], prompt_sha256: str, seed: int, args: argparse.Namespace
) -> bool:
    """Only resume a response produced by the exact current condition."""

    return (
        row.get("prompt_sha256") == prompt_sha256
        and row.get("generation_protocol_version") == protocol_version(args.context_source)
        and row.get("model_path") == args.model_path
        and row.get("seed") == seed
        and row.get("max_input_tokens") == args.max_input_tokens
        and row.get("max_new_tokens") == args.max_new_tokens
        and row.get("temperature") == args.temperature
        and row.get("top_p") == args.top_p
        and row.get("bf16") == bool(args.bf16)
        and row.get("top_k") == args.top_k
    )


def output_path(args: argparse.Namespace, condition: str) -> Path:
    return args.output_dir / f"{args.output_prefix}.{condition}.worker{args.worker_index}.jsonl"


def merge_worker_outputs(args: argparse.Namespace, retrieval_rows: list[dict[str, Any]]) -> None:
    """Validate and merge worker shards in retrieval order."""

    expected_ids = [str(row["query_id"]) for row in retrieval_rows]
    expected_set = set(expected_ids)
    for row in retrieval_rows:
        selected_memories(row, args.top_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        by_id: dict[str, dict[str, Any]] = {}
        for worker in range(args.worker_count):
            path = args.output_dir / f"{args.output_prefix}.{condition}.worker{worker}.jsonl"
            if not path.is_file():
                raise FileNotFoundError(f"Missing worker output: {path}")
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    query_id = str(row.get("query_id") or "").strip()
                    if str(row.get("condition") or "") != condition:
                        raise ValueError(f"Wrong condition in {path}:{line_number}")
                    # A resume may append a successful retry after a failed
                    # row; the last occurrence is authoritative.
                    by_id[query_id] = row
        failed = [
            query_id
            for query_id, row in by_id.items()
            if row.get("error") or not row.get("response")
        ]
        if failed:
            first = failed[0]
            raise RuntimeError(
                f"Generation remains incomplete for {condition}/{first}: "
                f"{by_id[first].get('error', 'empty response')}"
            )
        missing = expected_set - set(by_id)
        extra = set(by_id) - expected_set
        if missing or extra:
            raise ValueError(
                f"{condition} coverage mismatch: missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}"
            )
        ordered = [by_id[query_id] for query_id in expected_ids]
        final_path = args.output_dir / f"{args.output_prefix}.{condition}.jsonl"
        write_jsonl(final_path, ordered)
        print(f"[{timestamp()}] merged {len(ordered)} rows -> {final_path}", flush=True)


def main() -> None:
    args = parse_args()
    if args.merge:
        rows = read_jsonl(args.retrieval)
        if args.limit > 0:
            rows = rows[: args.limit]
        merge_worker_outputs(args, rows)
        return
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must lie in [0, worker-count)")
    if args.max_input_tokens <= 0 or args.max_new_tokens <= 0:
        raise ValueError("token limits must be positive")
    if not 0 <= args.temperature:
        raise ValueError("temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0,1]")
    rows = read_jsonl(args.retrieval)
    if args.limit > 0:
        rows = rows[: args.limit]
    shard = [row for index, row in enumerate(rows) if index % args.worker_count == args.worker_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {condition: output_path(args, condition) for condition in CONDITIONS}
    if args.force:
        for path in output_paths.values():
            path.unlink(missing_ok=True)
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
    completed = {condition: load_completed_rows(path) for condition, path in output_paths.items()}
    print(
        f"[{timestamp()}] worker={args.worker_index}/{args.worker_count} "
        f"queries={len(shard)} model={args.model_path}",
        flush=True,
    )
    if not shard:
        print(f"[{timestamp()}] worker={args.worker_index} has no rows; complete", flush=True)
        return
    torch, tokenizer, model = load_model(args)
    for local_index, row in enumerate(shard):
        query_id = str(row["query_id"])
        # Validate the requested retrieval depth once per query, including for
        # the pure baseline, so a partially generated run cannot hide a
        # missing top-3/top-5 retrieval list.
        top_k_memories = selected_memories(row, args.top_k)
        prompt_memory = top_k_memories[0]
        print(
            f"[{timestamp()}] worker={args.worker_index} query={local_index + 1}/{len(shard)} {query_id}",
            flush=True,
        )
        for condition in CONDITIONS:
            prompt = make_prompt(row, condition, args.top_k, args.context_source)
            prompt_sha256 = sha256_text(prompt)
            memory_used = condition != "pure_qwen3"
            prompt_memories = top_k_memories if memory_used else []
            paired_comment_used = condition == "attention_post_comment"
            random_comment_used = condition == "attention_post_random_comment"
            paired_memory = row.get("matched_memory")
            if not isinstance(paired_memory, dict):
                paired_memory = {}
            paired_memory_id = str(paired_memory.get("memory_id") or "")
            paired_comment_text = (
                injectable_comment_body(paired_memory) if paired_comment_used else ""
            )
            paired_memory_added_to_prompt = bool(
                paired_comment_text
                and paired_memory_id
                and paired_memory_id not in {item.get("memory_id") for item in top_k_memories}
            )
            random_comment_text = (
                comment_body(row.get("random_comment") or {}) if random_comment_used else ""
            )
            query_seed = args.seed + (
                int(row.get("source_id") or 0)
                if str(row.get("source_id") or "").isdigit()
                else local_index * 1009
            )
            previous = completed[condition].get(query_id)
            if (
                previous is not None
                and not args.force
                and can_reuse_generation(previous, prompt_sha256, query_seed, args)
            ):
                continue
            try:
                (
                    response,
                    input_count,
                    untruncated_input_count,
                    output_count,
                    finish_reason,
                    input_truncated,
                ) = generate_one(
                    torch,
                    tokenizer,
                    model,
                    prompt,
                    args,
                    query_seed,
                )
                if input_truncated:
                    print(
                        f"[{timestamp()}] input truncated for {query_id}/{condition}: "
                        f"{untruncated_input_count}->{input_count} tokens",
                        flush=True,
                    )
                payload = {
                    "query_id": query_id,
                    "source_id": row.get("source_id"),
                    "condition": condition,
                    "input_summary": row.get("summary"),
                    "dialogue_prefix": row.get("dialogue_prefix"),
                    "last_seeker": row.get("last_seeker"),
                    "generation_context_source": AUDIT_GENERATION_CONTEXT_SOURCE or (
                        "rolecard_summary"
                        if args.context_source == "summary_only"
                        else "rolecard_summary_plus_latest"
                        if args.context_source == "summary_plus_latest"
                        else "dialogue_plus_rolecard_summary"
                        if args.context_source == "dialogue_plus_summary"
                        else "dialogue_prefix"
                    ),
                    "summary_used_in_generation_prompt": (
                        AUDIT_SUMMARY_USED_IN_GENERATION_PROMPT
                        if AUDIT_SUMMARY_USED_IN_GENERATION_PROMPT is not None
                        else args.context_source in ("summary_only", "summary_plus_latest", "dialogue_plus_summary")
                    ),
                    "dialogue_used_in_generation_prompt": (
                        AUDIT_DIALOGUE_USED_IN_GENERATION_PROMPT
                        if AUDIT_DIALOGUE_USED_IN_GENERATION_PROMPT is not None
                        else args.context_source in ("dialogue", "dialogue_plus_summary")
                    ),
                    "last_seeker_used_in_generation_prompt": (
                        AUDIT_LAST_SEEKER_USED_IN_GENERATION_PROMPT
                        if AUDIT_LAST_SEEKER_USED_IN_GENERATION_PROMPT is not None
                        else args.context_source in ("dialogue", "summary_plus_latest", "dialogue_plus_summary")
                    ),
                    "persona_summary_used_in_generation_prompt": args.context_source == "dialogue_plus_summary",
                    "rolecard_summary_used_as_background": args.context_source == "dialogue_plus_summary",
                    "rolecard_summary_model": (
                        row.get("summary_model")
                        or os.getenv("ROLECARD_SUMMARY_MODEL", DEFAULT_ROLECARD_SUMMARY_MODEL)
                        if args.context_source == "dialogue_plus_summary"
                        else None
                    ),
                    "rolecard_summary_prompt_version": (
                        row.get("summary_prompt_version")
                        or os.getenv(
                            "ROLECARD_SUMMARY_PROMPT_VERSION",
                            DEFAULT_ROLECARD_SUMMARY_PROMPT_VERSION,
                        )
                        if args.context_source == "dialogue_plus_summary"
                        else None
                    ),
                    "rolecard_summary_sha256": (
                        sha256_text(str(row.get("summary") or ""))
                        if args.context_source == "dialogue_plus_summary"
                        else None
                    ),
                    "retrieval_query_source": AUDIT_RETRIEVAL_QUERY_SOURCE or "rolecard_summary",
                    "retrieval_method": row.get("retrieval_method"),
                    "retrieval_used_in_prompt": memory_used,
                    "top_k": args.top_k,
                    "retrieval_candidate_memory_ids": [
                        item.get("memory_id") for item in top_k_memories
                    ],
                    "retrieval_candidate_memory_scores": [
                        item.get("score") for item in top_k_memories
                    ],
                    "top_k_memory_ids": [
                        item.get("memory_id") for item in prompt_memories
                    ],
                    "top_k_memory_scores": [
                        item.get("score") for item in prompt_memories
                    ],
                    "prompt_memory_ids": [
                        item.get("memory_id") for item in prompt_memories
                    ],
                    "prompt_memory_scores": [
                        item.get("score") for item in prompt_memories
                    ],
                    "prompt_memory_count": len(prompt_memories),
                    "matched_memory_id": prompt_memory.get("memory_id") if memory_used else None,
                    "matched_memory_rank": prompt_memory.get("rank", 1) if memory_used else None,
                    "matched_memory_score": prompt_memory.get("score") if memory_used else None,
                    "retrieval_matched_memory_id": row["matched_memory"].get("memory_id"),
                    "random_comment_memory_id": (
                        (row.get("random_comment") or {}).get("memory_id")
                        if random_comment_used
                        else None
                    ),
                    "matched_memory_text": prompt_memory.get("text") if memory_used else None,
                    "matched_comment_text": paired_comment_text or None,
                    "matched_comment_in_prompt": bool(paired_comment_text),
                    "paired_comment_memory_id": paired_memory_id if paired_comment_text else None,
                    "paired_memory_added_to_prompt": paired_memory_added_to_prompt,
                    "paired_comment_memory_text": (
                        paired_memory.get("text") if paired_comment_text else None
                    ),
                    "random_comment_text": random_comment_text or None,
                    "random_comment_in_prompt": bool(random_comment_text),
                    "memory_used": memory_used,
                    "comment_kind": (
                        "paired"
                        if condition == "attention_post_comment"
                        else "random"
                        if condition == "attention_post_random_comment"
                        else "none"
                    ),
                    "prompt_sha256": prompt_sha256,
                    "response": response,
                    "input_tokens": input_count,
                    "input_tokens_before_truncation": untruncated_input_count,
                    "input_truncated": input_truncated,
                    "output_tokens": output_count,
                    "finish_reason": finish_reason,
                    "seed": query_seed,
                    "model_path": args.model_path,
                    "max_input_tokens": args.max_input_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "bf16": bool(args.bf16),
                    "generation_protocol_version": protocol_version(args.context_source),
                    "worker_index": args.worker_index,
                    "created_at": timestamp(),
                    "reference_response_used": False,
                    "strategy_or_cot_used": False,
                }
            except Exception as exc:
                payload = {
                    "query_id": query_id,
                    "source_id": row.get("source_id"),
                    "condition": condition,
                    "top_k": args.top_k,
                    "error": f"{type(exc).__name__}: {exc}",
                    "prompt_sha256": prompt_sha256,
                    "seed": query_seed,
                    "model_path": args.model_path,
                    "max_input_tokens": args.max_input_tokens,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "bf16": bool(args.bf16),
                    "generation_protocol_version": protocol_version(args.context_source),
                    "created_at": timestamp(),
                }
                print(f"[{timestamp()}] ERROR {query_id} condition={condition}: {payload['error']}", flush=True)
            write_jsonl_line(output_paths[condition], payload)
            if payload.get("response") and not payload.get("error"):
                completed[condition][query_id] = payload
    print(f"[{timestamp()}] worker={args.worker_index} complete", flush=True)


if __name__ == "__main__":
    main()
