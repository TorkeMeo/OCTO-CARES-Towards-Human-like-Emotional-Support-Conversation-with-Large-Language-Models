#!/usr/bin/env python3
"""Create one-pass seeker-only ESCoT summaries with prompt-level leakage control."""

from __future__ import annotations

import re
import time
from typing import Any

import summarize_escot_bailian_rolecard as base


base.SUMMARY_PROMPT_VERSION = "escot-seeker-only-prompt-constrained-single-pass-summary-v6"
base.SUMMARY_STYLE = "first_person_seeker_reddit_narrative"
base.SUMMARY_PIPELINE = "prompt_constrained_single_pass_narrative"

SUMMARY_SYSTEM_PROMPT = base.SUMMARY_SYSTEM_PROMPT
SUMMARY_SYSTEM_PROMPT += (
    "\n\nWrite between 100 and 400 words. Stay near the shorter end when the remaining "
    "Seeker material is sparse, and use more space only when the situation is detailed. "
    "The conversation may contain Supporter replies and the Seeker's reactions to them. "
    "Use those turns only to understand the exchange; do not present Supporter advice, "
    "suggestions, interpretations, reassurance, exercises, or plans as facts, beliefs, "
    "actions, or pre-existing plans of mine. If I merely agree with or thank the Supporter, "
    "preserve only my resulting state when useful, not the Supporter's proposed content."
)


def dialogue_turns(prefix: str) -> list[dict[str, str]]:
    pattern = re.compile(r"(?im)^\s*(Seeker|Supporter)\s*:\s*")
    matches = list(pattern.finditer(str(prefix or "")))
    turns: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(prefix)
        content = prefix[match.end() : end].strip()
        if content:
            turns.append({"speaker": match.group(1).lower(), "text": content})
    if not any(turn["speaker"] == "seeker" for turn in turns):
        raise ValueError("conversation contains no non-empty Seeker turns")
    return turns


def completion(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    args: Any,
) -> tuple[str, dict[str, Any]]:
    last_error = ""
    for attempt in range(max(1, args.max_retries)):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = base.clean_summary(base.response_content(response))
            if content:
                return content, {
                    "attempt": attempt + 1,
                    "model": str(getattr(response, "model", model) or model),
                    "finish_reason": str(
                        getattr(response.choices[0], "finish_reason", "") or ""
                    ),
                    "usage": str(getattr(response, "usage", "") or ""),
                }
            last_error = "empty response"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt + 1 < max(1, args.max_retries):
            time.sleep(min(16, 2**attempt))
    raise RuntimeError(last_error or "local completion failed")


def call_summary(
    client: Any, model: str, prefix: str, args: Any
) -> tuple[str, dict[str, Any]]:
    turns = dialogue_turns(prefix)
    seeker_count = sum(turn["speaker"] == "seeker" for turn in turns)
    supporter_count = sum(turn["speaker"] == "supporter" for turn in turns)
    user_prompt = base.make_user_prompt(prefix)
    summary, call_meta = completion(
        client, model, SUMMARY_SYSTEM_PROMPT, user_prompt, args
    )
    return summary, {
        "pipeline": "prompt_constrained_single_pass_narrative",
        "dialogue_turn_count": len(turns),
        "seeker_turn_count": seeker_count,
        "supporter_turn_count_in_prompt": supporter_count,
        "summary_call": call_meta,
    }


base.call_summary = call_summary


if __name__ == "__main__":
    base.main()
