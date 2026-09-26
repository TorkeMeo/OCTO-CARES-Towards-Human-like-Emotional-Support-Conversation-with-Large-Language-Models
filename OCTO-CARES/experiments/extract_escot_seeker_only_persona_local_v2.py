#!/usr/bin/env python3
"""Extract compact personas from the seeker-only background, never Supporter text."""

from __future__ import annotations

from pathlib import Path

import extract_simple_seeker_persona_local as base


base.PROTOCOL_VERSION = "escot-seeker-only-persona-local-qwen3-v2"
base.PROMPT_VERSION = "escot-seeker-only-persona-five-field-v2"
base.CONTEXT_SOURCE = "seeker_only_summary"
base.WORKER_ENTRYPOINT = Path(__file__)


def prompt_for(row: dict) -> str:
    summary = str(row.get("summary") or "").strip()
    if not summary:
        raise ValueError(f"{row.get('query_id')}: seeker-only summary is empty")
    return "\n\n".join(
        [
            base.INSTRUCTION.replace("conversation below", "seeker-only background below"),
            "<seeker_only_background>",
            summary,
            "</seeker_only_background>",
        ]
    )


base.prompt_for = prompt_for


if __name__ == "__main__":
    base.main()
