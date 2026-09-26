#!/usr/bin/env python3
"""Three- or four-worker launcher for seeker-only latest-priority MultiAgentESC."""

from pathlib import Path

import run_multiagentesc_official_no_rag_qwen3 as base


SCRIPT_DIR = Path(__file__).resolve().parent
base.ENGINE = SCRIPT_DIR / "generate_escot_seeker_only_reddit_multiagent_v2.py"
base.CONDITION = "reddit_multiagent"
base.PROTOCOL_VERSION = "escot-seeker-only-multiagentesc-reddit-latest-priority-v2"


if __name__ == "__main__":
    base.main()
