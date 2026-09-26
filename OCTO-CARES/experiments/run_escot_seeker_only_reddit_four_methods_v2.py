#!/usr/bin/env python3
"""Three- or four-worker launcher for the seeker-only Reddit mirror v2."""

from pathlib import Path

import run_lively_topk_generation as base


SCRIPT_DIR = Path(__file__).resolve().parent
base.ENGINE = SCRIPT_DIR / "generate_escot_seeker_only_reddit_four_methods_v2.py"
base.PROTOCOL_VERSION = "escot-seeker-only-reddit-four-method-latest-priority-v2"


if __name__ == "__main__":
    base.main()
