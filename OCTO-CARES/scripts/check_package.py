#!/usr/bin/env python3
"""Run standalone CPU regression tests, syntax checks, and release inventory."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check(command: list[str]) -> None:
    print("CHECK", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    scripts = sorted(ROOT.rglob("*.py"))
    for path in scripts:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print(f"SYNTAX_OK python_files={len(scripts)}", flush=True)
    for shell in sorted(ROOT.rglob("*.sh")):
        check(["bash", "-n", str(shell)])
    check([sys.executable, "-B", "-m", "unittest", "discover", "-s", "annotation", "-p", "test_*.py"])
    check([sys.executable, "-B", "-m", "unittest", "discover", "-s", "representation", "-p", "test_*.py"])
    check([sys.executable, "-B", "-m", "unittest", "discover", "-s", "experiments", "-p", "test_*.py"])
    check([sys.executable, "-B", "-m", "unittest", "discover", "-s", "scripts", "-p", "test_*.py"])
    check([sys.executable, "-B", "scripts/audit_release.py", "--ignore-interpreter-cache"])
    print("PACKAGE_CHECK_OK: CPU only; GPU/API/Docker execution not covered", flush=True)


if __name__ == "__main__":
    main()
