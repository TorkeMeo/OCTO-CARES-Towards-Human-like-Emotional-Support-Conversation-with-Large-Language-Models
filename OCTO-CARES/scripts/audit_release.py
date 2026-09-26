#!/usr/bin/env python3
"""Read-only inventory and conservative anonymity check for a release copy."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SUFFIXES = {".py", ".md", ".json", ".txt", ".sh", ".example", ""}
PRIVATE_PARTS = {".git", ".venv", "__pycache__", "outputs", "artifacts", "models", "private"}
PRIVATE_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".log", ".pyc", ".csv", ".tsv", ".jsonl"}
PATTERNS = {
    "personal_absolute_path": re.compile(r"/(?:Users|home|data)/(?:[A-Za-z][A-Za-z0-9_-]*)/(?:codes|Desktop|Documents|eight_type_classifier|miniforge3|\.ssh)"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "token": re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b"),
    "source_repository_name": re.compile("Scene" + "-Emotion-Memory-adapter-LLM|PythonProject" + "4|/fresh" + "new/"),
}


def inspect(root: Path, ignore_interpreter_cache: bool = False) -> tuple[list[tuple[str, str]], list[str]]:
    inventory, problems = [], []
    for path in sorted(root.rglob("*")):
        name = path.relative_to(root).as_posix()
        if ignore_interpreter_cache and "__pycache__" in path.relative_to(root).parts:
            continue
        if path.is_symlink():
            problems.append(f"Symlink is not allowed: {name}")
            continue
        if set(path.relative_to(root).parts) & PRIVATE_PARTS:
            problems.append(f"Private/runtime directory present: {name}")
            continue
        if not path.is_file():
            continue
        if path.suffix in PRIVATE_SUFFIXES or path.suffix not in ALLOWED_SUFFIXES:
            problems.append(f"Unreviewed file type: {name}")
            continue
        payload = path.read_bytes()
        if b"\x00" in payload:
            problems.append(f"Binary content: {name}")
            continue
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError:
            problems.append(f"Not UTF-8 text: {name}")
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                problems.append(f"{label}: {name}")
        inventory.append((name, hashlib.sha256(payload).hexdigest()))
    if not (root / "README.md").is_file():
        problems.append("Missing README.md")
    if not (root / "THIRD_PARTY.md").is_file():
        problems.append("Missing THIRD_PARTY.md")
    return inventory, problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Inspect the exact directory planned for upload")
    parser.add_argument("--list", action="store_true", help="Print every path and SHA-256")
    parser.add_argument("--ignore-interpreter-cache", action="store_true",
                        help="For local tests only; a publication directory must pass without this")
    args = parser.parse_args()
    inventory, problems = inspect(args.root.resolve(), args.ignore_interpreter_cache)
    if args.list:
        for name, digest in inventory:
            print(digest, name)
    print(f"RELEASE_AUDIT files={len(inventory)} problems={len(problems)}")
    for problem in problems:
        print("FAIL", problem)
    if problems:
        raise SystemExit(1)
    print("PASS: code inventory only. Licensing and data rights still require human review.")


if __name__ == "__main__":
    main()
