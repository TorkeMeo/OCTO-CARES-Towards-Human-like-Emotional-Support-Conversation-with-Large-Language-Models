"""Small process and GPU utilities used only by the portable launchers."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Job:
    argv: list[str]
    log: Path
    env: dict[str, str] = field(default_factory=dict)


def show(jobs: list[Job]) -> None:
    for job in jobs:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in job.env.items())
        print(f"{prefix} {shlex.join(job.argv)}\n  log: {job.log}", flush=True)


def run(jobs: list[Job]) -> None:
    """Run a parallel wave, and stop only children of this wave on failure."""
    handles, children = [], []
    try:
        for job in jobs:
            job.log.parent.mkdir(parents=True, exist_ok=True)
            handle = job.log.open("a", encoding="utf-8")
            handles.append(handle)
            handle.write("\nCOMMAND " + shlex.join(job.argv) + "\n")
            handle.flush()
            children.append(subprocess.Popen(
                job.argv, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1", **job.env},
                stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
            ))
        while any(child.poll() is None for child in children):
            failed = [child for child in children if child.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f"Child failed (exit {failed[0].returncode}); inspect wave logs")
            time.sleep(0.2)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("Child failed; inspect wave logs")
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for handle in handles:
            handle.close()


@contextmanager
def run_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pipeline.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another launcher owns {output}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def gpu_ids(value) -> list[int]:
    if not isinstance(value, list) or not value or any(type(v) is not int or v < 0 for v in value):
        raise ValueError("GPU IDs must be a nonempty list of nonnegative integers")
    if len(set(value)) != len(value):
        raise ValueError("Duplicate GPU IDs")
    return value


def gpu_memory(ids: list[int]) -> dict[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    rows = {int(parts[0]): int(parts[1]) for line in result.stdout.splitlines()
            if len(parts := line.split(",")) == 2}
    missing = set(ids) - set(rows)
    if missing:
        raise RuntimeError(f"Requested GPUs do not exist: {sorted(missing)}")
    return {index: rows[index] for index in ids}


def free_port(port: int) -> bool:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def wait_resources(ids: list[int], settings: dict, wait: bool, port: int | None = None):
    checks = 0
    required = int(settings.get("idle_checks", 2)) if wait else 1
    interval = float(settings.get("gpu_poll_seconds", 30))
    threshold = int(settings.get("gpu_idle_max_mib", 1024))
    if required <= 0 or interval <= 0 or threshold < 0:
        raise ValueError("Invalid GPU polling settings")
    while checks < required:
        memory = gpu_memory(ids)
        ready = max(memory.values()) <= threshold and (port is None or free_port(port))
        checks = checks + 1 if ready else 0
        print(f"GPU_CHECK memory_MiB={memory} port={port} ready={ready} checks={checks}/{required}", flush=True)
        if not wait and not ready:
            raise RuntimeError("GPU/port busy. No existing job was stopped; use --wait-for-gpus")
        if checks < required:
            time.sleep(interval)


def server_command(model_path: Path, name: str, port: int, ids: list[int], settings: dict, tag: str):
    # A single model-directory mount works even when models live on different disks.
    if "," in str(model_path):
        raise ValueError("Docker mount paths cannot contain a comma")
    command = [
        "docker", "run", "--detach", "--rm", "--name", tag, "--ipc=host",
        "--gpus", '"device=' + ",".join(map(str, ids)) + '"',
        "--publish", f"127.0.0.1:{port}:8000",
        "--mount", f"type=bind,src={model_path},dst=/model,readonly",
        settings.get("image", "vllm/vllm-openai:v0.29.0"),
        "/model", "--served-model-name", name,
        "--tensor-parallel-size", str(len(ids)), "--port", "8000",
        "--host", "0.0.0.0", "--trust-remote-code", "--disable-custom-all-reduce", "--enforce-eager",
        "--max-model-len", str(settings.get("max_model_len", 32768)),
        "--gpu-memory-utilization", str(settings.get("gpu_memory_utilization", 0.85)),
    ]
    if settings.get("reasoning_parser"):
        command.extend(["--reasoning-parser", settings["reasoning_parser"]])
    return command


@contextmanager
def managed_server(model_path: Path, name: str, port: int, ids: list[int],
                   settings: dict, output: Path, wait: bool):
    if not model_path.is_dir():
        raise FileNotFoundError(f"Missing local model: {model_path}")
    wait_resources(ids, settings, wait, port)
    tag = "octo-" + uuid.uuid4().hex[:16]
    command = server_command(model_path, name, port, ids, settings, tag)
    output.mkdir(parents=True, exist_ok=True)
    (output / "server.command.json").write_text(json.dumps(command, indent=2) + "\n")
    launched = False
    logger = None
    try:
        subprocess.run(command, check=True)
        launched = True
        with (output / "server.log").open("a") as handle:
            logger = subprocess.Popen(["docker", "logs", "--follow", tag], stdout=handle,
                                      stderr=subprocess.STDOUT)
            deadline = time.monotonic() + float(settings.get("startup_timeout", 1200))
            while True:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                        payload = json.load(response)
                    if name in {item.get("id") for item in payload.get("data", [])}:
                        break
                except (OSError, ValueError):
                    pass
                status = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", tag],
                                        capture_output=True, text=True)
                if status.returncode or status.stdout.strip() != "true":
                    raise RuntimeError(f"Server exited; inspect {output / 'server.log'}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Server startup timeout; inspect {output / 'server.log'}")
                time.sleep(2)
            yield f"http://127.0.0.1:{port}/v1"
    finally:
        # Never stop a container that this invocation did not create.
        if launched:
            subprocess.run(["docker", "stop", "--time", "20", tag], check=False)
        if logger is not None:
            try:
                logger.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.terminate()
                logger.wait()
