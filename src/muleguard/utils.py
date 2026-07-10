"""Shared helpers: hashing, JSON IO, seeding, timing, git introspection."""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if np.isnan(v) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, cls=NumpyJSONEncoder, ensure_ascii=False)
    os.replace(tmp, path)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_info(repo_root: Path) -> dict[str, str]:
    def _run(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, timeout=10
            )
            return out.stdout.strip() if out.returncode == 0 else ""
        except Exception:
            return ""

    return {
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "commit_sha": _run("rev-parse", "HEAD"),
        "is_dirty": "yes" if _run("status", "--porcelain") else "no",
    }


@contextmanager
def timer(label: str, sink: dict[str, float] | None = None):
    """Time a block; optionally record seconds into ``sink[label]``."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        if sink is not None:
            sink[label] = round(elapsed, 3)
