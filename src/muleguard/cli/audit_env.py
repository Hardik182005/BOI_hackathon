"""Record the execution environment to artifacts/environment_snapshot.json."""
from __future__ import annotations

import datetime as dt
import importlib.metadata as im
import platform
import shutil
import sys

import psutil

from muleguard import settings
from muleguard.logging import get_logger
from muleguard.utils import git_info, save_json

log = get_logger("cli.audit_env")

TRACKED_PACKAGES = [
    "numpy", "pandas", "polars", "pyarrow", "fastexcel", "openpyxl",
    "scikit-learn", "lightgbm", "xgboost", "catboost", "optuna", "shap",
    "scipy", "matplotlib", "pydantic", "fastapi", "uvicorn", "httpx",
    "jinja2", "joblib", "pytest", "pyyaml", "psutil",
]


def detect_compute_mode(ram_gb: float, has_cuda: bool) -> str:
    if has_cuda:
        return "gpu_24gb"
    return "cpu_16gb"


def main() -> dict:
    settings.ensure_dirs()
    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(settings.REPO_ROOT)
    has_cuda = False
    gpu_names: list[str] = []
    try:  # torch is optional in the venv; CUDA absent on this laptop
        import torch  # type: ignore

        has_cuda = bool(torch.cuda.is_available())
        if has_cuda:
            gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        pass

    packages = {}
    for p in TRACKED_PACKAGES:
        try:
            packages[p] = im.version(p)
        except im.PackageNotFoundError:
            packages[p] = None

    snapshot = {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_info(settings.REPO_ROOT),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "ram_available_gb": round(vm.available / 1e9, 2),
        "disk_free_gb": round(disk.free / 1e9, 2),
        "cuda_available": has_cuda,
        "gpu_names": gpu_names,
        "compute_mode": detect_compute_mode(vm.total / 1e9, has_cuda),
        "seed": settings.GLOBAL_SEED,
        "packages": packages,
    }
    out = settings.ARTIFACTS_DIR / "environment_snapshot.json"
    save_json(snapshot, out)
    log.info("environment snapshot written to %s (mode=%s)", out, snapshot["compute_mode"])
    return snapshot


if __name__ == "__main__":
    snap = main()
    print(f"compute_mode={snap['compute_mode']} commit={snap['git']['commit_sha'][:12]}")
