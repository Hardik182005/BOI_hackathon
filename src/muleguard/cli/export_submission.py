"""Export a clean submission package to dist/submission/.

Contents: source (git archive), docs, metrics, plots, evidence, manifests -
never the raw dataset, never .env, never the virtualenv.
"""
from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import zipfile
from pathlib import Path

from muleguard import settings
from muleguard.utils import git_info, save_json

OUT = settings.REPO_ROOT / "dist" / "submission"


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    # 1. tracked source at HEAD (respects .gitignore -> no data, no secrets)
    src_zip = OUT / "muleguard_source.zip"
    subprocess.run(["git", "archive", "-o", str(src_zip), "HEAD"],
                   cwd=settings.REPO_ROOT, check=True)

    # 2. evidence: docs + metrics + plots + reports + manifests
    evidence_zip = OUT / "muleguard_evidence.zip"
    include_dirs = [
        settings.DOCS_DIR,
        settings.METRICS_DIR,
        settings.PLOTS_DIR,
        settings.REPORTS_DIR,
        settings.EVIDENCE_DIR,
        settings.FEATURES_DIR,
        settings.REGISTRY_DIR,
    ]
    include_files = [
        settings.ARTIFACTS_DIR / "environment_snapshot.json",
        settings.ARTIFACTS_DIR / "release_manifest.json",
        settings.MODELS_DIR / "model_manifest.json",
        settings.REPO_ROOT / "data/interim/data_fingerprint.json",
    ]
    with zipfile.ZipFile(evidence_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for d in include_dirs:
            for f in sorted(d.rglob("*")):
                if f.is_file() and f.suffix not in {".joblib", ".db"}:
                    z.write(f, f.relative_to(settings.REPO_ROOT))
        for f in include_files:
            if f.exists():
                z.write(f, f.relative_to(settings.REPO_ROOT))

    save_json({
        "exported_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": git_info(settings.REPO_ROOT),
        "contents": ["muleguard_source.zip", "muleguard_evidence.zip"],
        "note": "raw dataset and model binaries intentionally excluded; "
                "rebuild via README quick start",
    }, OUT / "export_manifest.json")
    print(f"SUBMISSION EXPORTED -> {OUT}")


if __name__ == "__main__":
    main()
