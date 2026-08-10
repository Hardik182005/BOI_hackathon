"""Label-noise audit CLI (addendum UPDATE 5).

    python -m muleguard.cli.audit_labels

Reads the stored OOF predictions, asks which labelled mules every competent
model ranks as ordinary, and writes the report. It changes nothing: no label is
flipped and no row is removed, here or anywhere downstream.
"""
from __future__ import annotations

import argparse

from muleguard import settings
from muleguard.logging import get_logger
from muleguard.models.label_noise import audit_label_noise
from muleguard.utils import save_json

log = get_logger("cli.audit_labels")

OUT_JSON = settings.METRICS_DIR / "label_noise_audit_v2.json"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Report possible label noise (report only)")
    p.add_argument("--positive-percentile-max", type=float, default=None,
                   help="override the flagging percentile for labelled mules")
    a = p.parse_args(argv)

    overrides = {}
    if a.positive_percentile_max is not None:
        overrides["positive_percentile_max"] = a.positive_percentile_max
        log.warning("threshold overridden on the command line to %.1f; the "
                    "published default is %.1f and the artifact records which "
                    "was used", a.positive_percentile_max,
                    audit_label_noise.__globals__["NOISE_THRESHOLDS"]
                    ["positive_percentile_max"])

    payload = audit_label_noise(thresholds=overrides or None)
    payload["thresholds_overridden_on_cli"] = bool(overrides)
    save_json(payload, OUT_JSON)

    noise = payload["possible_label_noise"]
    high = payload["high_scoring_negatives"]
    log.info("written %s", OUT_JSON.name)
    log.info("%s: %d of %d positives (%.1f%%)", noise["flag"], noise["count"],
             payload["n_positives"], 100 * noise["share_of_positives"])
    log.info("%s: %d of %d negatives", high["flag"], high["count"],
             payload["n_negatives"])
    log.info("no label was changed and no row was removed")


if __name__ == "__main__":
    main()
