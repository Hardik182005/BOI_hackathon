"""Regenerate every Cohort Radar evidence artifact section 39 requires.

    python -m muleguard.cli.cohort_evidence

Writes the retrieval diagnostic, the leakage report and the determinism report,
then prints a single verdict. Exits non-zero if either release-blocking report
fails, so it can sit in a release script without anybody having to read it.
"""
from __future__ import annotations

import argparse

from muleguard.logging import get_logger
from muleguard.usp import cohort_audit, cohort_eval

log = get_logger("cli.cohort_evidence")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-retrieval", action="store_true",
                    help="skip the outer-fold retrieval evaluation (~90s)")
    ap.add_argument("--repeat", type=int, default=cohort_eval.DEFAULT_REPEAT,
                    help="which nested-CV repeat's outer folds to evaluate")
    args = ap.parse_args(argv)

    if not args.skip_retrieval:
        report = cohort_eval.write(repeat=args.repeat)
        pos = report["pooled"]["positive"]
        print(f"retrieval   : hit@5 {pos['hit_at_5']:.4f}  hit@10 "
              f"{pos['hit_at_10']:.4f}  lift@10 {pos['lift_at_10']:.1f}x  "
              f"over prevalence "
              f"{report['reference_positive_prevalence']['mean']:.6f}")

    verdicts = cohort_audit.write_all()
    for name, verdict in verdicts.items():
        print(f"{name:12s}: {verdict}")

    failed = [n for n, v in verdicts.items() if v != "PASS"]
    print(f"\nVERDICT     : {'PASS' if not failed else 'FAIL - ' + ', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
