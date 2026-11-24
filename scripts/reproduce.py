"""Re-run the full SatelliteGuard pipeline and diff every reported number against the commit.

Intended for a CLEAN CLONE in a fresh environment:

    git clone https://github.com/b-jagan-18/SatelliteGuard.git && cd SatelliteGuard
    python -m venv .venv && .venv/Scripts/activate      # or: source .venv/bin/activate
    pip install -r requirements-lock.txt && pip install -e .
    python scripts/reproduce.py

The frozen test was scored once (reports/test_lock.json). In a throwaway clone this script
re-scores it with --force purely to verify that the committed numbers are reproducible; the
committed lock file remains the record of the one official run.

Exit code 0 if every value in reports/metrics.json, comparisons.json and leakage_check.json
matches the committed reference within --tol (timestamps and commit ids excepted).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

STEPS = [
    ["satelliteguard.data.download"],
    ["satelliteguard.data.validate"],
    ["satelliteguard.data.episodes"],
    ["satelliteguard.eda"],
    ["satelliteguard.evaluation.leakage_check"],
    ["satelliteguard.models.tune", "logreg"],
    ["satelliteguard.models.tune", "hgb"],
    ["satelliteguard.models.tune", "pca_recon"],
    ["satelliteguard.models.tune", "iforest"],
    ["satelliteguard.models.train"],
    ["satelliteguard.evaluation.compare", "hgb", "logreg"],
    ["satelliteguard.evaluation.compare", "hgb", "pca_recon"],
    ["satelliteguard.evaluation.compare", "pca_recon", "logreg"],
    ["satelliteguard.models.explain"],
    ["satelliteguard.evaluation.evaluate_test", "--force"],
    ["satelliteguard.evaluation.plots"],
    ["satelliteguard.dashboard.extract"],
]

# Fields that legitimately change between runs.
VOLATILE = {"trained_utc", "evaluated_utc", "git_commit", "forced"}


def flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k not in VOLATILE:
                yield from flatten(v, f"{prefix}{k}.")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from flatten(v, f"{prefix}{i}.")
    else:
        yield prefix.rstrip("."), obj


def diff(ref: dict, new: dict, tol: float) -> list[tuple[str, object, object]]:
    a, b = dict(flatten(ref)), dict(flatten(new))
    bad = []
    for key in sorted(a.keys() | b.keys()):
        x, y = a.get(key), b.get(key)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if abs(x - y) > tol:
                bad.append((key, x, y))
        elif x != y:
            bad.append((key, x, y))
    return bad


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--skip-run", action="store_true", help="only compare (after a manual run)")
    args = p.parse_args()

    files = [ROOT / "reports" / n for n in ("metrics.json", "comparisons.json",
                                            "leakage_check.json")]
    refs = {f: json.loads(f.read_text(encoding="utf-8")) for f in files}

    if not args.skip_run:
        for step in STEPS:
            t0 = time.time()
            print(f"$ python -m {' '.join(step)}", flush=True)
            subprocess.run([sys.executable, "-m", *step], cwd=ROOT, check=True,
                           stdout=subprocess.DEVNULL)
            print(f"  ok ({time.time() - t0:.0f} s)", flush=True)
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)

    failed = False
    for f in files:
        bad = diff(refs[f], json.loads(f.read_text(encoding="utf-8")), args.tol)
        n = len(dict(flatten(refs[f])))
        print(f"{f.relative_to(ROOT)}: {n - len(bad)}/{n} values match (tol {args.tol:g})")
        for key, x, y in bad[:20]:
            print(f"  {key}: committed {x!r} vs reproduced {y!r}")
        failed |= bool(bad)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
