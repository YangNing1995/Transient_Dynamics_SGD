"""Recompute freezing times from cached endpoints at different Jaccard thresholds.

This script reads existing endpoint cache files (produced by compute_freezing_time.py)
and recomputes t_f for multiple Jaccard similarity thresholds WITHOUT re-running any
continuation training.  It can run locally on CPU in seconds.

Usage example:
    python recompute_freezing_thresholds.py \
        --endpoint_root /path/to/freezing_time_training_lr_r1-5_ce/jaccard09 \
        --thresholds 0.90 0.95 0.99 \
        --output_dir ./freezing_threshold_robustness
"""

from __future__ import annotations

import csv
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def parse_args():
    parser = ArgumentParser(description="Recompute t_f from cached endpoints at different Jaccard thresholds.")
    parser.add_argument(
        "--endpoint_root",
        type=Path,
        required=True,
        help=(
            "Root directory containing per-condition subdirectories (e.g., bs50_lr0.05/) "
            "each with an endpoints/ folder of .npz caches."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.90, 0.95, 0.99],
        help="Jaccard similarity thresholds to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for summary CSVs. Defaults to endpoint_root/../threshold_robustness.",
    )
    return parser.parse_args()


def jaccard_similarity(wrong_a: np.ndarray, wrong_b: np.ndarray) -> float:
    set_a = set(int(x) for x in wrong_a)
    set_b = set(int(x) for x in wrong_b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def load_endpoints(npz_path: Path) -> dict:
    """Load cached endpoints. Returns dict: checkpoint_t -> wrong_indices."""
    data = np.load(npz_path, allow_pickle=True)
    endpoints = {}
    for idx, t in enumerate(data["checkpoint_t"].astype(int)):
        endpoints[int(t)] = np.asarray(data["wrong_indices"][idx], dtype=np.int64)
    return endpoints


def compute_tf(endpoints: dict, threshold: float) -> int | None:
    """Compute freezing time for a given Jaccard threshold.

    t_f = earliest checkpoint after which all subsequent endpoints have
    Jaccard similarity >= threshold with the reference (latest) endpoint.
    """
    sorted_times = sorted(endpoints.keys())
    if not sorted_times:
        return None
    ref_t = max(sorted_times)
    ref_wrong = endpoints[ref_t]

    # Compute similarity for each checkpoint against reference
    same_by_time = {}
    for t in sorted_times:
        sim = jaccard_similarity(endpoints[t], ref_wrong)
        same_by_time[t] = sim >= threshold

    # Find earliest stable suffix
    for idx, t in enumerate(sorted_times):
        later = sorted_times[idx:]
        if all(same_by_time[tt] for tt in later):
            return t
    return None


def main():
    args = parse_args()
    endpoint_root = args.endpoint_root.resolve()

    if args.output_dir is None:
        output_dir = endpoint_root.parent / "threshold_robustness"
    else:
        output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all endpoint cache files
    npz_files = sorted(endpoint_root.rglob("endpoints/*.npz"))
    if not npz_files:
        print(f"No endpoint cache files found under {endpoint_root}")
        return

    print(f"Found {len(npz_files)} endpoint cache files")
    print(f"Thresholds: {args.thresholds}")

    # Parse trajectory metadata from filenames: bs{B}_lr{LR}_repeat{R}_endpoints.npz
    import re
    pattern = re.compile(r"bs(\d+)_lr([\d.]+(?:e[+-]?\d+)?)_repeat(\d+)_endpoints\.npz")

    # Collect results per threshold
    results = {th: [] for th in args.thresholds}

    for npz_path in npz_files:
        match = pattern.match(npz_path.name)
        if not match:
            print(f"  Skipping unrecognized file: {npz_path.name}")
            continue

        batch_size = int(match.group(1))
        learning_rate = float(match.group(2))
        repeat = int(match.group(3))

        endpoints = load_endpoints(npz_path)
        if not endpoints:
            continue

        ref_t = max(endpoints.keys())

        for th in args.thresholds:
            tf = compute_tf(endpoints, th)
            eta_tf = learning_rate * tf if tf is not None else None
            results[th].append({
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "repeat": repeat,
                "tf": tf if tf is not None else "",
                "eta_tf": f"{eta_tf:.6g}" if eta_tf is not None else "",
                "reference_iteration": ref_t,
                "jaccard_threshold": th,
                "num_endpoints": len(endpoints),
            })

    # Write one summary CSV per threshold
    fieldnames = ["batch_size", "learning_rate", "repeat", "tf", "eta_tf",
                  "reference_iteration", "jaccard_threshold", "num_endpoints"]

    for th in args.thresholds:
        rows = sorted(results[th], key=lambda r: (int(r["batch_size"]), float(r["learning_rate"]), int(r["repeat"])))
        out_path = output_dir / f"freezing_time_jaccard{th:.2f}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        n_ok = sum(1 for r in rows if r["tf"] != "")
        n_fail = len(rows) - n_ok
        print(f"  threshold={th:.2f}: {n_ok} ok, {n_fail} failed -> {out_path}")

    print(f"\nDone. Results written to {output_dir}")


if __name__ == "__main__":
    main()
