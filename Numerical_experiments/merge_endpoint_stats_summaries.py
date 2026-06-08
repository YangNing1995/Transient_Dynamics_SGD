"""Merge endpoint_stats_summary.csv files from split continuation jobs."""

from __future__ import annotations

import csv
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = args.input_root.resolve()
    output_csv = args.output_csv or input_root / "endpoint_stats_summary.csv"

    paths = sorted(input_root.glob("r*/endpoint_stats_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No r*/endpoint_stats_summary.csv files found under {input_root}")

    rows = []
    fieldnames = None
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = reader.fieldnames
            rows.extend(reader)

    def sort_key(row):
        return (
            int(row.get("repeat") or 0),
            int(row.get("checkpoint_t") or -1),
        )

    rows = sorted(rows, key=sort_key)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for row in rows if row.get("status") == "ok")
    print(f"Merged {len(rows)} rows ({ok} ok) from {len(paths)} files into {output_csv}")


if __name__ == "__main__":
    main()
