"""Compute flatness at the continuation-defined freezing checkpoint.

This script evaluates local geometry at the operational basin-commitment time
t_f rather than at the final checkpoint.  The freezing time is read from
compute_freezing_time.py summaries and rounded to a saved checkpoint.
"""

from __future__ import annotations

import csv
import math
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

import compute_relative_sharpness as crs
from model import FCN


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH_SIZES = [1000, 500, 200, 100, 50, 20, 10]
DEFAULT_LEARNING_RATES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def parse_args():
    parser = ArgumentParser(description="Compute random-filter flatness at freezing time t_f.")
    parser.add_argument("--dataset_name", type=str, default="MNIST")
    parser.add_argument("--data_dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--hidden_num", type=int, default=50)
    parser.add_argument("--train_num", type=int, default=100)
    parser.add_argument("--test_num", type=int, default=20)
    parser.add_argument("--checkpoint_dir", type=Path, default=SCRIPT_DIR / "save_checkpoint")
    parser.add_argument(
        "--checkpoint_subdir_template",
        type=str,
        default="bs{batch_size}_lr{learning_rate:g}_repeat{repeat}",
    )
    parser.add_argument(
        "--freezing_summary_root",
        type=Path,
        default=SCRIPT_DIR.parent / "freezing_time_all_ce" / "jaccard095",
        help="Root containing bs*_lr*/freezing_time_summary.csv files.",
    )
    parser.add_argument("--output_dir", type=Path, default=SCRIPT_DIR / "save_data_freezing_flatness")
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--learning_rates", nargs="+", type=float, default=DEFAULT_LEARNING_RATES)
    parser.add_argument("--repeat_start", type=int, default=1)
    parser.add_argument("--repeat_end", type=int, default=20)
    parser.add_argument(
        "--freeze_rounding",
        choices=["ceil", "nearest", "floor"],
        default="ceil",
        help="How to map t_f to the saved checkpoint grid.",
    )
    parser.add_argument(
        "--definition",
        choices=["random_tensor", "random_filter", "asam_element"],
        default="random_filter",
    )
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--num_directions", type=int, default=20)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--include_bias", action="store_true")
    parser.add_argument("--relative_floor", type=float, default=1e-12)
    parser.add_argument("--adaptive_epsilon", type=float, default=1e-12)
    parser.add_argument("--eval_batch_size", type=int, default=-1)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--loss_type", choices=["ce", "mse_logits", "mse_softmax"], default="ce")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def selected_trajectories(args):
    repeats = range(args.repeat_start, args.repeat_end + 1)
    return [
        (int(batch_size), float(learning_rate), int(repeat))
        for batch_size in args.batch_sizes
        for learning_rate in args.learning_rates
        for repeat in repeats
    ]


def load_freezing_rows(root: Path):
    rows = {}
    for path in sorted(root.glob("*/freezing_time_summary.csv")):
        text = path.read_text(errors="ignore").replace("\x00", "")
        for row in csv.DictReader(text.splitlines()):
            try:
                key = (
                    int(float(row["batch_size"])),
                    float(row["learning_rate"]),
                    int(row["repeat"]),
                )
            except Exception:
                continue
            rows[key] = row
    return rows


def choose_checkpoint(iterations: list[int], target: float, mode: str) -> int | None:
    if not iterations:
        return None
    values = np.asarray(iterations, dtype=float)
    if mode == "nearest":
        return int(values[np.argmin(np.abs(values - target))])
    if mode == "floor":
        candidates = values[values <= target]
        return int(candidates[-1]) if candidates.size else int(values[0])
    candidates = values[values >= target]
    return int(candidates[0]) if candidates.size else int(values[-1])


def output_path(args):
    return args.output_dir / "freezing_flatness_summary.csv"


def summary_key(row):
    return int(row["batch_size"]), float(row["learning_rate"]), int(row["repeat"])


def load_existing(path: Path):
    if not path.exists():
        return {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return {summary_key(row): row for row in reader}


def write_rows(path: Path, rows):
    fieldnames = [
        "batch_size",
        "learning_rate",
        "repeat",
        "tf",
        "tf_checkpoint",
        "final_checkpoint",
        "eta_tf",
        "eta_tf_checkpoint",
        "freeze_rounding",
        "confidence_flag",
        "definition",
        "rho",
        "num_directions",
        "symmetric",
        "include_bias",
        "loss_type",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "sharpness_mean_delta",
        "sharpness_median_delta",
        "sharpness_std_delta",
        "sharpness_mean_positive_delta",
        "flatness_neg_log10",
        "relative_flatness_inverse",
        "min_delta",
        "max_delta",
        "wall_time_sec",
        "status",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({name: rows[key].get(name, "") for name in fieldnames})


def make_error_row(args, batch_size, learning_rate, repeat, message):
    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "repeat": repeat,
        "freeze_rounding": args.freeze_rounding,
        "definition": args.definition,
        "rho": args.rho,
        "num_directions": args.num_directions,
        "symmetric": args.symmetric,
        "include_bias": args.include_bias,
        "loss_type": args.loss_type,
        "status": "error",
        "message": message,
    }


def compute_one(args, batch_size, learning_rate, repeat, model, train_loader, test_loader, criterion, device, freezing_rows):
    freeze_row = freezing_rows.get((batch_size, learning_rate, repeat))
    if freeze_row is None:
        return make_error_row(args, batch_size, learning_rate, repeat, "missing freezing summary row")

    saved_iterations = crs.list_checkpoint_iterations(
        args.checkpoint_dir,
        args.checkpoint_subdir_template,
        batch_size,
        learning_rate,
        repeat,
    )
    if not saved_iterations:
        return make_error_row(args, batch_size, learning_rate, repeat, "no saved checkpoints")

    tf = float(freeze_row["tf"])
    tf_checkpoint = choose_checkpoint(saved_iterations, tf, args.freeze_rounding)
    final_checkpoint = int(max(saved_iterations))
    if tf_checkpoint is None:
        return make_error_row(args, batch_size, learning_rate, repeat, "could not resolve tf checkpoint")

    checkpoint_file = crs.checkpoint_path(
        args.checkpoint_dir,
        args.checkpoint_subdir_template,
        batch_size,
        learning_rate,
        repeat,
        tf_checkpoint,
    )
    if not checkpoint_file.exists():
        return make_error_row(args, batch_size, learning_rate, repeat, f"missing checkpoint {checkpoint_file}")

    wall_start = time.perf_counter()
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    train_loss, train_acc, deltas, _, _ = crs.relative_sharpness_for_model(
        model,
        train_loader,
        criterion,
        device,
        args,
    )
    test_loss, test_acc = crs.evaluate_loss_accuracy(model, test_loader, criterion, device)
    positive_deltas = np.maximum(deltas, 0.0)
    mean_positive = float(np.mean(positive_deltas))
    flatness = -math.log10(max(mean_positive, 1e-12))

    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "repeat": repeat,
        "tf": tf,
        "tf_checkpoint": int(tf_checkpoint),
        "final_checkpoint": final_checkpoint,
        "eta_tf": float(freeze_row.get("eta_tf", learning_rate * tf)),
        "eta_tf_checkpoint": float(learning_rate * tf_checkpoint),
        "freeze_rounding": args.freeze_rounding,
        "confidence_flag": freeze_row.get("confidence_flag", ""),
        "definition": args.definition,
        "rho": args.rho,
        "num_directions": args.num_directions,
        "symmetric": args.symmetric,
        "include_bias": args.include_bias,
        "loss_type": args.loss_type,
        "train_loss": float(train_loss),
        "train_accuracy": float(train_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "sharpness_mean_delta": float(np.mean(deltas)),
        "sharpness_median_delta": float(np.median(deltas)),
        "sharpness_std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "sharpness_mean_positive_delta": mean_positive,
        "flatness_neg_log10": flatness,
        "relative_flatness_inverse": float(1.0 / (1e-12 + mean_positive)),
        "min_delta": float(np.min(deltas)),
        "max_delta": float(np.max(deltas)),
        "wall_time_sec": float(time.perf_counter() - wall_start),
        "status": "ok",
        "message": "",
    }


def print_summary(rows):
    ok_rows = [row for row in rows.values() if row.get("status") == "ok"]
    if not ok_rows:
        return
    flatness = np.array([float(row["flatness_neg_log10"]) for row in ok_rows], dtype=float)
    train_acc = np.array([float(row["train_accuracy"]) for row in ok_rows], dtype=float)
    test_loss = np.array([float(row["test_loss"]) for row in ok_rows], dtype=float)
    print(
        f"OK rows={len(ok_rows)}; flatness mean={np.mean(flatness):.4f}, "
        f"train_acc mean={np.mean(train_acc):.4f}, test_loss mean={np.mean(test_loss):.4f}"
    )


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.freezing_summary_root = args.freezing_summary_root.resolve()
    args.output_dir = args.output_dir.resolve()

    trajectories = selected_trajectories(args)
    if args.dry_run:
        print(f"Selected {len(trajectories)} trajectories")
        freezing_rows = load_freezing_rows(args.freezing_summary_root)
        for batch_size, learning_rate, repeat in trajectories[:30]:
            saved = crs.list_checkpoint_iterations(
                args.checkpoint_dir,
                args.checkpoint_subdir_template,
                batch_size,
                learning_rate,
                repeat,
            )
            freeze_row = freezing_rows.get((batch_size, learning_rate, repeat))
            tf = float(freeze_row["tf"]) if freeze_row else math.nan
            tf_checkpoint = choose_checkpoint(saved, tf, args.freeze_rounding) if freeze_row else None
            print(
                f"  bs={batch_size:g} lr={learning_rate:g} repeat={repeat}: "
                f"tf={tf}, tf_checkpoint={tf_checkpoint}, saved={len(saved)}"
            )
        if len(trajectories) > 30:
            print(f"... {len(trajectories) - 30} more")
        return

    device = crs.resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Checkpoint root: {args.checkpoint_dir}")
    print(f"Freezing summary root: {args.freezing_summary_root}")
    print(f"Output root: {args.output_dir}")
    print(
        f"Definition={args.definition}, rho={args.rho}, K={args.num_directions}, "
        f"rounding={args.freeze_rounding}"
    )

    train_loader, test_loader, train_count, test_count = crs.make_loaders(args)
    print(f"Train samples: {train_count}; test samples: {test_count}")
    criterion = crs.ClassificationLoss(args.loss_type)
    model = FCN(input_dim=crs.input_dim_for_dataset(args.dataset_name), hidden=args.hidden_num).to(device)
    freezing_rows = load_freezing_rows(args.freezing_summary_root)

    path = output_path(args)
    rows = load_existing(path)
    for idx, (batch_size, learning_rate, repeat) in enumerate(trajectories, start=1):
        key = (batch_size, learning_rate, repeat)
        if key in rows and not args.overwrite:
            print(f"[{idx}/{len(trajectories)}] bs={batch_size:g}, lr={learning_rate:g}, repeat={repeat}: cached")
            continue
        print(f"[{idx}/{len(trajectories)}] bs={batch_size:g}, lr={learning_rate:g}, repeat={repeat}")
        row = compute_one(
            args,
            batch_size,
            learning_rate,
            repeat,
            model,
            train_loader,
            test_loader,
            criterion,
            device,
            freezing_rows,
        )
        rows[key] = row
        write_rows(path, rows)
        if row.get("status") == "ok":
            print(
                f"  tf={float(row['tf']):.6g}, checkpoint={row['tf_checkpoint']}, "
                f"flat={float(row['flatness_neg_log10']):.4f}, "
                f"train_acc={float(row['train_accuracy']):.4f}, "
                f"test_loss={float(row['test_loss']):.4f}"
            )
        else:
            print(f"  {row.get('status')}: {row.get('message')}")

    print_summary(rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
