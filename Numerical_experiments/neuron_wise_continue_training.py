"""Compute neuron-wise relative flatness for Figure 4 continuation trajectories.

This is a drop-in replacement for the Hessian-based geometry pass in
hessian_continue_training.py.  It does not train models; it reads the saved
reference SGD checkpoints and the saved deterministic-continuation checkpoints,
then evaluates

    F = -log10 mean_k [L(theta + rho d_k) - L(theta)]_+

where each random direction d_k is normalized separately for each output-neuron
weight row.  The default parameters reproduce the original Fig. 4 setting:
bs=50, lr=0.05, repeat=2, continuation starts t_c=0:20:1000, and each
continuation is sampled at 101 checkpoints over 2000 GD iterations.
"""

from __future__ import annotations

import csv
import math
import time
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.io import savemat

import compute_relative_sharpness as crs
from model import FCN


SCRIPT_DIR = Path(__file__).resolve().parent
SHARPNESS_FLOOR = 1e-12


def parse_int_range(spec: str) -> list[int]:
    """Parse 'start:stop:step' or a comma-separated integer list."""
    text = str(spec).strip()
    if ":" in text:
        parts = [int(x) for x in text.split(":")]
        if len(parts) != 3:
            raise ValueError("range syntax must be start:stop:step")
        start, stop, step = parts
        return list(range(start, stop + 1, step))
    return [int(x) for x in text.split(",") if x.strip()]


def parse_args():
    parser = ArgumentParser(description="Compute neuron-wise flatness for continuation-training checkpoints.")
    parser.add_argument("--dataset_name", type=str, default="MNIST")
    parser.add_argument("--data_dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--hidden_num", type=int, default=50)
    parser.add_argument("--train_num", type=int, default=100)
    parser.add_argument("--test_num", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=-1, help="-1 uses the full train/test set.")
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--loss_type", choices=["ce", "mse_logits", "mse_softmax"], default="ce")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument("--load_batch_size", type=int, default=50)
    parser.add_argument("--load_learning_rate", type=float, default=0.05)
    parser.add_argument("--load_realization", type=int, default=2)
    parser.add_argument("--base_total_iterations", type=int, default=-1, help="-1 uses round(100 / learning_rate).")
    parser.add_argument("--continue_total_iterations", type=int, default=2000)
    parser.add_argument("--load_iteration_range", type=str, default="0:1000:20")
    parser.add_argument("--base_checkpoint_dir", type=Path, default=SCRIPT_DIR / "save_checkpoint")
    parser.add_argument(
        "--continue_checkpoint_dir",
        type=Path,
        default=SCRIPT_DIR / "save_checkpoint_continue_training",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SCRIPT_DIR / "save_data_continue_training_neuron_wise",
    )

    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--num_directions", type=int, default=20)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--include_bias", action="store_true")
    parser.add_argument("--relative_floor", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["base", "continue", "both"],
        default="both",
        help="Whether to compute the reference SGD trajectory, continuation trajectories, or both.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def evenly_spaced_iterations(total_iterations: int, count: int = 101) -> np.ndarray:
    return np.round(np.linspace(0, total_iterations, count)).astype(int)


def input_dim_for_dataset(dataset_name: str) -> int:
    if dataset_name == "CIFAR10":
        return 3072
    return 784


def base_checkpoint_path(args, iteration: int) -> Path:
    subdir = f"bs{args.load_batch_size}_lr{args.load_learning_rate:g}_repeat{args.load_realization}"
    return args.base_checkpoint_dir / subdir / f"iteration_{int(iteration)}.pt"


def continue_checkpoint_path(args, load_iteration: int, local_iteration: int) -> Path:
    subdir = (
        f"bs{args.load_batch_size}_lr{args.load_learning_rate:g}_"
        f"repeat{args.load_realization}_ct{int(load_iteration)}"
    )
    return args.continue_checkpoint_dir / subdir / f"iteration_{int(local_iteration)}.pt"


def output_condition_dir(args) -> Path:
    return args.output_dir / (
        f"bs{args.load_batch_size}_lr{args.load_learning_rate:g}_"
        f"repeat{args.load_realization}_ct"
    )


def make_sharpness_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        definition="neuron_wise",
        rho=args.rho,
        num_directions=args.num_directions,
        symmetric=args.symmetric,
        include_bias=args.include_bias,
        relative_floor=args.relative_floor,
        max_train_batches=args.max_train_batches,
        seed=args.seed,
    )


def evaluate_checkpoint(model, checkpoint_path: Path, train_loader, criterion, device, sharpness_args):
    if not checkpoint_path.exists():
        return {
            "status": "missing",
            "message": str(checkpoint_path),
            "train_loss": math.nan,
            "train_accuracy": math.nan,
            "sharpness": math.nan,
            "flatness": math.nan,
        }

    state_dict = torch.load(str(checkpoint_path), map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    train_loss, train_acc, deltas, _, _ = crs.relative_sharpness_for_model(
        model, train_loader, criterion, device, sharpness_args
    )
    positive = np.maximum(deltas, 0.0)
    sharpness = float(np.mean(positive))
    flatness = -math.log10(max(sharpness, SHARPNESS_FLOOR)) if np.isfinite(sharpness) else math.nan
    return {
        "status": "ok",
        "message": "",
        "train_loss": float(train_loss),
        "train_accuracy": float(train_acc),
        "sharpness": sharpness,
        "flatness": flatness,
    }


def write_long_summary(path: Path, rows: list[dict]):
    fieldnames = [
        "trajectory_type",
        "load_iteration",
        "local_iteration",
        "global_iteration",
        "train_loss",
        "train_accuracy",
        "sharpness_mean_positive_delta",
        "flatness_neg_log10",
        "status",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def compute_base(args, model, train_loader, criterion, device, sharpness_args, summary_rows):
    total_iterations = args.base_total_iterations
    if total_iterations == -1:
        total_iterations = int(round(100.0 / args.load_learning_rate))
    time_points = evenly_spaced_iterations(total_iterations)

    flatness = np.full(time_points.shape, np.nan, dtype=np.float64)
    sharpness = np.full(time_points.shape, np.nan, dtype=np.float64)
    train_loss = np.full(time_points.shape, np.nan, dtype=np.float64)
    train_accuracy = np.full(time_points.shape, np.nan, dtype=np.float64)

    for i, iteration in enumerate(time_points):
        result = evaluate_checkpoint(model, base_checkpoint_path(args, int(iteration)), train_loader, criterion, device, sharpness_args)
        flatness[i] = result["flatness"]
        sharpness[i] = result["sharpness"]
        train_loss[i] = result["train_loss"]
        train_accuracy[i] = result["train_accuracy"]
        summary_rows.append(
            {
                "trajectory_type": "base",
                "load_iteration": "",
                "local_iteration": int(iteration),
                "global_iteration": int(iteration),
                "train_loss": result["train_loss"],
                "train_accuracy": result["train_accuracy"],
                "sharpness_mean_positive_delta": result["sharpness"],
                "flatness_neg_log10": result["flatness"],
                "status": result["status"],
                "message": result["message"],
            }
        )
        print(f"base t={int(iteration)} flat={flatness[i]:.6g} status={result['status']}")

    out = output_condition_dir(args) / "save_neuron_wise_flatness_base.mat"
    savemat(
        str(out),
        {
            "Flatness": flatness,
            "Sharpness": sharpness,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "time_points": time_points,
            "rho": args.rho,
            "num_directions": args.num_directions,
        },
    )
    print(f"Wrote {out}")


def compute_continue(args, model, train_loader, criterion, device, sharpness_args, summary_rows):
    load_iterations = parse_int_range(args.load_iteration_range)
    time_points = evenly_spaced_iterations(args.continue_total_iterations)

    aggregate_flatness = np.full((len(load_iterations), len(time_points)), np.nan, dtype=np.float64)
    aggregate_sharpness = np.full_like(aggregate_flatness, np.nan)

    for row_idx, load_iteration in enumerate(load_iterations):
        flatness = np.full(time_points.shape, np.nan, dtype=np.float64)
        sharpness = np.full(time_points.shape, np.nan, dtype=np.float64)
        train_loss = np.full(time_points.shape, np.nan, dtype=np.float64)
        train_accuracy = np.full(time_points.shape, np.nan, dtype=np.float64)

        for col_idx, local_iteration in enumerate(time_points):
            ckpt = continue_checkpoint_path(args, int(load_iteration), int(local_iteration))
            result = evaluate_checkpoint(model, ckpt, train_loader, criterion, device, sharpness_args)
            flatness[col_idx] = result["flatness"]
            sharpness[col_idx] = result["sharpness"]
            train_loss[col_idx] = result["train_loss"]
            train_accuracy[col_idx] = result["train_accuracy"]
            summary_rows.append(
                {
                    "trajectory_type": "continue",
                    "load_iteration": int(load_iteration),
                    "local_iteration": int(local_iteration),
                    "global_iteration": int(load_iteration + local_iteration),
                    "train_loss": result["train_loss"],
                    "train_accuracy": result["train_accuracy"],
                    "sharpness_mean_positive_delta": result["sharpness"],
                    "flatness_neg_log10": result["flatness"],
                    "status": result["status"],
                    "message": result["message"],
                }
            )

        aggregate_flatness[row_idx, :] = flatness
        aggregate_sharpness[row_idx, :] = sharpness
        out = output_condition_dir(args) / f"save_neuron_wise_flatness_ct{int(load_iteration)}.mat"
        savemat(
            str(out),
            {
                "Flatness": flatness,
                "Sharpness": sharpness,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "time_points": time_points,
                "load_iteration": int(load_iteration),
                "rho": args.rho,
                "num_directions": args.num_directions,
            },
        )
        print(f"Wrote {out}")

    out = output_condition_dir(args) / "save_neuron_wise_flatness_continue_all.mat"
    savemat(
        str(out),
        {
            "Flatness_all": aggregate_flatness,
            "Sharpness_all": aggregate_sharpness,
            "time_points": time_points,
            "load_iteration_list": np.asarray(load_iterations, dtype=int),
            "rho": args.rho,
            "num_directions": args.num_directions,
        },
    )
    print(f"Wrote {out}")


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.base_checkpoint_dir = args.base_checkpoint_dir.resolve()
    args.continue_checkpoint_dir = args.continue_checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    output_condition_dir(args).mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        load_iterations = parse_int_range(args.load_iteration_range)
        base_total = args.base_total_iterations if args.base_total_iterations != -1 else int(round(100.0 / args.load_learning_rate))
        first_base = base_checkpoint_path(args, int(evenly_spaced_iterations(base_total)[0]))
        first_ct = continue_checkpoint_path(args, int(load_iterations[0]), 0)
        print(f"Base checkpoints: {len(evenly_spaced_iterations(base_total))}")
        print(f"Continuation starts: {len(load_iterations)} from {load_iterations[0]} to {load_iterations[-1]}")
        print(f"Continuation checkpoints per start: {len(evenly_spaced_iterations(args.continue_total_iterations))}")
        print(f"First base checkpoint: {first_base} exists={first_base.exists()}")
        print(f"First continuation checkpoint: {first_ct} exists={first_ct.exists()}")
        print(f"Output: {output_condition_dir(args)}")
        return

    device = crs.resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Output: {output_condition_dir(args)}")
    print(f"Definition: neuron_wise, rho={args.rho}, K={args.num_directions}")

    train_loader, _, train_count, _ = crs.make_loaders(args)
    print(f"Train samples for sharpness: {train_count}")

    model = FCN(input_dim=input_dim_for_dataset(args.dataset_name), hidden=args.hidden_num).to(device)
    criterion = crs.ClassificationLoss(args.loss_type)
    sharpness_args = make_sharpness_args(args)
    summary_rows: list[dict] = []

    start = time.time()
    if args.mode in ("base", "both"):
        compute_base(args, model, train_loader, criterion, device, sharpness_args, summary_rows)
    if args.mode in ("continue", "both"):
        compute_continue(args, model, train_loader, criterion, device, sharpness_args, summary_rows)

    summary_path = output_condition_dir(args) / "neuron_wise_continue_flatness_summary.csv"
    write_long_summary(summary_path, summary_rows)
    print(f"Wrote {summary_path}")
    print(f"Runtime: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
