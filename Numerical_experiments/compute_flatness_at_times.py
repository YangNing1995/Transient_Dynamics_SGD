"""Compare flatness at freezing, accuracy-convergence, and final checkpoints.

This pilot tests whether final-time flatness is contaminated by the long
cross-entropy tail after classification accuracy has already saturated.  For
each selected trajectory it evaluates the same flatness definition at:

  * t_f: continuation-based freezing time, rounded to a saved checkpoint;
  * t_acc: first saved checkpoint whose train accuracy reaches a threshold;
  * T: final saved checkpoint.
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
DEFAULT_CONDITIONS = ["50:0.05", "20:0.02", "100:0.002", "200:0.01"]


def parse_args():
    parser = ArgumentParser(description="Pilot flatness at t_f, t_acc, and final checkpoints.")
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
    parser.add_argument("--output_dir", type=Path, default=SCRIPT_DIR / "save_data_flatness_at_times")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help="Condition specs as batch_size:learning_rate.",
    )
    parser.add_argument("--repeat_start", type=int, default=1)
    parser.add_argument("--repeat_end", type=int, default=5)
    parser.add_argument("--acc_threshold", type=float, default=0.999)
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


def parse_condition(spec: str) -> tuple[int, float]:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Bad condition spec {spec!r}; expected batch_size:learning_rate")
    return int(parts[0]), float(parts[1])


def selected_trajectories(args):
    conditions = [parse_condition(spec) for spec in args.conditions]
    repeats = range(args.repeat_start, args.repeat_end + 1)
    return [(bs, lr, repeat) for bs, lr in conditions for repeat in repeats]


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


def load_checkpoint(model, checkpoint_file: Path, device: torch.device):
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)


def evaluate_checkpoint(model, checkpoint_file: Path, train_loader, test_loader, criterion, device):
    load_checkpoint(model, checkpoint_file, device)
    train_loss, train_acc = crs.evaluate_loss_accuracy(model, train_loader, criterion, device)
    test_loss, test_acc = crs.evaluate_loss_accuracy(model, test_loader, criterion, device)
    return train_loss, train_acc, test_loss, test_acc


def find_accuracy_checkpoint(
    model,
    iterations: list[int],
    checkpoint_root: Path,
    template: str,
    batch_size: int,
    learning_rate: float,
    repeat: int,
    train_loader,
    test_loader,
    criterion,
    device,
    threshold: float,
):
    last_metrics = None
    for iteration in iterations:
        checkpoint_file = crs.checkpoint_path(
            checkpoint_root,
            template,
            batch_size,
            learning_rate,
            repeat,
            iteration,
        )
        if not checkpoint_file.exists():
            continue
        load_checkpoint(model, checkpoint_file, device)
        train_loss, train_acc = crs.evaluate_loss_accuracy(model, train_loader, criterion, device)
        metrics = (train_loss, train_acc)
        last_metrics = (iteration, metrics)
        if train_acc >= threshold:
            return iteration, metrics, "ok"
    if last_metrics is None:
        return None, None, "missing_checkpoints"
    return last_metrics[0], last_metrics[1], "never_reached_threshold"


def flatness_checkpoint(model, checkpoint_file: Path, train_loader, test_loader, criterion, device, args):
    load_checkpoint(model, checkpoint_file, device)
    start = time.perf_counter()
    base_train_loss, base_train_acc, deltas, _, _ = crs.relative_sharpness_for_model(
        model,
        train_loader,
        criterion,
        device,
        args,
    )
    sharpness = float(np.mean(np.maximum(deltas, 0.0)))
    flatness = -math.log10(max(sharpness, 1e-12))
    test_loss, test_acc = crs.evaluate_loss_accuracy(model, test_loader, criterion, device)
    return {
        "train_loss": float(base_train_loss),
        "train_accuracy": float(base_train_acc),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "sharpness": sharpness,
        "flatness_neg_log10": flatness,
        "sharpness_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "min_delta": float(np.min(deltas)),
        "max_delta": float(np.max(deltas)),
        "wall_time_sec": float(time.perf_counter() - start),
    }


def output_path(args):
    return args.output_dir / "flatness_at_times_summary.csv"


def load_existing(path: Path):
    if not path.exists():
        return {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return {
            (int(row["batch_size"]), float(row["learning_rate"]), int(row["repeat"]), row["time_label"]): row
            for row in reader
        }


def write_rows(path: Path, rows):
    fieldnames = [
        "batch_size",
        "learning_rate",
        "repeat",
        "time_label",
        "checkpoint_iteration",
        "tf",
        "tf_checkpoint",
        "t_acc_checkpoint",
        "final_checkpoint",
        "acc_status",
        "definition",
        "rho",
        "num_directions",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "sharpness",
        "flatness_neg_log10",
        "sharpness_std",
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


def make_error_rows(args, batch_size, learning_rate, repeat, message):
    rows = {}
    for label in ["freeze", "acc", "final"]:
        rows[(batch_size, learning_rate, repeat, label)] = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "repeat": repeat,
            "time_label": label,
            "definition": args.definition,
            "rho": args.rho,
            "num_directions": args.num_directions,
            "status": "error",
            "message": message,
        }
    return rows


def scan_trajectory(
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
):
    saved_iterations = crs.list_checkpoint_iterations(
        args.checkpoint_dir,
        args.checkpoint_subdir_template,
        batch_size,
        learning_rate,
        repeat,
    )
    if not saved_iterations:
        return make_error_rows(args, batch_size, learning_rate, repeat, "no saved checkpoints")

    freeze_row = freezing_rows.get((batch_size, learning_rate, repeat))
    if freeze_row is None:
        return make_error_rows(args, batch_size, learning_rate, repeat, "missing freezing summary row")

    tf = float(freeze_row["tf"])
    tf_checkpoint = choose_checkpoint(saved_iterations, tf, args.freeze_rounding)
    final_checkpoint = int(max(saved_iterations))
    t_acc_checkpoint, _, acc_status = find_accuracy_checkpoint(
        model,
        saved_iterations,
        args.checkpoint_dir,
        args.checkpoint_subdir_template,
        batch_size,
        learning_rate,
        repeat,
        train_loader,
        test_loader,
        criterion,
        device,
        args.acc_threshold,
    )
    if t_acc_checkpoint is None:
        t_acc_checkpoint = final_checkpoint

    checkpoints = {
        "freeze": int(tf_checkpoint),
        "acc": int(t_acc_checkpoint),
        "final": int(final_checkpoint),
    }

    rows = {}
    for label, iteration in checkpoints.items():
        checkpoint_file = crs.checkpoint_path(
            args.checkpoint_dir,
            args.checkpoint_subdir_template,
            batch_size,
            learning_rate,
            repeat,
            iteration,
        )
        if not checkpoint_file.exists():
            metric = {}
            status = "error"
            message = f"missing checkpoint {checkpoint_file}"
        else:
            metric = flatness_checkpoint(
                model,
                checkpoint_file,
                train_loader,
                test_loader,
                criterion,
                device,
                args,
            )
            status = "ok"
            message = ""
        row = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "repeat": repeat,
            "time_label": label,
            "checkpoint_iteration": iteration,
            "tf": tf,
            "tf_checkpoint": tf_checkpoint,
            "t_acc_checkpoint": t_acc_checkpoint,
            "final_checkpoint": final_checkpoint,
            "acc_status": acc_status,
            "definition": args.definition,
            "rho": args.rho,
            "num_directions": args.num_directions,
            "status": status,
            "message": message,
        }
        row.update(metric)
        rows[(batch_size, learning_rate, repeat, label)] = row
    return rows


def print_pivot_summary(rows):
    ok_rows = [row for row in rows.values() if row.get("status") == "ok"]
    if not ok_rows:
        return
    by_label = {}
    for row in ok_rows:
        by_label.setdefault(row["time_label"], []).append(row)
    print("Mean metrics by time label:")
    for label in ["freeze", "acc", "final"]:
        group = by_label.get(label, [])
        if not group:
            continue
        flatness = np.array([float(row["flatness_neg_log10"]) for row in group], dtype=float)
        test_loss = np.array([float(row["test_loss"]) for row in group], dtype=float)
        test_acc = np.array([float(row["test_accuracy"]) for row in group], dtype=float)
        print(
            f"  {label}: n={len(group)}, flatness={np.mean(flatness):.4f}, "
            f"test_loss={np.mean(test_loss):.4f}, test_acc={np.mean(test_acc):.4f}"
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
        for batch_size, learning_rate, repeat in trajectories:
            saved = crs.list_checkpoint_iterations(
                args.checkpoint_dir,
                args.checkpoint_subdir_template,
                batch_size,
                learning_rate,
                repeat,
            )
            print(f"  bs={batch_size:g} lr={learning_rate:g} repeat={repeat}: saved={len(saved)}")
        return

    device = crs.resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Checkpoint root: {args.checkpoint_dir}")
    print(f"Freezing summary root: {args.freezing_summary_root}")
    print(f"Output root: {args.output_dir}")
    print(f"Definition={args.definition}, rho={args.rho}, K={args.num_directions}")

    train_loader, test_loader, train_count, test_count = crs.make_loaders(args)
    print(f"Train samples: {train_count}; test samples: {test_count}")
    criterion = crs.ClassificationLoss(args.loss_type)
    model = FCN(input_dim=crs.input_dim_for_dataset(args.dataset_name), hidden=args.hidden_num).to(device)
    freezing_rows = load_freezing_rows(args.freezing_summary_root)

    path = output_path(args)
    rows = load_existing(path)
    for idx, (batch_size, learning_rate, repeat) in enumerate(trajectories, start=1):
        expected_keys = [(batch_size, learning_rate, repeat, label) for label in ["freeze", "acc", "final"]]
        if all(key in rows for key in expected_keys) and not args.overwrite:
            print(f"[{idx}/{len(trajectories)}] bs={batch_size:g}, lr={learning_rate:g}, repeat={repeat}: cached")
            continue
        print(f"[{idx}/{len(trajectories)}] bs={batch_size:g}, lr={learning_rate:g}, repeat={repeat}")
        new_rows = scan_trajectory(
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
        rows.update(new_rows)
        write_rows(path, rows)
        for label in ["freeze", "acc", "final"]:
            row = new_rows[(batch_size, learning_rate, repeat, label)]
            if row.get("status") == "ok":
                print(
                    f"  {label}: t={row['checkpoint_iteration']}, "
                    f"flat={float(row['flatness_neg_log10']):.4f}, "
                    f"train_acc={float(row['train_accuracy']):.4f}, "
                    f"test_loss={float(row['test_loss']):.4f}"
                )
            else:
                print(f"  {label}: {row.get('status')}: {row.get('message')}")

    print_pivot_summary(rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
