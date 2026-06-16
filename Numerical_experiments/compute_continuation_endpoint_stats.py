"""Run deterministic continuations on a fixed grid and save endpoint metrics.

This is an endpoint-only replacement for the heavier continue_training.py +
hessian_continue_training.py workflow.  For each selected SGD checkpoint it
runs full-batch GD to a common reference iteration, evaluates endpoint
train/test metrics, computes Hessian top-k flatness at the endpoint, and saves
one cache per repeat plus a CSV summary.
"""

from __future__ import annotations

import csv
import math
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from compute_freezing_time import (
    ClassificationLoss,
    checkpoint_path,
    evaluate_endpoint,
    input_dim_for_dataset,
    list_checkpoint_iterations,
    make_loaders,
)
from compute_relative_sharpness import hessian_topk_sharpness_for_model
from model import FCN


SCRIPT_DIR = Path(__file__).resolve().parent

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_name", type=str, default="MNIST")
    parser.add_argument("--data_dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--hidden_num", type=int, default=50)
    parser.add_argument("--train_num", type=int, default=100)
    parser.add_argument("--test_num", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--repeat_start", type=int, default=1)
    parser.add_argument("--repeat_end", type=int, default=20)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=SCRIPT_DIR / "save_checkpoint",
        help="Root containing bs{B}_lr{eta}_repeat{r}/iteration_{t}.pt.",
    )
    parser.add_argument(
        "--checkpoint_subdir_template",
        type=str,
        default="bs{batch_size}_lr{learning_rate:g}_repeat{repeat}",
    )
    parser.add_argument("--output_dir", type=Path, default=SCRIPT_DIR / "continuation_endpoint_stats")
    parser.add_argument("--reference_iteration", type=int, default=2000)
    parser.add_argument("--checkpoint_start", type=int, default=0)
    parser.add_argument("--checkpoint_stop", type=int, default=2000)
    parser.add_argument("--checkpoint_stride", type=int, default=20)
    parser.add_argument(
        "--continuation_lr",
        type=str,
        default="training",
        help="'training' to reuse --learning_rate, otherwise a numeric GD learning rate.",
    )
    parser.add_argument(
        "--continuation_batch_size",
        type=int,
        default=-1,
        help="-1 uses the full training subset in one batch.",
    )
    parser.add_argument(
        "--continuation_micro_batch_size",
        type=int,
        default=-1,
        help="Positive value smaller than the full batch uses gradient accumulation.",
    )
    parser.add_argument("--loss_type", type=str, default="ce", choices=("ce", "mse", "mse_softmax"))
    parser.add_argument("--hessian_topk", type=int, default=10)
    parser.add_argument("--hessian_layer_index", type=int, default=2)
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=-1,
        help="Passed to Hessian helper; -1 uses the first full-batch loader batch.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def resolve_continuation_lr(args) -> float:
    spec = str(args.continuation_lr).strip().lower()
    if spec in {"training", "trajectory", "same", "match", "eta"}:
        return float(args.learning_rate)
    return float(args.continuation_lr)


def requested_checkpoints(args) -> list[int]:
    if args.checkpoint_stride <= 0:
        raise ValueError("--checkpoint_stride must be positive")
    stop = min(int(args.checkpoint_stop), int(args.reference_iteration))
    return list(range(int(args.checkpoint_start), stop + 1, int(args.checkpoint_stride)))


def run_full_batch_continuation(model, train_loader, criterion, device, steps: int, lr: float) -> float:
    start = time.perf_counter()
    if steps <= 0:
        return 0.0

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    num_loader_batches = len(train_loader)
    use_grad_accum = num_loader_batches > 1
    total_train_count = len(train_loader.dataset)

    for _ in range(steps):
        optimizer.zero_grad()
        if use_grad_accum:
            for batch_data, batch_target in train_loader:
                batch_data = batch_data.to(device)
                batch_target = batch_target.to(device)
                loss = criterion(model(batch_data), batch_target)
                (loss * (batch_data.size(0) / total_train_count)).backward()
        else:
            batch_data, batch_target = next(iter(train_loader))
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            loss = criterion(model(batch_data), batch_target)
            loss.backward()
        optimizer.step()
    return time.perf_counter() - start


def flatness_from_hessian(model, train_loader, criterion, device, args):
    train_loss, train_acc, sharpness_arr, topk, _ = hessian_topk_sharpness_for_model(
        model,
        train_loader,
        criterion,
        device,
        args,
    )
    sharpness = float(sharpness_arr[0]) if len(sharpness_arr) else math.nan
    if np.isfinite(sharpness) and sharpness > 0:
        flatness = float(1.0 / sharpness)
        flatness_neg_log10 = float(-math.log10(max(sharpness, 1e-12)))
    else:
        flatness = math.nan
        flatness_neg_log10 = math.nan
    lambda_max = float(topk[0]) if len(topk) and np.isfinite(topk[0]) else math.nan
    return train_loss, train_acc, sharpness, flatness, flatness_neg_log10, lambda_max, topk


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "batch_size",
        "learning_rate",
        "repeat",
        "checkpoint_t",
        "reference_iteration",
        "continuation_steps",
        "continuation_lr",
        "train_loss",
        "train_accuracy",
        "test_loss",
        "test_accuracy",
        "hessian_topk",
        "hessian_layer_index",
        "hessian_sharpness",
        "hessian_flatness",
        "flatness_neg_log10",
        "hessian_lambda_max",
        "continuation_wall_time_sec",
        "hessian_wall_time_sec",
        "endpoint_wall_time_sec",
        "status",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def save_repeat_cache(path: Path, rows: list[dict[str, object]], wrong_indices: list[np.ndarray], topk_eigs: list[np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    padded_topk = np.full((len(topk_eigs), int(rows[0]["hessian_topk"])), np.nan, dtype=np.float64)
    for i, eigs in enumerate(topk_eigs):
        n = min(padded_topk.shape[1], len(eigs))
        if n:
            padded_topk[i, :n] = np.asarray(eigs[:n], dtype=np.float64)
    np.savez_compressed(
        path,
        checkpoint_t=np.array([r["checkpoint_t"] for r in rows], dtype=np.int64),
        continuation_steps=np.array([r["continuation_steps"] for r in rows], dtype=np.int64),
        train_loss=np.array([r["train_loss"] for r in rows], dtype=np.float64),
        test_loss=np.array([r["test_loss"] for r in rows], dtype=np.float64),
        train_accuracy=np.array([r["train_accuracy"] for r in rows], dtype=np.float64),
        test_accuracy=np.array([r["test_accuracy"] for r in rows], dtype=np.float64),
        hessian_sharpness=np.array([r["hessian_sharpness"] for r in rows], dtype=np.float64),
        hessian_flatness=np.array([r["hessian_flatness"] for r in rows], dtype=np.float64),
        flatness_neg_log10=np.array([r["flatness_neg_log10"] for r in rows], dtype=np.float64),
        hessian_lambda_max=np.array([r["hessian_lambda_max"] for r in rows], dtype=np.float64),
        hessian_topk_eigs=padded_topk,
        wrong_indices=np.array(wrong_indices, dtype=object),
    )


def scan_repeat(repeat: int, model, train_loader, test_loader, criterion, device, args) -> list[dict[str, object]]:
    continuation_lr = resolve_continuation_lr(args)
    saved = set(
        list_checkpoint_iterations(
            args.checkpoint_dir,
            args.checkpoint_subdir_template,
            args.batch_size,
            args.learning_rate,
            repeat,
        )
    )
    points = [t for t in requested_checkpoints(args) if t in saved]
    if not points:
        raise FileNotFoundError(f"No requested checkpoints found for repeat {repeat}")

    repeat_rows: list[dict[str, object]] = []
    wrong_indices: list[np.ndarray] = []
    topk_eigs: list[np.ndarray] = []

    for checkpoint_t in points:
        endpoint_start = time.perf_counter()
        checkpoint_file = checkpoint_path(
            args.checkpoint_dir,
            args.checkpoint_subdir_template,
            args.batch_size,
            args.learning_rate,
            repeat,
            checkpoint_t,
        )
        state_dict = torch.load(checkpoint_file, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.train()

        continuation_steps = max(0, int(args.reference_iteration) - int(checkpoint_t))
        cont_wall = run_full_batch_continuation(
            model,
            train_loader,
            criterion,
            device,
            continuation_steps,
            continuation_lr,
        )

        train_loss, train_acc, _, _ = evaluate_endpoint(model, train_loader, criterion, device)
        test_loss, test_acc, _, wrong = evaluate_endpoint(model, test_loader, criterion, device)

        hessian_start = time.perf_counter()
        _, _, sharpness, flatness, flatness_neg_log10, lambda_max, topk = flatness_from_hessian(
            model,
            train_loader,
            criterion,
            device,
            args,
        )
        hessian_wall = time.perf_counter() - hessian_start

        row = {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "repeat": repeat,
            "checkpoint_t": checkpoint_t,
            "reference_iteration": args.reference_iteration,
            "continuation_steps": continuation_steps,
            "continuation_lr": continuation_lr,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "hessian_topk": args.hessian_topk,
            "hessian_layer_index": args.hessian_layer_index,
            "hessian_sharpness": sharpness,
            "hessian_flatness": flatness,
            "flatness_neg_log10": flatness_neg_log10,
            "hessian_lambda_max": lambda_max,
            "continuation_wall_time_sec": cont_wall,
            "hessian_wall_time_sec": hessian_wall,
            "endpoint_wall_time_sec": time.perf_counter() - endpoint_start,
            "status": "ok",
            "message": "",
        }
        repeat_rows.append(row)
        wrong_indices.append(wrong)
        topk_eigs.append(np.asarray(topk, dtype=np.float64))
        print(
            "repeat={repeat} checkpoint_t={checkpoint_t} steps={steps} "
            "test_loss={loss:.6f} flatness={flat:.6g} wall={wall:.1f}s".format(
                repeat=repeat,
                checkpoint_t=checkpoint_t,
                steps=continuation_steps,
                loss=test_loss,
                flat=flatness,
                wall=row["endpoint_wall_time_sec"],
            ),
            flush=True,
        )

    cache_name = f"bs{args.batch_size}_lr{args.learning_rate:g}_repeat{repeat}_endpoint_stats.npz"
    save_repeat_cache(args.output_dir / "endpoints" / cache_name, repeat_rows, wrong_indices, topk_eigs)
    return repeat_rows


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.data_dir = args.data_dir.resolve()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, train_count, test_count = make_loaders(args)
    input_dim = input_dim_for_dataset(args.dataset_name)
    model = FCN(input_dim=input_dim, hidden=args.hidden_num).to(device)
    criterion = ClassificationLoss(args.loss_type)

    points = requested_checkpoints(args)
    repeats = list(range(args.repeat_start, args.repeat_end + 1))
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")
    print(f"Train/test counts: {train_count}/{test_count}")
    print(f"Repeats: {repeats}")
    print(f"Requested checkpoints: {points[0]}..{points[-1]} stride={args.checkpoint_stride} n={len(points)}")
    print(f"Reference iteration: {args.reference_iteration}")
    print(f"Continuation lr: {resolve_continuation_lr(args)}")
    print(f"Hessian layer index/topk: {args.hessian_layer_index}/{args.hessian_topk}")

    if args.dry_run:
        for repeat in repeats:
            saved = set(
                list_checkpoint_iterations(
                    args.checkpoint_dir,
                    args.checkpoint_subdir_template,
                    args.batch_size,
                    args.learning_rate,
                    repeat,
                )
            )
            available = [t for t in points if t in saved]
            print(f"dry_run repeat={repeat}: requested={len(points)} available={len(available)}")
        return

    all_rows: list[dict[str, object]] = []
    for repeat in repeats:
        try:
            all_rows.extend(scan_repeat(repeat, model, train_loader, test_loader, criterion, device, args))
            write_summary(args.output_dir / "endpoint_stats_summary.csv", all_rows)
        except Exception as exc:
            row = {
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "repeat": repeat,
                "status": "error",
                "message": str(exc),
            }
            all_rows.append(row)
            write_summary(args.output_dir / "endpoint_stats_summary.csv", all_rows)
            raise

    write_summary(args.output_dir / "endpoint_stats_summary.csv", all_rows)
    print(f"Summary written to {args.output_dir / 'endpoint_stats_summary.csv'}")


if __name__ == "__main__":
    main()
