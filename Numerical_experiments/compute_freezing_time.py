"""Endpoint-only scanner for continuation-based freezing times.

This script is a lightweight alternative to continue_training.py.  It runs a
fixed deterministic continuation from selected SGD checkpoints, records only
the endpoint signatures, and estimates the earliest checkpoint after which the
continuation endpoint stays in the same basin as the final checkpoint.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets

from data_utils import get_transform
from model import FCN


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH_SIZES = [1000, 500, 200, 100, 50, 20, 10]
DEFAULT_LEARNING_RATES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
ENDPOINT_CACHE_METADATA_KEYS = [
    "dataset_name",
    "data_dir",
    "download_data",
    "train_num",
    "test_num",
    "hidden_num",
    "checkpoint_subdir_template",
    "reference_iteration",
    "continuation_lr",
    "continuation_batch_size",
    "max_continuation_steps",
    "continuation_budget_mode",
    "loss_type",
]


@dataclass
class Endpoint:
    checkpoint_t: int
    train_loss: float
    test_loss: float
    train_accuracy: float
    test_accuracy: float
    pred_test: np.ndarray
    wrong_indices: np.ndarray
    converged: bool
    continuation_steps: int
    runtime_seconds: float


def parse_args():
    parser = ArgumentParser(
        description=(
            "Compute continuation-based freezing times from SGD checkpoints "
            "using endpoint-only deterministic continuation."
        )
    )
    parser.add_argument("--dataset_name", type=str, default="MNIST")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=SCRIPT_DIR / "data",
        help=(
            "Dataset root. The original data_utils.py used ./data, which depends "
            "on the submission working directory; this option makes the path explicit."
        ),
    )
    parser.add_argument(
        "--download_data",
        action="store_true",
        help="Allow torchvision to download the dataset if it is missing.",
    )
    parser.add_argument("--hidden_num", type=int, default=50)
    parser.add_argument("--train_num", type=int, default=100)
    parser.add_argument("--test_num", type=int, default=20)
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default=SCRIPT_DIR / "save_checkpoint",
        help="Root directory containing bs{B}_lr{eta}_repeat{r}/iteration_{t}.pt.",
    )
    parser.add_argument(
        "--checkpoint_subdir_template",
        type=str,
        default="bs{batch_size}_lr{learning_rate:g}_repeat{repeat}",
        help=(
            "Template for checkpoint subfolders. Available fields: "
            "{batch_size}, {learning_rate}, {repeat}. Example for old MSE runs: "
            "'bs{batch_size}_lr{learning_rate:g}_mse_repeat{repeat}'."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SCRIPT_DIR / "save_data_freezing_time",
        help="Directory for endpoint caches and freezing_time_summary.csv.",
    )
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--learning_rates", nargs="+", type=float, default=DEFAULT_LEARNING_RATES)
    parser.add_argument("--repeat_start", type=int, default=1)
    parser.add_argument("--repeat_end", type=int, default=20)
    parser.add_argument(
        "--reference_iteration",
        type=str,
        default="auto",
        help=(
            "Final SGD checkpoint used as the reference basin b_*. Use 'auto' "
            "to choose the largest saved checkpoint in each trajectory, "
            "'product' to use --lr_iteration_product / lr, or an explicit integer."
        ),
    )
    parser.add_argument(
        "--lr_iteration_product",
        type=float,
        default=100.0,
        help="Used only when --reference_iteration product; total_iterations = value / lr.",
    )
    parser.add_argument(
        "--num_checkpoint_intervals",
        type=int,
        default=100,
        help=(
            "Documented checkpoint layout: training saved 101 equally spaced "
            "checkpoints over 0..total_iterations."
        ),
    )
    parser.add_argument(
        "--coarse_step",
        type=int,
        default=None,
        help=(
            "Raw-iteration coarse spacing before transition refinement. If not "
            "set, use --coarse_checkpoint_stride on the saved checkpoint grid."
        ),
    )
    parser.add_argument(
        "--fine_step",
        type=int,
        default=None,
        help=(
            "Raw-iteration local refinement spacing. If not set, use "
            "--fine_checkpoint_stride on the saved checkpoint grid."
        ),
    )
    parser.add_argument(
        "--coarse_checkpoint_stride",
        type=int,
        default=10,
        help="Coarse backward scan stride in units of saved checkpoints.",
    )
    parser.add_argument(
        "--fine_checkpoint_stride",
        type=int,
        default=1,
        help="Refinement stride in units of saved checkpoints.",
    )
    parser.add_argument(
        "--verify_suffix",
        action="store_true",
        help=(
            "After adaptive refinement, compute every fine-grid checkpoint from "
            "the provisional t_f to the reference checkpoint."
        ),
    )
    parser.add_argument(
        "--continuation_lr",
        type=str,
        default="0.05",
        help=(
            "Learning rate for the deterministic continuation probe. Use a "
            "number for a fixed probe LR, or 'training'/'trajectory' to reuse "
            "the learning rate of each scanned trajectory."
        ),
    )
    parser.add_argument(
        "--loss_type",
        choices=["ce", "mse_logits", "mse_softmax"],
        default="ce",
        help=(
            "Continuation loss. Use ce for CrossEntropyLoss, mse_logits for "
            "MSE(output, one_hot), or mse_softmax for MSE(softmax(output), one_hot)."
        ),
    )
    parser.add_argument(
        "--continuation_batch_size",
        type=int,
        default=-1,
        help="Batch size for continuation. Use -1 for full train set.",
    )
    parser.add_argument("--max_continuation_steps", type=int, default=2000)
    parser.add_argument(
        "--max_continuation_eta_product",
        type=float,
        default=None,
        help=(
            "If set in fixed-budget mode, choose max continuation steps per "
            "trajectory as ceil(value / continuation_lr), keeping "
            "continuation_lr * max_steps constant. Example: 100 preserves the "
            "old 0.05 * 2000 continuation horizon."
        ),
    )
    parser.add_argument(
        "--continuation_budget_mode",
        choices=["fixed", "to_reference"],
        default="fixed",
        help=(
            "fixed: every checkpoint uses --max_continuation_steps. "
            "to_reference: checkpoint t_c uses max(reference_iteration - t_c, 0), "
            "matching Fig4-style fixed global horizon comparisons."
        ),
    )
    parser.add_argument("--min_continuation_steps", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=20)
    parser.add_argument("--early_stop_loss", type=float, default=5e-3)
    parser.add_argument("--early_stop_acc", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument(
        "--basin_mode",
        choices=["pred", "jaccard", "both"],
        default="pred",
        help="Endpoint basin comparison rule against the final checkpoint endpoint.",
    )
    parser.add_argument("--pred_threshold", type=float, default=0.99)
    parser.add_argument("--jaccard_threshold", type=float, default=0.95)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto, cpu, cuda, or a torch device string such as cuda:0.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Ignore existing endpoint cache for the selected trajectories.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="List selected trajectories and checkpoint directories without training.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def input_dim_for_dataset(dataset_name: str) -> int:
    if dataset_name == "MNIST":
        return 784
    if dataset_name == "CIFAR10":
        return 3072
    return 784


def get_dataset_from_root(dataset_name, root: Path, train=True, transform=None, train_num=None, test_num=None, download=False):
    if dataset_name == "MNIST":
        dataset_class = datasets.MNIST
        max_per_class = 6000 if train else 1000
    elif dataset_name == "CIFAR10":
        dataset_class = datasets.CIFAR10
        max_per_class = 5000 if train else 1000
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset = dataset_class(root=str(root), train=train, download=download, transform=transform)
    target_num = train_num if train else test_num
    if target_num is None or target_num == -1 or target_num >= max_per_class:
        return dataset

    subset_indices = []
    targets = torch.tensor(dataset.targets) if not isinstance(dataset.targets, torch.Tensor) else dataset.targets
    for label in range(10):
        class_indices = (targets == label).nonzero(as_tuple=True)[0]
        subset_indices.extend(class_indices[:target_num])
    return data.Subset(dataset, subset_indices)


class ClassificationLoss:
    def __init__(self, loss_type: str, num_classes: int = 10):
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "ce":
            return self.ce(output, target)
        one_hot = output.new_zeros((target.size(0), self.num_classes))
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        if self.loss_type == "mse_softmax":
            output = torch.softmax(output, dim=1)
        return self.mse(output, one_hot)


def make_loaders(args):
    train_num = args.train_num
    if train_num is None or train_num == -1:
        train_num = 6000 if args.dataset_name == "MNIST" else 5000

    test_num = args.test_num
    transform = get_transform(args.dataset_name)
    train_dataset = get_dataset_from_root(
        args.dataset_name,
        args.data_dir,
        train=True,
        transform=transform,
        train_num=train_num,
        test_num=test_num,
        download=args.download_data,
    )
    test_dataset = get_dataset_from_root(
        args.dataset_name,
        args.data_dir,
        train=False,
        transform=transform,
        train_num=train_num,
        test_num=test_num,
        download=args.download_data,
    )

    continuation_batch_size = args.continuation_batch_size
    if continuation_batch_size == -1:
        continuation_batch_size = len(train_dataset)

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=continuation_batch_size,
        shuffle=False,
    )
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
    )
    return train_loader, test_loader, len(train_dataset), len(test_dataset)


def checkpoint_subdir(template: str, batch_size: int, learning_rate: float, repeat: int) -> str:
    return template.format(batch_size=batch_size, learning_rate=learning_rate, repeat=repeat)


def checkpoint_path(
    root: Path,
    template: str,
    batch_size: int,
    learning_rate: float,
    repeat: int,
    iteration: int,
) -> Path:
    return root / checkpoint_subdir(template, batch_size, learning_rate, repeat) / f"iteration_{iteration}.pt"


def trajectory_dir(root: Path, template: str, batch_size: int, learning_rate: float, repeat: int) -> Path:
    return root / checkpoint_subdir(template, batch_size, learning_rate, repeat)


def list_checkpoint_iterations(root: Path, template: str, batch_size: int, learning_rate: float, repeat: int) -> list[int]:
    directory = trajectory_dir(root, template, batch_size, learning_rate, repeat)
    if not directory.exists():
        return []
    iterations: list[int] = []
    pattern = re.compile(r"^iteration_(\d+)\.pt$")
    for file in directory.iterdir():
        match = pattern.match(file.name)
        if match:
            iterations.append(int(match.group(1)))
    return sorted(set(iterations))


def resolve_reference_iteration(args, learning_rate: float, saved_iterations: list[int]) -> int | None:
    spec = str(args.reference_iteration).strip().lower()
    if spec in {"auto", "max", "last"}:
        return max(saved_iterations) if saved_iterations else None
    if spec == "product":
        return int(round(args.lr_iteration_product / learning_rate))
    return int(spec)


def cache_path(output_dir: Path, batch_size: int, learning_rate: float, repeat: int) -> Path:
    return output_dir / "endpoints" / f"bs{batch_size}_lr{learning_rate:g}_repeat{repeat}_endpoints.npz"


def summary_key(row: dict[str, str]) -> tuple[int, float, int]:
    return int(row["batch_size"]), float(row["learning_rate"]), int(row["repeat"])


def load_summary(path: Path) -> dict[tuple[int, float, int], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return {summary_key(row): row for row in reader}


def write_summary(path: Path, rows: dict[tuple[int, float, int], dict[str, object]]) -> None:
    fieldnames = [
        "batch_size",
        "learning_rate",
        "repeat",
        "tf",
        "eta_tf",
        "reference_iteration",
        "reference_iteration_spec",
        "lr_iteration_product",
        "num_checkpoint_intervals",
        "coarse_step",
        "fine_step",
        "coarse_checkpoint_stride",
        "fine_checkpoint_stride",
        "verify_suffix",
        "basin_mode",
        "pred_threshold",
        "jaccard_threshold",
        "continuation_lr",
        "continuation_lr_spec",
        "loss_type",
        "continuation_budget_mode",
        "max_continuation_steps",
        "max_continuation_steps_spec",
        "max_continuation_eta_product",
        "num_endpoints_computed",
        "num_cached_endpoints",
        "computed_continuation_steps",
        "computed_endpoint_runtime_sec",
        "trajectory_wall_time_sec",
        "sec_per_continuation_step",
        "reference_test_loss",
        "reference_test_accuracy",
        "confidence_flag",
        "status",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in sorted(rows):
            row = {name: rows[key].get(name, "") for name in fieldnames}
            writer.writerow(row)


def load_endpoint_cache(path: Path) -> dict[int, Endpoint]:
    if not path.exists():
        return {}
    data_npz = np.load(path, allow_pickle=True)
    endpoints: dict[int, Endpoint] = {}
    runtime_seconds = (
        data_npz["runtime_seconds"]
        if "runtime_seconds" in data_npz.files
        else np.full_like(data_npz["continuation_steps"], np.nan, dtype=np.float64)
    )
    for idx, checkpoint_t in enumerate(data_npz["checkpoint_t"].astype(int)):
        endpoints[int(checkpoint_t)] = Endpoint(
            checkpoint_t=int(checkpoint_t),
            train_loss=float(data_npz["train_loss"][idx]),
            test_loss=float(data_npz["test_loss"][idx]),
            train_accuracy=float(data_npz["train_accuracy"][idx]),
            test_accuracy=float(data_npz["test_accuracy"][idx]),
            pred_test=np.asarray(data_npz["pred_test"][idx], dtype=np.int64),
            wrong_indices=np.asarray(data_npz["wrong_indices"][idx], dtype=np.int64),
            converged=bool(data_npz["converged"][idx]),
            continuation_steps=int(data_npz["continuation_steps"][idx]),
            runtime_seconds=float(runtime_seconds[idx]),
        )
    return endpoints


def load_cache_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    data_npz = np.load(path, allow_pickle=True)
    if "metadata_json" not in data_npz.files:
        return {}
    try:
        raw = data_npz["metadata_json"]
        if hasattr(raw, "item"):
            raw = raw.item()
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def cache_metadata_matches(path: Path, metadata: dict[str, object]) -> bool:
    cached = load_cache_metadata(path)
    if not cached:
        return False
    for key in ENDPOINT_CACHE_METADATA_KEYS:
        value = metadata.get(key)
        if cached.get(key) != value:
            return False
    return True


def save_endpoint_cache(path: Path, endpoints: dict[int, Endpoint], metadata: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [endpoints[t] for t in sorted(endpoints)]
    if not ordered:
        return
    np.savez_compressed(
        path,
        checkpoint_t=np.array([e.checkpoint_t for e in ordered], dtype=np.int64),
        train_loss=np.array([e.train_loss for e in ordered], dtype=np.float64),
        test_loss=np.array([e.test_loss for e in ordered], dtype=np.float64),
        train_accuracy=np.array([e.train_accuracy for e in ordered], dtype=np.float64),
        test_accuracy=np.array([e.test_accuracy for e in ordered], dtype=np.float64),
        pred_test=np.stack([e.pred_test for e in ordered]).astype(np.int64),
        wrong_indices=np.array([e.wrong_indices for e in ordered], dtype=object),
        converged=np.array([e.converged for e in ordered], dtype=bool),
        continuation_steps=np.array([e.continuation_steps for e in ordered], dtype=np.int64),
        runtime_seconds=np.array([e.runtime_seconds for e in ordered], dtype=np.float64),
        metadata_json=json.dumps(metadata, sort_keys=True),
    )


def evaluate_endpoint(model, loader, criterion, device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch_data, batch_target in loader:
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            output = model(batch_data)
            loss = criterion(output, batch_target)
            pred = output.argmax(dim=1)
            total_loss += loss.item() * batch_data.size(0)
            correct += pred.eq(batch_target).sum().item()
            total += batch_target.size(0)
            predictions.append(pred.detach().cpu().numpy())
            targets_all.append(batch_target.detach().cpu().numpy())
    pred_np = np.concatenate(predictions).astype(np.int64)
    target_np = np.concatenate(targets_all).astype(np.int64)
    wrong_indices = np.flatnonzero(pred_np != target_np).astype(np.int64)
    return total_loss / total, correct / total, pred_np, wrong_indices


def resolve_continuation_lr(args, learning_rate: float) -> float:
    spec = str(args.continuation_lr).strip().lower()
    if spec in {"training", "trajectory", "same", "match", "eta"}:
        return float(learning_rate)
    return float(args.continuation_lr)


def resolve_max_continuation_steps(args, continuation_lr: float) -> int:
    if args.max_continuation_eta_product is None:
        return int(args.max_continuation_steps)
    if continuation_lr <= 0:
        raise ValueError("continuation_lr must be positive when using --max_continuation_eta_product.")
    return int(math.ceil(float(args.max_continuation_eta_product) / float(continuation_lr)))


def run_continuation_to_endpoint(
    model: torch.nn.Module,
    checkpoint_file: Path,
    train_loader,
    test_loader,
    criterion,
    device: torch.device,
    args,
    continuation_steps_budget: int,
    continuation_lr: float,
) -> Endpoint:
    endpoint_start = time.perf_counter()
    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=continuation_lr)
    loader_iter = iter(train_loader)
    converged = False
    patience_count = 0

    step = 0
    for step in range(1, continuation_steps_budget + 1):
        try:
            batch_data, batch_target = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch_data, batch_target = next(loader_iter)

        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)
        optimizer.zero_grad()
        output = model(batch_data)
        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()

        if step >= args.min_continuation_steps and step % args.eval_every == 0:
            train_loss, train_acc, _, _ = evaluate_endpoint(model, train_loader, criterion, device)
            if train_loss <= args.early_stop_loss and train_acc >= args.early_stop_acc:
                patience_count += 1
                if patience_count >= args.early_stop_patience:
                    converged = True
                    break
            else:
                patience_count = 0
    else:
        step = continuation_steps_budget

    train_loss, train_acc, _, _ = evaluate_endpoint(model, train_loader, criterion, device)
    test_loss, test_acc, pred_test, wrong_indices = evaluate_endpoint(model, test_loader, criterion, device)
    model.train()
    return Endpoint(
        checkpoint_t=-1,
        train_loss=train_loss,
        test_loss=test_loss,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        pred_test=pred_test,
        wrong_indices=wrong_indices,
        converged=converged,
        continuation_steps=step,
        runtime_seconds=time.perf_counter() - endpoint_start,
    )


def pred_similarity(a: Endpoint, b: Endpoint) -> float:
    return float(np.mean(a.pred_test == b.pred_test))


def jaccard_similarity(a: Endpoint, b: Endpoint) -> float:
    set_a = set(int(x) for x in a.wrong_indices.tolist())
    set_b = set(int(x) for x in b.wrong_indices.tolist())
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def same_basin(endpoint: Endpoint, reference: Endpoint, args) -> tuple[bool, float, float]:
    pred_sim = pred_similarity(endpoint, reference)
    jac_sim = jaccard_similarity(endpoint, reference)
    pred_ok = pred_sim >= args.pred_threshold
    jac_ok = jac_sim >= args.jaccard_threshold
    if args.basin_mode == "pred":
        return pred_ok, pred_sim, jac_sim
    if args.basin_mode == "jaccard":
        return jac_ok, pred_sim, jac_sim
    return pred_ok and jac_ok, pred_sim, jac_sim


def grid_points(start: int, stop: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("Grid step must be positive.")
    values = list(range(start, stop + 1, step))
    if stop not in values:
        values.append(stop)
    return sorted(set(values))


def available_grid(saved_iterations: list[int], points: Iterable[int]) -> list[int]:
    saved = set(saved_iterations)
    return [int(t) for t in points if int(t) in saved]


def select_saved_by_stride(saved_iterations: list[int], start: int, stop: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("Checkpoint stride must be positive.")
    points = [t for t in saved_iterations if start <= t <= stop]
    if not points:
        return []
    selected = points[::stride]
    if points[-1] not in selected:
        selected.append(points[-1])
    if points[0] not in selected:
        selected.insert(0, points[0])
    return sorted(set(selected))


def select_saved_grid(saved_iterations: list[int], start: int, stop: int, raw_step: int | None, checkpoint_stride: int) -> list[int]:
    if raw_step is not None:
        return available_grid(saved_iterations, grid_points(start, stop, raw_step))
    return select_saved_by_stride(saved_iterations, start, stop, checkpoint_stride)


def earliest_stable_suffix(times: list[int], same_by_time: dict[int, bool]) -> int | None:
    sorted_times = sorted(t for t in times if t in same_by_time)
    for idx, checkpoint_t in enumerate(sorted_times):
        later = sorted_times[idx:]
        if later and all(same_by_time[t] for t in later):
            return checkpoint_t
    return None


def confidence_flag(
    endpoints: dict[int, Endpoint],
    same_by_time: dict[int, bool],
    tf: int | None,
    args,
    verified_suffix: bool,
) -> str:
    if tf is None:
        return "failed"
    suffix_times = [t for t in sorted(same_by_time) if t >= tf]
    if not suffix_times or not all(same_by_time[t] for t in suffix_times):
        return "low"
    if any(not endpoints[t].converged for t in suffix_times):
        return "medium_unconverged_continuation"
    if verified_suffix:
        return "high_verified_suffix"
    return "medium_adaptive"


def continuation_budget_for(checkpoint_t: int, reference_t: int, args, max_continuation_steps: int) -> int:
    if args.continuation_budget_mode == "to_reference":
        return max(int(reference_t) - int(checkpoint_t), 0)
    return max_continuation_steps


def get_or_compute_endpoint(
    checkpoint_t: int,
    endpoints: dict[int, Endpoint],
    model: torch.nn.Module,
    checkpoint_file: Path,
    train_loader,
    test_loader,
    criterion,
    device: torch.device,
    args,
    continuation_steps_budget: int,
    continuation_lr: float,
    cache_file: Path,
    metadata: dict[str, object],
) -> tuple[Endpoint, bool]:
    if checkpoint_t in endpoints:
        return endpoints[checkpoint_t], True
    endpoint = run_continuation_to_endpoint(
        model,
        checkpoint_file,
        train_loader,
        test_loader,
        criterion,
        device,
        args,
        continuation_steps_budget,
        continuation_lr,
    )
    endpoint.checkpoint_t = checkpoint_t
    endpoints[checkpoint_t] = endpoint
    save_endpoint_cache(cache_file, endpoints, metadata)
    return endpoint, False


def scan_trajectory(
    batch_size: int,
    learning_rate: float,
    repeat: int,
    model: torch.nn.Module,
    train_loader,
    test_loader,
    criterion,
    device: torch.device,
    args,
) -> dict[str, object]:
    trajectory_start = time.perf_counter()
    continuation_lr = resolve_continuation_lr(args, learning_rate)
    max_continuation_steps = resolve_max_continuation_steps(args, continuation_lr)
    root = args.checkpoint_dir
    saved_iterations = list_checkpoint_iterations(
        root, args.checkpoint_subdir_template, batch_size, learning_rate, repeat
    )
    ref_t = resolve_reference_iteration(args, learning_rate, saved_iterations)
    if ref_t is None:
        return {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "repeat": repeat,
            "tf": "",
            "eta_tf": "",
            "reference_iteration": "",
            "reference_iteration_spec": args.reference_iteration,
            "lr_iteration_product": args.lr_iteration_product,
            "num_checkpoint_intervals": args.num_checkpoint_intervals,
            "coarse_step": args.coarse_step if args.coarse_step is not None else "",
            "fine_step": args.fine_step if args.fine_step is not None else "",
            "coarse_checkpoint_stride": args.coarse_checkpoint_stride,
            "fine_checkpoint_stride": args.fine_checkpoint_stride,
            "verify_suffix": args.verify_suffix,
            "basin_mode": args.basin_mode,
            "pred_threshold": args.pred_threshold,
            "jaccard_threshold": args.jaccard_threshold,
            "continuation_lr": continuation_lr,
            "continuation_lr_spec": str(args.continuation_lr),
            "loss_type": args.loss_type,
            "continuation_budget_mode": args.continuation_budget_mode,
            "max_continuation_steps": max_continuation_steps,
            "max_continuation_steps_spec": args.max_continuation_steps,
            "max_continuation_eta_product": args.max_continuation_eta_product,
            "num_endpoints_computed": 0,
            "num_cached_endpoints": 0,
            "computed_continuation_steps": 0,
            "computed_endpoint_runtime_sec": 0.0,
            "trajectory_wall_time_sec": time.perf_counter() - trajectory_start,
            "sec_per_continuation_step": "",
            "reference_test_loss": "",
            "reference_test_accuracy": "",
            "confidence_flag": "failed",
            "status": "missing_checkpoint_directory",
            "message": str(
                trajectory_dir(root, args.checkpoint_subdir_template, batch_size, learning_rate, repeat)
            ),
        }
    ref_checkpoint = checkpoint_path(
        root, args.checkpoint_subdir_template, batch_size, learning_rate, repeat, ref_t
    )
    if not ref_checkpoint.exists():
        return {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "repeat": repeat,
            "tf": "",
            "eta_tf": "",
            "reference_iteration": ref_t,
            "reference_iteration_spec": args.reference_iteration,
            "lr_iteration_product": args.lr_iteration_product,
            "num_checkpoint_intervals": args.num_checkpoint_intervals,
            "coarse_step": args.coarse_step if args.coarse_step is not None else "",
            "fine_step": args.fine_step if args.fine_step is not None else "",
            "coarse_checkpoint_stride": args.coarse_checkpoint_stride,
            "fine_checkpoint_stride": args.fine_checkpoint_stride,
            "verify_suffix": args.verify_suffix,
            "basin_mode": args.basin_mode,
            "pred_threshold": args.pred_threshold,
            "jaccard_threshold": args.jaccard_threshold,
            "continuation_lr": continuation_lr,
            "continuation_lr_spec": str(args.continuation_lr),
            "loss_type": args.loss_type,
            "continuation_budget_mode": args.continuation_budget_mode,
            "max_continuation_steps": max_continuation_steps,
            "max_continuation_steps_spec": args.max_continuation_steps,
            "max_continuation_eta_product": args.max_continuation_eta_product,
            "num_endpoints_computed": 0,
            "num_cached_endpoints": 0,
            "computed_continuation_steps": 0,
            "computed_endpoint_runtime_sec": 0.0,
            "trajectory_wall_time_sec": time.perf_counter() - trajectory_start,
            "sec_per_continuation_step": "",
            "reference_test_loss": "",
            "reference_test_accuracy": "",
            "confidence_flag": "failed",
            "status": "missing_reference_checkpoint",
            "message": str(ref_checkpoint),
        }

    metadata = {
        "dataset_name": args.dataset_name,
        "data_dir": str(args.data_dir),
        "download_data": args.download_data,
        "train_num": args.train_num,
        "test_num": args.test_num,
        "hidden_num": args.hidden_num,
        "checkpoint_subdir_template": args.checkpoint_subdir_template,
        "reference_iteration": ref_t,
        "continuation_lr": continuation_lr,
        "continuation_lr_spec": str(args.continuation_lr),
        "continuation_batch_size": args.continuation_batch_size,
        "max_continuation_steps": max_continuation_steps,
        "max_continuation_steps_spec": args.max_continuation_steps,
        "max_continuation_eta_product": args.max_continuation_eta_product,
        "continuation_budget_mode": args.continuation_budget_mode,
        "loss_type": args.loss_type,
        "basin_mode": args.basin_mode,
        "pred_threshold": args.pred_threshold,
        "jaccard_threshold": args.jaccard_threshold,
    }
    cache_file = cache_path(args.output_dir, batch_size, learning_rate, repeat)
    use_cache = (not args.overwrite_cache) and cache_metadata_matches(cache_file, metadata)
    endpoints = load_endpoint_cache(cache_file) if use_cache else {}
    cached_before = len(endpoints)
    cached_times_before = set(endpoints)

    reference, _ = get_or_compute_endpoint(
        ref_t,
        endpoints,
        model,
        ref_checkpoint,
        train_loader,
        test_loader,
        criterion,
        device,
        args,
        continuation_budget_for(ref_t, ref_t, args, max_continuation_steps),
        continuation_lr,
        cache_file,
        metadata,
    )

    same_by_time: dict[int, bool] = {ref_t: True}
    saved_iterations = [t for t in saved_iterations if t <= ref_t]
    coarse_points = select_saved_grid(
        saved_iterations,
        0,
        ref_t,
        args.coarse_step,
        args.coarse_checkpoint_stride,
    )
    if ref_t not in coarse_points:
        coarse_points.append(ref_t)
    coarse_points = sorted(set(coarse_points))

    mismatch_t = None
    stable_after_t = ref_t
    for checkpoint_t in sorted([t for t in coarse_points if t < ref_t], reverse=True):
        endpoint, _ = get_or_compute_endpoint(
            checkpoint_t,
            endpoints,
            model,
            checkpoint_path(
                root, args.checkpoint_subdir_template, batch_size, learning_rate, repeat, checkpoint_t
            ),
            train_loader,
            test_loader,
            criterion,
            device,
            args,
            continuation_budget_for(checkpoint_t, ref_t, args, max_continuation_steps),
            continuation_lr,
            cache_file,
            metadata,
        )
        is_same, _, _ = same_basin(endpoint, reference, args)
        same_by_time[checkpoint_t] = is_same
        if is_same:
            stable_after_t = checkpoint_t
        else:
            mismatch_t = checkpoint_t
            break

    verified_suffix = False
    if mismatch_t is None:
        provisional_tf = min(coarse_points) if coarse_points else ref_t
    else:
        fine_points = select_saved_grid(
            saved_iterations,
            mismatch_t,
            stable_after_t,
            args.fine_step,
            args.fine_checkpoint_stride,
        )
        for checkpoint_t in fine_points:
            if checkpoint_t in same_by_time:
                continue
            endpoint, _ = get_or_compute_endpoint(
                checkpoint_t,
                endpoints,
                model,
                checkpoint_path(
                    root,
                    args.checkpoint_subdir_template,
                    batch_size,
                    learning_rate,
                    repeat,
                    checkpoint_t,
                ),
                train_loader,
                test_loader,
                criterion,
                device,
                args,
                continuation_budget_for(checkpoint_t, ref_t, args, max_continuation_steps),
                continuation_lr,
                cache_file,
                metadata,
            )
            is_same, _, _ = same_basin(endpoint, reference, args)
            same_by_time[checkpoint_t] = is_same
        provisional_tf = earliest_stable_suffix(sorted(same_by_time), same_by_time)

    if args.verify_suffix and provisional_tf is not None:
        fine_suffix_points = select_saved_grid(
            saved_iterations,
            provisional_tf,
            ref_t,
            args.fine_step,
            args.fine_checkpoint_stride,
        )
        for checkpoint_t in fine_suffix_points:
            if checkpoint_t in same_by_time:
                continue
            endpoint, _ = get_or_compute_endpoint(
                checkpoint_t,
                endpoints,
                model,
                checkpoint_path(
                    root,
                    args.checkpoint_subdir_template,
                    batch_size,
                    learning_rate,
                    repeat,
                    checkpoint_t,
                ),
                train_loader,
                test_loader,
                criterion,
                device,
                args,
                continuation_budget_for(checkpoint_t, ref_t, args, max_continuation_steps),
                continuation_lr,
                cache_file,
                metadata,
            )
            is_same, _, _ = same_basin(endpoint, reference, args)
            same_by_time[checkpoint_t] = is_same
        verified_suffix = True
        provisional_tf = earliest_stable_suffix(sorted(same_by_time), same_by_time)

    tf = provisional_tf
    conf = confidence_flag(endpoints, same_by_time, tf, args, verified_suffix)
    status = "ok" if tf is not None else "no_stable_suffix"
    computed_times = sorted(set(endpoints) - cached_times_before)
    num_computed = len(computed_times)
    computed_steps = sum(endpoints[t].continuation_steps for t in computed_times)
    computed_runtime = sum(
        endpoints[t].runtime_seconds
        for t in computed_times
        if not math.isnan(endpoints[t].runtime_seconds)
    )
    trajectory_wall_time = time.perf_counter() - trajectory_start
    sec_per_step = computed_runtime / computed_steps if computed_steps > 0 else ""
    save_endpoint_cache(cache_file, endpoints, metadata)

    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "repeat": repeat,
        "tf": "" if tf is None else tf,
        "eta_tf": "" if tf is None else learning_rate * tf,
        "reference_iteration": ref_t,
        "reference_iteration_spec": args.reference_iteration,
        "lr_iteration_product": args.lr_iteration_product,
        "num_checkpoint_intervals": args.num_checkpoint_intervals,
        "coarse_step": args.coarse_step if args.coarse_step is not None else "",
        "fine_step": args.fine_step if args.fine_step is not None else "",
        "coarse_checkpoint_stride": args.coarse_checkpoint_stride,
        "fine_checkpoint_stride": args.fine_checkpoint_stride,
        "verify_suffix": args.verify_suffix,
        "basin_mode": args.basin_mode,
        "pred_threshold": args.pred_threshold,
        "jaccard_threshold": args.jaccard_threshold,
        "continuation_lr": continuation_lr,
        "continuation_lr_spec": str(args.continuation_lr),
        "loss_type": args.loss_type,
        "continuation_budget_mode": args.continuation_budget_mode,
        "max_continuation_steps": max_continuation_steps,
        "max_continuation_steps_spec": args.max_continuation_steps,
        "max_continuation_eta_product": args.max_continuation_eta_product,
        "num_endpoints_computed": num_computed,
        "num_cached_endpoints": cached_before,
        "computed_continuation_steps": computed_steps,
        "computed_endpoint_runtime_sec": computed_runtime,
        "trajectory_wall_time_sec": trajectory_wall_time,
        "sec_per_continuation_step": sec_per_step,
        "reference_test_loss": reference.test_loss,
        "reference_test_accuracy": reference.test_accuracy,
        "confidence_flag": conf,
        "status": status,
        "message": "",
    }


def selected_trajectories(args) -> list[tuple[int, float, int]]:
    repeats = range(args.repeat_start, args.repeat_end + 1)
    return [
        (int(batch_size), float(learning_rate), int(repeat))
        for batch_size in args.batch_sizes
        for learning_rate in args.learning_rates
        for repeat in repeats
    ]


def main():
    run_start = time.perf_counter()
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    trajectories = selected_trajectories(args)
    if args.dry_run:
        print(f"Selected {len(trajectories)} trajectories")
        for batch_size, learning_rate, repeat in trajectories[:20]:
            saved_iterations = list_checkpoint_iterations(
                args.checkpoint_dir,
                args.checkpoint_subdir_template,
                batch_size,
                learning_rate,
                repeat,
            )
            ref_t = resolve_reference_iteration(args, learning_rate, saved_iterations)
            path = checkpoint_path(
                args.checkpoint_dir,
                args.checkpoint_subdir_template,
                batch_size,
                learning_rate,
                repeat,
                ref_t if ref_t is not None else -1,
            )
            if ref_t is None:
                print(
                    f"bs={batch_size:g} lr={learning_rate:g} repeat={repeat}: "
                    "no checkpoints found under "
                    f"{trajectory_dir(args.checkpoint_dir, args.checkpoint_subdir_template, batch_size, learning_rate, repeat)}"
                )
            else:
                print(
                    f"bs={batch_size:g} lr={learning_rate:g} repeat={repeat}: "
                    f"reference={ref_t}, checkpoints={len(saved_iterations)}, {path}"
                )
        if len(trajectories) > 20:
            print(f"... {len(trajectories) - 20} more")
        return

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Dataset root: {args.data_dir}")
    print(f"Checkpoint root: {args.checkpoint_dir}")
    print(f"Output root: {args.output_dir}")

    train_loader, test_loader, train_count, test_count = make_loaders(args)
    print(f"Train samples: {train_count}; test samples: {test_count}")

    model = FCN(input_dim=input_dim_for_dataset(args.dataset_name), hidden=args.hidden_num).to(device)
    criterion = ClassificationLoss(args.loss_type)

    summary_path = args.output_dir / "freezing_time_summary.csv"
    summary_rows: dict[tuple[int, float, int], dict[str, object]] = load_summary(summary_path)

    for idx, (batch_size, learning_rate, repeat) in enumerate(trajectories, start=1):
        print(
            f"[{idx}/{len(trajectories)}] "
            f"bs={batch_size:g}, lr={learning_rate:g}, repeat={repeat}"
        )
        row = scan_trajectory(
            batch_size,
            learning_rate,
            repeat,
            model,
            train_loader,
            test_loader,
            criterion,
            device,
            args,
        )
        summary_rows[(batch_size, learning_rate, repeat)] = row
        write_summary(summary_path, summary_rows)
        if row["status"] == "ok":
            sec_per_step = row["sec_per_continuation_step"]
            sec_per_step_text = (
                f"{float(sec_per_step):.4g}s/step" if sec_per_step != "" else "n/a"
            )
            print(
                "  "
                f"tf={row['tf']}, eta_tf={float(row['eta_tf']):.6g}, "
                f"computed={row['num_endpoints_computed']}, "
                f"steps={row['computed_continuation_steps']}, "
                f"wall={float(row['trajectory_wall_time_sec']):.1f}s, "
                f"{sec_per_step_text}, "
                f"confidence={row['confidence_flag']}"
            )
        else:
            print(f"  status={row['status']}: {row['message']}")

    print(f"Summary written to {summary_path}")
    print(f"Total wall time: {time.perf_counter() - run_start:.1f}s")


if __name__ == "__main__":
    # Keep torch deterministic enough for a deterministic continuation probe.
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
