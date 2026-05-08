"""Compare flatness definitions under function-preserving rescalings.

The goal is to screen candidate flatness metrics before running them on every
checkpoint.  For each trusted solution, the script constructs several ReLU
scale-equivalent parameterizations that compute the same function, then
measures how much each flatness definition changes.  A useful definition should
be stable under these rescalings.
"""

from __future__ import annotations

import csv
import math
import re
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils import data
from torchvision import datasets

from data_utils import get_transform
from model import FCN


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DEFINITIONS = [
    "random_global",
    "random_tensor",
    "random_filter",
    "sam_global",
    "asam_tensor",
    "asam_element",
]
DEFAULT_CONDITIONS = [
    "50:0.05:2",
    "20:0.02:1",
    "100:0.002:1",
    "200:0.01:1",
]
DEFAULT_SCALE_PAIRS = [
    "1:1",
    "4:0.25",
    "0.25:4",
    "10:0.1",
    "0.1:10",
    "4:4",
    "0.25:0.25",
]


def parse_args():
    parser = ArgumentParser(
        description="Pilot multiple flatness definitions under ReLU scale symmetries."
    )
    parser.add_argument("--dataset_name", type=str, default="MNIST")
    parser.add_argument("--data_dir", type=Path, default=SCRIPT_DIR / "data")
    parser.add_argument("--download_data", action="store_true")
    parser.add_argument("--hidden_num", type=int, default=50)
    parser.add_argument("--train_num", type=int, default=100)
    parser.add_argument("--test_num", type=int, default=20)
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
    parser.add_argument("--output_dir", type=Path, default=SCRIPT_DIR / "save_data_flatness_pilot")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help="Condition specs as batch_size:learning_rate:repeat.",
    )
    parser.add_argument(
        "--scale_pairs",
        nargs="+",
        default=DEFAULT_SCALE_PAIRS,
        help=(
            "Positive hidden-layer rescalings c1:c2. For FCN, layer1 is scaled "
            "by c1 and layer2 by c2 while preserving the represented function."
        ),
    )
    parser.add_argument(
        "--definitions",
        nargs="+",
        default=DEFAULT_DEFINITIONS,
        choices=DEFAULT_DEFINITIONS,
    )
    parser.add_argument(
        "--checkpoint_iteration",
        type=str,
        default="auto",
        help="'auto'/'last', 'product', or an explicit integer.",
    )
    parser.add_argument("--lr_iteration_product", type=float, default=100.0)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--num_directions", type=int, default=20)
    parser.add_argument("--symmetric_random", action="store_true")
    parser.add_argument("--include_bias", action="store_true")
    parser.add_argument("--adaptive_epsilon", type=float, default=1e-12)
    parser.add_argument("--relative_floor", type=float, default=1e-12)
    parser.add_argument("--eval_batch_size", type=int, default=-1)
    parser.add_argument("--max_train_batches", type=int, default=-1)
    parser.add_argument("--loss_type", choices=["ce", "mse_logits", "mse_softmax"], default="ce")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
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


def parse_condition(spec: str) -> tuple[int, float, int]:
    parts = spec.split(":")
    if len(parts) != 3:
        raise ValueError(f"Bad condition spec {spec!r}; expected batch_size:learning_rate:repeat")
    return int(parts[0]), float(parts[1]), int(parts[2])


def parse_scale_pair(spec: str) -> tuple[str, float, float]:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Bad scale spec {spec!r}; expected c1:c2")
    c1 = float(parts[0])
    c2 = float(parts[1])
    if c1 <= 0 or c2 <= 0:
        raise ValueError(f"Scale factors must be positive: {spec!r}")
    return f"c1={c1:g}_c2={c2:g}", c1, c2


def get_dataset_from_root(
    dataset_name,
    root: Path,
    train=True,
    transform=None,
    train_num=None,
    test_num=None,
    download=False,
):
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

    transform = get_transform(args.dataset_name)
    train_dataset = get_dataset_from_root(
        args.dataset_name,
        args.data_dir,
        train=True,
        transform=transform,
        train_num=train_num,
        test_num=args.test_num,
        download=args.download_data,
    )
    test_dataset = get_dataset_from_root(
        args.dataset_name,
        args.data_dir,
        train=False,
        transform=transform,
        train_num=train_num,
        test_num=args.test_num,
        download=args.download_data,
    )
    eval_batch_size = args.eval_batch_size
    if eval_batch_size == -1:
        eval_batch_size = max(len(train_dataset), len(test_dataset))
    train_loader = data.DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    return train_loader, test_loader, len(train_dataset), len(test_dataset)


def checkpoint_subdir(template: str, batch_size: int, learning_rate: float, repeat: int) -> str:
    return template.format(batch_size=batch_size, learning_rate=learning_rate, repeat=repeat)


def trajectory_dir(root: Path, template: str, batch_size: int, learning_rate: float, repeat: int) -> Path:
    return root / checkpoint_subdir(template, batch_size, learning_rate, repeat)


def list_checkpoint_iterations(root: Path, template: str, batch_size: int, learning_rate: float, repeat: int):
    directory = trajectory_dir(root, template, batch_size, learning_rate, repeat)
    if not directory.exists():
        return []
    pattern = re.compile(r"^iteration_(\d+)\.pt$")
    iterations = []
    for file in directory.iterdir():
        match = pattern.match(file.name)
        if match:
            iterations.append(int(match.group(1)))
    return sorted(set(iterations))


def resolve_checkpoint_iteration(args, learning_rate: float, saved_iterations):
    spec = str(args.checkpoint_iteration).strip().lower()
    if spec in {"auto", "max", "last"}:
        return max(saved_iterations) if saved_iterations else None
    if spec == "product":
        return int(round(args.lr_iteration_product / learning_rate))
    return int(spec)


def checkpoint_path(
    root: Path,
    template: str,
    batch_size: int,
    learning_rate: float,
    repeat: int,
    iteration: int,
) -> Path:
    return root / checkpoint_subdir(template, batch_size, learning_rate, repeat) / f"iteration_{iteration}.pt"


def clone_state_dict(state_dict):
    return {key: value.detach().clone() for key, value in state_dict.items()}


def rescale_fcn_state_dict(state_dict, c1: float, c2: float):
    """Apply an exact positive-homogeneous rescaling for FCN's two ReLU layers."""
    scaled = clone_state_dict(state_dict)
    required = [
        "net.1.weight",
        "net.1.bias",
        "net.3.weight",
        "net.3.bias",
        "net.5.weight",
    ]
    missing = [key for key in required if key not in scaled]
    if missing:
        raise KeyError(f"State dict does not match FCN keys; missing {missing}")
    scaled["net.1.weight"].mul_(c1)
    scaled["net.1.bias"].mul_(c1)
    scaled["net.3.weight"].mul_(c2 / c1)
    scaled["net.3.bias"].mul_(c2)
    scaled["net.5.weight"].div_(c2)
    return scaled


def evaluate_loss_accuracy(model, loader, criterion, device: torch.device, max_batches: int = -1):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(loader):
            if max_batches != -1 and batch_idx >= max_batches:
                break
            batch_data = batch_data.to(device)
            batch_target = batch_target.to(device)
            output = model(batch_data)
            loss = criterion(output, batch_target)
            pred = output.argmax(dim=1)
            total_loss += loss.item() * batch_target.size(0)
            correct += pred.eq(batch_target).sum().item()
            total += batch_target.size(0)
    if total == 0:
        return math.nan, math.nan
    return total_loss / total, correct / total


def functional_difference(model_a, model_b, loader, device: torch.device):
    model_a.eval()
    model_b.eval()
    sq_sum = 0.0
    count = 0
    max_abs = 0.0
    with torch.no_grad():
        for batch_data, _ in loader:
            batch_data = batch_data.to(device)
            out_a = model_a(batch_data)
            out_b = model_b(batch_data)
            diff = out_a - out_b
            sq_sum += diff.pow(2).sum().item()
            count += diff.numel()
            max_abs = max(max_abs, diff.abs().max().item())
    mse = sq_sum / count if count else math.nan
    return mse, max_abs


def selected_params(model, include_bias: bool):
    params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not include_bias and param.ndim < 2:
            continue
        params.append((name, param))
    return params


def tensor_norm(tensor: torch.Tensor) -> torch.Tensor:
    return torch.norm(tensor.reshape(-1))


def global_param_norm(params, floor: float) -> torch.Tensor:
    total = None
    for _, param in params:
        term = torch.sum(param.detach() * param.detach())
        total = term if total is None else total + term
    if total is None:
        return torch.tensor(float(floor))
    return torch.sqrt(torch.clamp(total, min=floor * floor))


def add_perturbations(perturbations, alpha: float = 1.0):
    with torch.no_grad():
        for param, eps in perturbations:
            param.add_(eps * alpha)


def random_global_perturbations(model, include_bias: bool, relative_floor: float):
    params = selected_params(model, include_bias)
    theta_norm = global_param_norm(params, relative_floor)
    randoms = [(param, torch.randn_like(param)) for _, param in params]
    total = None
    for _, rand in randoms:
        term = torch.sum(rand * rand)
        total = term if total is None else total + term
    rand_norm = torch.sqrt(torch.clamp(total, min=relative_floor * relative_floor))
    return [(param, rand * (theta_norm / rand_norm)) for param, rand in randoms]


def random_tensor_perturbations(model, include_bias: bool, relative_floor: float):
    perturbations = []
    for _, param in selected_params(model, include_bias):
        rand = torch.randn_like(param)
        rand_norm = torch.clamp(tensor_norm(rand), min=relative_floor)
        theta_norm = torch.clamp(tensor_norm(param.detach()), min=relative_floor)
        perturbations.append((param, rand * (theta_norm / rand_norm)))
    return perturbations


def normalize_rows(rand: torch.Tensor, reference: torch.Tensor, floor: float) -> torch.Tensor:
    flat_rand = rand.reshape(rand.shape[0], -1)
    flat_ref = reference.reshape(reference.shape[0], -1)
    rand_norm = torch.norm(flat_rand, dim=1, keepdim=True).clamp(min=floor)
    ref_norm = torch.norm(flat_ref, dim=1, keepdim=True).clamp(min=floor)
    return (flat_rand * (ref_norm / rand_norm)).reshape_as(rand)


def random_filter_perturbations(model, include_bias: bool, relative_floor: float):
    perturbations = []
    for _, param in selected_params(model, include_bias):
        rand = torch.randn_like(param)
        if param.ndim >= 2:
            eps = normalize_rows(rand, param.detach(), relative_floor)
        else:
            rand_norm = torch.clamp(tensor_norm(rand), min=relative_floor)
            theta_norm = torch.clamp(tensor_norm(param.detach()), min=relative_floor)
            eps = rand * (theta_norm / rand_norm)
        perturbations.append((param, eps))
    return perturbations


def random_sharpness(model, train_loader, criterion, device, args, definition: str):
    base_loss, _ = evaluate_loss_accuracy(model, train_loader, criterion, device, args.max_train_batches)
    deltas = []
    for direction_idx in range(args.num_directions):
        torch.manual_seed(args.seed + 1009 * direction_idx + 17 * len(definition))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 1009 * direction_idx + 17 * len(definition))
        if definition == "random_global":
            perturbations = random_global_perturbations(model, args.include_bias, args.relative_floor)
        elif definition == "random_tensor":
            perturbations = random_tensor_perturbations(model, args.include_bias, args.relative_floor)
        elif definition == "random_filter":
            perturbations = random_filter_perturbations(model, args.include_bias, args.relative_floor)
        else:
            raise ValueError(definition)

        add_perturbations(perturbations, args.rho)
        plus_loss, _ = evaluate_loss_accuracy(model, train_loader, criterion, device, args.max_train_batches)
        if args.symmetric_random:
            add_perturbations(perturbations, -2.0 * args.rho)
            minus_loss, _ = evaluate_loss_accuracy(model, train_loader, criterion, device, args.max_train_batches)
            add_perturbations(perturbations, args.rho)
            delta = 0.5 * (plus_loss + minus_loss) - base_loss
        else:
            add_perturbations(perturbations, -args.rho)
            delta = plus_loss - base_loss
        deltas.append(max(delta, 0.0))
    deltas_np = np.asarray(deltas, dtype=float)
    return float(np.mean(deltas_np)), float(np.std(deltas_np, ddof=1)) if deltas_np.size > 1 else 0.0


def compute_gradients(model, train_loader, criterion, device, max_batches: int):
    model.train()
    model.zero_grad()
    total = 0
    total_loss = 0.0
    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        if max_batches != -1 and batch_idx >= max_batches:
            break
        batch_data = batch_data.to(device)
        batch_target = batch_target.to(device)
        output = model(batch_data)
        loss = criterion(output, batch_target)
        weighted_loss = loss * batch_target.size(0)
        weighted_loss.backward()
        total_loss += weighted_loss.item()
        total += batch_target.size(0)
    if total == 0:
        return math.nan, []
    params = selected_params(model, include_bias=True)
    grads = []
    for name, param in params:
        if param.grad is None:
            grads.append((name, param, torch.zeros_like(param)))
        else:
            grads.append((name, param, param.grad.detach().clone() / float(total)))
    model.zero_grad()
    return total_loss / float(total), grads


def sam_global_perturbations(model, grads, include_bias: bool, rho: float, floor: float):
    selected = [(name, param, grad) for name, param, grad in grads if include_bias or param.ndim >= 2]
    theta_norm = global_param_norm([(name, param) for name, param, _ in selected], floor)
    total = None
    for _, _, grad in selected:
        term = torch.sum(grad * grad)
        total = term if total is None else total + term
    grad_norm = torch.sqrt(torch.clamp(total, min=floor * floor))
    return [(param, grad * (rho * theta_norm / grad_norm)) for _, param, grad in selected]


def asam_tensor_perturbations(grads, include_bias: bool, rho: float, floor: float):
    selected = [(name, param, grad) for name, param, grad in grads if include_bias or param.ndim >= 2]
    denom_sq = None
    scales = []
    for _, param, grad in selected:
        scale = torch.clamp(tensor_norm(param.detach()), min=floor)
        scales.append(scale)
        term = (scale * scale) * torch.sum(grad * grad)
        denom_sq = term if denom_sq is None else denom_sq + term
    denom = torch.sqrt(torch.clamp(denom_sq, min=floor * floor))
    perturbations = []
    for (_, param, grad), scale in zip(selected, scales):
        perturbations.append((param, grad * (rho * scale * scale / denom)))
    return perturbations


def asam_element_perturbations(grads, include_bias: bool, rho: float, adaptive_epsilon: float, floor: float):
    selected = [(name, param, grad) for name, param, grad in grads if include_bias or param.ndim >= 2]
    denom_sq = None
    scales = []
    for _, param, grad in selected:
        scale = param.detach().abs() + adaptive_epsilon
        scales.append(scale)
        term = torch.sum((scale * grad) * (scale * grad))
        denom_sq = term if denom_sq is None else denom_sq + term
    denom = torch.sqrt(torch.clamp(denom_sq, min=floor * floor))
    perturbations = []
    for (_, param, grad), scale in zip(selected, scales):
        perturbations.append((param, rho * scale * scale * grad / denom))
    return perturbations


def adversarial_sharpness(model, train_loader, criterion, device, args, definition: str):
    base_loss, grads = compute_gradients(model, train_loader, criterion, device, args.max_train_batches)
    if not np.isfinite(base_loss):
        return math.nan, math.nan
    if definition == "sam_global":
        perturbations = sam_global_perturbations(model, grads, args.include_bias, args.rho, args.relative_floor)
    elif definition == "asam_tensor":
        perturbations = asam_tensor_perturbations(grads, args.include_bias, args.rho, args.relative_floor)
    elif definition == "asam_element":
        perturbations = asam_element_perturbations(
            grads,
            args.include_bias,
            args.rho,
            args.adaptive_epsilon,
            args.relative_floor,
        )
    else:
        raise ValueError(definition)
    add_perturbations(perturbations, 1.0)
    perturbed_loss, _ = evaluate_loss_accuracy(model, train_loader, criterion, device, args.max_train_batches)
    add_perturbations(perturbations, -1.0)
    return float(max(perturbed_loss - base_loss, 0.0)), 0.0


def compute_flatness(model, train_loader, criterion, device, args, definition: str):
    if definition.startswith("random_"):
        return random_sharpness(model, train_loader, criterion, device, args, definition)
    return adversarial_sharpness(model, train_loader, criterion, device, args, definition)


def output_paths(output_dir: Path):
    return output_dir / "flatness_definition_pilot.csv", output_dir / "flatness_definition_scale_summary.csv"


def write_rows(path: Path, rows):
    fieldnames = [
        "condition",
        "batch_size",
        "learning_rate",
        "repeat",
        "checkpoint_iteration",
        "scale_name",
        "scale_hidden1",
        "scale_hidden2",
        "function_mse",
        "function_max_abs",
        "definition",
        "rho",
        "num_directions",
        "symmetric_random",
        "include_bias",
        "base_train_loss",
        "scaled_train_loss",
        "scaled_train_accuracy",
        "scaled_test_loss",
        "scaled_test_accuracy",
        "sharpness",
        "sharpness_std",
        "flatness_neg_log10",
        "wall_time_sec",
        "status",
        "message",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def summarize_scale_sensitivity(rows):
    grouped = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        key = (row["condition"], row["definition"])
        grouped.setdefault(key, []).append(row)

    per_condition = []
    for (condition, definition), group in grouped.items():
        flatness = np.asarray([float(row["flatness_neg_log10"]) for row in group], dtype=float)
        sharpness = np.asarray([float(row["sharpness"]) for row in group], dtype=float)
        sharpness = sharpness[np.isfinite(sharpness)]
        mean_sharp = float(np.mean(sharpness)) if sharpness.size else math.nan
        cv_sharp = (
            float(np.std(sharpness, ddof=1) / abs(mean_sharp))
            if sharpness.size > 1 and abs(mean_sharp) > 0
            else math.nan
        )
        per_condition.append(
            {
                "condition": condition,
                "definition": definition,
                "num_scale_variants": len(group),
                "flatness_range": float(np.nanmax(flatness) - np.nanmin(flatness)),
                "flatness_std": float(np.nanstd(flatness, ddof=1)) if flatness.size > 1 else 0.0,
                "sharpness_cv": cv_sharp,
                "max_function_abs": float(max(float(row["function_max_abs"]) for row in group)),
            }
        )

    by_definition = {}
    for row in per_condition:
        by_definition.setdefault(row["definition"], []).append(row)

    summary = []
    for definition, group in by_definition.items():
        ranges = np.asarray([row["flatness_range"] for row in group], dtype=float)
        stds = np.asarray([row["flatness_std"] for row in group], dtype=float)
        cvs = np.asarray([row["sharpness_cv"] for row in group], dtype=float)
        cvs = cvs[np.isfinite(cvs)]
        summary.append(
            {
                "definition": definition,
                "num_conditions": len(group),
                "median_flatness_range": float(np.median(ranges)),
                "mean_flatness_range": float(np.mean(ranges)),
                "max_flatness_range": float(np.max(ranges)),
                "median_flatness_std": float(np.median(stds)),
                "median_sharpness_cv": float(np.median(cvs)) if cvs.size else math.nan,
                "max_function_abs": float(max(row["max_function_abs"] for row in group)),
            }
        )
    return per_condition, sorted(summary, key=lambda row: row["median_flatness_range"])


def write_summary(path: Path, rows):
    fieldnames = [
        "definition",
        "num_conditions",
        "median_flatness_range",
        "mean_flatness_range",
        "max_flatness_range",
        "median_flatness_std",
        "median_sharpness_cv",
        "max_function_abs",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main():
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    conditions = [parse_condition(spec) for spec in args.conditions]
    scale_pairs = [parse_scale_pair(spec) for spec in args.scale_pairs]
    if args.dry_run:
        print("Conditions:")
        for bs, lr, repeat in conditions:
            saved = list_checkpoint_iterations(args.checkpoint_dir, args.checkpoint_subdir_template, bs, lr, repeat)
            ckpt = resolve_checkpoint_iteration(args, lr, saved)
            print(f"  bs={bs:g} lr={lr:g} repeat={repeat}: checkpoint={ckpt}, saved={len(saved)}")
        print("Scale pairs:")
        for name, c1, c2 in scale_pairs:
            print(f"  {name}: c1={c1:g}, c2={c2:g}")
        print("Definitions:", ", ".join(args.definitions))
        return

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Dataset root: {args.data_dir}")
    print(f"Checkpoint root: {args.checkpoint_dir}")
    print(f"Output root: {args.output_dir}")
    train_loader, test_loader, train_count, test_count = make_loaders(args)
    print(f"Train samples: {train_count}; test samples: {test_count}")

    criterion = ClassificationLoss(args.loss_type)
    base_model = FCN(input_dim=input_dim_for_dataset(args.dataset_name), hidden=args.hidden_num).to(device)
    scaled_model = FCN(input_dim=input_dim_for_dataset(args.dataset_name), hidden=args.hidden_num).to(device)

    output_csv, summary_csv = output_paths(args.output_dir)
    rows = []
    for cond_idx, (bs, lr, repeat) in enumerate(conditions, start=1):
        saved = list_checkpoint_iterations(args.checkpoint_dir, args.checkpoint_subdir_template, bs, lr, repeat)
        checkpoint_iteration = resolve_checkpoint_iteration(args, lr, saved)
        if checkpoint_iteration is None:
            print(f"[{cond_idx}/{len(conditions)}] bs={bs:g} lr={lr:g} repeat={repeat}: missing checkpoints")
            continue
        ckpt = checkpoint_path(args.checkpoint_dir, args.checkpoint_subdir_template, bs, lr, repeat, checkpoint_iteration)
        if not ckpt.exists():
            print(f"[{cond_idx}/{len(conditions)}] missing {ckpt}")
            continue
        print(f"[{cond_idx}/{len(conditions)}] bs={bs:g} lr={lr:g} repeat={repeat} checkpoint={checkpoint_iteration}")
        base_state = torch.load(ckpt, map_location=device)
        base_model.load_state_dict(base_state)
        base_model.to(device)
        base_train_loss, base_train_acc = evaluate_loss_accuracy(
            base_model,
            train_loader,
            criterion,
            device,
            args.max_train_batches,
        )
        print(f"  base train loss={base_train_loss:.6g}, acc={base_train_acc:.4f}")

        condition_name = f"bs{bs:g}_lr{lr:g}_repeat{repeat}"
        for scale_name, c1, c2 in scale_pairs:
            scaled_state = rescale_fcn_state_dict(base_state, c1, c2)
            scaled_model.load_state_dict(scaled_state)
            scaled_model.to(device)
            function_mse, function_max_abs = functional_difference(base_model, scaled_model, train_loader, device)
            scaled_train_loss, scaled_train_acc = evaluate_loss_accuracy(
                scaled_model,
                train_loader,
                criterion,
                device,
                args.max_train_batches,
            )
            scaled_test_loss, scaled_test_acc = evaluate_loss_accuracy(scaled_model, test_loader, criterion, device)
            print(
                f"  scale {scale_name}: max|df|={function_max_abs:.3g}, "
                f"train_loss={scaled_train_loss:.6g}, test_acc={scaled_test_acc:.4f}"
            )
            for definition in args.definitions:
                start = time.perf_counter()
                status = "ok"
                message = ""
                try:
                    sharpness, sharpness_std = compute_flatness(
                        scaled_model,
                        train_loader,
                        criterion,
                        device,
                        args,
                        definition,
                    )
                except Exception as exc:
                    status = "error"
                    message = str(exc)
                    sharpness = math.nan
                    sharpness_std = math.nan
                flatness = -math.log10(max(float(sharpness), 1e-12)) if np.isfinite(sharpness) else math.nan
                row = {
                    "condition": condition_name,
                    "batch_size": bs,
                    "learning_rate": lr,
                    "repeat": repeat,
                    "checkpoint_iteration": checkpoint_iteration,
                    "scale_name": scale_name,
                    "scale_hidden1": c1,
                    "scale_hidden2": c2,
                    "function_mse": function_mse,
                    "function_max_abs": function_max_abs,
                    "definition": definition,
                    "rho": args.rho,
                    "num_directions": args.num_directions,
                    "symmetric_random": args.symmetric_random,
                    "include_bias": args.include_bias,
                    "base_train_loss": base_train_loss,
                    "scaled_train_loss": scaled_train_loss,
                    "scaled_train_accuracy": scaled_train_acc,
                    "scaled_test_loss": scaled_test_loss,
                    "scaled_test_accuracy": scaled_test_acc,
                    "sharpness": sharpness,
                    "sharpness_std": sharpness_std,
                    "flatness_neg_log10": flatness,
                    "wall_time_sec": time.perf_counter() - start,
                    "status": status,
                    "message": message,
                }
                rows.append(row)
                print(
                    f"    {definition}: sharp={sharpness:.6g}, "
                    f"flat={flatness:.4f}, status={status}"
                )
                write_rows(output_csv, rows)

    _, summary = summarize_scale_sensitivity(rows)
    write_summary(summary_csv, summary)
    print(f"Wrote {output_csv}")
    print(f"Wrote {summary_csv}")
    print("Scale-sensitivity ranking; lower median_flatness_range is better:")
    for row in summary:
        print(
            "  {definition}: median_range={median_flatness_range:.4g}, "
            "max_range={max_flatness_range:.4g}, median_cv={median_sharpness_cv:.4g}".format(**row)
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
