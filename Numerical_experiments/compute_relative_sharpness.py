"""Estimate local sharpness/flatness of saved SGD solutions.

Supported definitions:
  - tensor_wise: random directions normalized by each full parameter tensor.
  - neuron_wise: random directions normalized by each output neuron/filter row.
  - hessian_topk: geometric mean of the largest Hessian eigenvalues.
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
import torch.nn.functional as F
from torch.utils import data
from torchvision import datasets
try:
    from torch.func import hessian
    HESSIAN_IS_CURRIED = True
except ImportError:  # Older torch versions used on some clusters.
    from torch.autograd.functional import hessian
    HESSIAN_IS_CURRIED = False

from data_utils import get_transform
from model import FCN


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BATCH_SIZES = [1000, 500, 200, 100, 50, 20, 10]
DEFAULT_LEARNING_RATES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
SUPPORTED_DEFINITIONS = ("tensor_wise", "neuron_wise", "hessian_topk")


torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def parse_args():
    parser = ArgumentParser(
        description=(
            "Compute a simple relative sharpness proxy from saved checkpoints. "
            "Perturbations are normalized tensor-by-tensor by the parameter norm."
        )
    )
    parser.add_argument("--dataset_name", type=str, default="MNIST")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=SCRIPT_DIR / "data",
        help="Dataset root. Use --download_data only if the dataset is missing.",
    )
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
        help=(
            "Template for checkpoint subfolders. Available fields: "
            "{batch_size}, {learning_rate}, {repeat}."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=SCRIPT_DIR / "save_data_relative_sharpness",
    )
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--learning_rates", nargs="+", type=float, default=DEFAULT_LEARNING_RATES)
    parser.add_argument("--repeat_start", type=int, default=1)
    parser.add_argument("--repeat_end", type=int, default=20)
    parser.add_argument(
        "--checkpoint_iteration",
        type=str,
        default="auto",
        help=(
            "Checkpoint to evaluate. Use 'auto'/'last' for the largest saved "
            "checkpoint, 'product' for --lr_iteration_product / lr, 'freezing' "
            "for the continuation-defined t_f checkpoint, or an integer."
        ),
    )
    parser.add_argument(
        "--lr_iteration_product",
        type=float,
        default=100.0,
        help="Used only when --checkpoint_iteration product.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.05,
        help="Relative perturbation radius.",
    )
    parser.add_argument(
        "--definition",
        choices=SUPPORTED_DEFINITIONS,
        default="tensor_wise",
        help=(
            "Flatness definition. tensor_wise normalizes each full parameter tensor; "
            "neuron_wise normalizes each output-neuron row/filter; hessian_topk "
            "uses the geometric mean of the largest Hessian eigenvalues."
        ),
    )
    parser.add_argument(
        "--num_directions",
        type=int,
        default=10,
        help="Number of random directions per checkpoint; ignored by --definition hessian_topk.",
    )
    parser.add_argument(
        "--hessian_topk",
        type=int,
        default=10,
        help="Number of largest Hessian eigenvalues used by --definition hessian_topk.",
    )
    parser.add_argument(
        "--hessian_layer_index",
        type=int,
        default=2,
        help=(
            "Index in model.state_dict() of the weight tensor used for Hessian "
            "flatness. For the current FCN, 2 is the second Linear layer weight."
        ),
    )
    parser.add_argument(
        "--freezing_summary_root",
        type=Path,
        default=SCRIPT_DIR.parent / "freezing_time_all_ce" / "jaccard095",
        help="Root containing bs*_lr*/freezing_time_summary.csv for --checkpoint_iteration freezing.",
    )
    parser.add_argument(
        "--freeze_rounding",
        choices=["ceil", "nearest", "floor"],
        default="ceil",
        help="How to map t_f to the saved checkpoint grid for --checkpoint_iteration freezing.",
    )
    parser.add_argument(
        "--symmetric",
        action="store_true",
        help=(
            "Use 0.5 * [L(theta+delta)+L(theta-delta)] - L(theta). "
            "This is more curvature-like but roughly doubles runtime."
        ),
    )
    parser.add_argument(
        "--include_bias",
        action="store_true",
        help="Also perturb bias tensors. Default perturbs weight tensors only.",
    )
    parser.add_argument(
        "--relative_floor",
        type=float,
        default=1e-12,
        help="Minimum tensor norm used when a parameter tensor is nearly zero.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=-1,
        help="Batch size for loss evaluation. Use -1 for full train/test sets.",
    )
    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=-1,
        help="Use only the first N train batches for a faster pilot; -1 uses all.",
    )
    parser.add_argument(
        "--loss_type",
        choices=["ce", "mse_logits", "mse_softmax"],
        default="ce",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=0)
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
    if spec == "freezing":
        raise ValueError("Use resolve_checkpoint_for_trajectory for freezing checkpoints.")
    return int(spec)


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


def choose_checkpoint(iterations, target: float, mode: str):
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


def resolve_checkpoint_for_trajectory(args, batch_size, learning_rate, repeat, saved_iterations, freezing_rows):
    spec = str(args.checkpoint_iteration).strip().lower()
    if spec != "freezing":
        checkpoint = resolve_checkpoint_iteration(args, learning_rate, saved_iterations)
        return checkpoint, math.nan, "", ""

    row = freezing_rows.get((batch_size, learning_rate, repeat))
    if row is None:
        return None, math.nan, "", "missing freezing summary row"
    tf = float(row["tf"])
    checkpoint = choose_checkpoint(saved_iterations, tf, args.freeze_rounding)
    return checkpoint, tf, row.get("confidence_flag", ""), ""


def checkpoint_path(
    root: Path,
    template: str,
    batch_size: int,
    learning_rate: float,
    repeat: int,
    iteration: int,
) -> Path:
    return root / checkpoint_subdir(template, batch_size, learning_rate, repeat) / f"iteration_{iteration}.pt"


def selected_trajectories(args):
    repeats = range(args.repeat_start, args.repeat_end + 1)
    return [
        (int(batch_size), float(learning_rate), int(repeat))
        for batch_size in args.batch_sizes
        for learning_rate in args.learning_rates
        for repeat in repeats
    ]


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


def first_eval_batch(loader, device: torch.device, max_batches: int = -1):
    chunks = []
    targets = []
    for batch_idx, (batch_data, batch_target) in enumerate(loader):
        if max_batches != -1 and batch_idx >= max_batches:
            break
        chunks.append(batch_data)
        targets.append(batch_target)
    if not chunks:
        return None, None
    return torch.cat(chunks, dim=0).to(device), torch.cat(targets, dim=0).to(device)


# ---------------------------------------------------------------------------
# Relative random perturbation definitions
# ---------------------------------------------------------------------------

def iter_perturbed_parameters(model, include_bias: bool):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not include_bias and param.ndim < 2:
            continue
        yield name, param


def sample_tensor_relative_direction(model, include_bias: bool, relative_floor: float):
    directions = []
    with torch.no_grad():
        for name, param in iter_perturbed_parameters(model, include_bias):
            direction = torch.randn_like(param)
            direction_norm = torch.norm(direction)
            param_norm = torch.norm(param)
            scale = torch.clamp(param_norm, min=relative_floor)
            if direction_norm.item() == 0.0:
                direction.zero_()
            else:
                direction.mul_(scale / direction_norm)
            directions.append((name, param, direction))
    return directions


def normalize_rows(direction: torch.Tensor, reference: torch.Tensor, relative_floor: float) -> torch.Tensor:
    """Normalize each output row/filter to the matching parameter row norm."""
    flat_direction = direction.reshape(direction.shape[0], -1)
    flat_reference = reference.reshape(reference.shape[0], -1)
    direction_norm = torch.norm(flat_direction, dim=1, keepdim=True).clamp(min=relative_floor)
    reference_norm = torch.norm(flat_reference, dim=1, keepdim=True).clamp(min=relative_floor)
    return (flat_direction * (reference_norm / direction_norm)).reshape_as(direction)


def sample_neuron_relative_direction(model, include_bias: bool, relative_floor: float):
    """Sample a neuron-wise relative direction without changing the existing math."""
    directions = []
    with torch.no_grad():
        for name, param in iter_perturbed_parameters(model, include_bias):
            direction = torch.randn_like(param)
            if param.ndim >= 2:
                direction = normalize_rows(direction, param.detach(), relative_floor)
            else:
                direction_norm = torch.norm(direction).clamp(min=relative_floor)
                param_norm = torch.norm(param.detach()).clamp(min=relative_floor)
                direction.mul_(param_norm / direction_norm)
            directions.append((name, param, direction))
    return directions


def add_direction(directions, alpha: float):
    with torch.no_grad():
        for _, param, direction in directions:
            # Avoid the newer add_(tensor, alpha=...) signature for PyTorch 1.1.
            param.add_(direction * alpha)


def random_relative_sharpness_for_model(model, train_loader, criterion, device: torch.device, args):
    base_loss, base_acc = evaluate_loss_accuracy(
        model,
        train_loader,
        criterion,
        device,
        max_batches=args.max_train_batches,
    )
    deltas = []
    plus_losses = []
    minus_losses = []
    for direction_idx in range(args.num_directions):
        torch.manual_seed(args.seed + 1009 * direction_idx)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + 1009 * direction_idx)

        if args.definition == "neuron_wise":
            directions = sample_neuron_relative_direction(model, args.include_bias, args.relative_floor)
        else:
            directions = sample_tensor_relative_direction(model, args.include_bias, args.relative_floor)
        add_direction(directions, args.rho)
        plus_loss, _ = evaluate_loss_accuracy(
            model,
            train_loader,
            criterion,
            device,
            max_batches=args.max_train_batches,
        )

        if args.symmetric:
            add_direction(directions, -2.0 * args.rho)
            minus_loss, _ = evaluate_loss_accuracy(
                model,
                train_loader,
                criterion,
                device,
                max_batches=args.max_train_batches,
            )
            add_direction(directions, args.rho)
            delta = 0.5 * (plus_loss + minus_loss) - base_loss
            minus_losses.append(minus_loss)
        else:
            add_direction(directions, -args.rho)
            delta = plus_loss - base_loss
            minus_losses.append(math.nan)

        plus_losses.append(plus_loss)
        deltas.append(delta)

    return base_loss, base_acc, np.asarray(deltas), np.asarray(plus_losses), np.asarray(minus_losses)


def relative_sharpness_for_model(model, train_loader, criterion, device: torch.device, args):
    if args.definition == "hessian_topk":
        return hessian_topk_sharpness_for_model(model, train_loader, criterion, device, args)
    return random_relative_sharpness_for_model(model, train_loader, criterion, device, args)


# ---------------------------------------------------------------------------
# Hessian top-k definition
# ---------------------------------------------------------------------------

def fcn_forward_with_replacement(model, batch_data, target_layer_name: str, replacement: torch.Tensor):
    """Forward pass for FCN with one parameter tensor replaced.

    This avoids torch.func/torch.nn.utils.stateless.functional_call, which is
    unavailable in the cluster's older PyTorch environment.
    """
    if not hasattr(model, "net") or len(model.net) < 6:
        raise TypeError("Manual Hessian path currently supports the FCN model only")

    def param(name: str, module, attr: str):
        return replacement if name == target_layer_name else getattr(module, attr)

    x = torch.flatten(batch_data, start_dim=1)
    layer1 = model.net[1]
    layer2 = model.net[3]
    layer3 = model.net[5]
    x = F.linear(x, param("net.1.weight", layer1, "weight"), param("net.1.bias", layer1, "bias"))
    x = F.relu(x)
    x = F.linear(x, param("net.3.weight", layer2, "weight"), param("net.3.bias", layer2, "bias"))
    x = F.relu(x)
    x = F.linear(x, param("net.5.weight", layer3, "weight"), param("net.5.bias", layer3, "bias"))
    return x


def compute_loss_stateless(model, batch_data, batch_target, criterion, target_layer_name, replacement):
    output = fcn_forward_with_replacement(model, batch_data, target_layer_name, replacement)
    return criterion(output, batch_target)


def hessian_target_layer_name(model, layer_index: int) -> str:
    names = list(model.state_dict().keys())
    if layer_index < 0 or layer_index >= len(names):
        raise IndexError(f"hessian_layer_index={layer_index} outside model.state_dict() with {len(names)} tensors")
    name = names[layer_index]
    if name not in dict(model.named_parameters()):
        raise ValueError(f"Selected Hessian tensor {name!r} is not a trainable parameter")
    return name


def eigvalsh_descending(matrix: torch.Tensor) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.t())

    try:
        return torch.linalg.eigvalsh(matrix).numpy()[::-1]
    except Exception:
        pass

    if hasattr(torch, "symeig"):
        try:
            eigvals, _ = torch.symeig(matrix, eigenvectors=False)
            return eigvals.numpy()[::-1]
        except Exception:
            pass

    matrix_np = matrix.double().numpy()
    matrix_np = 0.5 * (matrix_np + matrix_np.T)
    try:
        return np.linalg.eigvalsh(matrix_np)[::-1]
    except Exception:
        return np.asarray([math.nan])


def hessian_topk_sharpness_for_model(model, train_loader, criterion, device: torch.device, args):
    base_loss, base_acc = evaluate_loss_accuracy(
        model,
        train_loader,
        criterion,
        device,
        max_batches=args.max_train_batches,
    )
    batch_data, batch_target = first_eval_batch(train_loader, device, args.max_train_batches)
    if batch_data is None:
        return base_loss, base_acc, np.asarray([math.nan]), np.asarray([math.nan]), np.asarray([math.nan])

    target_layer_name = hessian_target_layer_name(model, args.hessian_layer_index)
    params = dict(model.named_parameters())
    target_weight = params[target_layer_name]

    def loss_wrt_target_layer(w_target):
        return compute_loss_stateless(model, batch_data, batch_target, criterion, target_layer_name, w_target)

    if HESSIAN_IS_CURRIED:
        H = hessian(loss_wrt_target_layer)(target_weight)
    else:
        H = hessian(loss_wrt_target_layer, target_weight)
    H = H.reshape(target_weight.numel(), target_weight.numel())
    H_cpu = H.detach().cpu()
    if torch.isnan(H_cpu).any() or torch.isinf(H_cpu).any():
        return base_loss, base_acc, np.asarray([math.nan]), np.asarray([math.nan]), np.asarray([math.nan])

    eigvals = eigvalsh_descending(H_cpu)
    k = min(int(args.hessian_topk), eigvals.size)
    topk = np.real_if_close(eigvals[:k], tol=1000).real
    if topk.size == 0 or np.any(topk <= 0):
        sharpness = math.nan
    else:
        # Legacy flatness was prod(lambda_1...lambda_k)^(-1/k);
        # therefore the corresponding sharpness is the geometric mean.
        sharpness = float(np.exp(np.mean(np.log(topk))))
    return base_loss, base_acc, np.asarray([sharpness]), topk, np.asarray([math.nan])


def load_summary(path: Path):
    if not path.exists():
        return {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return {
            (int(row["batch_size"]), float(row["learning_rate"]), int(row["repeat"])): row
            for row in reader
        }


def write_summary(path: Path, rows):
    fieldnames = [
        "batch_size",
        "learning_rate",
        "repeat",
        "checkpoint_iteration",
        "tf",
        "eta_checkpoint",
        "freeze_rounding",
        "confidence_flag",
        "definition",
        "rho",
        "num_directions",
        "hessian_topk",
        "hessian_layer_index",
        "symmetric",
        "include_bias",
        "max_train_batches",
        "loss_type",
        "base_train_loss",
        "base_train_accuracy",
        "base_test_loss",
        "base_test_accuracy",
        "sharpness_mean_delta",
        "sharpness_median_delta",
        "sharpness_std_delta",
        "sharpness_mean_positive_delta",
        "flatness_neg_log10",
        "relative_flatness_inverse",
        "hessian_flatness",
        "hessian_lambda_max",
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
            row = {name: rows[key].get(name, "") for name in fieldnames}
            writer.writerow(row)


def make_error_row(batch_size, learning_rate, repeat, args, message):
    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "repeat": repeat,
        "checkpoint_iteration": "",
        "tf": "",
        "eta_checkpoint": "",
        "freeze_rounding": args.freeze_rounding,
        "confidence_flag": "",
        "definition": args.definition,
        "rho": args.rho,
        "num_directions": args.num_directions,
        "hessian_topk": args.hessian_topk,
        "hessian_layer_index": args.hessian_layer_index,
        "symmetric": args.symmetric,
        "include_bias": args.include_bias,
        "max_train_batches": args.max_train_batches,
        "loss_type": args.loss_type,
        "status": "missing",
        "message": message,
    }


def scan_checkpoint(
    batch_size,
    learning_rate,
    repeat,
    model,
    train_loader,
    test_loader,
    criterion,
    device,
    args,
    freezing_rows,
):
    wall_start = time.perf_counter()
    saved_iterations = list_checkpoint_iterations(
        args.checkpoint_dir,
        args.checkpoint_subdir_template,
        batch_size,
        learning_rate,
        repeat,
    )
    checkpoint_iteration, tf, confidence_flag, checkpoint_error = resolve_checkpoint_for_trajectory(
        args,
        batch_size,
        learning_rate,
        repeat,
        saved_iterations,
        freezing_rows,
    )
    if checkpoint_iteration is None:
        directory = trajectory_dir(
            args.checkpoint_dir,
            args.checkpoint_subdir_template,
            batch_size,
            learning_rate,
            repeat,
        )
        message = checkpoint_error or f"No checkpoints found in {directory}"
        return make_error_row(batch_size, learning_rate, repeat, args, message)

    checkpoint_file = checkpoint_path(
        args.checkpoint_dir,
        args.checkpoint_subdir_template,
        batch_size,
        learning_rate,
        repeat,
        checkpoint_iteration,
    )
    if not checkpoint_file.exists():
        return make_error_row(batch_size, learning_rate, repeat, args, f"Missing {checkpoint_file}")

    state_dict = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

    base_train_loss, base_train_acc, deltas, plus_losses, _ = relative_sharpness_for_model(
        model,
        train_loader,
        criterion,
        device,
        args,
    )
    base_test_loss, base_test_acc = evaluate_loss_accuracy(model, test_loader, criterion, device)
    positive_deltas = np.maximum(deltas, 0.0)
    mean_positive = float(np.mean(positive_deltas))
    flatness_neg_log10 = -math.log10(max(mean_positive, 1e-12)) if np.isfinite(mean_positive) else math.nan
    flatness_inverse = float(1.0 / (1e-12 + mean_positive))
    hessian_lambda_max = (
        float(np.max(plus_losses))
        if args.definition == "hessian_topk" and len(plus_losses) and np.isfinite(plus_losses).any()
        else ""
    )
    hessian_flatness = flatness_inverse if args.definition == "hessian_topk" else ""

    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "repeat": repeat,
        "checkpoint_iteration": checkpoint_iteration,
        "tf": float(tf) if np.isfinite(tf) else "",
        "eta_checkpoint": float(learning_rate * checkpoint_iteration),
        "freeze_rounding": args.freeze_rounding if str(args.checkpoint_iteration).strip().lower() == "freezing" else "",
        "confidence_flag": confidence_flag,
        "definition": args.definition,
        "rho": args.rho,
        "num_directions": args.num_directions,
        "hessian_topk": args.hessian_topk,
        "hessian_layer_index": args.hessian_layer_index,
        "symmetric": args.symmetric,
        "include_bias": args.include_bias,
        "max_train_batches": args.max_train_batches,
        "loss_type": args.loss_type,
        "base_train_loss": float(base_train_loss),
        "base_train_accuracy": float(base_train_acc),
        "base_test_loss": float(base_test_loss),
        "base_test_accuracy": float(base_test_acc),
        "sharpness_mean_delta": float(np.mean(deltas)),
        "sharpness_median_delta": float(np.median(deltas)),
        "sharpness_std_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
        "sharpness_mean_positive_delta": mean_positive,
        "flatness_neg_log10": flatness_neg_log10,
        "relative_flatness_inverse": flatness_inverse,
        "hessian_flatness": hessian_flatness,
        "hessian_lambda_max": hessian_lambda_max,
        "min_delta": float(np.min(deltas)),
        "max_delta": float(np.max(deltas)),
        "wall_time_sec": float(time.perf_counter() - wall_start),
        "status": "ok",
        "message": "",
    }


def main():
    run_start = time.perf_counter()
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.freezing_summary_root = args.freezing_summary_root.resolve()

    trajectories = selected_trajectories(args)
    freezing_rows = (
        load_freezing_rows(args.freezing_summary_root)
        if str(args.checkpoint_iteration).strip().lower() == "freezing"
        else {}
    )
    if args.dry_run:
        print(f"Selected {len(trajectories)} trajectories")
        if str(args.checkpoint_iteration).strip().lower() == "freezing":
            print(f"Loaded {len(freezing_rows)} freezing rows from {args.freezing_summary_root}")
        for batch_size, learning_rate, repeat in trajectories[:30]:
            saved_iterations = list_checkpoint_iterations(
                args.checkpoint_dir,
                args.checkpoint_subdir_template,
                batch_size,
                learning_rate,
                repeat,
            )
            checkpoint_iteration, tf, confidence_flag, error = resolve_checkpoint_for_trajectory(
                args,
                batch_size,
                learning_rate,
                repeat,
                saved_iterations,
                freezing_rows,
            )
            print(
                f"bs={batch_size:g} lr={learning_rate:g} repeat={repeat}: "
                f"checkpoint={checkpoint_iteration}, tf={tf}, saved={len(saved_iterations)}, "
                f"flag={confidence_flag}, error={error}"
            )
        if len(trajectories) > 30:
            print(f"... {len(trajectories) - 30} more")
        return

    device = resolve_device(args.device)
    print(f"Using device: {device}")
    print(f"Dataset root: {args.data_dir}")
    print(f"Checkpoint root: {args.checkpoint_dir}")
    print(f"Output root: {args.output_dir}")
    if str(args.checkpoint_iteration).strip().lower() == "freezing":
        print(f"Freezing summary root: {args.freezing_summary_root}; rows={len(freezing_rows)}")
    print(
        "Sharpness settings: "
        f"definition={args.definition}, rho={args.rho}, directions={args.num_directions}, "
        f"symmetric={args.symmetric}, include_bias={args.include_bias}, "
        f"max_train_batches={args.max_train_batches}, "
        f"hessian_topk={args.hessian_topk}, hessian_layer_index={args.hessian_layer_index}"
    )

    train_loader, test_loader, train_count, test_count = make_loaders(args)
    print(f"Train samples: {train_count}; test samples: {test_count}")

    model = FCN(input_dim=input_dim_for_dataset(args.dataset_name), hidden=args.hidden_num).to(device)
    criterion = ClassificationLoss(args.loss_type)

    summary_path = args.output_dir / "relative_sharpness_summary.csv"
    summary_rows = load_summary(summary_path)

    for idx, (batch_size, learning_rate, repeat) in enumerate(trajectories, start=1):
        key = (batch_size, learning_rate, repeat)
        if key in summary_rows and not args.overwrite:
            print(f"[{idx}/{len(trajectories)}] bs={batch_size:g}, lr={learning_rate:g}, repeat={repeat}: cached")
            continue

        print(f"[{idx}/{len(trajectories)}] bs={batch_size:g}, lr={learning_rate:g}, repeat={repeat}")
        row = scan_checkpoint(
            batch_size,
            learning_rate,
            repeat,
            model,
            train_loader,
            test_loader,
            criterion,
            device,
            args,
            freezing_rows,
        )
        summary_rows[key] = row
        write_summary(summary_path, summary_rows)

        if row["status"] == "ok":
            print(
                "  "
                f"iter={row['checkpoint_iteration']}, "
                f"sharp={float(row['sharpness_mean_positive_delta']):.6g}, "
                f"flat_inv={float(row['relative_flatness_inverse']):.6g}, "
                f"test_acc={float(row['base_test_accuracy']):.4f}, "
                f"wall={float(row['wall_time_sec']):.1f}s"
            )
        else:
            print(f"  status={row['status']}: {row['message']}")

    print(f"Summary written to {summary_path}")
    print(f"Total wall time: {time.perf_counter() - run_start:.1f}s")


if __name__ == "__main__":
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
