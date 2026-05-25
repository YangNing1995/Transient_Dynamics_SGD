"""Four-panel Fig. 2 draft from current freezing and solution data.

Panels:
A: matched-lr freezing coordinate, median eta t_f from jaccard09.
B: final-solution spread, log10 trace covariance of converged final weights.
C: median final flatness over converged repeats.
D: best final test accuracy over converged repeats.

The heatmaps use x = batch size and y = learning rate. Batch size decreases
from left to right and learning rate increases from bottom to top, so the
largest SGD-noise conditions appear in the upper-right corner.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.stats import linregress, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT.parent / "Data" / "Train_with_different_hyperparas"
FREEZING_ROOT = REPO_ROOT / "freezing_time_training_lr_r1-5_ce" / "jaccard09"
FINAL_METRICS = (
    REPO_ROOT
    / "relative_sharpness_all_ce"
    / "random_filter_rho0.05_k20"
    / "relative_sharpness_summary.csv"
)
OUT_DIR = REPO_ROOT / "PRX_revison" / "Figures_revision" / "Fig2_four_panel_current"

BS_LIST = np.array([10, 20, 50, 100, 200, 500])
LR_LIST = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002])
BS_DISPLAY = np.array([500, 200, 100, 50, 20, 10])
LR_DISPLAY = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
TRAIN_ACC_THRESHOLD = 0.999
SHARPNESS_FLOOR = 1e-12
N_REPEATS_FOR_SPREAD = 20


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.85,
        "axes.labelsize": 8.2,
        "axes.titlesize": 8.6,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def condition_allowed(bs: int, lr: float) -> bool:
    return bs in set(BS_LIST) and np.any(np.isclose(LR_LIST, lr))


def load_freezing() -> dict[tuple[int, float], dict[str, float]]:
    by_condition: dict[tuple[int, float], list[dict[str, float]]] = {}
    for path in sorted(FREEZING_ROOT.glob("bs*_lr*/freezing_time_summary.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                bs = int(float(row["batch_size"]))
                lr = float(row["learning_rate"])
                if not condition_allowed(bs, lr):
                    continue
                tf = float(row["tf"])
                ref = float(row["reference_iteration"])
                by_condition.setdefault((bs, lr), []).append({
                    "eta_tf": float(row["eta_tf"]),
                    "censored": float(abs(tf - ref) < 1e-9),
                })

    out: dict[tuple[int, float], dict[str, float]] = {}
    for key, rows in by_condition.items():
        eta_tf = np.array([r["eta_tf"] for r in rows], dtype=float)
        censored = np.array([r["censored"] for r in rows], dtype=float)
        out[key] = {
            "median_eta_tf": float(np.median(eta_tf)),
            "mean_eta_tf": float(np.mean(eta_tf)),
            "n": float(len(rows)),
            "n_censored": float(np.sum(censored)),
        }
    return out


def load_final_metrics() -> dict[tuple[int, float], dict[str, float]]:
    by_condition: dict[tuple[int, float], list[dict[str, float]]] = {}
    with FINAL_METRICS.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            bs = int(float(row["batch_size"]))
            lr = float(row["learning_rate"])
            if not condition_allowed(bs, lr):
                continue
            train_acc = float(row["base_train_accuracy"])
            if train_acc < TRAIN_ACC_THRESHOLD:
                continue
            inv = float(row["relative_flatness_inverse"])
            flatness = np.log10(max(inv, SHARPNESS_FLOOR))
            by_condition.setdefault((bs, lr), []).append({
                "test_acc": float(row["base_test_accuracy"]),
                "test_loss": float(row["base_test_loss"]),
                "flatness": float(flatness),
            })

    out: dict[tuple[int, float], dict[str, float]] = {}
    for key, rows in by_condition.items():
        acc = np.array([r["test_acc"] for r in rows], dtype=float)
        flat = np.array([r["flatness"] for r in rows], dtype=float)
        top_n = max(1, int(np.ceil(0.2 * len(acc))))
        out[key] = {
            "median_flatness": float(np.median(flat)),
            "mean_flatness": float(np.mean(flat)),
            "max_flatness": float(np.max(flat)),
            "max_test_acc": float(np.max(acc)),
            "top20_test_acc": float(np.mean(np.sort(acc)[-top_n:])),
            "median_test_acc": float(np.median(acc)),
            "n_converged": float(len(rows)),
        }
    return out


def load_solution_spread() -> dict[tuple[int, float], dict[str, float]]:
    cache_path = OUT_DIR / "solution_spread_summary.csv"
    if cache_path.exists():
        out: dict[tuple[int, float], dict[str, float]] = {}
        with cache_path.open(newline="") as f:
            for row in csv.DictReader(f):
                bs = int(float(row["batch_size"]))
                lr = float(row["learning_rate"])
                out[(bs, lr)] = {
                    "log10_weight_var": float(row["log10_weight_var"]) if row["log10_weight_var"] else np.nan,
                    "rms_radius": float(row["rms_radius"]) if row["rms_radius"] else np.nan,
                    "n_weight_converged": float(row["n_weight_converged"]),
                }
        return out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[tuple[int, float], dict[str, float]] = {}
    rows_for_cache = []
    for bs in BS_LIST:
        for lr in LR_LIST:
            weights = []
            for repeat in range(1, N_REPEATS_FOR_SPREAD + 1):
                path = DATA_ROOT / f"bs{int(bs)}_lr{lr:g}" / f"save_metrics_repeat{repeat}.mat"
                if not path.exists():
                    continue
                mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
                train_acc = np.asarray(mat["train_accuracy"]).reshape(-1)[-1]
                if float(train_acc) < TRAIN_ACC_THRESHOLD:
                    continue
                weight = np.asarray(mat["weight_all"])[-1].astype(np.float64, copy=False)
                weights.append(weight)

            if len(weights) >= 2:
                w = np.vstack(weights)
                centered = w - np.mean(w, axis=0, keepdims=True)
                second_moment = float(np.mean(np.sum(centered * centered, axis=1)))
                rms_radius = float(np.sqrt(max(second_moment, 0.0)))
                log_var = float(np.log10(max(second_moment, 1e-30)))
            else:
                second_moment = np.nan
                rms_radius = np.nan
                log_var = np.nan

            value = {
                "log10_weight_var": log_var,
                "rms_radius": rms_radius,
                "n_weight_converged": float(len(weights)),
            }
            out[(int(bs), float(lr))] = value
            rows_for_cache.append({
                "batch_size": int(bs),
                "learning_rate": float(lr),
                "n_weight_converged": int(len(weights)),
                "weight_second_moment": second_moment,
                "rms_radius": rms_radius,
                "log10_weight_var": log_var,
            })

    with cache_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_for_cache[0].keys()))
        writer.writeheader()
        writer.writerows(rows_for_cache)
    return out


def matrix(data: dict[tuple[int, float], dict[str, float]], field: str) -> np.ndarray:
    arr = np.full((len(LR_DISPLAY), len(BS_DISPLAY)), np.nan)
    for i, lr in enumerate(LR_DISPLAY):
        for j, bs in enumerate(BS_DISPLAY):
            row = data.get((int(bs), float(lr)))
            if row is not None:
                arr[i, j] = row.get(field, np.nan)
    return arr


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.09, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def draw_heatmap(
    ax: plt.Axes,
    fig: plt.Figure,
    data: np.ndarray,
    title: str,
    cbar_label: str,
    *,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    fmt: str = ".2g",
    annotate: str = "value",
) -> None:
    valid = np.isfinite(data)
    if np.any(valid):
        inferred_vmin = np.nanmin(data[valid]) if vmin is None else vmin
        inferred_vmax = np.nanmax(data[valid]) if vmax is None else vmax
    else:
        inferred_vmin = 0 if vmin is None else vmin
        inferred_vmax = 1 if vmax is None else vmax

    im = ax.imshow(
        data,
        origin="lower",
        aspect="equal",
        cmap=cmap,
        vmin=inferred_vmin,
        vmax=inferred_vmax,
    )
    ax.set_xticks(np.arange(len(BS_DISPLAY)))
    ax.set_xticklabels([f"{x:g}" for x in BS_DISPLAY])
    ax.set_yticks(np.arange(len(LR_DISPLAY)))
    ax.set_yticklabels([f"{x:g}" for x in LR_DISPLAY])
    ax.set_xlabel(r"Batch size $B$ (decreases $\rightarrow$)")
    ax.set_ylabel(r"Learning rate $\eta$")
    ax.set_title(title, pad=7)
    ax.set_xticks(np.arange(-0.5, len(BS_DISPLAY), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(LR_DISPLAY), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.55, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if not np.isfinite(value):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="0.72", zorder=3))
                ax.text(j, i, "--", ha="center", va="center", fontsize=6.8, color="0.2", zorder=4)
                continue
            normed = (value - inferred_vmin) / max(inferred_vmax - inferred_vmin, 1e-12)
            color = "white" if normed > 0.52 else "black"
            if annotate == "censored" and np.isclose(value, 100.0):
                text = r"$\geq100$"
            else:
                text = format(value, fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=6.7, color=color, zorder=4)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cb.set_label(cbar_label, fontsize=7.4)
    cb.ax.tick_params(labelsize=6.8)


def write_summary(freezing: dict, final: dict, spread: dict, path: Path) -> None:
    fields = [
        "batch_size",
        "learning_rate",
        "noise",
        "median_eta_tf",
        "mean_eta_tf",
        "n_freezing",
        "n_censored",
        "log10_weight_var",
        "rms_radius",
        "n_weight_converged",
        "median_final_flatness",
        "mean_final_flatness",
        "max_final_flatness",
        "max_test_acc",
        "top20_test_acc",
        "median_test_acc",
        "n_converged_final",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for bs in BS_LIST:
            for lr in LR_LIST:
                key = (int(bs), float(lr))
                fr = freezing.get(key, {})
                fi = final.get(key, {})
                sp = spread.get(key, {})
                writer.writerow({
                    "batch_size": int(bs),
                    "learning_rate": float(lr),
                    "noise": float(lr / bs),
                    "median_eta_tf": fr.get("median_eta_tf", ""),
                    "mean_eta_tf": fr.get("mean_eta_tf", ""),
                    "n_freezing": fr.get("n", ""),
                    "n_censored": fr.get("n_censored", ""),
                    "log10_weight_var": sp.get("log10_weight_var", ""),
                    "rms_radius": sp.get("rms_radius", ""),
                    "n_weight_converged": sp.get("n_weight_converged", ""),
                    "median_final_flatness": fi.get("median_flatness", ""),
                    "mean_final_flatness": fi.get("mean_flatness", ""),
                    "max_final_flatness": fi.get("max_flatness", ""),
                    "max_test_acc": fi.get("max_test_acc", ""),
                    "top20_test_acc": fi.get("top20_test_acc", ""),
                    "median_test_acc": fi.get("median_test_acc", ""),
                    "n_converged_final": fi.get("n_converged", ""),
                })


def print_trends(freezing: dict, final: dict, spread: dict) -> None:
    rows = []
    for bs in BS_LIST:
        for lr in LR_LIST:
            key = (int(bs), float(lr))
            row = {"noise": float(lr / bs)}
            row.update(freezing.get(key, {}))
            row.update({f"final_{k}": v for k, v in final.get(key, {}).items()})
            row.update({f"spread_{k}": v for k, v in spread.get(key, {}).items()})
            rows.append(row)

    fields = [
        "median_eta_tf",
        "spread_log10_weight_var",
        "final_median_flatness",
        "final_mean_flatness",
        "final_max_test_acc",
        "final_top20_test_acc",
    ]
    for field in fields:
        x, y = [], []
        for row in rows:
            if field in row and np.isfinite(row[field]):
                x.append(np.log10(row["noise"]))
                y.append(row[field])
        if len(x) >= 3:
            fit = linregress(np.array(x), np.array(y))
            sp = spearmanr(x, y)
            print(
                f"{field}: n={len(x)}, R2={fit.rvalue**2:.3f}, "
                f"p={fit.pvalue:.2g}, rho={sp.statistic:.3f}, rho_p={sp.pvalue:.2g}"
            )


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    freezing = load_freezing()
    final = load_final_metrics()
    spread = load_solution_spread()
    write_summary(freezing, final, spread, OUT_DIR / "Fig2_four_panel_current_summary.csv")
    print_trends(freezing, final, spread)

    eta_tf = matrix(freezing, "median_eta_tf")
    weight_var = matrix(spread, "log10_weight_var")
    flatness = matrix(final, "median_flatness")
    max_acc = matrix(final, "max_test_acc")

    fig = plt.figure(figsize=(6.9, 6.3))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, bottom=0.105, top=0.92, wspace=0.36, hspace=0.48)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    draw_heatmap(
        axes[0],
        fig,
        eta_tf,
        r"Freezing coordinate",
        r"median $\eta t_f$",
        cmap="viridis",
        vmin=0,
        vmax=100,
        annotate="censored",
    )
    add_panel_label(axes[0], "A")

    draw_heatmap(
        axes[1],
        fig,
        weight_var,
        r"Final-solution spread",
        r"$\log_{10}\langle\|\theta_f-\bar{\theta}_f\|^2\rangle$",
        cmap="plasma",
        fmt=".2f",
    )
    add_panel_label(axes[1], "B")

    draw_heatmap(
        axes[2],
        fig,
        flatness,
        r"Final flatness",
        r"median $\log_{10}F$",
        cmap="magma",
        fmt=".2f",
    )
    add_panel_label(axes[2], "C")

    draw_heatmap(
        axes[3],
        fig,
        max_acc,
        r"Best final accuracy",
        r"max test accuracy",
        cmap="cividis",
        vmin=0.90,
        vmax=0.93,
        fmt=".3f",
    )
    add_panel_label(axes[3], "D")

    fig.text(
        0.5,
        0.035,
        (
            r"B=1000 and $\eta=0.001$ omitted; B/C/D use final-converged repeats only; "
            r"D is an upper-envelope statistic."
        ),
        ha="center",
        va="bottom",
        fontsize=7.2,
    )
    for ext in ["png", "pdf"]:
        out = OUT_DIR / f"Fig2_four_panel_current.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
