
"""Three-panel Fig. 2 draft from current freezing and final-solution data.

Panels:
A: median matched-lr freezing coordinate eta t_f from jaccard09.
C: median final flatness over converged repeats.
D: best final test accuracy over converged repeats (upper envelope).

B=1000 and lr=0.001 are omitted by design.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
FREEZING_ROOT = REPO_ROOT / "freezing_time_training_lr_r1-5_ce" / "jaccard09"
FINAL_METRICS = REPO_ROOT / "relative_sharpness_all_ce" / "random_filter_rho0.05_k20" / "relative_sharpness_summary.csv"
OUT_DIR = REPO_ROOT / "PRX_revison" / "Figures_revision" / "Fig2_three_panel_current"

BS_LIST = np.array([10, 20, 50, 100, 200, 500])
LR_LIST = np.array([0.1, 0.05, 0.02, 0.01, 0.005, 0.002])
TRAIN_ACC_THRESHOLD = 0.999
SHARPNESS_FLOOR = 1e-12


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.85,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.4,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_freezing() -> dict[tuple[int, float], dict[str, float]]:
    by_condition: dict[tuple[int, float], list[dict[str, float]]] = {}
    for path in sorted(FREEZING_ROOT.glob("bs*_lr*/freezing_time_summary.csv")):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "ok":
                    continue
                bs = int(float(row["batch_size"]))
                lr = float(row["learning_rate"])
                if bs not in set(BS_LIST) or not np.any(np.isclose(LR_LIST, lr)):
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
            if bs not in set(BS_LIST) or not np.any(np.isclose(LR_LIST, lr)):
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
        out[key] = {
            "median_flatness": float(np.median(flat)),
            "mean_flatness": float(np.mean(flat)),
            "max_flatness": float(np.max(flat)),
            "max_test_acc": float(np.max(acc)),
            "top20_test_acc": float(np.mean(np.sort(acc)[-max(1, int(np.ceil(0.2 * len(acc)))):])),
            "median_test_acc": float(np.median(acc)),
            "n_converged": float(len(rows)),
        }
    return out


def matrix(data: dict[tuple[int, float], dict[str, float]], field: str) -> np.ndarray:
    arr = np.full((len(BS_LIST), len(LR_LIST)), np.nan)
    for i, bs in enumerate(BS_LIST):
        for j, lr in enumerate(LR_LIST):
            row = data.get((int(bs), float(lr)))
            if row is not None:
                arr[i, j] = row.get(field, np.nan)
    return arr


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.08, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def draw_heatmap(ax: plt.Axes, fig: plt.Figure, data: np.ndarray, title: str, cbar_label: str, *,
                 text_data: np.ndarray | None = None, cmap: str = "viridis", vmin=None, vmax=None,
                 fmt: str = ".2g", annotate: str = "value") -> None:
    valid = np.isfinite(data)
    im = ax.imshow(data, origin="lower", aspect="equal", cmap=cmap,
                   vmin=np.nanmin(data[valid]) if vmin is None and np.any(valid) else vmin,
                   vmax=np.nanmax(data[valid]) if vmax is None and np.any(valid) else vmax)
    ax.set_xticks(np.arange(len(LR_LIST)))
    ax.set_xticklabels([f"{x:g}" for x in LR_LIST], rotation=42, ha="right")
    ax.set_yticks(np.arange(len(BS_LIST)))
    ax.set_yticklabels([f"{x:g}" for x in BS_LIST])
    ax.set_xlabel(r"Learning rate $\eta$")
    ax.set_ylabel(r"Batch size $B$")
    ax.set_title(title, pad=7)
    ax.set_xticks(np.arange(-0.5, len(LR_LIST), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BS_LIST), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.55, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    vmax_eff = np.nanmax(data[valid]) if np.any(valid) else 1.0
    vmin_eff = np.nanmin(data[valid]) if np.any(valid) else 0.0
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if not np.isfinite(value):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="0.72", zorder=3))
                ax.text(j, i, "--", ha="center", va="center", fontsize=6.8, color="0.2", zorder=4)
                continue
            text_value = value if text_data is None else text_data[i, j]
            normed = (value - vmin_eff) / max(vmax_eff - vmin_eff, 1e-12)
            color = "white" if normed > 0.52 else "black"
            if annotate == "censored" and np.isclose(value, 100.0):
                text = r"$\geq100$"
            else:
                text = format(text_value, fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=6.7, color=color, zorder=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cb.set_label(cbar_label, fontsize=7.4)
    cb.ax.tick_params(labelsize=6.8)


def write_summary(freezing: dict, final: dict, path: Path) -> None:
    fields = [
        "batch_size", "learning_rate", "noise", "median_eta_tf", "mean_eta_tf", "n_freezing", "n_censored",
        "median_final_flatness", "mean_final_flatness", "max_final_flatness", "max_test_acc", "top20_test_acc",
        "median_test_acc", "n_converged_final",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for bs in BS_LIST:
            for lr in LR_LIST:
                fr = freezing.get((int(bs), float(lr)), {})
                fi = final.get((int(bs), float(lr)), {})
                writer.writerow({
                    "batch_size": int(bs),
                    "learning_rate": float(lr),
                    "noise": float(lr / bs),
                    "median_eta_tf": fr.get("median_eta_tf", ""),
                    "mean_eta_tf": fr.get("mean_eta_tf", ""),
                    "n_freezing": fr.get("n", ""),
                    "n_censored": fr.get("n_censored", ""),
                    "median_final_flatness": fi.get("median_flatness", ""),
                    "mean_final_flatness": fi.get("mean_flatness", ""),
                    "max_final_flatness": fi.get("max_flatness", ""),
                    "max_test_acc": fi.get("max_test_acc", ""),
                    "top20_test_acc": fi.get("top20_test_acc", ""),
                    "median_test_acc": fi.get("median_test_acc", ""),
                    "n_converged_final": fi.get("n_converged", ""),
                })


def print_trends(freezing: dict, final: dict) -> None:
    rows = []
    for bs in BS_LIST:
        for lr in LR_LIST:
            key = (int(bs), float(lr))
            row = {"noise": float(lr / bs)}
            row.update(freezing.get(key, {}))
            row.update({f"final_{k}": v for k, v in final.get(key, {}).items()})
            rows.append(row)
    for field in ["median_eta_tf", "final_median_flatness", "final_mean_flatness", "final_max_test_acc", "final_top20_test_acc"]:
        x, y = [], []
        for row in rows:
            if field in row and np.isfinite(row[field]):
                x.append(np.log10(row["noise"]))
                y.append(row[field])
        if len(x) >= 3:
            fit = linregress(np.array(x), np.array(y))
            sp = spearmanr(x, y)
            print(f"{field}: n={len(x)}, R2={fit.rvalue**2:.3f}, p={fit.pvalue:.2g}, rho={sp.statistic:.3f}, rho_p={sp.pvalue:.2g}")


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    freezing = load_freezing()
    final = load_final_metrics()
    write_summary(freezing, final, OUT_DIR / "Fig2_three_panel_current_summary.csv")
    print_trends(freezing, final)

    eta_tf = matrix(freezing, "median_eta_tf")
    flatness = matrix(final, "median_flatness")
    max_acc = matrix(final, "max_test_acc")

    fig = plt.figure(figsize=(7.6, 2.85))
    gs = fig.add_gridspec(1, 3, left=0.06, right=0.985, bottom=0.20, top=0.86, wspace=0.42)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    draw_heatmap(
        axes[0], fig, eta_tf,
        r"Freezing coordinate", r"median $\eta t_f$",
        cmap="viridis", vmin=0, vmax=100, annotate="censored",
    )
    add_panel_label(axes[0], "A")

    draw_heatmap(
        axes[1], fig, flatness,
        r"Final flatness", r"median $\log_{10}F$",
        cmap="magma", fmt=".2f",
    )
    add_panel_label(axes[1], "C")

    draw_heatmap(
        axes[2], fig, max_acc,
        r"Best final accuracy", r"max test accuracy",
        cmap="cividis", vmin=0.90, vmax=0.93, fmt=".3f",
    )
    add_panel_label(axes[2], "D")

    fig.text(
        0.5, 0.045,
        r"B=1000 and $\eta=0.001$ omitted; C/D use final-converged repeats only; D is an upper-envelope statistic.",
        ha="center", va="bottom", fontsize=7.2,
    )
    for ext in ["png", "pdf"]:
        out = OUT_DIR / f"Fig2_three_panel_current.{ext}"
        fig.savefig(out, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
