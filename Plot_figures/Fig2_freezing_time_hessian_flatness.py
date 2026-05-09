"""Hessian-based companion version of Fig. 2.

This reproduces the current final-converged-stable Fig. 2 logic, but Panel C
uses the original top-Hessian-eigenvalue flatness measured at theta(t_f).
It is intended as a robustness/comparison figure rather than the main figure.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import linregress


BS_LIST = np.array([1000, 500, 200, 100, 50, 20, 10])
LR_LIST = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
COMPARE_LR_MAX = 0.1
SHARPNESS_FLOOR = 1e-12

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZING_FLATNESS_CSV = (
    REPO_ROOT
    / "freezing_hessian_flatness_all_ce"
    / "top10_layer2_ceil"
    / "relative_sharpness_summary.csv"
)
FINAL_METRICS_CSV = (
    REPO_ROOT
    / "relative_sharpness_all_ce"
    / "random_filter_rho0.05_k20"
    / "relative_sharpness_summary.csv"
)
OUT_DIR = REPO_ROOT / "PRX_revison" / "Figures_revision" / "Fig2_hessian_top10_layer2"


@dataclass
class RunRecord:
    batch_size: int
    learning_rate: float
    repeat: int
    effective_noise: float
    tf: float
    eta_tf: float
    relative_flatness: float
    relative_sharpness: float
    final_test_acc: float
    final_test_loss: float
    final_train_converged: bool
    continuation_converged: bool

    @property
    def stable(self) -> bool:
        return (
            self.final_train_converged
            and self.continuation_converged
            and np.isfinite(self.tf)
            and np.isfinite(self.eta_tf)
            and np.isfinite(self.relative_flatness)
            and np.isfinite(self.final_test_acc)
        )


@dataclass
class ConditionSummary:
    batch_size: int
    learning_rate: float
    effective_noise: float
    n_total: int
    n_stable: int
    stable_fraction: float
    eta_tf_mean: float
    eta_tf_median: float
    eta_tf_sem: float
    relative_flatness_mean: float
    relative_flatness_sem: float
    relative_sharpness_mean: float
    relative_sharpness_sem: float
    final_test_acc_mean: float
    final_test_acc_sem: float
    final_test_loss_mean: float
    final_test_loss_sem: float


def sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def nanmean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def nanmedian(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else np.nan


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.0,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.16,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def load_freezing_flatness_rows() -> dict[tuple[int, float, int], dict[str, str]]:
    rows: dict[tuple[int, float, int], dict[str, str]] = {}
    with FREEZING_FLATNESS_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            key = (
                int(float(row["batch_size"])),
                float(row["learning_rate"]),
                int(row["repeat"]),
            )
            rows[key] = row
    return rows


def load_final_metric_rows() -> dict[tuple[int, float, int], dict[str, str]]:
    rows: dict[tuple[int, float, int], dict[str, str]] = {}
    with FINAL_METRICS_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok":
                continue
            key = (
                int(float(row["batch_size"])),
                float(row["learning_rate"]),
                int(row["repeat"]),
            )
            rows[key] = row
    return rows


def load_records() -> list[RunRecord]:
    freezing_flatness_rows = load_freezing_flatness_rows()
    final_metric_rows = load_final_metric_rows()
    records: list[RunRecord] = []
    for key, row in freezing_flatness_rows.items():
        bs, lr, repeat = key
        final = final_metric_rows.get(key)
        if final is None:
            continue
        relative_sharpness = float(row.get("sharpness_mean_positive_delta", row.get("hessian_lambda_max", "nan")))
        relative_flatness = float(row.get("flatness_neg_log10", "nan"))
        if not np.isfinite(relative_flatness):
            hessian_flatness = float(row.get("hessian_flatness", "nan"))
            relative_flatness = np.log10(max(hessian_flatness, SHARPNESS_FLOOR)) if np.isfinite(hessian_flatness) else np.nan
        continuation_converged = "unconverged" not in row.get("confidence_flag", "")
        final_train_converged = float(final["base_train_accuracy"]) >= 0.999
        records.append(
            RunRecord(
                batch_size=bs,
                learning_rate=lr,
                repeat=repeat,
                effective_noise=float(lr / bs),
                tf=float(row["tf"]),
                eta_tf=float(row.get("eta_tf", row.get("eta_checkpoint", "nan"))),
                relative_flatness=float(relative_flatness),
                relative_sharpness=float(relative_sharpness),
                final_test_acc=float(final["base_test_accuracy"]),
                final_test_loss=float(final["base_test_loss"]),
                final_train_converged=final_train_converged,
                continuation_converged=continuation_converged,
            )
        )
    return records


def summarize_conditions(records: list[RunRecord]) -> list[ConditionSummary]:
    summaries: list[ConditionSummary] = []
    for bs in BS_LIST:
        for lr in LR_LIST:
            condition = [r for r in records if r.batch_size == int(bs) and np.isclose(r.learning_rate, lr)]
            stable = [r for r in condition if r.stable]
            eta_tf = np.array([r.eta_tf for r in stable], dtype=float)
            flatness = np.array([r.relative_flatness for r in stable], dtype=float)
            sharpness = np.array([r.relative_sharpness for r in stable], dtype=float)
            test_acc = np.array([r.final_test_acc for r in stable], dtype=float)
            test_loss = np.array([r.final_test_loss for r in stable], dtype=float)
            summaries.append(
                ConditionSummary(
                    batch_size=int(bs),
                    learning_rate=float(lr),
                    effective_noise=float(lr / bs),
                    n_total=len(condition),
                    n_stable=len(stable),
                    stable_fraction=len(stable) / len(condition) if condition else 0.0,
                    eta_tf_mean=nanmean(eta_tf),
                    eta_tf_median=nanmedian(eta_tf),
                    eta_tf_sem=sem(eta_tf),
                    relative_flatness_mean=nanmean(flatness),
                    relative_flatness_sem=sem(flatness),
                    relative_sharpness_mean=nanmean(sharpness),
                    relative_sharpness_sem=sem(sharpness),
                    final_test_acc_mean=nanmean(test_acc),
                    final_test_acc_sem=sem(test_acc),
                    final_test_loss_mean=nanmean(test_loss),
                    final_test_loss_sem=sem(test_loss),
                )
            )
    return summaries


def lr_color_map(summaries: list[ConditionSummary]) -> tuple[dict[float, tuple], list[float]]:
    learning_rates = sorted(
        {s.learning_rate for s in summaries if s.learning_rate < COMPARE_LR_MAX and s.n_stable > 0}
    )
    cmap = plt.colormaps["viridis"].resampled(len(learning_rates))
    return {lr: cmap(i) for i, lr in enumerate(learning_rates)}, learning_rates


def condition_arrays(summaries: list[ConditionSummary]) -> dict[str, np.ndarray]:
    shape = (len(BS_LIST), len(LR_LIST))
    arrays = {
        "eta_tf": np.full(shape, np.nan),
        "stable_fraction": np.full(shape, np.nan),
        "n_stable": np.full(shape, np.nan),
    }
    for s in summaries:
        i = np.flatnonzero(BS_LIST == s.batch_size)[0]
        j = np.flatnonzero(np.isclose(LR_LIST, s.learning_rate))[0]
        arrays["eta_tf"][i, j] = s.eta_tf_median
        arrays["stable_fraction"][i, j] = s.stable_fraction
        arrays["n_stable"][i, j] = s.n_stable
    return arrays


def plot_freezing_heatmap(ax: plt.Axes, summaries: list[ConditionSummary]) -> None:
    arrays = condition_arrays(summaries)
    data = arrays["eta_tf"].copy()
    stable_fraction = arrays["stable_fraction"]
    valid = np.isfinite(data)
    im = ax.imshow(
        data,
        origin="lower",
        aspect="equal",
        cmap="viridis",
        vmin=np.nanmin(data[valid]),
        vmax=np.nanmax(data[valid]),
    )
    ax.set_xticks(np.arange(len(LR_LIST)))
    ax.set_xticklabels([f"{x:g}" for x in LR_LIST], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(BS_LIST)))
    ax.set_yticklabels([f"{x:g}" for x in BS_LIST])
    ax.set_xlabel(r"Learning rate $\eta$")
    ax.set_ylabel(r"Batch size $B$")

    for i in range(stable_fraction.shape[0]):
        for j in range(stable_fraction.shape[1]):
            frac = stable_fraction[i, j]
            if not np.isfinite(frac) or frac == 0:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="0.65", zorder=3))
            elif frac < 0.8:
                ax.text(j, i, f"{frac:.0%}", ha="center", va="center", fontsize=6, color="white")

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"Median $\eta t_f$")
    ax.text(
        0.03,
        0.08,
        "stable trajectories only\ntext: stable fraction < 80%",
        transform=ax.transAxes,
        fontsize=7,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 2},
    )
    add_panel_label(ax, "A")


def plot_freezing_distribution(ax: plt.Axes, records: list[RunRecord]) -> None:
    stable_records = [r for r in records if r.stable and r.learning_rate < COMPARE_LR_MAX]
    x = np.array([r.effective_noise for r in stable_records], dtype=float)
    y = np.array([r.eta_tf for r in stable_records], dtype=float)
    c = np.array([np.log10(r.learning_rate) for r in stable_records], dtype=float)

    ax.scatter(x, y, c=c, cmap="viridis", s=8, alpha=0.28, edgecolor="none")
    ax.set_xscale("log")
    ax.set_xlabel(r"Effective noise proxy $\eta/B$")
    ax.set_ylabel(r"Continuation freezing coordinate $\eta t_f$")
    ax.grid(True, lw=0.4, alpha=0.35)
    cb = plt.colorbar(
        plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=np.nanmin(c), vmax=np.nanmax(c))),
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cb.set_label(r"$\log_{10}\eta$")
    ax.text(0.04, 0.92, "individual stable runs", transform=ax.transAxes, fontsize=8, va="top")
    add_panel_label(ax, "B")


def rows_for_comparison(
    summaries: list[ConditionSummary],
    learning_rates: list[float],
    metric_attr: str,
) -> list[ConditionSummary]:
    return [
        s
        for s in summaries
        if any(np.isclose(s.learning_rate, lr) for lr in learning_rates)
        and s.n_stable > 0
        and np.isfinite(s.eta_tf_mean)
        and np.isfinite(getattr(s, metric_attr))
    ]


def draw_metric_scatter(
    ax: plt.Axes,
    rows: list[ConditionSummary],
    metric_attr: str,
    x_attr: str,
    xlabel: str,
    ylabel: str,
    color_by_lr: dict[float, tuple],
    learning_rates: list[float],
    *,
    xscale: str = "linear",
) -> None:
    for lr in learning_rates:
        lr_rows = [s for s in rows if np.isclose(s.learning_rate, lr)]
        lr_rows = sorted(lr_rows, key=lambda s: getattr(s, x_attr))
        x = np.array([getattr(s, x_attr) for s in lr_rows], dtype=float)
        y = np.array([getattr(s, metric_attr) for s in lr_rows], dtype=float)
        n = np.array([s.n_stable for s in lr_rows], dtype=float)
        if x.size == 0:
            continue
        sizes = 14 + 1.7 * n
        ax.scatter(x, y, s=sizes, color=color_by_lr[lr], edgecolor="k", linewidth=0.3)

    all_x = np.array([getattr(s, x_attr) for s in rows], dtype=float)
    all_y = np.array([getattr(s, metric_attr) for s in rows], dtype=float)
    mask = np.isfinite(all_x) & np.isfinite(all_y)
    all_x = all_x[mask]
    all_y = all_y[mask]
    if all_x.size >= 3:
        fit_x = np.log10(all_x) if xscale == "log" else all_x
        fit = linregress(fit_x, all_y)
        if xscale == "log":
            xs = np.logspace(np.log10(np.nanmin(all_x)), np.log10(np.nanmax(all_x)), 200)
            xs_fit = np.log10(xs)
        else:
            xs = np.linspace(np.nanmin(all_x), np.nanmax(all_x), 200)
            xs_fit = xs
        ax.plot(xs, fit.intercept + fit.slope * xs_fit, color="#cc6677", lw=1.1)
        ax.text(0.04, 0.88, rf"$R^2={fit.rvalue**2:.2f}$", transform=ax.transAxes, fontsize=6.5)

    ax.set_xscale(xscale)
    ax.set_ylabel(ylabel, fontsize=6.5)
    ax.set_xlabel(xlabel, fontsize=6.5)
    ax.grid(True, lw=0.35, alpha=0.35)
    ax.tick_params(labelsize=5.8)


def plot_metric_comparison_panel(
    container_ax: plt.Axes,
    summaries: list[ConditionSummary],
    metric_attr: str,
    ylabel: str,
    panel_label: str,
    color_by_lr: dict[float, tuple],
    learning_rates: list[float],
    *,
    show_legend: bool = False,
) -> None:
    container_ax.axis("off")
    left = container_ax.inset_axes([0.08, 0.16, 0.37, 0.70])
    right = container_ax.inset_axes([0.62, 0.16, 0.37, 0.70])
    rows = rows_for_comparison(summaries, learning_rates, metric_attr)

    draw_metric_scatter(
        left,
        rows,
        metric_attr,
        "effective_noise",
        r"$\eta/B$",
        ylabel,
        color_by_lr,
        learning_rates,
        xscale="log",
    )
    draw_metric_scatter(
        right,
        rows,
        metric_attr,
        "eta_tf_mean",
        r"$\langle\eta t_f\rangle_{\rm stable}$",
        ylabel,
        color_by_lr,
        learning_rates,
        xscale="linear",
    )
    y_values = np.array([getattr(s, metric_attr) for s in rows], dtype=float)
    y_values = y_values[np.isfinite(y_values)]
    if y_values.size:
        y_pad = 0.08 * (np.nanmax(y_values) - np.nanmin(y_values))
        if y_pad == 0:
            y_pad = 0.05 * abs(np.nanmean(y_values))
        left.set_ylim(np.nanmin(y_values) - y_pad, np.nanmax(y_values) + y_pad)
        right.set_ylim(np.nanmin(y_values) - y_pad, np.nanmax(y_values) + y_pad)
    right.set_ylabel("")
    right.set_yticklabels([])

    if show_legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                markerfacecolor=color_by_lr[lr],
                markeredgecolor="k",
                markeredgewidth=0.3,
                markersize=4.5,
                label=rf"$\eta={lr:g}$",
            )
            for lr in learning_rates
        ]
        container_ax.legend(
            handles=handles,
            frameon=False,
            fontsize=5.4,
            loc="upper center",
            bbox_to_anchor=(0.55, 1.00),
            ncol=3,
            handletextpad=0.3,
            columnspacing=0.7,
        )

    container_ax.text(
        -0.02,
        1.04,
        panel_label,
        transform=container_ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def write_summary(summaries: list[ConditionSummary], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([field.name for field in ConditionSummary.__dataclass_fields__.values()])
        for s in summaries:
            writer.writerow([getattr(s, field.name) for field in ConditionSummary.__dataclass_fields__.values()])


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records()
    summaries = summarize_conditions(records)
    summary_path = OUT_DIR / "Fig2_freezing_time_hessian_flatness_summary.csv"
    write_summary(summaries, summary_path)

    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.4), constrained_layout=True)
    color_by_lr, learning_rates = lr_color_map(summaries)

    plot_freezing_heatmap(axs[0, 0], summaries)
    plot_freezing_distribution(axs[0, 1], records)
    plot_metric_comparison_panel(
        axs[1, 0],
        summaries,
        "relative_flatness_mean",
        r"Hessian flatness at $t_f$, $-\log_{10}\bar{\lambda}_{1:10}$",
        "C",
        color_by_lr,
        learning_rates,
        show_legend=True,
    )
    plot_metric_comparison_panel(
        axs[1, 1],
        summaries,
        "final_test_acc_mean",
        r"Final test accuracy",
        "D",
        color_by_lr,
        learning_rates,
    )

    png_path = OUT_DIR / "Fig2_freezing_time_hessian_flatness.png"
    pdf_path = OUT_DIR / "Fig2_freezing_time_hessian_flatness.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
