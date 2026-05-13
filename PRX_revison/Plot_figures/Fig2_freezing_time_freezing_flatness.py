"""PRX Fig. 2: freezing coordinate, commitment geometry, and test accuracy.

This is the single canonical plotting script for the current Fig. 2. It keeps
only panels A/C/D from the previous draft:
  A: median continuation-based freezing coordinate across hyperparameters.
  C: neuron-wise flatness at the freezing checkpoint vs eta/B and eta t_f.
  D: final test accuracy vs the same two organizing variables.

The stable set is unchanged: continuation-stable trajectories whose final
checkpoint reaches the training-accuracy convergence criterion.
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
TRAIN_ACC_THRESHOLD = 0.999
LOW_STABLE_FRACTION = 0.8
LOW_STABLE_MIN_COUNT = 5
LARGE_SEM_THRESHOLD = 5.0

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "PRX_revison" / "Figures_revision" / "Fig2_final_converged_stable"

FREEZING_FLATNESS_CANDIDATES = [
    REPO_ROOT
    / "freezing_flatness_three_methods_ce"
    / "neuron_wise_rho0.05_k20_ceil"
    / "relative_sharpness_summary.csv",
    REPO_ROOT
    / "freezing_flatness_all_ce"
    / "random_filter_rho0.05_k20_ceil"
    / "freezing_flatness_summary.csv",
]
FINAL_METRICS_CSV = (
    REPO_ROOT
    / "relative_sharpness_all_ce"
    / "random_filter_rho0.05_k20"
    / "relative_sharpness_summary.csv"
)


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


@dataclass(frozen=True)
class HeatmapConfig:
    value_attr: str = "eta_tf_median"
    title: str = r"Median freezing coordinate $\eta t_f$"
    note: str = "stable runs only\ntext: stable fraction < 80%"
    mask_low_stability: bool = False
    annotate_sem: bool = False


def first_existing_path(candidates: list[Path]) -> Path:
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"None of the candidate input files exists:\n{tried}")


def finite_values(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def sem(values: np.ndarray) -> float:
    values = finite_values(values)
    if values.size <= 1:
        return np.nan
    return float(np.std(values, ddof=1) / np.sqrt(values.size))


def nanmean(values: np.ndarray) -> float:
    values = finite_values(values)
    return float(np.mean(values)) if values.size else np.nan


def nanmedian(values: np.ndarray) -> float:
    values = finite_values(values)
    return float(np.median(values)) if values.size else np.nan


def float_from(row: dict[str, str], *names: str, default: float = np.nan) -> float:
    for name in names:
        value = row.get(name, "")
        if value not in ("", None):
            return float(value)
    return default


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.9,
            "axes.labelsize": 7.6,
            "axes.titlesize": 8.0,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.4,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.07,
        label,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
        ha="left",
    )


def load_rows(path: Path, *, require_ok: bool = True) -> dict[tuple[int, float, int], dict[str, str]]:
    rows: dict[tuple[int, float, int], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if require_ok and row.get("status") != "ok":
                continue
            key = (int(float(row["batch_size"])), float(row["learning_rate"]), int(row["repeat"]))
            rows[key] = row
    return rows


def load_records() -> list[RunRecord]:
    freezing_path = first_existing_path(FREEZING_FLATNESS_CANDIDATES)
    freezing_rows = load_rows(freezing_path)
    final_rows = load_rows(FINAL_METRICS_CSV)

    records: list[RunRecord] = []
    for key, row in freezing_rows.items():
        final = final_rows.get(key)
        if final is None:
            continue
        bs, lr, repeat = key
        sharpness = float_from(row, "sharpness_mean_positive_delta")
        flatness = float_from(row, "flatness_neg_log10")
        if not np.isfinite(flatness):
            flatness = -np.log10(max(sharpness, SHARPNESS_FLOOR))
        records.append(
            RunRecord(
                batch_size=bs,
                learning_rate=lr,
                repeat=repeat,
                effective_noise=float(lr / bs),
                tf=float_from(row, "tf"),
                eta_tf=float_from(row, "eta_tf", "eta_checkpoint"),
                relative_flatness=float(flatness),
                relative_sharpness=float(sharpness),
                final_test_acc=float(final["base_test_accuracy"]),
                final_test_loss=float(final["base_test_loss"]),
                final_train_converged=float(final["base_train_accuracy"]) >= TRAIN_ACC_THRESHOLD,
                continuation_converged="unconverged" not in row.get("confidence_flag", ""),
            )
        )
    print(f"Loaded freezing flatness from {freezing_path}")
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


def write_summary(summaries: list[ConditionSummary], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([field.name for field in ConditionSummary.__dataclass_fields__.values()])
        for summary in summaries:
            writer.writerow([getattr(summary, field.name) for field in ConditionSummary.__dataclass_fields__.values()])


def lr_color_map(summaries: list[ConditionSummary]) -> tuple[dict[float, tuple], list[float]]:
    learning_rates = sorted({s.learning_rate for s in summaries if s.learning_rate < COMPARE_LR_MAX and s.n_stable > 0})
    cmap = plt.colormaps["viridis"].resampled(len(learning_rates))
    return {lr: cmap(i) for i, lr in enumerate(learning_rates)}, learning_rates


def condition_arrays(summaries: list[ConditionSummary], value_attr: str) -> dict[str, np.ndarray]:
    shape = (len(BS_LIST), len(LR_LIST))
    arrays = {
        "value": np.full(shape, np.nan),
        "eta_tf_sem": np.full(shape, np.nan),
        "stable_fraction": np.full(shape, np.nan),
        "n_stable": np.zeros(shape, dtype=int),
    }
    for s in summaries:
        i = np.flatnonzero(BS_LIST == s.batch_size)[0]
        j = np.flatnonzero(np.isclose(LR_LIST, s.learning_rate))[0]
        arrays["value"][i, j] = getattr(s, value_attr)
        arrays["eta_tf_sem"][i, j] = s.eta_tf_sem
        arrays["stable_fraction"][i, j] = s.stable_fraction
        arrays["n_stable"][i, j] = s.n_stable
    return arrays


def plot_freezing_heatmap(
    ax: plt.Axes,
    summaries: list[ConditionSummary],
    fig: plt.Figure,
    cax: plt.Axes | None = None,
    config: HeatmapConfig = HeatmapConfig(),
) -> None:
    arrays = condition_arrays(summaries, config.value_attr)
    data = arrays["value"].copy()
    stable_fraction = arrays["stable_fraction"]
    n_stable = arrays["n_stable"]
    if config.mask_low_stability:
        low_stability = (stable_fraction < LOW_STABLE_FRACTION) | (n_stable < LOW_STABLE_MIN_COUNT)
        data[low_stability] = np.nan

    valid = np.isfinite(data)
    im = ax.imshow(
        data,
        origin="lower",
        aspect="equal",
        cmap="viridis",
        vmin=np.nanmin(data[valid]) if np.any(valid) else 0,
        vmax=np.nanmax(data[valid]) if np.any(valid) else 1,
    )

    ax.set_xticks(np.arange(len(LR_LIST)))
    ax.set_xticklabels([f"{x:g}" for x in LR_LIST], rotation=42, ha="right")
    ax.set_yticks(np.arange(len(BS_LIST)))
    ax.set_yticklabels([f"{x:g}" for x in BS_LIST])
    ax.set_xlabel(r"Learning rate $\eta$")
    ax.set_ylabel(r"Batch size $B$")
    ax.set_title(config.title, pad=7)

    ax.set_xticks(np.arange(-0.5, len(LR_LIST), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BS_LIST), 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.6, alpha=0.55)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(stable_fraction.shape[0]):
        for j in range(stable_fraction.shape[1]):
            frac = stable_fraction[i, j]
            masked = config.mask_low_stability and (
                (np.isfinite(frac) and frac < LOW_STABLE_FRACTION) or n_stable[i, j] < LOW_STABLE_MIN_COUNT
            )
            if not np.isfinite(frac) or frac == 0 or masked:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="0.68", zorder=3))
            elif frac < LOW_STABLE_FRACTION:
                ax.text(j, i, f"{frac:.0%}", ha="center", va="center", fontsize=6.6, color="white")
            if config.annotate_sem and np.isfinite(data[i, j]) and arrays["eta_tf_sem"][i, j] > LARGE_SEM_THRESHOLD:
                ax.plot(
                    j,
                    i,
                    marker="o",
                    markersize=5.2,
                    markerfacecolor="none",
                    markeredgecolor="black",
                    markeredgewidth=0.7,
                    lw=0,
                    zorder=4,
                )

    cb = fig.colorbar(im, cax=cax) if cax is not None else fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cb.ax.set_title(r"$\eta t_f$", fontsize=8, pad=4)
    cb.ax.tick_params(labelsize=7)
    ax.text(
        0.03,
        0.06,
        config.note,
        transform=ax.transAxes,
        fontsize=6.6,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.34, "edgecolor": "none", "pad": 2.2},
    )
    add_panel_label(ax, "A")


def rows_for_comparison(summaries: list[ConditionSummary], learning_rates: list[float], metric_attr: str) -> list[ConditionSummary]:
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
        lr_rows = sorted([s for s in rows if np.isclose(s.learning_rate, lr)], key=lambda s: getattr(s, x_attr))
        if not lr_rows:
            continue
        x = np.array([getattr(s, x_attr) for s in lr_rows], dtype=float)
        y = np.array([getattr(s, metric_attr) for s in lr_rows], dtype=float)
        n = np.array([s.n_stable for s in lr_rows], dtype=float)
        ax.scatter(x, y, s=18 + 1.75 * n, color=color_by_lr[lr], edgecolor="0.12", linewidth=0.35, alpha=0.95)

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
        ax.plot(xs, fit.intercept + fit.slope * xs_fit, color="#c95f75", lw=1.25, zorder=0)
        ax.text(
            0.06,
            0.88,
            rf"$R^2={fit.rvalue**2:.2f}$",
            transform=ax.transAxes,
            fontsize=7.1,
            bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5},
        )

    ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_facecolor("#fbfbf7")
    ax.grid(True, lw=0.42, alpha=0.32)
    ax.tick_params(length=3.0, width=0.8)


def plot_metric_pair(
    left: plt.Axes,
    right: plt.Axes,
    summaries: list[ConditionSummary],
    metric_attr: str,
    ylabel: str,
    label: str,
    color_by_lr: dict[float, tuple],
    learning_rates: list[float],
) -> None:
    rows = rows_for_comparison(summaries, learning_rates, metric_attr)
    draw_metric_scatter(
        left,
        rows,
        metric_attr,
        "effective_noise",
        r"Noise proxy $\eta/B$",
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
        r"Freezing coordinate $\langle\eta t_f\rangle$",
        "",
        color_by_lr,
        learning_rates,
    )

    y_values = finite_values(np.array([getattr(s, metric_attr) for s in rows], dtype=float))
    if y_values.size:
        pad = 0.09 * (np.max(y_values) - np.min(y_values))
        pad = pad if pad > 0 else 0.03 * max(abs(np.mean(y_values)), 1.0)
        ylim = (np.min(y_values) - pad, np.max(y_values) + pad)
        left.set_ylim(*ylim)
        right.set_ylim(*ylim)
    right.tick_params(labelleft=False)
    add_panel_label(left, label)


def add_learning_rate_legend(fig: plt.Figure, color_by_lr: dict[float, tuple], learning_rates: list[float]) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=color_by_lr[lr],
            markeredgecolor="0.12",
            markeredgewidth=0.35,
            markersize=4.8,
            label=rf"$\eta={lr:g}$",
        )
        for lr in learning_rates
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.69, 0.982),
        ncol=6,
        columnspacing=0.72,
        handletextpad=0.28,
    )


def make_figure(summaries: list[ConditionSummary], heatmap_config: HeatmapConfig = HeatmapConfig()) -> plt.Figure:
    color_by_lr, learning_rates = lr_color_map(summaries)
    fig = plt.figure(figsize=(7.5, 4.65))

    # Manual axes positions keep the square heatmap top-aligned with the
    # two right-hand comparison rows, avoiding the large blank space produced
    # by an equal-aspect heatmap inside a tall GridSpec cell.
    fig_w, fig_h = fig.get_size_inches()
    heat_left = 0.065
    heat_width = 0.300
    heat_height = heat_width * fig_w / fig_h
    heat_bottom = 0.835 - heat_height
    cbar_gap = 0.012
    cbar_width = 0.012
    right_left = 0.505
    right_width = 0.190
    right_gap = 0.075
    row_height = 0.285
    row_top = 0.555
    row_bottom = 0.14

    ax_a = fig.add_axes([heat_left, heat_bottom, heat_width, heat_height])
    cax_a = fig.add_axes([heat_left + heat_width + cbar_gap, heat_bottom, cbar_width, heat_height])
    ax_c1 = fig.add_axes([right_left, row_top, right_width, row_height])
    ax_c2 = fig.add_axes([right_left + right_width + right_gap, row_top, right_width, row_height])
    ax_d1 = fig.add_axes([right_left, row_bottom, right_width, row_height])
    ax_d2 = fig.add_axes([right_left + right_width + right_gap, row_bottom, right_width, row_height])

    plot_freezing_heatmap(ax_a, summaries, fig, cax_a, heatmap_config)
    plot_metric_pair(
        ax_c1,
        ax_c2,
        summaries,
        "relative_flatness_mean",
        "Freezing flatness\n" + r"$F_f=-\log_{10}S_{\rm nw}$",
        "C",
        color_by_lr,
        learning_rates,
    )
    ax_c1.set_xlabel("")
    ax_c2.set_xlabel("")
    plot_metric_pair(
        ax_d1,
        ax_d2,
        summaries,
        "final_test_acc_mean",
        r"Final test accuracy",
        "D",
        color_by_lr,
        learning_rates,
    )
    ax_c1.set_title("Raw hyperparameter scale", pad=6)
    ax_c2.set_title("Freezing-time coordinate", pad=6)
    add_learning_rate_legend(fig, color_by_lr, learning_rates)
    return fig


def main() -> None:
    setup_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_records()
    summaries = summarize_conditions(records)
    summary_path = OUT_DIR / "Fig2_freezing_time_freezing_flatness_summary.csv"
    write_summary(summaries, summary_path)

    figure_specs = [
        (
            "Fig2_freezing_time_freezing_flatness",
            HeatmapConfig(),
        ),
        (
            "Fig2_freezing_time_freezing_flatness_A_mean",
            HeatmapConfig(
                value_attr="eta_tf_mean",
                title=r"Mean freezing coordinate $\eta t_f$",
                note=rf"stable runs only" "\n" rf"circle: SEM $>{LARGE_SEM_THRESHOLD:g}$",
                annotate_sem=True,
            ),
        ),
        (
            "Fig2_freezing_time_freezing_flatness_A_median_mask080",
            HeatmapConfig(
                title=r"Median freezing coordinate $\eta t_f$",
                note=f"gray: stable fraction < {LOW_STABLE_FRACTION:.0%}\n"
                + rf"or $n_\mathrm{{stable}}<{LOW_STABLE_MIN_COUNT}$",
                mask_low_stability=True,
            ),
        ),
    ]

    for stem, heatmap_config in figure_specs:
        fig = make_figure(summaries, heatmap_config)
        png_path = OUT_DIR / f"{stem}.png"
        pdf_path = OUT_DIR / f"{stem}.pdf"
        fig.savefig(png_path, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {png_path}")
        print(f"Wrote {pdf_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
