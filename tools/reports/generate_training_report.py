#!/usr/bin/env python3
"""Generate the final RotGS report figures and LaTeX tables from frozen exports.

This script is intentionally independent of training, W&B, rendering, and the
corrected-evaluation program.  It reads finalized local CSV/JSON inputs only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


ROOT = Path("/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis")
REPORT = ROOT / "report"
FIGURES = REPORT / "figures"
TABLES = REPORT / "tables"

RUNS = {
    "pjqalc78": "Chips blue (down)",
    "xxa8u1lf": "Chips blue",
    "k91zqmrx": "Kaegifret (down)",
    "nzn0toxf": "Kaegifret (top)",
    "tihtfc2s": "Leermond",
    "razrfwg4": "Pepsi (down)",
    "gb34gtza": "Pepsi",
    "legtvway": "Quoellfrisch (top, 30°)",
    "jhyezl5j": "Ramseier Tetra",
    "irrf5t67": "Schwamm (down)",
    "eir65c8l": "Schwamm (top)",
    "x34n08fc": "Weisswein",
}
RUN_IDS = list(RUNS)
COLORS = dict(zip(RUN_IDS, plt.cm.tab20(np.linspace(0, 1, len(RUN_IDS)))))
RESETS = [500, 3000, 6000, 9000, 12000]

INPUTS = [
    "final_analysis_context.md", "source_verification_summary.md",
    "corrected_evaluation_methodology.md", "code_metric_map.md",
    "run_configs.json", "run_metadata.csv", "run_selection.csv",
    "timeseries_analysis.csv", "training_events.csv", "metric_summary.csv",
    "cross_run_metrics.csv", "cross_run_relationships.csv",
    "metric_definitions.csv", "data_quality_report.csv",
    "corrected_evaluation_30k.csv", "corrected_evaluation_30k_per_view.csv",
    "corrected_evaluation_relationships.csv", "corrected_evaluation_validation.csv",
    "corrected_evaluation_view_manifest.csv", "evaluation_25k_vs_30k.csv",
    "evaluation_25k_vs_30k_per_view.csv", "local_enrichment.csv",
]

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 7, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 110, "savefig.dpi": 220, "axes.grid": True,
    "grid.alpha": .22, "lines.linewidth": 1.0, "pdf.fonttype": 42,
    "ps.fonttype": 42, "figure.constrained_layout.use": True,
})


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name)


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        if column in out:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def groups(ts: pd.DataFrame):
    for run_id in RUN_IDS:
        yield run_id, ts.loc[ts.run_id == run_id].sort_values("iteration")


def event_line(ax, x: int, label: str | None = None, *, color="0.35", ls="--", lw=.7):
    ax.axvline(x, color=color, linestyle=ls, linewidth=lw, alpha=.65, zorder=0)
    if label:
        ax.text(x, .98, label, transform=ax.get_xaxis_transform(), rotation=90,
                ha="right", va="top", fontsize=6.5, color=color)


def common_boundaries(ax, labels=False):
    event_line(ax, 2000, "center active after 2k" if labels else None)
    event_line(ax, 12000, "motion stop / reset" if labels else None)
    event_line(ax, 13000, "topology frozen" if labels else None)


def shared_legend(fig, axes, ncol=4):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=ncol,
               frameon=False, columnspacing=1.1, handlelength=2.0)


def save(fig, stem: str):
    for suffix in ("pdf", "png"):
        path = FIGURES / f"{stem}.{suffix}"
        metadata = ({"Creator": "RotGS final report generator", "CreationDate": None,
                     "ModDate": None} if suffix == "pdf" else
                    {"Software": "RotGS final report generator"})
        fig.savefig(path, bbox_inches="tight", metadata=metadata)
    plt.close(fig)


def plot_all(ax, ts, metric, *, alpha=.45, lw=.65, marker=None, ms=2.5, label=True):
    for run_id, frame in groups(ts):
        data = frame[["iteration", metric]].dropna()
        ax.plot(data.iteration, data[metric], color=COLORS[run_id], alpha=alpha,
                lw=lw, marker=marker, ms=ms,
                label=RUNS[run_id] if label else None)


def median_line(ax, ts, metric, *, color="black", label="Across-run median"):
    pivot = ts.pivot_table(index="iteration", columns="run_id", values=metric)
    med = pivot.median(axis=1, skipna=True)
    ax.plot(med.index, med, color=color, lw=1.7, label=label, zorder=10)


def figure_01(ts):
    fig, axes = plt.subplots(2, 1, figsize=(7.25, 6.4), sharex=True)
    for ax, metric, title, ylabel in [
        (axes[0], "train/loss_total", "Total training loss", "Loss"),
        (axes[1], "train/raw/alpha_l1", "Alpha loss", "L1 loss"),
    ]:
        plot_all(ax, ts, metric, alpha=.23, lw=.5)
        median_line(ax, ts, metric)
        for reset in RESETS:
            event_line(ax, reset, color="#b34a3c", ls=":", lw=.7)
        common_boundaries(ax, labels=ax is axes[0])
        ax.set(title=title, ylabel=ylabel, xlim=(0, 30000))
    axes[1].set_xlabel("Training iteration")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Training convergence and scheduled state disturbances")
    save(fig, "figure_01_training_convergence")


def figure_02(ts):
    fig, axes = plt.subplots(2, 1, figsize=(7.25, 7.0), sharex=False)
    plot_all(axes[0], ts, "scene/gaussians_after_update", alpha=.9, lw=1.0)
    common_boundaries(axes[0], labels=True)
    event_line(axes[0], 600, "first update", color="#555")
    event_line(axes[0], 12900, "final update", color="#555")
    axes[0].axhline(450000, color="black", ls=":", lw=.8, label="Configured maximum")
    axes[0].set(title="Gaussian population after topology update", ylabel="Gaussians",
                xlim=(0, 30000))
    for run_id, frame in groups(ts):
        data = frame[["iteration", "densification/net_change"]].dropna()
        axes[1].plot(data.iteration, data["densification/net_change"], marker="o",
                     ms=2.5, color=COLORS[run_id], lw=.8, label=RUNS[run_id])
    axes[1].axhline(0, color="black", lw=.6)
    event_line(axes[1], 600, "first")
    event_line(axes[1], 12900, "final")
    event_line(axes[1], 13000, "freeze")
    axes[1].set(title="Exact net change at each densification event",
                xlabel="Training iteration", ylabel="Net Gaussian change", xlim=(400, 13200))
    shared_legend(fig, axes, ncol=3)
    fig.suptitle("Adaptive representation growth and densification")
    save(fig, "figure_02_gaussian_population")


def figure_03(ts):
    fig, axes = plt.subplots(3, 1, figsize=(7.25, 7.6), sharex=True)
    plot_all(axes[0], ts, "scene/gaussians_after_update", alpha=.9, lw=1.0)
    axes[0].set_ylabel("Gaussians")
    for run_id, frame in groups(ts):
        data = frame[["iteration", "densification/net_change"]].dropna()
        axes[1].plot(data.iteration, data["densification/net_change"], marker="o", ms=3,
                     color=COLORS[run_id], lw=.8)
    axes[1].axhline(0, color="black", lw=.6)
    axes[1].set_ylabel("Net change")
    for run_id, frame in groups(ts):
        data = frame[["iteration", "scene/opacity_mean"]].dropna()
        axes[2].scatter(data.iteration, data["scene/opacity_mean"], s=11,
                        color=COLORS[run_id], label=RUNS[run_id])
    axes[2].set(ylabel="Mean opacity", xlabel="Training iteration")
    for ax in axes:
        ax.set_xlim(11500, 13200)
        for x, label in [(12000, "reset / motion stop"), (12300, "12.3k"),
                         (12600, "12.6k"), (12900, "final update"), (13000, "freeze")]:
            event_line(ax, x, label if ax is axes[0] else None,
                       color="#b34a3c" if x == 12000 else "0.35")
    shared_legend(fig, axes, ncol=3)
    fig.suptitle("Final topology transition")
    save(fig, "figure_03_final_topology_transition")


def figure_04(ts):
    fig, axes = plt.subplots(3, 1, figsize=(7.25, 7.7), sharex=True)
    for run_id, frame in groups(ts):
        data = frame[["iteration", "scene/opacity_mean"]].dropna()
        axes[0].scatter(data.iteration, data["scene/opacity_mean"], s=9,
                        color=COLORS[run_id], alpha=.8, label=RUNS[run_id])
    axes[0].set(title="Sparse post-event opacity observations", ylabel="Mean opacity")
    for ax, metric, title, ylabel in [
        (axes[1], "train/raw/alpha_l1", "Pre-topology alpha loss", "L1 loss"),
        (axes[2], "train/loss_total", "Pre-topology total loss", "Loss"),
    ]:
        plot_all(ax, ts, metric, alpha=.20, lw=.45, label=False)
        median_line(ax, ts, metric)
        ax.set(title=title, ylabel=ylabel)
    for ax in axes:
        ax.set_xlim(0, 13000)
        for reset in RESETS:
            event_line(ax, reset, f"reset {reset/1000:g}k" if ax is axes[0] else None,
                       color="#b34a3c", ls=":")
    axes[2].set_xlabel("Training iteration")
    axes[1].legend(loc="upper right", frameon=False)
    shared_legend(fig, axes, ncol=3)
    fig.suptitle("Opacity-reset disturbance and recovery")
    save(fig, "figure_04_opacity_reset_dynamics")


def figure_05(ts):
    fig, axes = plt.subplots(3, 1, figsize=(7.25, 7.4), sharex=True)
    specs = [("motion/axis_tilt_deg", "Axis tilt", "Degrees"),
             ("motion/axis_side_deg", "Axis-side correction", "Degrees"),
             ("motion/center_shift", "Center shift", "World units")]
    for ax, (metric, title, ylabel) in zip(axes, specs):
        plot_all(ax, ts, metric, alpha=.8, lw=.75)
        event_line(ax, 2000, "center active after 2k" if ax is axes[0] else None)
        event_line(ax, 12000, "axis/center stop" if ax is axes[0] else None)
        ax.set(title=title, ylabel=ylabel, xlim=(0, 30000))
    axes[-1].set_xlabel("Training iteration")
    shared_legend(fig, axes, ncol=3)
    fig.suptitle("Rotation-axis and center convergence")
    save(fig, "figure_05_motion_convergence")


def figure_06(ts):
    fig, ax = plt.subplots(figsize=(7.25, 4.5))
    plot_all(ax, ts, "motion/sweep_error_deg", alpha=.9, lw=.8)
    ax.axhline(-4, color="#b34a3c", ls="--", lw=1, label="Configured ±4° bounds")
    ax.axhline(4, color="#b34a3c", ls="--", lw=1)
    event_line(ax, 12000, "axis/center stop")
    event_line(ax, 15000, "residual LR terminal value")
    ax.set(title="Bounded global sweep correction", xlabel="Training iteration",
           ylabel="Sweep correction (degrees)", xlim=(0, 30000), ylim=(-4.25, 4.25))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), frameon=False)
    save(fig, "figure_06_sweep_correction")


def figure_07(ts):
    fig, ax = plt.subplots(figsize=(7.25, 4.7))
    plot_all(ax, ts, "eval/test/foreground_psnr", alpha=.95, lw=1.05,
             marker="o", ms=3.5)
    event_line(ax, 12000, "motion stop")
    event_line(ax, 13000, "topology frozen")
    ax.axvspan(13000, 30000, color="#4c78a8", alpha=.06, label="Fixed-topology refinement")
    ax.set(title="Held-out-view foreground PSNR at logged evaluation checkpoints",
           xlabel="Training iteration", ylabel="Foreground PSNR (dB)", xlim=(4500, 30500))
    ax.legend(loc="center left", bbox_to_anchor=(1.01, .5), frameon=False)
    save(fig, "figure_07_quality_convergence")


def descriptive_guide(ax, x, y, color):
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(min(x), max(x), 100)
    ax.plot(xx, np.polyval(coef, xx), color=color, ls="--", lw=.9, alpha=.75)


def figure_08(corrected, relationships):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.35), sharex=True)
    specs = [
        ("PSNR", "old_full_psnr", "foreground_psnr_mean", "PSNR (dB)"),
        ("SSIM", "old_full_ssim", "foreground_crop_ssim_mean", "SSIM"),
        ("LPIPS", "old_full_lpips", "foreground_crop_lpips_mean", "LPIPS"),
    ]
    x = corrected.mean_foreground_coverage.to_numpy()
    for ax, (name, old, new, ylabel) in zip(axes, specs):
        ax.scatter(x, corrected[old], marker="x", s=28, color="#d95f02", label="Historical full frame")
        ax.scatter(x, corrected[new], marker="o", s=22, facecolor="none", edgecolor="#1b9e77",
                   label="Corrected foreground-aware")
        descriptive_guide(ax, x, corrected[old].to_numpy(), "#d95f02")
        descriptive_guide(ax, x, corrected[new].to_numpy(), "#1b9e77")
        old_rel = relationships.loc[relationships.outcome == old].iloc[0]
        new_rel = relationships.loc[relationships.outcome == new].iloc[0]
        ax.text(.03, .97, f"Historical: r={old_rel.pearson_r:+.3f}, ρ={old_rel.spearman_rho:+.3f}\n"
                          f"Corrected: r={new_rel.pearson_r:+.3f}, ρ={new_rel.spearman_rho:+.3f}\nN=12",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
        ax.set(title=name, xlabel="Mean foreground coverage", ylabel=ylabel)
    axes[0].legend(loc="lower left", frameon=False, fontsize=6.5)
    fig.suptitle("Foreground coverage association before and after evaluation correction\n"
                 "Dashed lines are descriptive linear guides, not predictive models")
    save(fig, "figure_08_evaluation_bias")


def figure_09(endpoint):
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.55))
    specs = [
        ("foreground_psnr", "Foreground PSNR (dB)", "Higher is better"),
        ("foreground_crop_ssim", "Foreground-crop SSIM", "Higher is better"),
        ("foreground_crop_lpips", "Foreground-crop LPIPS", "Lower is better"),
    ]
    for ax, (prefix, ylabel, direction) in zip(axes, specs):
        a, b = f"{prefix}_25000_mean", f"{prefix}_30000_mean"
        for _, row in endpoint.iterrows():
            rid = row.run_id
            ax.plot([0, 1], [row[a], row[b]], marker="o", ms=3,
                    color=COLORS[rid], lw=.9, label=RUNS[rid])
        delta = endpoint[f"{prefix}_delta_30000_minus_25000"].mean()
        ax.text(.04, .96, f"{direction}\nMean Δ={delta:+.5g}", transform=ax.transAxes,
                ha="left", va="top", fontsize=7)
        ax.set(xticks=[0, 1], xticklabels=["25k", "30k"], ylabel=ylabel)
    shared_legend(fig, axes, ncol=3)
    fig.suptitle("Fixed-topology parameter refinement from 25k to 30k")
    save(fig, "figure_09_fixed_topology_refinement")


def figure_10(ts):
    post = ts.loc[ts.iteration >= 13000].copy()
    fig, axes = plt.subplots(3, 1, figsize=(7.25, 7.5), sharex=True)
    for run_id, frame in groups(post):
        final = frame["scene/gaussians_after_update"].dropna().iloc[-1]
        data = frame[["iteration", "scene/gaussians_after_update"]].dropna()
        axes[0].plot(data.iteration, data["scene/gaussians_after_update"] / final,
                     color=COLORS[run_id], lw=.8, label=RUNS[run_id])
        opacity = frame[["iteration", "scene/opacity_mean"]].dropna()
        axes[1].scatter(opacity.iteration, opacity["scene/opacity_mean"], s=12,
                        color=COLORS[run_id])
        timing = frame[["iteration", "performance/iteration_ms"]].dropna().copy()
        baseline = timing.loc[timing.iteration < 14000, "performance/iteration_ms"].median()
        timing["bin"] = ((timing.iteration - 13000) // 500).astype(int)
        binned = timing.groupby("bin", as_index=False).agg(
            iteration=("iteration", "median"), value=("performance/iteration_ms", "median"))
        axes[2].plot(binned.iteration, binned.value / baseline, color=COLORS[run_id], lw=.9)
    axes[0].set(title="Gaussian population (normalized by final count)", ylabel="Relative count", ylim=(.995, 1.005))
    axes[1].set(title="Sparse event-triggered opacity observations", ylabel="Mean opacity")
    axes[2].set(title="CUDA forward/loss/backward timing (500-iteration bin medians)",
                ylabel="Relative timing", xlabel="Training iteration")
    axes[2].axhline(1, color="black", ls=":", lw=.7)
    for ax in axes:
        ax.set_xlim(13000, 30000)
    shared_legend(fig, axes, ncol=3)
    fig.suptitle("Post-densification computational behaviour")
    save(fig, "figure_10_post_densification_compute")


def tex(value) -> str:
    if pd.isna(value):
        return "--"
    text = str(value)
    for old, new in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#")]:
        text = text.replace(old, new)
    return text


def write_table(name: str, body: str):
    (TABLES / name).write_text(body.rstrip() + "\n", encoding="utf-8")


def make_tables(corrected, endpoint, cross, metadata, validation, manifest, metric_defs):
    selected = metadata.loc[metadata.selected_for_analysis == True].copy()  # noqa: E712
    overview = selected.merge(cross[["run_id", "final_gaussians", "wandb_total_runtime_seconds"]], on="run_id")
    dims = manifest.loc[manifest.iteration == 30000].groupby("run_id").agg(
        resolution=("run_id", "size")).reset_index()
    # Resolution is recovered from the per-view metric file below; the manifest itself intentionally has no dimensions.
    per_view = read("corrected_evaluation_30k_per_view.csv")
    sizes = per_view.groupby("run_id").first().reset_index()[["run_id", "image_width", "image_height"]]
    warns = validation[["run_id", "warning_expected"]]
    overview = overview.merge(sizes, on="run_id").merge(warns, on="run_id")
    lines = [r"\begin{longtable}{@{}llrrrrl@{}}", r"\caption{Completed-run overview.}\label{tab:overview}\\",
             r"\toprule Capture & Run ID & Resolution & Final G & Runtime (s) & Views & RGB/$\alpha$ warning\\\midrule", r"\endfirsthead",
             r"\toprule Capture & Run ID & Resolution & Final G & Runtime (s) & Views & RGB/$\alpha$ warning\\\midrule", r"\endhead"]
    for rid in RUN_IDS:
        r = overview.loc[overview.run_id == rid].iloc[0]
        lines.append(f"{tex(RUNS[rid])} & {rid} & {int(r.image_width)}$\\times${int(r.image_height)} & {int(r.final_gaussians):,} & {r.wandb_total_runtime_seconds:.0f} & 73 & {'yes' if r.warning_expected else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    write_table("experimental_overview.tex", "\n".join(lines))

    lines = [r"\begin{longtable}{@{}lrrrr@{}}", r"\caption{Corrected 30k reconstruction metrics.}\label{tab:corrected30k}\\",
             r"\toprule Capture & FG PSNR (dB) & Crop SSIM & Crop LPIPS & Coverage\\\midrule", r"\endfirsthead",
             r"\toprule Capture & FG PSNR (dB) & Crop SSIM & Crop LPIPS & Coverage\\\midrule", r"\endhead"]
    for rid in RUN_IDS:
        r = corrected.loc[corrected.run_id == rid].iloc[0]
        lines.append(f"{tex(RUNS[rid])} & {r.foreground_psnr_mean:.3f} & {r.foreground_crop_ssim_mean:.4f} & {r.foreground_crop_lpips_mean:.4f} & {r.mean_foreground_coverage:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    write_table("corrected_metrics_30k.tex", "\n".join(lines))

    lines = [r"\begin{longtable}{@{}lrrrrr@{}}", r"\caption{Fixed-topology change from 25k to 30k; every delta is 30k minus 25k.}\label{tab:endpoints}\\",
             r"\toprule Capture & $\Delta$PSNR & $\Delta$SSIM & $\Delta$LPIPS & $\Delta$G & Extra runtime (s)\\\midrule", r"\endfirsthead",
             r"\toprule Capture & $\Delta$PSNR & $\Delta$SSIM & $\Delta$LPIPS & $\Delta$G & Extra runtime (s)\\\midrule", r"\endhead"]
    for rid in RUN_IDS:
        r = endpoint.loc[endpoint.run_id == rid].iloc[0]
        lines.append(f"{tex(RUNS[rid])} & {r.foreground_psnr_delta_30000_minus_25000:+.3f} & {r.foreground_crop_ssim_delta_30000_minus_25000:+.5f} & {r.foreground_crop_lpips_delta_30000_minus_25000:+.5f} & {int(r.gaussian_count_delta_30000_minus_25000)} & {r.wandb_runtime_seconds_delta_30000_minus_25000:.0f} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    write_table("refinement_25k_30k.tex", "\n".join(lines))

    schedule = [
        ("Center warmup", "1--2000", "Center fixed; first active iteration is 2001"),
        ("Densification", "600, 900, $\\ldots$, 12900", "42 exact topology-update events"),
        ("Opacity reset", "500, 3000, 6000, 9000, 12000", "Opacity capped at 0.01 after loss measurement"),
        ("Axis/center stop", "12000", "Axis and center cease updating"),
        ("Topology freeze", "13000", "No topology modification at or after this boundary"),
        ("Validation/checkpoints", "5000, 10000, $\\ldots$, 30000", "Six held-out-view evaluations"),
        ("Residual LR terminal value", "15000", "$2.5\\times10^{-5}$; remains nonzero"),
    ]
    lines = [r"\begin{tabularx}{\linewidth}{@{}l l X@{}}", r"\toprule Event & Iteration(s) & Verified interpretation\\\midrule"]
    lines += [f"{a} & {b} & {c} \\\\" for a, b, c in schedule]
    lines += [r"\bottomrule", r"\end{tabularx}"]
    write_table("training_event_schedule.tex", "\n".join(lines))

    keep = ["train/loss_total", "train/loss_image", "train/raw/alpha_l1",
            "scene/gaussians", "scene/gaussians_after_update", "scene/visible_gaussians",
            "scene/visible_fraction", "scene/opacity_mean", "motion/axis_tilt_deg",
            "motion/axis_side_deg", "motion/center_shift", "motion/sweep_error_deg",
            "performance/iteration_ms", "eval/test/foreground_psnr",
            "eval/test/foreground_ssim", "eval/train/foreground_psnr"]
    lines = [r"\begin{longtable}{@{}p{.24\linewidth}p{.13\linewidth}p{.25\linewidth}p{.28\linewidth}@{}}",
             r"\caption{Selected W\&B metric reference.}\label{tab:metrics}\\",
             r"\toprule Metric & Unit & Measurement stage/cadence & Interpretation warning\\\midrule", r"\endfirsthead",
             r"\toprule Metric & Unit & Measurement stage/cadence & Interpretation warning\\\midrule", r"\endhead"]
    for metric in keep:
        r = metric_defs.loc[metric_defs.metric == metric].iloc[0]
        warning = r.interpretation_warning if pd.notna(r.interpretation_warning) and r.interpretation_warning else "--"
        lines.append(f"\\texttt{{{tex(metric)}}} & {tex(r.unit)} & {tex(r.measurement_stage)}; {tex(r.logging_cadence)} & {tex(warning)} \\\\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    write_table("metric_reference.tex", "\n".join(lines))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def row_count(path: Path):
    if path.suffix != ".csv":
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def validate_inputs(ts, corrected, endpoint, events, quality, validation):
    assert set(ts.run_id.unique()) == set(RUN_IDS)
    assert len(ts) == 36000 and len(corrected) == len(endpoint) == 12
    assert len(quality) == 240 and set(quality.status) == {"PASS"}
    assert validation.validation_passed.astype(str).str.lower().eq("true").all()
    assert endpoint.gaussian_count_delta_30000_minus_25000.eq(0).all()
    for rid in RUN_IDS:
        e = events.loc[events.run_id == rid]
        assert len(e.loc[e.event_type == "densification"]) == 42
        assert len(e.loc[e.event_type == "opacity_reset"]) == 5


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    ts = numeric(read("timeseries_analysis.csv"), [c for c in read("timeseries_analysis.csv").columns if c not in ("run_id", "run_name")])
    corrected = read("corrected_evaluation_30k.csv")
    endpoint = read("evaluation_25k_vs_30k.csv")
    relationships = read("corrected_evaluation_relationships.csv")
    events = read("training_events.csv")
    quality = read("data_quality_report.csv")
    validation = read("corrected_evaluation_validation.csv")
    cross = read("cross_run_metrics.csv")
    metadata = read("run_metadata.csv")
    view_manifest = read("corrected_evaluation_view_manifest.csv")
    metric_defs = read("metric_definitions.csv")
    validate_inputs(ts, corrected, endpoint, events, quality, validation)

    figure_01(ts); figure_02(ts); figure_03(ts); figure_04(ts); figure_05(ts)
    figure_06(ts); figure_07(ts); figure_08(corrected, relationships)
    figure_09(endpoint); figure_10(ts)
    make_tables(corrected, endpoint, cross, metadata, validation, view_manifest, metric_defs)

    outputs = sorted(str(p.relative_to(ROOT)) for p in [*FIGURES.glob("figure_*.*"), *TABLES.glob("*.tex")])
    manifest = {
        "generator": str(Path(__file__).resolve()),
        "inputs": [{"path": name, "sha256": sha256(ROOT / name), "row_count": row_count(ROOT / name)} for name in INPUTS],
        "outputs": outputs,
        "packages": {"python": platform.python_version(), "matplotlib": matplotlib.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        "run_ids": RUN_IDS,
        "cross_run_analysis_run_ids": RUN_IDS,
        "transformations": {
            "figures_1_and_4": "raw traces plus across-run median; raw traces remain visible",
            "figure_10_timing": "per-run 500-iteration medians normalized by the per-run 13k-to-14k median",
            "figure_10_gaussians": "per-run counts normalized by each run's final count",
            "opacity": "sparse event-triggered observations shown as markers",
            "densification": "exact event values; no smoothing",
            "evaluation": "actual logged checkpoints; no interpolation",
        },
    }
    (REPORT / "generated_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(outputs)} figure/table files and manifest for {len(RUN_IDS)} runs.")


if __name__ == "__main__":
    main()
