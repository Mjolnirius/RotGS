#!/usr/bin/env python3
"""Export and derive the reproducible RotGS W&B analysis dataset.

The W&B scalar history written by ``scan_history`` is canonical and untouched.
All derived tables are regenerated from that history plus explicitly identified
configuration and local files.  The script never reads or writes credentials.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import wandb


SCRIPT_VERSION = "1.0.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis")

NORMALIZED_CONFIG_FIELDS = [
    "iterations", "sh_degree", "white_background", "rotation_direction",
    "fixed_camera", "densify_from_iter", "densify_until_iter",
    "densification_interval", "densify_grad_threshold", "percent_dense",
    "opacity_reset_interval", "max_gaussians", "axis_mode", "freeze_axis",
    "wo_axis", "freeze_center", "axis_tilt_init_deg", "axis_tilt_min_deg",
    "axis_tilt_max_deg", "axis_side_limit_deg", "center_max_offset",
    "center_warmup_iterations", "center_axis_lr_max_steps",
    "max_residual_angle_deg", "max_sweep_error_deg", "wo_tiny", "wo_flow",
    "wo_UDFS", "warmup_iteration", "flow_schedule_lambda", "lambda_flow",
    "flow_downsampling", "lambda_dssim", "lambda_foreground_rgb",
    "lambda_full_rgb", "lambda_alpha", "lambda_center_reg",
    "ssim_crop_padding", "position_lr_init", "position_lr_final",
    "position_lr_max_steps", "center_axis_lr_init", "center_axis_lr_final",
    "axis_lr_init", "axis_lr_final", "center_lr_init", "center_lr_final",
    "residual_lr_init", "residual_lr_final", "residual_lr_max_steps",
    "dataset_id", "dataset_session", "preprocessing_variant", "model_path",
    "git_commit", "git_dirty", "wandb_log_interval", "test_iterations",
    "save_iterations", "checkpoint_iterations",
]

PHASE_METRICS = [
    "train/loss_total", "train/raw/alpha_l1", "scene/visible_fraction",
    "performance/iteration_ms", "motion/center_shift",
    "motion/axis_tilt_deg", "motion/axis_side_deg", "motion/sweep_error_deg",
]

LOSS_METRICS = [
    "train/loss_total", "train/loss_image", "train/raw/foreground_l1",
    "train/raw/full_l1", "train/raw/foreground_dssim", "train/raw/alpha_l1",
    "train/raw/flow", "train/raw/center_regularization",
    "train/weighted/foreground_l1", "train/weighted/full_l1",
    "train/weighted/foreground_dssim", "train/weighted/alpha_l1",
    "train/weighted/flow", "train/weighted/center_regularization",
]


def jsonable(value: Any) -> Any:
    if not isinstance(value, Mapping) and hasattr(value, "items"):
        value = dict(value.items())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))
    return value


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in fields})


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def repository_provenance() -> dict[str, Any]:
    commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain")
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=REPO_ROOT
    )
    return {
        "git_commit": commit,
        "git_dirty": bool(status),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_diff_hash_scope": "tracked changes relative to HEAD; untracked file contents excluded",
    }


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def finite_series(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in frame:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[metric], errors="coerce").dropna()


def value_at(frame: pd.DataFrame, iteration: int, metric: str) -> Any:
    if "iteration" not in frame or metric not in frame:
        return None
    iterations = pd.to_numeric(frame["iteration"], errors="coerce")
    values = pd.to_numeric(frame.loc[iterations == iteration, metric], errors="coerce").dropna()
    return None if values.empty else values.iloc[-1]


def metric_definitions() -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = []

    def add(metric: str, group: str, typ: str, line: int, function: str,
            expression: str, unit: str, value_range: str, cadence: str,
            stage: str, deps: str, mechanism: str, relevance: str = "all runs",
            warning: str = "") -> None:
        definitions.append({
            "metric": metric, "conceptual_group": group, "type": typ,
            "source_file": "train.py" if line else "utils/experiment_tracking.py",
            "source_line": line or 107, "function": function,
            "source_expression": expression, "unit": unit,
            "valid_or_expected_range": value_range, "logging_cadence": cadence,
            "measurement_stage": stage, "config_dependencies": deps,
            "mechanism_that_changes_it": mechanism,
            "current_run_relevance": relevance, "interpretation_warning": warning,
        })

    base = "Every wandb_log_interval and forced validation/densification/reset/final rows"
    pre = "pre_topology_after_backward"
    add("iteration", "step", "state", 0, "ExperimentTracker.log", "iteration", "iteration", "1..iterations", "every W&B row", "log_payload", "iterations", "training loop")
    add("train/loss_total", "training_objective", "loss", 400, "training", "image_loss + lambda_flow*flow_loss + lambda_center_reg*center_reg_loss", "loss", ">=0 expected", base, pre, "all loss weights", "gradient optimization")
    add("train/loss_image", "training_objective", "loss", 392, "training", "weighted RGB+DSSIM+alpha terms", "loss", ">=0 expected", base, pre, "lambda_dssim,lambda_foreground_rgb,lambda_full_rgb,lambda_alpha", "appearance, geometry, motion, opacity")
    raw = [
        ("foreground_l1", "masked_l1_loss(image,gt,foreground_mask)", "[0,1] expected", "lambda_foreground_rgb"),
        ("full_l1", "l1_loss(image,gt)", "[0,1] expected", "lambda_full_rgb"),
        ("foreground_dssim", "1-ssim(foreground_bbox_crop)", "[0,2] theoretical", "lambda_dssim,ssim_crop_padding"),
        ("alpha_l1", "l1_loss(rendered_alpha,foreground_mask)", "[0,1] expected", "lambda_alpha"),
        ("flow", "UDFS mixture or L1_flow; zero if inactive", ">=0", "wo_flow,wo_UDFS,warmup_iteration,flow_schedule_lambda,flow_downsampling"),
        ("center_regularization", "mean((effective_center-center_initial)^2)", "[0,center_max_offset^2)", "lambda_center_reg,center_max_offset,freeze_center"),
    ]
    for name, expr, rng, deps in raw:
        add(f"train/raw/{name}", "training_objective", "loss", 465, "training", expr, "loss", rng, base, pre, deps, "render/flow/motion optimization", "zero for flow in current wo_flow runs" if name == "flow" else "all runs", "Flow is exactly zero when inactive" if name == "flow" else "")
    weighted_expr = {
        "foreground_l1": "(1-lambda_dssim)*lambda_foreground_rgb*foreground_l1",
        "full_l1": "lambda_full_rgb*full_l1",
        "foreground_dssim": "lambda_dssim*foreground_dssim",
        "alpha_l1": "lambda_alpha*alpha_l1", "flow": "lambda_flow*flow_loss",
        "center_regularization": "lambda_center_reg*center_reg_loss",
    }
    for name, expr in weighted_expr.items():
        add(f"train/weighted/{name}", "training_objective", "loss_component", 481, "training", expr, "loss", ">=0 expected", base, pre, "corresponding lambda", "corresponding raw loss")
    add("train/raw/flow_direction", "flow", "diagnostic_loss", 516, "training/cosine_flow", "1-mean(masked cosine similarity)", "loss", "[0,2]", "active flow+UDFS rows", pre, "wo_flow,wo_UDFS,flow_downsampling", "flow supervision", "absent in current wo_flow runs", "Absent rather than zero when not computed")
    add("train/raw/flow_magnitude_direction", "flow", "diagnostic_loss", 517, "training/L1_flow", "mean pixelwise L2 flow error after mask", "downsampled pixels", ">=0 unbounded", "active flow rows", pre, "wo_flow,flow_downsampling", "flow supervision", "absent in current wo_flow runs")
    add("train/flow_udfs_weight", "flow", "state", 521, "training/flow_schedule", "1 through warmup then exp(-gamma*progress)", "fraction", "(0,1]", "active flow+UDFS rows", pre, "warmup_iteration,flow_schedule_lambda,center_axis_lr_max_steps", "scheduled mixture", "absent in current wo_flow runs")
    add("performance/iteration_ms", "performance", "performance", 437, "training", "CUDA elapsed from loop start through backward", "ms", ">=0", base, pre, "resolution,flow,Gaussian count", "forward/backward workload", warning="Excludes validation, densification, optimizer, logging, checkpointing, and artifacts")
    scene = [
        ("gaussians", "N before topology update", "count", ">=1", pre),
        ("gaussians_after_update", "N after clone/split/prune", "count", ">=1", "post_topology_or_reset"),
        ("gaussian_limit", "max_gaussians", "count", ">=1", pre),
        ("visible_gaussians", "sum(radii>0)", "count", "[0,N]", pre),
        ("visible_fraction", "visible/max(1,N)", "fraction", "[0,1]", pre),
    ]
    for name, expr, unit, rng, stage in scene:
        add(f"scene/{name}", "scene_topology", "state" if name != "visible_fraction" else "diagnostic", 463, "training", expr, unit, rng, base, stage, "densification,pruning,max_gaussians,camera/model", "topology and visibility")
    for name, expr in [("opacity_mean", "mean(sigmoid(_opacity))"), ("opacity_min", "min(sigmoid(_opacity))"), ("opacity_max", "max(sigmoid(_opacity))")]:
        add(f"scene/{name}", "opacity", "state", 631, "training", expr, "opacity", "(0,1)", "validation/densification/opacity-reset rows only", "post_topology_or_reset", "opacity_lr,opacity_reset_interval,densification", "opacity optimization/reset/pruning", warning="Sparse event-triggered observations; do not interpolate as continuous measurement")
    motion = [
        ("axis_tilt_deg", "deg(atan2(axis_z,axis_y))", "degree", "configured bounds in bounded_tilt; otherwise [-180,180]", "axis_mode,axis LR,freeze_axis"),
        ("axis_side_deg", "deg(asin(axis_x))", "degree", "configured side bounds in bounded_tilt; otherwise [-90,90]", "axis_mode,axis_side_limit_deg,freeze_axis"),
        ("center_shift", "norm(effective_center-center_initial)", "world units", "[0,sqrt(3)*center_max_offset)", "center_max_offset,center_warmup_iterations,freeze_center"),
        ("sweep_error_deg", "deg(effective_sweep_error)", "degree", "[-max_sweep_error_deg,+max_sweep_error_deg]", "max_sweep_error_deg,residual LR"),
    ]
    for name, expr, unit, rng, deps in motion:
        add(f"motion/{name}", "motion", "state", 502, "training/motion_summary", expr, unit, rng, base, pre, deps, "motion optimization")
    lr_names = ["xyz", "axis", "center_point", "f_dc", "f_rest", "opacity", "scaling", "rotation", "exposure", "residual"]
    for name in lr_names:
        warning = "Exposure is unused by the render path; schedule is not causally relevant" if name == "exposure" else "Residual LR is logged before its end-of-iteration scheduler update" if name == "residual" else ""
        add(f"learning_rate/{name}", "learning_rate", "state", 507, "training", f"optimizer param-group LR for {name}", "learning rate", ">=0", base, "pre_optimizer_step", "corresponding LR config and schedule", "scheduler", warning=warning)
    density = [
        ("event", "1", "indicator", "{1}"), ("count_before", "N before update", "count", ">=1"),
        ("count_after", "N after update", "count", ">=1"), ("net_change", "count_after-count_before", "count", "integer"),
        ("limit_reached", "int(count_after>=max_gaussians)", "indicator", "{0,1}"),
        ("opacity_reset", "1", "indicator", "{1}"),
    ]
    for name, expr, unit, rng in density:
        cadence = "opacity reset rows" if name == "opacity_reset" else "densification rows"
        add(f"densification/{name}", "densification", "event" if name in {"event", "opacity_reset"} else "diagnostic", 611, "training", expr, unit, rng, cadence, "post_topology_or_reset", "densification/reset config", "clone/split/prune/reset", warning="Clone/split/prune contributions are not separately logged" if name == "net_change" else "")
    for split in ("test", "train"):
        for name, expr, unit, rng in [
            ("l1", "mean full-image L1 across views", "loss", "[0,1]"),
            ("psnr", "mean 20*log10(1/sqrt(MSE))", "dB", "[0,+inf] for clamped images"),
            ("foreground_psnr", "mean masked PSNR", "dB", "[0,+inf] for clamped images"),
            ("foreground_ssim", "mean SSIM on foreground bbox crops", "SSIM", "approximately [-1,1]"),
            ("num_views", "number of evaluated camera entries", "count", ">=1"),
        ]:
            add(f"eval/{split}/{name}", "evaluation", "evaluation", 815, "training_report", expr, unit, rng, "test_iterations", "validation_pre_topology", "test_iterations,dataset split,ssim_crop_padding", "model quality", warning="Train evaluation uses five modulo-selected entries and may repeat views" if split == "train" else "")
        add(f"media/{split}/comparisons", "media", "diagnostic_media", 768, "training_report", "up to 3 GT|render|absolute-error images", "image", "n/a", "test_iterations", "validation_pre_topology", "test_iterations", "render quality")
    add("best/eval_test_foreground_psnr", "summary", "evaluation_summary", 570, "training", "best test foreground PSNR seen so far", "dB", "[0,+inf] for clamped images", "on a new best", "run_summary", "test_iterations", "validation quality", warning="W&B summary field, not a history series")
    add("best/iteration", "summary", "state", 570, "training", "iteration of best test foreground PSNR", "iteration", "[1,iterations]", "on a new best", "run_summary", "test_iterations", "validation quality", warning="W&B summary field, not a history series")
    for metric, unit, expression in [("_step", "W&B step", "W&B internal history row index"), ("_timestamp", "Unix seconds", "W&B row timestamp"), ("_runtime", "seconds", "W&B run runtime at the row")]:
        add(metric, "wandb_internal", "provenance", 0, "W&B SDK", expression, unit, "implementation-defined", "every available history row", "wandb_internal", "W&B SDK", "tracking runtime", warning="SDK-generated rather than a RotGS application metric")
    return definitions


def scan_media(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for key, value in row.items():
            if not key.startswith("media/") or not isinstance(value, Mapping):
                continue
            filenames = value.get("filenames", [])
            captions = value.get("captions", [])
            for index, filename in enumerate(filenames):
                output.append({
                    "iteration": row.get("iteration"), "_step": row.get("_step"),
                    "metric": key, "index": index, "filename": filename,
                    "caption": captions[index] if index < len(captions) else None,
                    "width": value.get("width"), "height": value.get("height"),
                    "format": value.get("format"), "count": value.get("count"),
                })
    return output


def artifact_manifest(run: Any) -> dict[str, Any]:
    artifacts = []
    try:
        for artifact in run.logged_artifacts():
            artifacts.append({
                "id": artifact.id, "name": artifact.name, "type": artifact.type,
                "version": artifact.version, "aliases": list(artifact.aliases or []),
                "description": artifact.description, "metadata": artifact.metadata,
                "created_at": getattr(artifact, "created_at", None),
                "digest": getattr(artifact, "digest", None),
                "size": getattr(artifact, "size", None),
            })
    except Exception as exc:  # artifact absence/permissions must not abort raw export
        return {"artifacts": artifacts, "error": f"{type(exc).__name__}: {exc}"}
    return {"artifacts": artifacts}


def event_rows(run_id: str, run_name: str, config: dict[str, Any], history: pd.DataFrame) -> list[dict[str, Any]]:
    iterations = as_int(config.get("iterations"), 30000)
    densify_from = as_int(config.get("densify_from_iter"), 500)
    densify_until = as_int(config.get("densify_until_iter"), 15000)
    interval = max(1, as_int(config.get("densification_interval"), 300))
    opacity_interval = max(1, as_int(config.get("opacity_reset_interval"), 3000))
    white = as_bool(config.get("white_background"), True)
    rows: list[dict[str, Any]] = []

    def add(iteration: int, event_type: str, **extra: Any) -> None:
        rows.append({
            "run_id": run_id, "run_name": run_name, "iteration": iteration,
            "event_type": event_type, "source": "reconstructed_config",
            "logged_event_present": None, "mismatch": None,
            "active": True, "implementation_effective": True,
            "render_effective": True, **extra,
        })

    densification_iterations = [
        i for i in range(interval, min(iterations, densify_until - 1) + 1, interval)
        if i > densify_from
    ]
    for iteration in densification_iterations:
        logged = value_at(history, iteration, "densification/event")
        add(
            iteration, "densification", logged_event_present=logged is not None,
            mismatch=logged is None, count_before=value_at(history, iteration, "densification/count_before"),
            count_after=value_at(history, iteration, "densification/count_after"),
            net_change=value_at(history, iteration, "densification/net_change"),
            limit_reached=value_at(history, iteration, "densification/limit_reached"),
        )
    if densification_iterations:
        add(densification_iterations[0], "densification_start")
    add(densify_until, "densification_stop", phase_boundary=True)

    reset_iterations = set(range(opacity_interval, min(iterations, densify_until - 1) + 1, opacity_interval))
    if white and densify_from < densify_until and densify_from <= iterations:
        reset_iterations.add(densify_from)
    for iteration in sorted(reset_iterations):
        logged = value_at(history, iteration, "densification/opacity_reset")
        add(iteration, "opacity_reset", logged_event_present=logged is not None, mismatch=logged is None)

    warmup = as_int(config.get("center_warmup_iterations"), 2000)
    max_steps = as_int(config.get("center_axis_lr_max_steps"), 12000)
    center_active = not as_bool(config.get("freeze_center")) and not as_bool(config.get("wo_axis")) and float(config.get("center_max_offset", 0.25) or 0) > 0
    axis_active = not as_bool(config.get("freeze_axis")) and not as_bool(config.get("wo_axis"))
    add(warmup + 1, "center_warmup_end", active=center_active)
    add(max_steps, "center_optimization_stop", active=center_active)
    add(max_steps, "axis_optimization_stop", active=axis_active)

    if not as_bool(config.get("wo_flow")):
        flow_warmup = as_int(config.get("warmup_iteration"), 3000)
        add(flow_warmup, "flow_warmup_end")
        add(flow_warmup + 1, "flow_mixture_transition")
        add(max_steps, "flow_shutoff")
    else:
        add(1, "flow_supervision_inactive", active=False)

    local_active = not as_bool(config.get("wo_tiny"))
    sweep_active = float(config.get("max_sweep_error_deg", 0) or 0) > 0
    add(1, "local_residual_correction", active=local_active)
    add(1, "bounded_sweep_correction", active=sweep_active)
    for iteration in config.get("test_iterations", []) or []:
        if int(iteration) <= iterations:
            add(int(iteration), "validation")
    add(iterations, "final_iteration")
    sh_degree = as_int(config.get("sh_degree"), 3)
    for degree in range(1, sh_degree + 1):
        iteration = degree * 1000
        if iteration <= iterations:
            add(iteration, "nominal_sh_degree_increase", implementation_effective=False, target_degree=degree)
    residual_terminal = as_int(config.get("residual_lr_max_steps"), 15000)
    if residual_terminal <= iterations:
        add(residual_terminal, "residual_lr_terminal_value")
    xyz_terminal = as_int(config.get("position_lr_max_steps"), 30000)
    if xyz_terminal <= iterations:
        add(xyz_terminal, "xyz_lr_terminal_value")
    add(1, "exposure_lr_schedule", render_effective=False)
    return rows


def add_stat(rows: list[dict[str, Any]], run_id: str, run_name: str,
             statistic: str, value: Any, unit: str = "", context: str = "") -> None:
    rows.append({"run_id": run_id, "run_name": run_name, "statistic": statistic,
                 "value": value, "unit": unit, "context": context})


def summarize_run(run_id: str, run_name: str, config: dict[str, Any],
                  summary: dict[str, Any], history: pd.DataFrame,
                  events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    iterations = pd.to_numeric(history.get("iteration"), errors="coerce")
    densify_until = as_int(config.get("densify_until_iter"), 15000)
    before = history.loc[iterations < densify_until]
    after = history.loc[iterations >= densify_until]

    gauss = finite_series(history, "scene/gaussians_after_update")
    dens = history.loc[pd.to_numeric(history.get("densification/event"), errors="coerce").notna()] if "densification/event" in history else pd.DataFrame()
    initial = gauss.iloc[0] if not gauss.empty else None
    final = gauss.iloc[-1] if not gauss.empty else None
    add_stat(rows, run_id, run_name, "initial_gaussians", initial, "count")
    add_stat(rows, run_id, run_name, "final_gaussians", final, "count")
    last_density = finite_series(dens, "scene/gaussians_after_update")
    add_stat(rows, run_id, run_name, "gaussians_after_last_densification", last_density.iloc[-1] if not last_density.empty else None, "count")
    add_stat(rows, run_id, run_name, "gaussian_growth_absolute", None if initial is None or final is None else final - initial, "count")
    add_stat(rows, run_id, run_name, "gaussian_growth_factor", None if initial in (None, 0) or final is None else final / initial, "ratio")
    add_stat(rows, run_id, run_name, "maximum_gaussians", gauss.max() if not gauss.empty else None, "count")
    limit = float(config.get("max_gaussians", np.nan) or np.nan)
    add_stat(rows, run_id, run_name, "maximum_fraction_of_limit", gauss.max() / limit if not gauss.empty and np.isfinite(limit) and limit else None, "fraction")
    net = finite_series(dens, "densification/net_change")
    positive_net = net.loc[net > 0]
    negative_net = net.loc[net < 0]
    add_stat(rows, run_id, run_name, "number_of_densification_events", len(net), "count")
    add_stat(rows, run_id, run_name, "sum_densification_net_change", net.sum() if not net.empty else 0, "count")
    add_stat(rows, run_id, run_name, "largest_positive_net_change", positive_net.max() if not positive_net.empty else None, "count")
    add_stat(rows, run_id, run_name, "largest_negative_net_change", negative_net.min() if not negative_net.empty else None, "count")
    if not net.empty:
        idx = net.abs().idxmax()
        add_stat(rows, run_id, run_name, "iteration_of_largest_net_change", history.loc[idx, "iteration"], "iteration")
    reached = finite_series(dens, "densification/limit_reached")
    add_stat(rows, run_id, run_name, "limit_ever_reached", bool((reached == 1).any()) if not reached.empty else False, "boolean")

    visible = finite_series(history, "scene/visible_fraction")
    for name, value in [
        ("visible_fraction_initial", visible.iloc[0] if not visible.empty else None),
        ("visible_fraction_final", visible.iloc[-1] if not visible.empty else None),
        ("visible_fraction_median", visible.median() if not visible.empty else None),
        ("visible_fraction_p10", visible.quantile(.10) if not visible.empty else None),
        ("visible_fraction_p90", visible.quantile(.90) if not visible.empty else None),
        ("visible_fraction_pre_densification_end_median", finite_series(before, "scene/visible_fraction").median() if not finite_series(before, "scene/visible_fraction").empty else None),
        ("visible_fraction_post_densification_end_median", finite_series(after, "scene/visible_fraction").median() if not finite_series(after, "scene/visible_fraction").empty else None),
    ]:
        add_stat(rows, run_id, run_name, name, value, "fraction")

    reset_events = [e for e in events if e["event_type"] == "opacity_reset"]
    add_stat(rows, run_id, run_name, "number_of_opacity_resets", len(reset_events), "count")
    for metric in ("mean", "max", "min"):
        series = finite_series(history, f"scene/opacity_{metric}")
        add_stat(rows, run_id, run_name, f"opacity_{metric}_final", series.iloc[-1] if not series.empty else None, "opacity")
    opacity_obs = history.loc[finite_series(history, "scene/opacity_mean").index] if "scene/opacity_mean" in history else pd.DataFrame()
    for event in reset_events:
        it = event["iteration"]
        obs_iter = pd.to_numeric(opacity_obs.get("iteration"), errors="coerce") if not opacity_obs.empty else pd.Series(dtype=float)
        at = opacity_obs.loc[obs_iter == it]
        previous = opacity_obs.loc[obs_iter < it].tail(1)
        subsequent = opacity_obs.loc[obs_iter > it].head(1)
        for label, frame in (("at", at), ("pre", previous), ("next", subsequent)):
            for metric in ("mean", "max"):
                values = finite_series(frame, f"scene/opacity_{metric}")
                add_stat(rows, run_id, run_name, f"opacity_reset_{it}_{label}_{metric}", values.iloc[-1] if not values.empty else None, "opacity", "No interpolation; nearest actual sparse observation")
        pre_v = finite_series(previous, "scene/opacity_mean")
        at_v = finite_series(at, "scene/opacity_mean")
        next_v = finite_series(subsequent, "scene/opacity_mean")
        add_stat(rows, run_id, run_name, f"opacity_reset_{it}_drop_mean", pre_v.iloc[-1] - at_v.iloc[-1] if not pre_v.empty and not at_v.empty else None, "opacity")
        add_stat(rows, run_id, run_name, f"opacity_reset_{it}_next_recovery_mean", next_v.iloc[-1] - at_v.iloc[-1] if not next_v.empty and not at_v.empty else None, "opacity")

    for metric, prefix, unit in [
        ("motion/axis_tilt_deg", "axis_tilt", "degree"),
        ("motion/axis_side_deg", "axis_side", "degree"),
    ]:
        s = finite_series(history, metric)
        add_stat(rows, run_id, run_name, f"{prefix}_initial", s.iloc[0] if not s.empty else None, unit)
        add_stat(rows, run_id, run_name, f"{prefix}_final", s.iloc[-1] if not s.empty else None, unit)
        add_stat(rows, run_id, run_name, f"{prefix}_delta", s.iloc[-1] - s.iloc[0] if len(s) else None, unit)
    for metric, names, unit in [
        ("motion/center_shift", ("center_shift_final", "center_shift_max"), "world units"),
        ("motion/sweep_error_deg", ("sweep_error_final_deg", "sweep_error_max_abs_deg"), "degree"),
    ]:
        s = finite_series(history, metric)
        add_stat(rows, run_id, run_name, names[0], s.iloc[-1] if not s.empty else None, unit)
        add_stat(rows, run_id, run_name, names[1], s.abs().max() if not s.empty else None, unit)

    for metric in LOSS_METRICS:
        s = finite_series(history, metric)
        if s.empty:
            continue
        slug = metric.replace("/", "__")
        trailing_n = max(1, math.ceil(len(s) * .1))
        first, last = s.iloc[0], s.iloc[-1]
        add_stat(rows, run_id, run_name, f"{slug}__first", first, "loss")
        add_stat(rows, run_id, run_name, f"{slug}__final", last, "loss")
        add_stat(rows, run_id, run_name, f"{slug}__minimum", s.min(), "loss")
        add_stat(rows, run_id, run_name, f"{slug}__trailing_10pct_median", s.tail(trailing_n).median(), "loss")
        add_stat(rows, run_id, run_name, f"{slug}__relative_reduction", (first - last) / abs(first) if abs(first) > 1e-12 else None, "fraction", "Undefined for zero/nearly-zero first value")

    evaluation_stats = {
        "final_test_foreground_psnr": "eval/test/foreground_psnr",
        "final_test_foreground_ssim": "eval/test/foreground_ssim",
        "final_test_psnr": "eval/test/psnr",
        "final_train_foreground_psnr": "eval/train/foreground_psnr",
    }
    for name, metric in evaluation_stats.items():
        s = finite_series(history, metric)
        add_stat(rows, run_id, run_name, name, s.iloc[-1] if not s.empty else None, "dB" if "psnr" in name else "SSIM")
    test_fg = finite_series(history, "eval/test/foreground_psnr")
    add_stat(rows, run_id, run_name, "best_test_foreground_psnr", test_fg.max() if not test_fg.empty else summary.get("best/eval_test_foreground_psnr"), "dB")
    add_stat(rows, run_id, run_name, "best_iteration", summary.get("best/iteration"), "iteration")
    for suffix in ("foreground_psnr", "foreground_ssim"):
        test_col, train_col = f"eval/test/{suffix}", f"eval/train/{suffix}"
        paired = history[[test_col, train_col]].apply(pd.to_numeric, errors="coerce").dropna() if test_col in history and train_col in history else pd.DataFrame()
        unit = "dB" if suffix.endswith("psnr") else "SSIM"
        add_stat(rows, run_id, run_name, f"train_test_{suffix}_gap_final", paired[train_col].iloc[-1] - paired[test_col].iloc[-1] if not paired.empty else None, unit, "train minus test at matched logged evaluation")
    if not test_fg.empty:
        eval_rows = history.loc[test_fg.index]
        initial_psnr, best_psnr = test_fg.iloc[0], test_fg.max()
        threshold = initial_psnr + .95 * (best_psnr - initial_psnr)
        candidates = eval_rows.loc[pd.to_numeric(eval_rows["eval/test/foreground_psnr"], errors="coerce") >= threshold]
        conv = pd.to_numeric(candidates["iteration"], errors="coerce").iloc[0] if not candidates.empty else None
        add_stat(rows, run_id, run_name, "first_logged_eval_iteration_reaching_95pct_best_improvement", conv, "iteration", f"threshold={threshold}; discrete logged evaluations only")

    perf = finite_series(history, "performance/iteration_ms")
    for name, value in [
        ("iteration_ms_median", perf.median() if not perf.empty else None),
        ("iteration_ms_p25", perf.quantile(.25) if not perf.empty else None),
        ("iteration_ms_p75", perf.quantile(.75) if not perf.empty else None),
        ("iteration_ms_p90", perf.quantile(.90) if not perf.empty else None),
        ("iteration_ms_p95", perf.quantile(.95) if not perf.empty else None),
        ("iteration_ms_pre_densification_end_median", finite_series(before, "performance/iteration_ms").median() if not finite_series(before, "performance/iteration_ms").empty else None),
        ("iteration_ms_post_densification_end_median", finite_series(after, "performance/iteration_ms").median() if not finite_series(after, "performance/iteration_ms").empty else None),
    ]:
        add_stat(rows, run_id, run_name, name, value, "ms", "CUDA forward/backward timing only")
    add_stat(rows, run_id, run_name, "wandb_total_runtime_seconds", summary.get("_runtime", (summary.get("_wandb") or {}).get("runtime") if isinstance(summary.get("_wandb"), dict) else None), "seconds")

    warm_end = as_int(config.get("center_warmup_iterations"), 2000)
    axis_stop = as_int(config.get("center_axis_lr_max_steps"), 12000)
    phases = [("phase1_initial_center_warmup", 1, min(warm_end, as_int(config.get("iterations"), 30000)))]
    phase2_end = min(axis_stop - 1, densify_until - 1)
    if warm_end + 1 <= phase2_end:
        phases.append(("phase2_center_axis_and_densification", warm_end + 1, phase2_end))
    if axis_stop < densify_until:
        phases.append(("phase3_motion_frozen_densification_active", axis_stop, densify_until - 1))
    if densify_until <= as_int(config.get("iterations"), 30000):
        phases.append(("phase4_post_densification_refinement", densify_until, as_int(config.get("iterations"), 30000)))
    for phase, start, end in phases:
        subset = history.loc[(iterations >= start) & (iterations <= end)]
        for metric in PHASE_METRICS:
            s = finite_series(subset, metric)
            if not s.empty:
                slug = metric.replace("/", "__")
                add_stat(rows, run_id, run_name, f"{phase}__{slug}__median", s.median(), "metric units", f"iterations {start}..{end}")
                add_stat(rows, run_id, run_name, f"{phase}__{slug}__change", s.iloc[-1] - s.iloc[0], "metric units", f"last minus first logged value; iterations {start}..{end}")
    return rows


def quality_checks(run_id: str, run_name: str, config: dict[str, Any], history: pd.DataFrame,
                   events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    def check(name: str, ok: bool | None, details: str) -> None:
        output.append({"run_id": run_id, "run_name": run_name, "check": name,
                       "status": "PASS" if ok is True else "FAIL" if ok is False else "WARN",
                       "details": details})
    it = pd.to_numeric(history.get("iteration"), errors="coerce").dropna()
    check("monotonically_increasing_iteration", bool(it.is_monotonic_increasing), f"rows={len(it)}")
    duplicates = int(it.duplicated().sum())
    check("duplicate_iteration_rows", duplicates == 0, f"duplicates={duplicates}")
    interval = as_int(config.get("wandb_log_interval"), 10)
    regular_expected = set(range(interval, as_int(config.get("iterations"), 30000) + 1, interval))
    missing = sorted(regular_expected - set(it.astype(int)))
    check("missing_regular_history_intervals", len(missing) == 0, f"missing_count={len(missing)}; first={missing[:20]}")
    decreases = int((it.diff().dropna() <= 0).sum())
    check("restart_resume_discontinuity", decreases == 0, f"nonpositive_iteration_deltas={decreases}")
    for metric, event_type, check_name in (("densification/event", "densification", "reconstructed_densification_vs_logged"), ("densification/opacity_reset", "opacity_reset", "reconstructed_opacity_reset_vs_logged")):
        expected = {int(e["iteration"]) for e in events if e["event_type"] == event_type}
        logged = set(pd.to_numeric(history.loc[finite_series(history, metric).index, "iteration"], errors="coerce").dropna().astype(int)) if metric in history else set()
        missing, extra = sorted(expected - logged), sorted(logged - expected)
        check(check_name, not missing and not extra, f"missing_logged_markers={missing}; unexpected_logged_markers={extra}")
    final_iteration = as_int(config.get("iterations"), 30000)
    check("final_iteration_present", final_iteration in set(it.astype(int)), f"configured_final={final_iteration}; observed_max={it.max() if not it.empty else None}")
    actual_eval = set(pd.to_numeric(history.loc[finite_series(history, "eval/test/foreground_psnr").index, "iteration"], errors="coerce").dropna().astype(int)) if "eval/test/foreground_psnr" in history else set()
    expected_eval = {int(x) for x in config.get("test_iterations", []) or [] if int(x) <= final_iteration}
    check("configured_test_iterations_vs_eval_rows", actual_eval == expected_eval, f"expected={sorted(expected_eval)}; actual={sorted(actual_eval)}")
    density_rows = history.loc[finite_series(history, "densification/event").index] if "densification/event" in history else pd.DataFrame()
    consistency = True
    detail = []
    for _, row in density_rows.iterrows():
        before = row.get("densification/count_before")
        after = row.get("densification/count_after")
        scene_before = row.get("scene/gaussians")
        scene_after = row.get("scene/gaussians_after_update")
        if pd.notna(before) and pd.notna(scene_before) and float(before) != float(scene_before):
            consistency = False; detail.append(f"i={row.get('iteration')}:before")
        if pd.notna(after) and pd.notna(scene_after) and float(after) != float(scene_after):
            consistency = False; detail.append(f"i={row.get('iteration')}:after")
    check("gaussian_count_consistency", consistency, ";".join(detail[:20]) or "count_before/after match scene counts")
    for metric in ("scene/visible_fraction", "scene/opacity_mean", "scene/opacity_min", "scene/opacity_max"):
        s = finite_series(history, metric)
        check(f"range_{metric}", bool(((s >= 0) & (s <= 1)).all()), f"min={s.min() if not s.empty else None}; max={s.max() if not s.empty else None}")
    center = finite_series(history, "motion/center_shift")
    center_bound = math.sqrt(3) * float(config.get("center_max_offset", .25) or 0)
    check("center_shift_bound", bool((center <= center_bound + 1e-6).all()), f"max={center.max() if not center.empty else None}; bound={center_bound}")
    sweep = finite_series(history, "motion/sweep_error_deg")
    sweep_bound = float(config.get("max_sweep_error_deg", 0) or 0)
    check("sweep_error_bound", bool((sweep.abs() <= sweep_bound + 1e-5).all()), f"max_abs={sweep.abs().max() if not sweep.empty else None}; bound={sweep_bound}")
    if config.get("axis_mode") == "bounded_tilt":
        tilt, side = finite_series(history, "motion/axis_tilt_deg"), finite_series(history, "motion/axis_side_deg")
        lo, hi = float(config.get("axis_tilt_min_deg", 0)), float(config.get("axis_tilt_max_deg", 90))
        side_lim = float(config.get("axis_side_limit_deg", 5))
        check("bounded_axis_tilt", bool(((tilt >= lo - 1e-5) & (tilt <= hi + 1e-5)).all()), f"observed=[{tilt.min()},{tilt.max()}]; configured=[{lo},{hi}]")
        check("bounded_axis_side", bool((side.abs() <= side_lim + 1e-5).all()), f"max_abs={side.abs().max()}; configured={side_lim}")
    flow = finite_series(history, "train/raw/flow")
    detailed_present = any(col in history and finite_series(history, col).size for col in ("train/raw/flow_direction", "train/raw/flow_magnitude_direction", "train/flow_udfs_weight"))
    if as_bool(config.get("wo_flow")):
        check("flow_metrics_consistent_with_wo_flow", bool((flow.fillna(0) == 0).all()) and not detailed_present, f"raw_flow_nonzero={int((flow != 0).sum())}; detailed_present={detailed_present}")
    if as_int(config.get("dataset_session"), -1) == 3:
        expected_queue = {"iterations": 30000, "densify_until_iter": 13000, "fixed_camera": True, "axis_mode": "bounded_tilt", "wo_tiny": True, "wo_flow": True, "max_sweep_error_deg": 4}
        mismatches = {key: {"expected": expected, "actual": config.get(key)} for key, expected in expected_queue.items() if config.get(key) != expected}
        check("current_session03_queue_config", not mismatches, f"verified={not mismatches}; mismatches={mismatches}")
    required = ["train/loss_total", "scene/gaussians_after_update", "scene/visible_fraction", "performance/iteration_ms"]
    missing_metrics = [m for m in required if m not in history or finite_series(history, m).empty]
    check("missing_expected_metrics", not missing_metrics, f"missing={missing_metrics}")
    return output


def plot_lines(histories: dict[str, pd.DataFrame], names: dict[str, str], metric: str,
               title: str, ylabel: str, out: Path, markers: bool = False, verticals: list[tuple[int, str]] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2))
    plotted = False
    for run_id, frame in histories.items():
        if metric not in frame:
            continue
        data = frame[["iteration", metric]].apply(pd.to_numeric, errors="coerce").dropna()
        if data.empty:
            continue
        ax.plot(data["iteration"], data[metric], lw=1.15, marker="o" if markers else None,
                ms=3 if markers else None, label=names[run_id])
        if metric == "scene/gaussians_after_update" and "densification/event" in frame:
            event_mask = pd.to_numeric(frame["densification/event"], errors="coerce").notna()
            event_data = frame.loc[event_mask, ["iteration", metric]].apply(pd.to_numeric, errors="coerce").dropna()
            ax.scatter(event_data["iteration"], event_data[metric], color=ax.lines[-1].get_color(), s=8, alpha=.7)
        plotted = True
    if not plotted:
        plt.close(fig); return
    for boundary_iteration, boundary_label in verticals or []:
        ax.axvline(boundary_iteration, color="black", lw=.7, alpha=.35, linestyle="--")
        ax.text(boundary_iteration, .99, boundary_label, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=6, alpha=.7)
    ax.set(title=title, xlabel="Iteration", ylabel=ylabel)
    ax.grid(alpha=.25)
    ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(fig)


def plot_train_test_matched(histories: dict[str, pd.DataFrame], names: dict[str, str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    colors = plt.cm.tab20(np.linspace(0, 1, len(histories)))
    plotted = False
    for color, (run_id, frame) in zip(colors, histories.items()):
        columns = ["iteration", "eval/train/foreground_psnr", "eval/test/foreground_psnr"]
        if not all(column in frame for column in columns):
            continue
        data = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
        if data.empty:
            continue
        ax.plot(data["iteration"], data[columns[1]], color=color, linestyle="--", marker="o", ms=3, lw=1, label=f"{names[run_id]} train")
        ax.plot(data["iteration"], data[columns[2]], color=color, linestyle="-", marker="o", ms=3, lw=1, label=f"{names[run_id]} test")
        plotted = True
    if plotted:
        ax.set(title="Matched train/test foreground PSNR", xlabel="Iteration", ylabel="dB")
        ax.grid(alpha=.25); ax.legend(fontsize=5.5, ncol=2); fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(out.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(fig)



def plot_weighted_components(histories: dict[str, pd.DataFrame], names: dict[str, str], out: Path) -> None:
    metrics = ["train/weighted/foreground_l1", "train/weighted/foreground_dssim",
               "train/weighted/full_l1", "train/weighted/alpha_l1",
               "train/weighted/center_regularization"]
    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)
    for ax, metric in zip(axes.flat, metrics):
        for run_id, frame in histories.items():
            if metric in frame:
                data = frame[["iteration", metric]].apply(pd.to_numeric, errors="coerce").dropna()
                ax.plot(data["iteration"], data[metric], lw=.9, label=names[run_id])
        ax.set_title(metric, fontsize=9); ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=6.5, ncol=2)
    fig.suptitle("Weighted loss components (raw, unsmoothed W&B history)")
    fig.tight_layout(rect=(0, .05, 1, .97))
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(fig)


def plot_bar(cross: pd.DataFrame, metric: str, title: str, ylabel: str, out: Path) -> None:
    data = cross[["run_name", metric]].dropna().sort_values(metric)
    if data.empty: return
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.barh(data["run_name"], data[metric]); ax.set(title=title, xlabel=ylabel)
    ax.grid(axis="x", alpha=.25); fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(fig)


def plot_scatter(cross: pd.DataFrame, x: str, y: str, title: str, out: Path) -> None:
    data = cross[["run_id", "run_name", x, y]].dropna()
    if len(data) < 2: return
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(data[x], data[y], s=38)
    for _, row in data.iterrows():
        ax.annotate(row["run_name"], (row[x], row[y]), xytext=(4, 3), textcoords="offset points", fontsize=6.5)
    ax.set(title=title, xlabel=x.replace("_", " "), ylabel=y.replace("_", " "))
    ax.grid(alpha=.25); fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(fig)


def parse_cfg_args(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text.startswith("Namespace(") or not text.endswith(")"):
        return {"raw": text}
    expression = ast.parse(text, mode="eval").body
    if not isinstance(expression, ast.Call):
        return {"raw": text}
    return {kw.arg: ast.literal_eval(kw.value) for kw in expression.keywords if kw.arg}


def copy_bundle_files(report_root: Path) -> None:
    bundle = report_root / "chat_analysis_bundle"
    if bundle.exists(): shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    names = [
        "analysis_context.md", "export_wandb_analysis.py", "run_selection.csv", "run_metadata.csv",
        "run_configs.json", "training_events.csv", "metric_definitions.csv",
        "metric_summary.csv", "cross_run_metrics.csv",
        "cross_run_relationships.csv", "data_quality_report.csv",
        "code_metric_map.md", "local_enrichment.csv", "timeseries_analysis.csv",
        "external_eval_metrics.csv",
    ]
    for name in names:
        source = report_root / name
        if source.exists(): shutil.copy2(source, bundle / name)
    shutil.copytree(report_root / "figures", bundle / "figures")
    archive = report_root / "chat_analysis_bundle.zip"
    if archive.exists(): archive.unlink()
    shutil.make_archive(str(archive.with_suffix("")), "zip", bundle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="3D-Scanning-MT")
    parser.add_argument("--project", default="RotGS")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--system-samples", type=int, default=100000,
                        help="Upper bound for W&B events-stream export; API marks this path sampled")
    args = parser.parse_args()
    report_root = args.output.resolve()
    full_export = report_root / "full_export"
    figures = report_root / "figures"
    for generated in (full_export, figures, report_root / "chat_analysis_bundle"):
        if generated.exists(): shutil.rmtree(generated)
        generated.mkdir(parents=True)

    exported_at = datetime.now(timezone.utc).isoformat()
    provenance = repository_provenance()
    api = wandb.Api()
    project = api.project(args.project, entity=args.entity)
    entity, project_name = project.entity, project.name
    runs = list(api.runs(f"{entity}/{project_name}"))
    runs.sort(key=lambda run: (run.created_at or "", run.id))

    selection_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    all_configs: dict[str, Any] = {}
    selected: list[Any] = []
    for run in runs:
        config = jsonable(dict(run.config or {}))
        summary = jsonable(dict(run.summary or {}))
        runtime = summary.get("_runtime")
        if runtime is None and isinstance(summary.get("_wandb"), dict): runtime = summary["_wandb"].get("runtime")
        is_selected = run.state == "finished"
        reason = "" if is_selected else f"state={run.state}; default analysis requires completed runs"
        if is_selected: selected.append(run)
        attrs = run._attrs
        created = attrs.get("createdAt") or run.created_at
        ended = attrs.get("heartbeatAt") if run.state in {"finished", "failed", "crashed", "killed"} else None
        row = {
            "run_id": run.id, "run_name": run.name, "display_name": run.name,
            "state": run.state, "created_at": created, "started_at": created,
            "ended_at": ended, "runtime_seconds": runtime,
            "group": run.group, "tags": list(run.tags or []), "notes": run.notes,
            "job_type": attrs.get("jobType"), "sweep_id": attrs.get("sweepName"),
            "model_path": config.get("model_path"), "dataset_id": config.get("dataset_id"),
            "dataset_session": config.get("dataset_session"),
            "preprocessing_variant": config.get("preprocessing_variant"),
            "git_commit": config.get("git_commit"), "git_dirty": config.get("git_dirty"),
            "iterations": config.get("iterations"), "selected_for_analysis": is_selected,
            "exclusion_reason": reason,
        }
        selection_rows.append(row)
        metadata_rows.append({**row, "url": run.url, "history_line_count": attrs.get("historyLineCount"),
                              "timestamp_semantics": "W&B public API exposes createdAt and heartbeatAt; started_at uses createdAt and terminal ended_at uses heartbeatAt"})
        all_configs[run.id] = {"run_name": run.name, "selected_for_analysis": is_selected,
                               "full_config": config,
                               "normalized_config": {key: config.get(key) for key in NORMALIZED_CONFIG_FIELDS}}
    write_rows(report_root / "run_selection.csv", selection_rows)
    write_rows(report_root / "run_metadata.csv", metadata_rows)
    write_json(report_root / "run_configs.json", all_configs)

    histories: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    names: dict[str, str] = {}
    manifest_runs: dict[str, Any] = {}
    for index, run in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] Exporting {run.id} {run.name}", flush=True)
        run_dir = full_export / "runs" / run.id
        run_dir.mkdir(parents=True, exist_ok=True)
        config = jsonable(dict(run.config or {})); summary = jsonable(dict(run.summary or {}))
        write_json(run_dir / "config.json", config); write_json(run_dir / "summary.json", summary)
        history_rows = [jsonable(dict(row)) for row in run.scan_history(page_size=1000, use_cache=False)]
        write_rows(run_dir / "history.csv", history_rows)
        system_error = None
        try:
            system_rows = [jsonable(dict(row)) for row in run.history(stream="events", samples=args.system_samples, pandas=False)]
        except Exception as exc:
            system_rows = []; system_error = f"{type(exc).__name__}: {exc}"
        write_rows(run_dir / "system_metrics.csv", system_rows)
        media_rows = scan_media(history_rows); write_rows(run_dir / "media_manifest.csv", media_rows)
        artifacts = artifact_manifest(run); write_json(run_dir / "artifact_manifest.json", artifacts)
        frame = pd.DataFrame(history_rows)
        histories[run.id] = frame; summaries[run.id] = summary; names[run.id] = run.name
        manifest_runs[run.id] = {
            "run_name": run.name, "history_rows": len(history_rows),
            "history_columns": sorted({key for row in history_rows for key in row}),
            "system_metric_rows": len(system_rows),
            "system_metric_columns": sorted({key for row in system_rows for key in row}),
            "system_metrics_error": system_error, "media_manifest_rows": len(media_rows),
        }

    event_output: list[dict[str, Any]] = []
    quality_output: list[dict[str, Any]] = []
    summary_output: list[dict[str, Any]] = []
    timeseries_frames: list[pd.DataFrame] = []
    for run in selected:
        frame = histories[run.id]; config = dict(run.config or {})
        events = event_rows(run.id, run.name, config, frame)
        event_output.extend(events)
        quality_output.extend(quality_checks(run.id, run.name, config, frame, events))
        summary_output.extend(summarize_run(run.id, run.name, config, summaries[run.id], frame, events))
        copy = frame.copy(); copy.insert(0, "run_name", run.name); copy.insert(0, "run_id", run.id)
        timeseries_frames.append(copy)
    write_rows(report_root / "training_events.csv", event_output)
    write_rows(report_root / "data_quality_report.csv", quality_output)
    write_rows(report_root / "metric_summary.csv", summary_output)
    timeseries = pd.concat(timeseries_frames, ignore_index=True, sort=False)
    timeseries.to_csv(report_root / "timeseries_analysis.csv", index=False)

    definitions = metric_definitions()
    write_rows(report_root / "metric_definitions.csv", definitions)
    md = ["# RotGS code-to-metric map", "", "Generated from the audited single-camera training implementation.", ""]
    for definition in definitions:
        md.extend([f"## `{definition['metric']}`", "",
                   f"- Source: `{definition['source_file']}:{definition['source_line']}` — `{definition['function']}`",
                   f"- Expression: `{definition['source_expression']}`",
                   f"- Stage/cadence: {definition['measurement_stage']}; {definition['logging_cadence']}",
                   f"- Dependencies: {definition['config_dependencies']}",
                   f"- Warning: {definition['interpretation_warning'] or 'None specific.'}", ""])
    (report_root / "code_metric_map.md").write_text("\n".join(md), encoding="utf-8")

    enrichment: list[dict[str, Any]] = []
    external: list[dict[str, Any]] = []
    output_root = Path("/shared/datasets/3D_Scanning_MT/output_RotGS/session_03")
    for run in selected:
        config = dict(run.config or {}); model = Path(str(config.get("model_path", "")))
        if not model.is_absolute(): model = (REPO_ROOT / model).resolve()
        id_file = model / "wandb_run_id"
        local_id = id_file.read_text().strip() if id_file.is_file() else None
        exact = local_id == run.id
        for field, value, source, method in [
            ("resolved_model_path", str(model), model, "W&B config model_path"),
            ("local_wandb_run_id", local_id, id_file, "exact run-ID file match" if exact else "run-ID mismatch or unavailable"),
        ]:
            enrichment.append({"run_id": run.id, "field": field, "value": value,
                               "source_file": str(source), "confidence": "exact" if exact else "config_only",
                               "association_method": method,
                               "wandb_value": config.get(field), "discrepancy": bool(field == "local_wandb_run_id" and local_id and local_id != run.id)})
        if exact:
            cfg_path = model / "cfg_args"
            if cfg_path.is_file():
                local_cfg = parse_cfg_args(cfg_path)
                enrichment.append({"run_id": run.id, "field": "cfg_args", "value": local_cfg,
                                   "source_file": str(cfg_path), "confidence": "exact",
                                   "association_method": "model_path plus exact wandb_run_id",
                                   "wandb_value": dict(run.config or {}),
                                   "discrepancy": local_cfg != dict(run.config or {})})
                source_path = local_cfg.get("source_path")
                if source_path:
                    prep = Path(source_path) / "preprocessing_metadata.json"
                    if prep.is_file():
                        enrichment.append({"run_id": run.id, "field": "preprocessing_metadata",
                                           "value": json.loads(prep.read_text()), "source_file": str(prep),
                                           "confidence": "exact", "association_method": "cfg_args source_path",
                                           "wandb_value": config.get("preprocessing_variant"), "discrepancy": False})
            for motion in sorted(model.glob("point_cloud/iteration_*/motion.json")):
                enrichment.append({"run_id": run.id, "field": f"motion_{motion.parent.name}",
                                   "value": json.loads(motion.read_text()), "source_file": str(motion),
                                   "confidence": "exact", "association_method": "model_path plus exact wandb_run_id",
                                   "wandb_value": None, "discrepancy": False})
            result_path = model / "results.json"
            if result_path.is_file():
                result_data = json.loads(result_path.read_text())
                for checkpoint, metrics in result_data.items():
                    match = re.search(r"(\d+)$", checkpoint)
                    checkpoint_iteration = int(match.group(1)) if match else None
                    if isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            if metric.upper() in {"LPIPS", "PSNR", "SSIM"}:
                                external.append({"run_id": run.id, "run_name": run.name,
                                                 "checkpoint_iteration": checkpoint_iteration,
                                                 "split": "test", "metric": metric.upper(), "value": value,
                                                 "source_file": str(result_path),
                                                 "association_method": "model_path plus exact wandb_run_id"})
    write_rows(report_root / "local_enrichment.csv", enrichment)
    write_rows(report_root / "external_eval_metrics.csv", external)

    summary_df = pd.DataFrame(summary_output)
    pivot = summary_df.pivot_table(index=["run_id", "run_name"], columns="statistic", values="value", aggfunc="last").reset_index()
    for source, target in (("axis_tilt_delta", "abs_axis_tilt_delta"), ("axis_side_delta", "abs_axis_side_delta"), ("sweep_error_final_deg", "abs_sweep_error_final_deg")):
        if source in pivot:
            pivot[target] = pd.to_numeric(pivot[source], errors="coerce").abs()
    if external:
        ext = pd.DataFrame(external).pivot_table(index="run_id", columns="metric", values="value", aggfunc="last").add_prefix("external_test_").reset_index()
        pivot = pivot.merge(ext, on="run_id", how="left")
    pivot.to_csv(report_root / "cross_run_metrics.csv", index=False)
    relationships = [
        ("final_gaussians", "best_test_foreground_psnr"),
        ("final_gaussians", "final_test_foreground_ssim"),
        ("final_gaussians", "external_test_LPIPS"),
        ("final_gaussians", "wandb_total_runtime_seconds"),
        ("final_gaussians", "iteration_ms_post_densification_end_median"),
        ("gaussian_growth_factor", "best_test_foreground_psnr"),
        ("visible_fraction_median", "best_test_foreground_psnr"),
        ("center_shift_final", "best_test_foreground_psnr"),
        ("center_shift_max", "best_test_foreground_psnr"),
        ("abs_axis_tilt_delta", "best_test_foreground_psnr"),
        ("abs_axis_side_delta", "best_test_foreground_psnr"),
        ("abs_sweep_error_final_deg", "best_test_foreground_psnr"),
        ("wandb_total_runtime_seconds", "best_test_foreground_psnr"),
    ]
    relationship_rows = []
    for x, y in relationships:
        if x not in pivot or y not in pivot: continue
        data = pivot[["run_id", "run_name", x, y]].copy()
        data[x] = pd.to_numeric(data[x], errors="coerce"); data[y] = pd.to_numeric(data[y], errors="coerce")
        data = data.dropna()
        pearson = pearsonr(data[x], data[y]).statistic if len(data) >= 2 and data[x].nunique() > 1 and data[y].nunique() > 1 else None
        spearman = spearmanr(data[x], data[y]).statistic if len(data) >= 2 and data[x].nunique() > 1 and data[y].nunique() > 1 else None
        relationship_rows.append({"x_metric": x, "y_metric": y, "n": len(data),
                                  "pearson_r": pearson, "spearman_rho": spearman,
                                  "interpretation": "descriptive/exploratory only; no causal or significance claim",
                                  "raw_points_json": data.to_dict(orient="records")})
    write_rows(report_root / "cross_run_relationships.csv", relationship_rows)

    def unique_boundaries(field: str, offset: int, label: str) -> list[tuple[int, str]]:
        values = sorted({as_int(dict(run.config or {}).get(field), 0) + offset for run in selected})
        return [(value, label if len(values) == 1 else f"{label} ({value})") for value in values]
    density_boundaries = unique_boundaries("densify_until_iter", 0, "densification stop")
    opacity_reset_iterations: set[int] = set()
    for run in selected:
        config = dict(run.config or {})
        reset_interval = max(1, as_int(config.get("opacity_reset_interval"), 3000))
        reset_stop = min(as_int(config.get("iterations"), 30000), as_int(config.get("densify_until_iter"), 15000) - 1)
        opacity_reset_iterations.update(range(reset_interval, reset_stop + 1, reset_interval))
        if as_bool(config.get("white_background"), True):
            density_start = as_int(config.get("densify_from_iter"), 500)
            if density_start <= reset_stop:
                opacity_reset_iterations.add(density_start)
    opacity_boundaries = [(value, "opacity reset") for value in sorted(opacity_reset_iterations)]
    motion_boundaries = (unique_boundaries("center_warmup_iterations", 1, "center warmup end") + unique_boundaries("center_axis_lr_max_steps", 0, "center/axis stop") + density_boundaries)
    plot_specs = [
        ("train/loss_total", "1a_loss_total", "Training total loss", "Loss", False),
        ("train/loss_image", "1b_loss_image", "Training image loss", "Loss", False),
        ("scene/gaussians_after_update", "2a_gaussians", "Gaussian population after topology update", "Gaussians", False),
        ("densification/net_change", "2b_densification_net_change", "Densification net population change", "Gaussians", True),
        ("scene/visible_fraction", "3a_visible_fraction", "Visible Gaussian fraction", "Fraction", False),
        ("scene/visible_gaussians", "3b_visible_gaussians", "Visible Gaussian count", "Gaussians", False),
        ("scene/opacity_mean", "4a_opacity_mean", "Sparse opacity mean observations", "Opacity", True),
        ("scene/opacity_min", "4b_opacity_min", "Sparse opacity minimum observations", "Opacity", True),
        ("scene/opacity_max", "4c_opacity_max", "Sparse opacity maximum observations", "Opacity", True),
        ("motion/axis_tilt_deg", "5a_axis_tilt", "Rotation-axis tilt", "Degrees", False),
        ("motion/axis_side_deg", "5b_axis_side", "Rotation-axis sideways angle", "Degrees", False),
        ("motion/center_shift", "5c_center_shift", "Rotation-center shift", "World units", False),
        ("motion/sweep_error_deg", "5d_sweep_error", "Bounded sweep correction", "Degrees", False),
        ("learning_rate/axis", "5e_axis_lr", "Axis learning rate", "Learning rate", False),
        ("learning_rate/center_point", "5f_center_lr", "Center learning rate", "Learning rate", False),
        ("eval/test/foreground_psnr", "6a_test_fg_psnr", "Test foreground PSNR", "dB", True),
        ("eval/test/foreground_ssim", "6b_test_fg_ssim", "Test foreground SSIM", "SSIM", True),
        ("eval/train/foreground_psnr", "6c_train_fg_psnr", "Train foreground PSNR", "dB", True),
        ("performance/iteration_ms", "7a_iteration_ms", "CUDA forward/backward iteration time", "ms", False),
    ]
    for metric, filename, title, ylabel, markers in plot_specs:
        plot_lines(histories, names, metric, title, ylabel, figures / filename, markers, motion_boundaries if filename.startswith("5") else density_boundaries if filename == "2a_gaussians" else opacity_boundaries if filename.startswith("4") else None)
    plot_weighted_components(histories, names, figures / "1c_weighted_components")
    plot_train_test_matched(histories, names, figures / "6c_train_vs_test_fg_psnr")
    plot_bar(pivot, "final_gaussians", "Final Gaussian count by run", "Gaussians", figures / "2c_final_gaussians")
    scatter_specs = [
        ("final_gaussians", "best_test_foreground_psnr"),
        ("final_gaussians", "final_test_foreground_ssim"),
        ("final_gaussians", "wandb_total_runtime_seconds"),
        ("final_gaussians", "iteration_ms_post_densification_end_median"),
        ("visible_fraction_median", "best_test_foreground_psnr"),
        ("center_shift_final", "best_test_foreground_psnr"),
        ("center_shift_max", "best_test_foreground_psnr"),
        ("abs_sweep_error_final_deg", "best_test_foreground_psnr"),
        ("wandb_total_runtime_seconds", "best_test_foreground_psnr"),
        ("final_gaussians", "external_test_LPIPS"),
        ("wandb_total_runtime_seconds", "external_test_LPIPS"),
    ]
    for index, (x, y) in enumerate(scatter_specs, 1):
        if x in pivot and y in pivot:
            plot_scatter(pivot, x, y, f"{x.replace('_',' ')} vs {y.replace('_',' ')}", figures / f"8_{index:02d}_{x}_vs_{y}")
    # Matched iteration-time/Gaussian observations.
    fig, ax = plt.subplots(figsize=(8.5, 5.5)); plotted = False
    for run_id, frame in histories.items():
        cols = ["scene/gaussians_after_update", "performance/iteration_ms"]
        if all(c in frame for c in cols):
            data = frame[cols].apply(pd.to_numeric, errors="coerce").dropna()
            if not data.empty:
                ax.scatter(data[cols[0]], data[cols[1]], s=5, alpha=.35, label=names[run_id]); plotted = True
    if plotted:
        ax.set(title="Matched iteration time versus Gaussian count", xlabel="Gaussians after update", ylabel="CUDA forward/backward ms")
        ax.grid(alpha=.25); ax.legend(fontsize=6, ncol=2); fig.tight_layout()
        for suffix in ("png", "pdf"): fig.savefig((figures / "7b_iteration_ms_vs_gaussians").with_suffix(f".{suffix}"), dpi=220 if suffix == "png" else None)
    plt.close(fig)
    plot_bar(pivot, "wandb_total_runtime_seconds", "Total W&B runtime by run", "Seconds", figures / "7c_total_runtime")

    selected_ids = [run.id for run in selected]
    manifest = {
        "entity": entity, "project": project_name, "export_timestamp_utc": exported_at,
        **provenance, "exporter_version": SCRIPT_VERSION,
        "exporter_script": str(Path(__file__).resolve()),
        "exporter_command": " ".join([sys.executable, *sys.argv]),
        "python_version": platform.python_version(), "wandb_version": wandb.__version__,
        "queried_run_ids": [run.id for run in runs], "selected_run_ids": selected_ids,
        "canonical_history_api": "Run.scan_history(page_size=1000,use_cache=False)",
        "canonical_history_sampled": False,
        "any_wandb_api_call_sampled": True,
        "sampling_note": "Only the separate W&B events/system-metrics endpoint uses history(samples=100000). Canonical scalar/event history and timeseries_analysis are complete scan_history output.",
        "timeseries_sampling": "full", "runs": manifest_runs,
    }
    write_json(full_export / "export_manifest.json", manifest)

    excluded = [r for r in selection_rows if not r["selected_for_analysis"]]
    context = f"""# RotGS W&B analysis dataset

## Purpose

This dataset supports code-grounded analysis of RotGS training dynamics, scheduled events, reconstruction quality, and computational cost. It deliberately separates raw observations, reconstructed events, local enrichment, derived statistics, and hypotheses. No scientific prose or causal conclusions are generated here.

## Export identity

- W&B project: `{entity}/{project_name}`
- Exported UTC: `{exported_at}`
- Repository commit: `{provenance['git_commit']}`
- Repository dirty: `{provenance['git_dirty']}`
- Tracked-diff SHA-256: `{provenance['git_diff_sha256']}`
- Python: `{platform.python_version()}`
- wandb: `{wandb.__version__}`
- Selected finished runs: {len(selected)} ({', '.join(selected_ids)})
- Excluded runs: {len(excluded)} ({'; '.join(f"{r['run_id']} {r['exclusion_reason']}" for r in excluded)})
- `timeseries_sampling = full`
- Canonical history sampled: false
- Any API call sampled: true, solely because W&B's separate system-events API accepts a `samples` argument. It was requested with an upper bound of {args.system_samples}; this does not affect canonical history.

## Source classes

1. `full_export/runs/<run_id>/history.csv`: directly observed, complete unsmoothed W&B scalar/event history from `scan_history`; source of truth.
2. `training_events.csv`: deterministic reconstruction from each run's config, with logged-marker comparisons.
3. `local_enrichment.csv` and `external_eval_metrics.csv`: local files joined only through model path plus exact `wandb_run_id` where scientific values are involved.
4. `metric_summary.csv`, `cross_run_metrics.csv`, and `cross_run_relationships.csv`: reproducible derived values.
5. Hypotheses are intentionally absent.

## Measurement and implementation semantics

Within an iteration the order is: forward/loss, backward, base metric capture, optional validation, topology/reset, post-topology metric capture, optimizer step, W&B log. Losses and `scene/gaussians` are pre-topology; `scene/gaussians_after_update` and opacity statistics are post-topology/reset.

Current Session-03 configs are verified per run in `run_configs.json`. Flow-disabled runs have zero aggregate flow and absent detailed flow metrics; local residual correction can be off while bounded sweep correction remains active. Nominal SH increments are marked `implementation_effective=false` because the audited rasterizer captures degree zero once. Exposure schedule is marked `render_effective=false` because exposure is unused. `performance/iteration_ms` covers CUDA work only through backward. Opacity observations are sparse/event-triggered. Deterministic view cycling can cause periodic curves, but W&B history lacks view identity.

## Bundle files

- `run_selection.csv`: all project runs, selection flags, and exclusion reasons.
- `run_metadata.csv`: expanded run metadata and timestamp semantics.
- `run_configs.json`: complete plus normalized config for every discovered run.
- `training_events.csv`: per-run reconstructed phase/event inventory.
- `metric_definitions.csv`, `code_metric_map.md`: code-grounded metric dictionary.
- `metric_summary.csv`: long-form per-run and phase statistics.
- `cross_run_metrics.csv`: one row per selected run for comparisons.
- `cross_run_relationships.csv`: N, Pearson r, Spearman rho, and raw labeled points; exploratory only.
- `data_quality_report.csv`: checks without silent repair.
- `local_enrichment.csv`: local values beside W&B values, sources, confidence, and discrepancies.
- `external_eval_metrics.csv`: existing PSNR/SSIM/LPIPS joined by exact local run ID; no new evaluation was run.
- `timeseries_analysis.csv`: complete selected-run scalar history, not downsampled or forward-filled.
- `figures/`: raw-history publication plots in PNG and PDF.

The full per-run raw export and its manifest remain in `full_export/` outside the compact chat bundle.
"""
    (report_root / "analysis_context.md").write_text(context, encoding="utf-8")
    copy_bundle_files(report_root)
    print(f"Export complete: {report_root}")
    print(f"Selected runs: {len(selected)}; excluded runs: {len(excluded)}")


if __name__ == "__main__":
    main()
