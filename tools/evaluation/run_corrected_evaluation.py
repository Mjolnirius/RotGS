"""Run the isolated foreground-aware evaluation for the 12 selected RotGS runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
from scipy.stats import pearsonr, spearmanr
import torch

from corrected_metrics import CroppedLPIPS, cropped_ssim, foreground_crop, foreground_psnr, load_rgb, load_rgba


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "analysis" / "wandb_report"
SELECTION = REPORT / "run_selection.csv"
THRESHOLD = 0.001
PADDING = 8
PSNR_TOLERANCE_DB = 0.05
WARNED_DATASETS = {
    "Chips_blau_down_und_rn_roi_sqr_PNGaR_3029",
    "Chips_blau_und_rn_roi_sqr_PNGaR_3029",
    "Kaegifret_top_und_rn_roi_sqr_PNGaR_2226",
    "Quoellfrisch_top_30d_und_rn_roi_sqr_PNGaR_1777",
}


def selected_runs() -> list[dict[str, str]]:
    with SELECTION.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["selected_for_analysis"] == "True"]
    if len(rows) != 12:
        raise RuntimeError(f"expected 12 selected runs, found {len(rows)}")
    for row in rows:
        cfg = Path(row["model_path"]) / "cfg_args"
        text = cfg.read_text(encoding="utf-8")
        match = re.search(r"source_path='([^']+)'", text)
        if not match:
            raise RuntimeError(f"cannot resolve source_path from {cfg}")
        row["source_path"] = match.group(1)
        row["cfg_args"] = str(cfg)
    return rows


def source_metadata(run: dict[str, str]) -> dict[str, Any]:
    path = Path(run["source_path"]) / "preprocessing_metadata.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(runs: list[dict[str, str]], iteration: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for run in runs:
        source = Path(run["source_path"])
        image_files = sorted(
            path for path in (source / "images").iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        test_sources = image_files[::8]
        method = Path(run["model_path"]) / "test" / f"ours_{iteration}"
        renders = sorted((method / "renders").glob("*.png")) if (method / "renders").is_dir() else []
        gts = sorted((method / "gt").glob("*.png")) if (method / "gt").is_dir() else []
        metadata = source_metadata(run)
        renamed = {item["output"]: item for item in metadata.get("renamed_images", [])}
        errors: list[str] = []
        if len(image_files) != 73:
            errors.append(f"source image count is {len(image_files)}, expected 73")
        if len(test_sources) != 10:
            errors.append(f"test source count is {len(test_sources)}, expected 10")
        if len(renders) != 10 or len(gts) != 10:
            errors.append(f"render/GT count is {len(renders)}/{len(gts)}, expected 10/10")
        if errors:
            raise RuntimeError(f"{run['run_id']} iteration {iteration}: {'; '.join(errors)}")

        exact_gt = True
        dimensions_match = True
        for index, (source_image, render_path, gt_path) in enumerate(zip(test_sources, renders, gts)):
            expected_name = f"{index:05d}.png"
            if render_path.name != expected_name or gt_path.name != expected_name:
                raise RuntimeError(
                    f"{run['run_id']}: ordinal mismatch at {index}: "
                    f"{render_path.name}, {gt_path.name}"
                )
            with Image.open(source_image) as src_image, Image.open(gt_path) as gt_image, Image.open(render_path) as render_image:
                if src_image.mode != "RGBA":
                    raise RuntimeError(f"source lacks RGBA alpha: {source_image}")
                white = Image.new("RGBA", src_image.size, "white")
                composited = Image.alpha_composite(white, src_image.convert("RGBA")).convert("RGB")
                gt_rgb = gt_image.convert("RGB")
                exact = np.array_equal(np.asarray(composited), np.asarray(gt_rgb))
                shape_ok = src_image.size == gt_image.size == render_image.size
                exact_gt = exact_gt and exact
                dimensions_match = dimensions_match and shape_ok
            mapping = renamed.get(source_image.name, {})
            manifest.append({
                "run_id": run["run_id"],
                "run_name": run["run_name"],
                "dataset_id": run["dataset_id"],
                "iteration": iteration,
                "view_index": index,
                "render_file": str(render_path),
                "gt_file": str(gt_path),
                "source_image": str(source_image),
                "source_image_name": source_image.name,
                "original_source_name": mapping.get("source", ""),
                "angle_degrees": mapping.get("angle_degrees", ""),
                "gt_matches_white_composited_source": exact,
                "dimensions_match": shape_ok,
            })
        warnings = metadata.get("warnings", [])
        validations.append({
            "run_id": run["run_id"],
            "run_name": run["run_name"],
            "dataset_id": run["dataset_id"],
            "iteration": iteration,
            "num_views": len(test_sources),
            "view_identity_policy": "sorted source images at indices 0,8,...,72",
            "ordinal_filenames_valid": True,
            "dimensions_match": dimensions_match,
            "gt_matches_white_composited_source": exact_gt,
            "preprocessing_warning": " | ".join(warnings),
            "warning_expected": run["dataset_id"] in WARNED_DATASETS,
        })
    return manifest, validations


def evaluate_manifest(
    manifest: list[dict[str, Any]], device: torch.device, make_overlays: bool = False
) -> list[dict[str, Any]]:
    lpips_metric = CroppedLPIPS(device)
    results: list[dict[str, Any]] = []
    overlay_root = REPORT / "corrected_evaluation_overlays"
    if make_overlays:
        overlay_root.mkdir(parents=True, exist_ok=True)
    for item_index, item in enumerate(manifest, 1):
        render = load_rgb(Path(item["render_file"]))
        gt = load_rgb(Path(item["gt_file"]))
        _, alpha = load_rgba(Path(item["source_image"]))
        crop = foreground_crop(alpha, THRESHOLD, PADDING)
        alpha_crop = alpha[crop.y0:crop.y1, crop.x0:crop.x1]
        result = dict(item)
        result.update({
            "image_width": render.shape[1],
            "image_height": render.shape[0],
            "bbox_threshold": THRESHOLD,
            "bbox_padding_px": PADDING,
            "bbox_raw_x0": crop.raw_x0,
            "bbox_raw_y0": crop.raw_y0,
            "bbox_raw_x1": crop.raw_x1,
            "bbox_raw_y1": crop.raw_y1,
            "bbox_x0": crop.x0,
            "bbox_y0": crop.y0,
            "bbox_x1": crop.x1,
            "bbox_y1": crop.y1,
            "crop_width": crop.width,
            "crop_height": crop.height,
            "foreground_coverage": float(alpha.mean(dtype=np.float64)),
            "foreground_nonzero_coverage": float(np.mean(alpha > THRESHOLD)),
            "crop_foreground_coverage": float(alpha_crop.mean(dtype=np.float64)),
            "foreground_psnr": foreground_psnr(render, gt, alpha),
            "foreground_crop_ssim": cropped_ssim(render, gt, crop, device),
            "foreground_crop_lpips": lpips_metric(render, gt, crop),
        })
        results.append(result)
        warned = item["dataset_id"] in WARNED_DATASETS
        if make_overlays and (warned or item["view_index"] == 0):
            overlay_path = overlay_root / (
                f"{item['run_id']}_iter{item['iteration']}_view{item['view_index']:02d}.png"
            )
            save_overlay(Path(item["source_image"]), Path(item["gt_file"]), Path(item["render_file"]), crop, overlay_path)
        print(
            f"[{item_index:03d}/{len(manifest):03d}] {item['run_id']} "
            f"iter {item['iteration']} view {item['view_index']}"
        )
    del lpips_metric
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def save_overlay(source: Path, gt: Path, render: Path, crop: Any, output: Path) -> None:
    with Image.open(source) as source_image:
        white = Image.new("RGBA", source_image.size, "white")
        source_rgb = Image.alpha_composite(white, source_image.convert("RGBA")).convert("RGB")
        alpha = source_image.convert("RGBA").getchannel("A").convert("RGB")
    with Image.open(gt) as gt_image, Image.open(render) as render_image:
        panels = [source_rgb, alpha, gt_image.convert("RGB"), render_image.convert("RGB")]
    labels = ["source RGB", "alpha mask", "saved GT", "render"]
    rendered_panels = []
    for panel, label in zip(panels, labels):
        panel = panel.copy()
        draw = ImageDraw.Draw(panel)
        draw.rectangle((crop.raw_x0, crop.raw_y0, crop.raw_x1 - 1, crop.raw_y1 - 1), outline="lime", width=6)
        draw.rectangle((crop.x0, crop.y0, crop.x1 - 1, crop.y1 - 1), outline="red", width=6)
        panel.thumbnail((480, 480), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (480, 510), "white")
        canvas.paste(panel, ((480 - panel.width) // 2, 30))
        ImageDraw.Draw(canvas).text((8, 8), label, fill="black")
        rendered_panels.append(canvas)
    combined = Image.new("RGB", (480 * len(rendered_panels), 510), "white")
    for index, panel in enumerate(rendered_panels):
        combined.paste(panel, (index * 480, 0))
    combined.save(output)


def aggregate_30k(
    runs: list[dict[str, str]], per_view: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        values = [row for row in per_view if row["run_id"] == run["run_id"]]
        old = old_full_metrics(Path(run["model_path"]), 30000)
        row: dict[str, Any] = {
            "run_id": run["run_id"], "run_name": run["run_name"],
            "iteration": 30000, "num_views": len(values),
        }
        for metric in ("foreground_psnr", "foreground_crop_ssim", "foreground_crop_lpips"):
            data = np.asarray([value[metric] for value in values], dtype=np.float64)
            row[f"{metric}_mean"] = float(data.mean())
            row[f"{metric}_std"] = float(data.std(ddof=1))
        row.update({
            "old_full_psnr": old["PSNR"],
            "old_full_ssim": old["SSIM"],
            "old_full_lpips": old["LPIPS"],
            "mean_foreground_coverage": float(np.mean([value["foreground_coverage"] for value in values])),
        })
        rows.append(row)
    return rows


def old_full_metrics(model: Path, iteration: int) -> dict[str, float]:
    data = json.loads((model / "results.json").read_text(encoding="utf-8"))
    key = f"ours_{iteration}"
    if key not in data:
        raise KeyError(f"{model / 'results.json'} has no {key}")
    return data[key]


def wandb_row(run_id: str, iteration: int) -> dict[str, str]:
    path = REPORT / "full_export" / "runs" / run_id / "history.csv"
    matches = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                current = int(float(row.get("iteration", "nan")))
            except ValueError:
                continue
            if current == iteration:
                matches.append(row)
    if not matches:
        raise RuntimeError(f"no W&B row for {run_id} iteration {iteration}")
    return matches[-1]


def validate_30k(
    runs: list[dict[str, str]], summary: list[dict[str, Any]], validations: list[dict[str, Any]]
) -> bool:
    by_run = {row["run_id"]: row for row in summary}
    valid = True
    for run, validation in zip(runs, validations):
        wandb = wandb_row(run["run_id"], 30000)
        expected = float(wandb["eval/test/foreground_psnr"])
        observed = float(by_run[run["run_id"]]["foreground_psnr_mean"])
        delta = observed - expected
        validation.update({
            "wandb_foreground_psnr": expected,
            "png_foreground_psnr": observed,
            "foreground_psnr_signed_delta_db": delta,
            "foreground_psnr_abs_delta_db": abs(delta),
            "foreground_psnr_tolerance_db": PSNR_TOLERANCE_DB,
            "foreground_psnr_within_tolerance": abs(delta) <= PSNR_TOLERANCE_DB,
        })
        row_valid = (
            validation["dimensions_match"]
            and validation["gt_matches_white_composited_source"]
            and abs(delta) <= PSNR_TOLERANCE_DB
        )
        validation["validation_passed"] = row_valid
        valid = valid and row_valid
    return valid


def render_25k(runs: list[dict[str, str]]) -> dict[str, float]:
    costs: dict[str, float] = {}
    for index, run in enumerate(runs, 1):
        model = Path(run["model_path"])
        checkpoint = model / "point_cloud" / "iteration_25000" / "point_cloud.ply"
        residual = model / "residual_predictor" / "iteration_25000"
        target = model / "test" / "ours_25000"
        if not checkpoint.is_file() or not residual.is_dir():
            raise RuntimeError(f"missing 25k checkpoint components for {run['run_id']}")
        existing = list(target.rglob("*")) if target.exists() else []
        if existing:
            renders = list((target / "renders").glob("*.png"))
            gts = list((target / "gt").glob("*.png"))
            if len(renders) == 10 and len(gts) == 10:
                print(f"[{index:02d}/12] reusing complete 25k render: {run['run_id']}")
                continue
            raise RuntimeError(f"refusing partial/nonempty 25k target: {target}")
        command = [
            str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "render.py"),
            "--name", str(model), "--iteration", "25000", "--skip_train", "--quiet",
        ]
        print(f"[{index:02d}/12] rendering 25k test views: {run['run_id']}")
        start = time.monotonic()
        subprocess.run(command, cwd=ROOT, check=True)
        costs[run["run_id"]] = time.monotonic() - start
    return costs


def paired_outputs(
    runs: list[dict[str, str]], rows25: list[dict[str, Any]], rows30: list[dict[str, Any]], render_costs: dict[str, float]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    key = lambda row: (row["run_id"], int(row["view_index"]))
    map25, map30 = {key(row): row for row in rows25}, {key(row): row for row in rows30}
    per_view = []
    metrics = ("foreground_psnr", "foreground_crop_ssim", "foreground_crop_lpips")
    for item_key in sorted(map25):
        a, b = map25[item_key], map30[item_key]
        if a["source_image_name"] != b["source_image_name"]:
            raise RuntimeError(f"25k/30k view mismatch: {item_key}")
        row = {
            "run_id": a["run_id"], "run_name": a["run_name"],
            "view_index": a["view_index"], "source_image_name": a["source_image_name"],
            "original_source_name": a["original_source_name"], "angle_degrees": a["angle_degrees"],
            "foreground_coverage": a["foreground_coverage"],
            "crop_width": a["crop_width"], "crop_height": a["crop_height"],
        }
        for metric in metrics:
            row[f"{metric}_25000"] = a[metric]
            row[f"{metric}_30000"] = b[metric]
            row[f"{metric}_delta_30000_minus_25000"] = b[metric] - a[metric]
        per_view.append(row)

    per_run = []
    for run in runs:
        values = [row for row in per_view if row["run_id"] == run["run_id"]]
        row: dict[str, Any] = {
            "run_id": run["run_id"], "run_name": run["run_name"], "num_views": len(values),
            "mean_foreground_coverage": float(np.mean([v["foreground_coverage"] for v in values])),
        }
        for metric in metrics:
            for iteration in (25000, 30000):
                data = np.asarray([v[f"{metric}_{iteration}"] for v in values], dtype=np.float64)
                row[f"{metric}_{iteration}_mean"] = float(data.mean())
                row[f"{metric}_{iteration}_std"] = float(data.std(ddof=1))
            row[f"{metric}_delta_30000_minus_25000"] = (
                row[f"{metric}_30000_mean"] - row[f"{metric}_25000_mean"]
            )
        row["gaussian_count_25000"] = ply_vertex_count(Path(run["model_path"]) / "point_cloud/iteration_25000/point_cloud.ply")
        row["gaussian_count_30000"] = ply_vertex_count(Path(run["model_path"]) / "point_cloud/iteration_30000/point_cloud.ply")
        row["gaussian_count_delta_30000_minus_25000"] = row["gaussian_count_30000"] - row["gaussian_count_25000"]
        wb25, wb30 = wandb_row(run["run_id"], 25000), wandb_row(run["run_id"], 30000)
        for name, column in (("wandb_runtime_seconds", "_runtime"), ("iteration_ms", "performance/iteration_ms")):
            v25 = float(wb25[column]) if wb25.get(column) else math.nan
            v30 = float(wb30[column]) if wb30.get(column) else math.nan
            row[f"{name}_25000"] = v25
            row[f"{name}_30000"] = v30
            row[f"{name}_delta_30000_minus_25000"] = v30 - v25
        row["render_25000_wall_seconds"] = render_costs.get(run["run_id"], math.nan)
        per_run.append(row)
    return per_run, per_view


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for raw_line in handle:
            line = raw_line.decode("ascii", errors="strict").strip()
            match = re.fullmatch(r"element vertex (\d+)", line)
            if match:
                return int(match.group(1))
            if line == "end_header":
                break
    raise RuntimeError(f"PLY vertex count not found: {path}")


def relationship_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cross = read_indexed_csv(REPORT / "cross_run_metrics.csv", "run_id")
    predictors = {
        "mean_foreground_coverage": lambda row: float(row["mean_foreground_coverage"]),
        "final_gaussians": lambda row: float(cross[row["run_id"]]["final_gaussians"]),
        "wandb_total_runtime_seconds": lambda row: float(cross[row["run_id"]]["wandb_total_runtime_seconds"]),
    }
    outcomes = (
        "foreground_psnr_mean", "foreground_crop_ssim_mean", "foreground_crop_lpips_mean",
        "old_full_psnr", "old_full_ssim", "old_full_lpips",
    )
    rows = []
    for predictor_name, getter in predictors.items():
        for outcome in outcomes:
            x = np.asarray([getter(row) for row in summary], dtype=np.float64)
            y = np.asarray([float(row[outcome]) for row in summary], dtype=np.float64)
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            rows.append({
                "predictor": predictor_name, "outcome": outcome, "n": len(x),
                "pearson_r": float(pearson.statistic), "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic), "spearman_p": float(spearman.pvalue),
                "interpretation": "exploratory descriptive association; N=12",
            })
    return rows


def read_indexed_csv(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_methodology(validations: list[dict[str, Any]], relationships: list[dict[str, Any]]) -> None:
    warning_rows = [row for row in validations if row["preprocessing_warning"]]
    coverage_rows = [row for row in relationships if row["predictor"] == "mean_foreground_coverage"]
    weight_paths = [
        Path.home() / ".cache/torch/hub/checkpoints/vgg.pth",
        Path.home() / ".cache/torch/hub/checkpoints/vgg16-397923af.pth",
    ]
    lines = [
        "# Corrected reconstruction evaluation methodology", "",
        "- Foreground PSNR uses the original soft RGBA alpha and equal-view averaging.",
        "- Crops use `alpha > 0.001`, 8 px fixed padding, and image-bound clipping.",
        "- Cropped SSIM and LPIPS operate over the complete crop; neither is pixel-masked.",
        "- VGG LPIPS v0.1 receives RGB transformed from `[0,1]` to `[-1,1]` and no crops are resized.",
        "- Standard deviations are sample standard deviations (`ddof=1`).",
        "- Historical full-frame metrics are copied from existing `results.json`; those files are not changed.",
        "- Exact W&B runtime and iteration cost are retained; 25k render wall time is `NaN` when rendering and evaluation are run as separate stages.",
        "- Paired 25k/30k tables define every delta as `30000 - 25000`.", "",
        "## Validation", "",
        f"All {len(validations)} 30k runs passed: {all(row.get('validation_passed') for row in validations)}.",
        f"Foreground-PSNR comparison tolerance: {PSNR_TOLERANCE_DB} dB.",
        "Differences are expected because W&B evaluated float renders in memory whereas this tool reloads quantized PNGs.", "",
        "- 48 mask/crop overlays were generated; one representative overlay from each warned dataset was visually inspected, with no mask repair applied.", "",
        "## Preprocessing warnings retained", "",
    ]
    for row in warning_rows:
        lines.append(f"- `{row['dataset_id']}`: {row['preprocessing_warning']}")
    lines.extend(["", "## LPIPS weight provenance", ""])
    for path in weight_paths:
        if path.is_file():
            lines.append(f"- `{path}`: SHA-256 `{sha256(path)}`")
    lines.extend(["", "## Coverage associations", ""])
    for row in coverage_rows:
        lines.append(
            f"- `{row['outcome']}`: Pearson r={row['pearson_r']:.4f}; "
            f"Spearman rho={row['spearman_rho']:.4f}; N={row['n']}."
        )
    path = REPORT / "corrected_evaluation_methodology.md"
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = {
        "view_index": int, "iteration": int,
        "foreground_coverage": float, "foreground_psnr": float,
        "foreground_crop_ssim": float, "foreground_crop_lpips": float,
        "crop_width": int, "crop_height": int,
    }
    for row in rows:
        for key, converter in numeric.items():
            if key in row and row[key] != "":
                row[key] = converter(row[key])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "30k", "render-25k", "25k", "relationships"), default="all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    runs = selected_runs()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    summary30_path = REPORT / "corrected_evaluation_30k.csv"
    per_view30_path = REPORT / "corrected_evaluation_30k_per_view.csv"
    validation_path = REPORT / "corrected_evaluation_validation.csv"
    manifest_path = REPORT / "corrected_evaluation_view_manifest.csv"

    render_costs: dict[str, float] = {}
    if args.stage in {"all", "30k"}:
        manifest30, validations30 = build_manifest(runs, 30000)
        rows30 = evaluate_manifest(manifest30, device, make_overlays=True)
        summary30 = aggregate_30k(runs, rows30)
        passed = validate_30k(runs, summary30, validations30)
        write_csv(manifest_path, manifest30)
        write_csv(per_view30_path, rows30)
        write_csv(summary30_path, summary30)
        write_csv(validation_path, validations30)
        print(f"30k validation passed: {passed}")
        if not passed:
            raise RuntimeError("30k validation gate failed; refusing 25k rendering")
        if args.stage == "30k":
            return
    else:
        summary30 = load_csv(summary30_path)
        rows30 = load_csv(per_view30_path)
        validations30 = load_csv(validation_path)

    if args.stage in {"all", "render-25k"}:
        render_costs = render_25k(runs)
        if args.stage == "render-25k":
            return

    if args.stage in {"all", "25k"}:
        manifest25, validations25 = build_manifest(runs, 25000)
        rows25 = evaluate_manifest(manifest25, device, make_overlays=False)
        paired, paired_views = paired_outputs(runs, rows25, rows30, render_costs)
        write_csv(REPORT / "evaluation_25k_vs_30k.csv", paired)
        write_csv(REPORT / "evaluation_25k_vs_30k_per_view.csv", paired_views)
        # Keep both iterations in the auditable manifest.
        write_csv(manifest_path, load_csv(manifest_path) + manifest25)

    relationships = relationship_rows(summary30)
    write_csv(REPORT / "corrected_evaluation_relationships.csv", relationships)
    write_methodology(validations30, relationships)
    print("Corrected evaluation complete")


if __name__ == "__main__":
    main()
