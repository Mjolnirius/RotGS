#!/usr/bin/env python3
"""Export RotGS per-view angles, rigid transforms, and extrinsics plots.

Run: .venv/bin/python tools/inspection/export_view_poses.py RUN [RUN ...] \
         --output-dir REPORT

Import: extract_view_poses(run, iteration=-1) returns a JSON-compatible dict.
Angle extraction runs on CPU. Optional --render-views COUNT loads the Gaussian
PLY and uses CUDA to append rendered acquisition views to each report.
"""
from __future__ import annotations

import argparse
import ast
import csv
import html
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scene.residual_predictor import ResidualPredictor
from scene.tae_calibration import load_tae_calibration
from utils.general_utils import add_noise


def _config(model_path: Path) -> argparse.Namespace:
    """Parse the literal Namespace saved by training without evaluating code."""
    node = ast.parse((model_path / "cfg_args").read_text(), mode="eval").body
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
            or node.func.id != "Namespace" or node.args
            or any(k.arg is None for k in node.keywords)):
        raise ValueError("cfg_args must contain a literal Namespace(...)")
    return argparse.Namespace(**{k.arg: ast.literal_eval(k.value) for k in node.keywords})


def saved_iterations(model_path: str | Path) -> list[int]:
    return sorted(
        int(p.name.removeprefix("iteration_"))
        for p in (Path(model_path) / "point_cloud").glob("iteration_*")
        if p.name.removeprefix("iteration_").isdigit()
        and (p / "axis.npy").is_file() and (p / "center.npy").is_file()
    )


def rigid_transform(axis, center, angle_rad: float) -> np.ndarray:
    """Match rotate_gaussian: x' = R @ x + (I - R) @ center, column vectors."""
    axis, center = np.asarray(axis, dtype=np.float64), np.asarray(center, dtype=np.float64)
    if axis.shape != (3,) or center.shape != (3,):
        raise ValueError("axis and center must be 3-vectors")
    norm = np.linalg.norm(axis)
    if norm < 1e-8 or not np.isfinite(np.r_[axis, center, angle_rad]).all():
        raise ValueError("motion parameters must be finite with a nonzero axis")
    x, y, z = axis / norm
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    rotation = np.eye(3) + math.sin(angle_rad) * skew + (1 - math.cos(angle_rad)) * (skew @ skew)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = center - rotation @ center
    return transform


def _aruco_rows(path: Path, names: list[str], renamed: dict) -> list[dict | None]:
    """Join the existing TAE CSV by original filename, allowing partial tables."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"filename", "nominal_angle_deg", "estimated_angle_deg"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"ArUco CSV {path} requires columns {sorted(required)}")
        lookup = {}
        for row in reader:
            name = Path(row["filename"]).name
            if name in lookup:
                raise ValueError(f"duplicate ArUco filename: {name}")
            lookup[name] = row
    result = []
    for name in names:
        source = Path(renamed.get(name, {}).get("source", name)).name
        row = lookup.get(source)
        if row is not None and name in renamed:
            if not math.isclose(float(row["nominal_angle_deg"]), float(renamed[name]["angle_degrees"]), abs_tol=1e-6):
                raise ValueError(f"ArUco nominal angle disagrees with preprocessing: {name}")
        result.append(row)
    if not any(row is not None for row in result):
        raise ValueError(f"no input images match ArUco CSV {path}")
    return result


def _measured(row: dict | None, key: str) -> float | None:
    if row is None or row.get("confidence", "").strip().lower() == "unresolved" or not row.get(key, "").strip():
        return None
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite ArUco {key}")
    if key == "angle_uncertainty_deg" and value < 0:
        raise ValueError("ArUco uncertainty must be nonnegative")
    return value


def _predictor(model_path, cfg, iteration):
    use_local = not getattr(cfg, "wo_tiny", False)
    sweep_bound = getattr(cfg, "max_sweep_error_deg", 0.0)
    weights_iteration = iteration
    # PLY is saved before the optimizer step. Saving a .pth checkpoint then
    # overwrites the same-iteration residual file after that step. The retained
    # pre-validation residual at i-1 matches the PLY at i exactly.
    overwritten = iteration < cfg.iterations and iteration in getattr(cfg, "checkpoint_iterations", [])
    if overwritten and iteration in getattr(cfg, "test_iterations", []):
        previous = model_path / "residual_predictor" / f"iteration_{iteration - 1}" / "residual_predictor.pth"
        if previous.is_file():
            weights_iteration = iteration - 1
    weights = model_path / "residual_predictor" / f"iteration_{weights_iteration}" / "residual_predictor.pth"
    notes = []
    if overwritten and weights_iteration == iteration and (use_local or sweep_bound > 0):
        notes.append("Residual snapshot may be one optimizer update later than the saved axis/center; matching pre-validation residuals are unavailable.")
    state = torch.load(weights, map_location="cpu", weights_only=True) if weights.is_file() else None
    if state is None and (use_local or sweep_bound > 0):
        raise FileNotFoundError(f"missing learned angle corrections: {weights}")
    controls = state["residuals"].shape[1] - 1 if state is not None else 1
    predictor = ResidualPredictor(
        device="cpu", num_ctrl_points=controls,
        max_residual_angle_deg=getattr(cfg, "max_residual_angle_deg", 0.0),
        max_sweep_error_deg=sweep_bound,
        anchor_local_endpoints=getattr(cfg, "leverage_TAE", False),
    )
    if state is not None:
        missing, unexpected = predictor.load_state_dict(state, strict=False)
        # Legacy predictors had no sweep parameter, so its initial zero applies.
        if set(missing) - {"sweep_error"} or unexpected:
            raise ValueError(f"incompatible residual checkpoint: missing={missing}, unexpected={unexpected}")
    predictor.eval()
    return predictor, weights if state is not None else None, notes


def extract_view_poses(
    model_path: str | Path, iteration: int = -1, *,
    source_path: str | Path | None = None, aruco_csv: str | Path | None = None,
) -> dict:
    """Recover every view's saved angle and 4x4 object/camera transforms on CPU.

    Reuses ResidualPredictor.angle_correction and the training loader's input
    angle/time conventions. Reads saved effective axis.npy/center.npy directly.
    rotgs_render_angle_deg retains the signed learned starting offset;
    rotgs_angle_deg = direction * (render_angle - render_angle_at_view_0).
    Comparison angles remain unwrapped, including the final full-turn view.
    world_to_camera acts on column vectors in canonical model coordinates;
    camera_to_world is its inverse. Translations use RotGS scene units.
    Optional ArUco angles join by original source filename, never row position.
    """
    model_path = Path(model_path).expanduser().resolve()
    cfg = _config(model_path)
    if not getattr(cfg, "fixed_camera", False) or getattr(cfg, "multi_camera", False):
        raise ValueError("only fixed, single-camera RotGS runs are supported")
    dataset = Path(source_path or cfg.source_path).expanduser().resolve()
    iterations = saved_iterations(model_path)
    if not iterations:
        raise FileNotFoundError(f"no saved axis/center under {model_path}")
    iteration = iterations[-1] if iteration == -1 else iteration
    snapshot = model_path / "point_cloud" / f"iteration_{iteration}"
    axis = np.load(snapshot / "axis.npy").reshape(-1, 3)
    center = np.load(snapshot / "center.npy").reshape(-1, 3)
    if axis.shape != (1, 3) or center.shape != (1, 3):
        raise ValueError("expected saved axis/center for exactly one camera")
    axis, center = axis[0].astype(np.float64), center[0].astype(np.float64)
    rigid_transform(axis, center, 0)  # Validate before normalizing.
    axis /= np.linalg.norm(axis)
    # Match readFixedCameras, including filename suffix indexing.
    names = sorted(p.name for p in (dataset / (getattr(cfg, "images", None) or "images")).iterdir()
                   if p.name.endswith((".jpg", ".png", "jpeg")))
    if len(names) < 2:
        raise ValueError("a turntable sequence requires at least two images")
    metadata_path = dataset / "preprocessing_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    renamed = {}
    for item in metadata.get("renamed_images", []):
        name = Path(item["output"]).name
        if name in renamed:
            raise ValueError(f"duplicate preprocessing output: {name}")
        renamed[name] = item
    if renamed and set(renamed) != set(names):
        raise ValueError("preprocessing output names do not match the dataset images")
    nominal = np.array([float(renamed[name]["angle_degrees"]) if renamed else 360 * i / (len(names) - 1)
                        for i, name in enumerate(names)])
    nominal -= nominal[0]
    if not np.isfinite(nominal).all():
        raise ValueError("nominal angles must be finite")
    direction = getattr(cfg, "rotation_direction", -1)
    if direction not in (-1, 1):
        raise ValueError("rotation_direction must be -1 or +1")
    if getattr(cfg, "leverage_TAE", False):
        calibration = load_tae_calibration(dataset, image_names=names)
        measured = np.array([calibration.angles_by_image_deg[name] for name in names])
        relative = measured - measured[0]
        coarse = torch.tensor(np.deg2rad(direction * relative), dtype=torch.float32)
        times = torch.tensor(relative / calibration.measured_total_rotation_deg, dtype=torch.float32)
        angle_source = "ArUco / TAE"
    else:
        coarse_array = torch.linspace(0, direction * 2 * np.pi, steps=len(names))
        noise = getattr(cfg, "angle_noise_std", 0.0)
        if noise > 0:
            with torch.random.fork_rng(devices=[]):
                coarse_array = add_noise(coarse_array, noise_std=noise, seed=1304)
        indices = []
        for name in names:
            match = re.search(r"(\d+)\.(jpg|png)$", name)
            if match is None:
                raise ValueError(f"cannot reproduce training's angle index: {name}")
            indices.append(int(match.group(1)))
        if sorted(indices) != list(range(len(names))):
            raise ValueError("nominal training requires contiguous filename indices 0..N-1")
        coarse = coarse_array[indices]
        times = torch.tensor([i / (len(names) - 1) for i in indices], dtype=torch.float32)
        angle_source = "Nominal" + (" + synthetic noise" if noise else "")
    predictor, weights, notes = _predictor(model_path, cfg, iteration)
    use_local = not getattr(cfg, "wo_tiny", False)
    with torch.no_grad():
        sweep = times * predictor.effective_sweep_error(0)
        local = predictor(times, 0) if use_local else torch.zeros_like(times)
        render_angles = coarse + predictor.angle_correction(times, 0, use_local_residual=use_local)
    render_angles = render_angles.numpy().astype(np.float64)
    comparison = np.rad2deg(direction * (render_angles - render_angles[0]))
    csv_path = (Path(aruco_csv).expanduser().resolve() if aruco_csv is not None
                else dataset / "turntable_angle_estimation" / "turntable_angles.csv")
    if aruco_csv is not None and not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    reference = _aruco_rows(csv_path, names, renamed) if csv_path.is_file() else [None] * len(names)
    aruco_zero = _measured(reference[0], "estimated_angle_deg")
    if any(row is not None for row in reference) and aruco_zero is None:
        notes.append("ArUco view 0 is unresolved; relative ArUco comparisons are unavailable.")
    fixed_w2c = np.eye(4)
    fixed_w2c[:3, 3] = [0, 0, 5]  # readFixedCameras: R=I, T=(0,0,5).
    rows = []
    for i, name in enumerate(names):
        transform = rigid_transform(axis, center, render_angles[i])
        relative_transform = rigid_transform(axis, center, render_angles[i] - render_angles[0])
        w2c = fixed_w2c @ transform
        c2w = np.linalg.inv(w2c)
        measured = _measured(reference[i], "estimated_angle_deg")
        aruco_angle = measured - aruco_zero if measured is not None and aruco_zero is not None else None
        rows.append({
            "view_index": i, "image_name": name,
            "source_image": Path(renamed.get(name, {}).get("source", name)).name,
            "split": "test" if getattr(cfg, "eval", False) and i % 8 == 0 else "train",
            "time": float(times[i]), "nominal_angle_deg": float(nominal[i]),
            "input_angle_deg": math.degrees(float(coarse[i])),
            "local_correction_deg": math.degrees(float(local[i])),
            "sweep_correction_deg": math.degrees(float(sweep[i])),
            "rotgs_render_angle_deg": math.degrees(render_angles[i]),
            "rotgs_angle_deg": float(comparison[i]),
            "aruco_measured_angle_deg": measured, "aruco_angle_deg": aruco_angle,
            "aruco_uncertainty_deg": _measured(reference[i], "angle_uncertainty_deg"),
            "aruco_confidence": reference[i].get("confidence") if reference[i] else None,
            "rotgs_minus_nominal_deg": float(comparison[i] - nominal[i]),
            "rotgs_minus_aruco_deg": float(comparison[i] - aruco_angle) if aruco_angle is not None else None,
            "object_transform": transform.tolist(),
            "relative_object_transform": relative_transform.tolist(),
            "world_to_camera": w2c.tolist(), "camera_to_world": c2w.tolist(),
            "effective_camera_center": c2w[:3, 3].tolist(),
        })
    return {
        "schema_version": 1, "model_path": str(model_path), "source_path": str(dataset),
        "iteration": iteration, "configured_final_iteration": cfg.iterations,
        "residual_checkpoint": str(weights) if weights is not None else None,
        "aruco_csv": str(csv_path) if csv_path.is_file() else None,
        "angle_source": angle_source, "rotation_direction": direction,
        "axis": axis.tolist(), "center": center.tolist(),
        "axis_tilt_deg": math.degrees(math.atan2(axis[2], axis[1])),
        "axis_side_deg": math.degrees(math.asin(np.clip(axis[0], -1, 1))),
        "render_zero_deg": rows[0]["rotgs_render_angle_deg"],
        "total_rotation_deg": rows[-1]["rotgs_angle_deg"],
        "nominal_total_deg": float(nominal[-1]), "aruco_total_deg": rows[-1]["aruco_angle_deg"],
        "notes": notes,
        "conventions": {
            "matrices": "Active right-handed transforms acting on column vectors; camera coordinates are OpenCV x-right/y-down/z-forward.",
            "angles": "Raw render angles retain the learned zero. Comparison angles are unwrapped and relative to view 0; positive follows acquisition direction.",
            "orbit": "Equivalent camera motion for a static object. Comparison paths share the RotGS axis/center and align view 0; only their angles differ.",
            "units": "Translations use RotGS scene units, not millimetres. The physical camera is fixed at (0,0,-5).",
        },
        "views": rows,
    }


def _gallery_indices(report: dict, count: int) -> list[int]:
    if not 1 <= count <= len(report["views"]):
        raise ValueError("rendered view count must be between 1 and the number of views")
    held_out = [i for i, row in enumerate(report["views"]) if row["split"] == "test"]
    if len(held_out) == count:
        return held_out
    return np.rint(np.linspace(0, len(report["views"]) - 1, count)).astype(int).tolist()


def _saved_gallery(report: dict, output_dir: Path) -> list[dict]:
    """Keep matching gallery images when regenerating the plots on CPU."""
    path = output_dir / "rendered_views.json"
    if not path.is_file():
        return []
    saved = json.loads(path.read_text())
    for key in ("model_path", "source_path", "iteration", "axis", "center"):
        if saved.get(key) != report[key]:
            return []
    gallery = []
    for item in saved["views"]:
        row = report["views"][item["view_index"]]
        if item["image_name"] != row["image_name"] or item["rotgs_render_angle_deg"] != row["rotgs_render_angle_deg"]:
            return []
        if not (output_dir / item["file"]).is_file():
            return []
        # Comparison data can change independently of the rendered pose.
        # Refresh captions while retaining the matching image and dimensions.
        gallery.append({**item, **{key: row[key] for key in (
            "image_name", "split", "nominal_angle_deg", "rotgs_angle_deg", "aruco_angle_deg",
        )}})
    return gallery


def render_view_gallery(report: dict, output_dir: str | Path, count: int = 10) -> list[dict]:
    """Render selected acquisition frames at the exact raw angles in the report.

    Load the PLY directly and only one fixed-camera template. This avoids Scene
    initialization, random point-cloud generation, and writes to the dataset.
    """
    from PIL import Image
    from types import SimpleNamespace
    from gaussian_renderer import GaussianModel, render, set_rasterizer
    from scene.colmap_loader import read_intrinsics_binary, read_intrinsics_text
    from scene.dataset_readers import readFixedCameras
    from utils.camera_utils import loadCam

    indices = _gallery_indices(report, count)
    if not torch.cuda.is_available():
        raise RuntimeError("rendering view galleries requires CUDA")
    model_path = Path(report["model_path"])
    dataset = Path(report["source_path"])
    cfg = _config(model_path)
    cfg.source_path, cfg.model_path, cfg.data_device = str(dataset), str(model_path), "cuda"
    images_dir = dataset / (getattr(cfg, "images", None) or "images")
    sparse = dataset / "sparse" / "0"
    intrinsics = (read_intrinsics_binary(str(sparse / "cameras.bin"))
                  if (sparse / "cameras.bin").is_file()
                  else read_intrinsics_text(str(sparse / "cameras.txt")))
    calibration = load_tae_calibration(dataset) if getattr(cfg, "leverage_TAE", False) else None
    infos, _distance = readFixedCameras(
        intrinsics, str(images_dir),
        [row["image_name"] for row in report["views"] if row["split"] == "test"],
        rotation_direction=report["rotation_direction"], tae_calibration=calibration,
    )
    cfg.resolution = getattr(cfg, "resolution", 1)
    cfg.train_test_exp = getattr(cfg, "train_test_exp", False)
    template = loadCam(cfg, 0, infos[0], 1.0, False, infos[0].is_test)
    gaussians = GaussianModel(cfg.sh_degree, fixed_camera=True)
    ply_path = model_path / "point_cloud" / f"iteration_{report['iteration']}" / "point_cloud.ply"
    gaussians.load_ply(str(ply_path), use_train_test_exp=cfg.train_test_exp)
    background = torch.tensor([1., 1., 1.] if cfg.white_background else [0., 0., 0.], device="cuda")
    rasterizer = set_rasterizer(template, gaussians, SimpleNamespace(debug=False), background)
    axis = torch.tensor(report["axis"], dtype=torch.float32, device="cuda")
    center = torch.tensor(report["center"], dtype=torch.float32, device="cuda")
    output_dir = Path(output_dir)
    (output_dir / "renders").mkdir(parents=True, exist_ok=True)
    gallery = []
    try:
        with torch.no_grad():
            for i in indices:
                row = report["views"][i]
                angle = torch.tensor([math.radians(row["rotgs_render_angle_deg"])],
                                     dtype=torch.float32, device="cuda")
                package = render(rasterizer, gaussians, axis=axis, center=center, angle=angle)
                image = package["render"].clamp(0, 1)
                if cfg.train_test_exp:
                    image = image[..., image.shape[-1] // 2:]
                rgb = (image.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                filename = f"renders/view_{row['view_index']:04d}.png"
                Image.fromarray(rgb).save(output_dir / filename)
                gallery.append({
                    **{key: row[key] for key in ("view_index", "image_name", "split", "nominal_angle_deg",
                                                "rotgs_angle_deg", "aruco_angle_deg", "rotgs_render_angle_deg")},
                    "file": filename, "width": int(rgb.shape[1]), "height": int(rgb.shape[0]),
                })
                del package, image, rgb
                print(f"Rendered {model_path.name} view {i:02d} ({row['image_name']})", flush=True)
    finally:
        del gaussians, rasterizer, template
        torch.cuda.empty_cache()
    manifest = {key: report[key] for key in ("model_path", "source_path", "iteration", "axis", "center")}
    manifest.update(schema_version=1, views=gallery)
    (output_dir / "rendered_views.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return gallery


def _gallery_html(gallery: list[dict]) -> str:
    if not gallery:
        return ""
    figures = []
    for item in gallery:
        filename = html.escape(item["file"], quote=True)
        image_name = html.escape(item["image_name"])
        caption = (f"<strong>View {item['view_index']:02d}</strong> · {image_name} · {html.escape(item['split'])}<br>"
                   f"Nominal {item['nominal_angle_deg']:.2f}° · RotGS {item['rotgs_angle_deg']:.2f}°")
        if item["aruco_angle_deg"] is not None:
            caption += f"<br>ArUco {item['aruco_angle_deg']:.2f}°"
        figures.append(
            f'<figure><a href="{filename}" target="_blank" rel="noopener">'
            f'<img src="{filename}" width="{item["width"]}" height="{item["height"]}" '
            f'loading="lazy" alt="RotGS render of view {item["view_index"]}: {image_name}"></a>'
            f'<figcaption>{caption}</figcaption></figure>'
        )
    return (f'<section id="rendered-views"><h2>Rendered views ({len(gallery)})</h2>'
            '<p>Views sampled across the acquisition sequence, rendered from this report’s saved model '
            'with its learned per-view rotation. Caption angles are relative to view 0. '
            'Click an image to open the full-resolution render.</p>'
            '<div class="render-gallery">' + "".join(figures) + '</div></section>')


def _values(report, key):
    return np.array([np.nan if row[key] is None else row[key] for row in report["views"]])


def _html(content):
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>RotGS view angles and extrinsics</title><style>'
            'body{font:15px system-ui;margin:32px auto;padding:0 24px;max-width:1250px;color:#213040}'
            'table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border:1px solid #d5dce5;text-align:right}'
            'th{background:#edf2f8}tr:nth-child(even){background:#f7f9fb}a{color:#2676d2}'
            'img{width:100%;max-width:1200px}code{font-size:13px}'
            '.render-gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:20px}'
            '.render-gallery figure{margin:0;border:1px solid #d5dce5;border-radius:10px;overflow:hidden}'
            '.render-gallery img{display:block;width:100%;height:auto;background:#fff}'
            '.render-gallery figcaption{padding:12px;font-size:13px;line-height:1.65}'
            '#rendered-views{margin-top:36px}</style></head><body>'
            + content + '</body></html>\n')


def write_view_pose_report(
    report: dict, output_dir: str | Path, history: list[dict] | None = None,
    rendered_views: list[dict] | None = None,
) -> None:
    """Write JSON/CSV, angle and extrinsics plots, and a browsable view table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery = _saved_gallery(report, output_dir) if rendered_views is None else rendered_views
    report = {**report, "rendered_views": gallery}
    (output_dir / "view_poses.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    columns = [key for key, value in report["views"][0].items() if not isinstance(value, list)]
    with (output_dir / "view_angles.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report["views"])
    x = _values(report, "nominal_angle_deg")
    series = [("Nominal", x, "#7f8c8d"),
              ("RotGS", _values(report, "rotgs_angle_deg"), "#2676d2"),
              ("ArUco / TAE", _values(report, "aruco_angle_deg"), "#dc7932")]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    for label, angles, color in series:
        if np.isfinite(angles).any():
            axes[0].plot(x, angles, label=label, color=color, marker=".", markersize=4)
            axes[1].plot(x, angles - x, label=label, color=color)
            axes[2].plot(x[1:], np.diff(angles), label=label, color=color)
    uncertainty = _values(report, "aruco_uncertainty_deg")
    aruco = series[2][1]
    if np.isfinite(uncertainty).any() and np.isfinite(aruco).any():
        axes[1].fill_between(x, aruco - x - uncertainty, aruco - x + uncertainty,
                             color=series[2][2], alpha=0.18, label="Reported ArUco uncertainty")
    test = [i for i, row in enumerate(report["views"]) if row["split"] == "test"]
    if test:
        axes[1].scatter(x[test], series[1][1][test] - x[test], facecolors="none",
                        edgecolors=series[1][2], label="Held-out view", zorder=5)
    for ax, ylabel in zip(axes, ["Angle relative to view 0 (deg)", "Angle minus nominal (deg)", "Rotation since previous view (deg)"]):
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Nominal acquisition angle (deg)")
    fig.suptitle(f"{Path(report['model_path']).name}\nSaved iteration {report['iteration']}")
    fig.savefig(output_dir / "angles.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax3d = fig.add_subplot(121, projection="3d")
    ax_plane = fig.add_subplot(122)
    axis, center = np.array(report["axis"]), np.array(report["center"])
    camera = np.array([0., 0., -5.])
    radial = camera - center - np.dot(camera - center, axis) * axis
    # A camera on the axis has a degenerate positional orbit but valid poses.
    if np.linalg.norm(radial) < 1e-8:
        hint = np.eye(3)[np.argmin(np.abs(axis))]
        radial = hint - np.dot(hint, axis) * axis
    u = radial / np.linalg.norm(radial)
    v = np.cross(axis, u)
    all_points = [center]
    for label, angles, color in series:
        points = np.array([
            rigid_transform(axis, center, report["rotation_direction"] * math.radians(a))[:3, :3].T @ (camera - center) + center
            if np.isfinite(a) else [np.nan] * 3 for a in angles
        ])
        if not np.isfinite(points).any():
            continue
        all_points.extend(points[np.isfinite(points).all(axis=1)])
        ax3d.plot(*points.T, color=color, label=label, marker=".", markersize=3)
        planar = (points - center) @ np.column_stack([u, v])
        ax_plane.plot(*planar.T, color=color, label=label, marker=".", markersize=4)
        if label == "RotGS":
            for i in sorted(set([0, len(points) - 1, *range(12, len(points), 12)])):
                ax3d.text(*points[i], str(i), fontsize=8)
                ax_plane.annotate(str(i), planar[i], xytext=(4, 4), textcoords="offset points", fontsize=8)
            if test:
                ax_plane.scatter(*planar[test].T, facecolors="none", edgecolors=color, s=50)
    ax3d.plot(*np.stack([center - axis, center + axis]).T, color="#333333", label="Learned rotation axis")
    ax3d.scatter(*center, color="#333333", s=25)
    # Equal scale on all axes preserves the orbit geometry.
    points = np.array(all_points + [center - axis, center + axis])
    midpoint = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = max(float(np.ptp(points, axis=0).max()) * 0.55, 1e-3)
    ax3d.set(xlim=(midpoint[0] - radius, midpoint[0] + radius),
             ylim=(midpoint[1] - radius, midpoint[1] + radius),
             zlim=(midpoint[2] - radius, midpoint[2] + radius),
             xlabel="x", ylabel="y", zlabel="z", title="Equivalent camera extrinsics · aligned to view 0")
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.legend(fontsize=8)
    ax_plane.set(xlabel="Orbit-plane u (scene units)", ylabel="Orbit-plane v (scene units)",
                 title="View positions · labels are frame indices")
    ax_plane.set_aspect("equal")
    ax_plane.grid(alpha=0.25)
    ax_plane.legend(fontsize=8)
    fig.savefig(output_dir / "extrinsics.png", dpi=150)
    plt.close(fig)

    plots = ["angles.png", "extrinsics.png"]
    if history and len(history) > 1:
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
        iterations = [item["iteration"] for item in history]
        axes[0].plot(iterations, [item["total_rotation_deg"] for item in history], "o-", label="RotGS total")
        axes[0].axhline(report["nominal_total_deg"], color=series[0][2], label="Nominal")
        if report["aruco_total_deg"] is not None:
            axes[0].axhline(report["aruco_total_deg"], color=series[2][2], label="ArUco / TAE")
        axes[0].set(xlabel="Saved training iteration", ylabel="Total rotation (deg)")
        axes[1].plot(iterations, [item["axis_tilt_deg"] for item in history], "o-", label="Axis tilt", color="#2676d2")
        side_ax = axes[1].twinx()
        side_ax.plot(iterations, [item["axis_side_deg"] for item in history], "o--", label="Axis side", color="#dc7932")
        axes[1].set(xlabel="Saved training iteration", ylabel="Axis tilt (deg)")
        side_ax.set_ylabel("Axis side (deg)")
        side_ax.legend(fontsize=8, loc="upper right")
        for i, item in enumerate(history):
            axes[2].plot(x, _values(item, "rotgs_minus_nominal_deg"), label=str(item["iteration"]),
                         color=plt.cm.viridis(i / (len(history) - 1)))
        axes[2].set(xlabel="Nominal acquisition angle (deg)", ylabel="RotGS minus nominal (deg)")
        for ax in axes:
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8, loc="upper left")
        fig.savefig(output_dir / "convergence.png", dpi=150)
        plt.close(fig)
        plots.append("convergence.png")
        (output_dir / "convergence.json").write_text(json.dumps([
            {key: item[key] for key in ["iteration", "residual_checkpoint", "total_rotation_deg",
                                       "axis_tilt_deg", "axis_side_deg", "center", "notes"]}
            for item in history
        ], indent=2, allow_nan=False) + "\n")

    def display(value):
        return "—" if value is None else f"{value:.6f}" if isinstance(value, float) else html.escape(str(value))

    fields = ["view_index", "image_name", "split", "nominal_angle_deg", "rotgs_angle_deg",
              "aruco_angle_deg", "rotgs_minus_nominal_deg", "rotgs_minus_aruco_deg", "rotgs_render_angle_deg"]
    table = "".join("<tr>" + "".join(f"<td>{display(row[field])}</td>" for field in fields) + "</tr>" for row in report["views"])
    notes = list(dict.fromkeys(note for item in (history or [report]) for note in item["notes"]))
    content = f"<h1>{html.escape(Path(report['model_path']).name)}</h1><p>Iteration {report['iteration']} / {report['configured_final_iteration']} · {len(report['views'])} views · input angles: {report['angle_source']}</p>"
    content += f"<p>Nominal total: {display(report['nominal_total_deg'])}° · RotGS total: {display(report['total_rotation_deg'])}° · ArUco total: {display(report['aruco_total_deg'])}° · raw first-view offset: {display(report['render_zero_deg'])}°</p>"
    content += "<p>" + html.escape(report["conventions"]["angles"]) + " " + html.escape(report["conventions"]["orbit"]) + "</p>"
    content += "<p>" + html.escape(report["conventions"]["units"]) + " TAE-trained runs use ArUco as an input and prior; agreement is not an independent validation.</p>"
    content += "<p>The uncertainty band shows the values reported by ArUco/TAE, without assuming independent errors between frames. Held-out angles are predictions from the shared correction function.</p>"
    content += "".join(f"<p>{html.escape(note)}</p>" for note in notes)
    content += '<p><a href="view_angles.csv">Angle CSV</a> · <a href="view_poses.json">Matrices and angles JSON</a></p>'
    content += "".join(f'<img src="{plot}" alt="{plot}">' for plot in plots)
    content += "<div style='overflow-x:auto'><table><thead><tr>" + "".join(f"<th>{field}</th>" for field in fields) + "</tr></thead><tbody>" + table + "</tbody></table></div>"
    content += _gallery_html(gallery)
    (output_dir / "index.html").write_text(_html(content))


def write_report_index(report_dir: str | Path) -> None:
    """Index every existing report, including when only one run is refreshed."""
    report_dir = Path(report_dir)
    content = (
        '<h1>RotGS view angles and extrinsics</h1>'
        '<p>Angles relative to view 0; all input views, including held-out frames '
        'and the final full-turn frame. Iterations refer to the saved model used '
        'in each report. Run names may describe the planned training target.</p>'
        '<table><thead><tr><th>Run</th><th>Saved iteration / target</th><th>Checkpoint status</th>'
        '<th>RotGS total (deg)</th><th>ArUco total (deg)</th>'
        '<th>Raw first-view offset (deg)</th></tr></thead><tbody>'
    )
    for path in sorted(report_dir.glob('*/view_poses.json')):
        report = json.loads(path.read_text())
        name = path.parent.name
        target = report['configured_final_iteration']
        status = 'Final checkpoint' if report['iteration'] >= target else 'Earlier checkpoint'
        aruco = '—' if report['aruco_total_deg'] is None else f"{report['aruco_total_deg']:.6f}"
        content += (
            f'<tr><td><a href="{html.escape(name, quote=True)}/index.html">{html.escape(name)}</a></td>'
            f'<td>{report["iteration"]:,} / {target:,}</td><td>{status}</td>'
            f'<td>{report["total_rotation_deg"]:.6f}</td><td>{aruco}</td>'
            f'<td>{report["render_zero_deg"]:.6f}</td></tr>'
        )
    (report_dir / 'index.html').write_text(_html(content + '</tbody></table>'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_paths", nargs="+", type=Path)
    parser.add_argument("--iteration", type=int, default=-1, help="saved iteration; default: latest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, help="override a moved dataset path")
    parser.add_argument("--aruco-csv", type=Path, help="optional existing TAE-format angle table")
    parser.add_argument("--no-history", action="store_true", help="skip saved-checkpoint convergence plots")
    parser.add_argument("--render-views", type=int, default=0, metavar="COUNT",
                        help="render COUNT acquisition views into each report (requires CUDA)")
    args = parser.parse_args(argv)
    if args.render_views < 0:
        parser.error("--render-views must be nonnegative")
    if len({p.resolve().name for p in args.model_paths}) != len(args.model_paths):
        parser.error("model paths must have unique directory names")
    for path in args.model_paths:
        report = extract_view_poses(path, args.iteration, source_path=args.source_path, aruco_csv=args.aruco_csv)
        history = []
        if not args.no_history:
            for iteration in saved_iterations(path):
                if iteration <= report["iteration"]:
                    history.append(report if iteration == report["iteration"] else extract_view_poses(path, iteration, source_path=args.source_path, aruco_csv=args.aruco_csv))
        name = path.resolve().name
        output_dir = args.output_dir / name
        gallery = render_view_gallery(report, output_dir, args.render_views) if args.render_views else None
        write_view_pose_report(report, output_dir, history, rendered_views=gallery)
        print(f"{name}: {len(report['views'])} views, iteration {report['iteration']}, total {report['total_rotation_deg']:.6f} deg, offset {report['render_zero_deg']:.6f} deg")
    write_report_index(args.output_dir)


if __name__ == "__main__":
    main()
