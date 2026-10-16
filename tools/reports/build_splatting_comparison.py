"""Build E10 from saved image pairs and read-only Gaussian checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from PIL import Image
from plyfile import PlyData
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.reports.research_paths import DATA_ROOT, RESEARCH_ROOT, REPORTS_ROOT
from tools.evaluation.corrected_metrics import (
    CroppedLPIPS, cropped_ssim, foreground_crop, foreground_psnr, load_rgb, load_rgba,
)

COMPARISON = REPORTS_ROOT / 'experiments/E07_axis/comparison'
RUNS = {
    'R058': DATA_ROOT / 'output_RotGS/session_03/Quoellfrisch_top_30d_und_PNGaR_1777_center_xyz_noflow_ccw_densify13k_30k_wandb',
    'R082': DATA_ROOT / 'output_RotGS/session_04/4_Quoellfrisch_Top_30d_SAM31AC_1435_TAE_axis_projected_30k',
    'R089': DATA_ROOT / 'output_3DGS/R089_quoellfrisch_top_colmap_30k_seed0',
}
LABELS = {'R058': 'Earlier RotGS · Session 03', 'R082': 'Selected RotGS · Session 04',
          'R089': 'COLMAP + calibrated 3DGS · Session 04'}
SOURCE = DATA_ROOT / 'cleaned/Session_04/4_Quoellfrisch_Top_30d_SAM31AC_1435/images'
ORDER = sorted(RUNS, reverse=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def unit(v):
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if length < 1e-10:
        raise ValueError('Degenerate camera frame')
    return v / length


def fit_camera_ring(cameras):
    """Recover an arbitrary SfM frame from the camera ring, without TAE."""
    positions = np.array([c['position'] for c in cameras], dtype=float)
    origin = positions.mean(axis=0)
    _, singular, basis = np.linalg.svd(positions - origin, full_matrices=False)
    if singular[1] < 1e-8:
        raise ValueError('Camera centers do not span an orbit plane')
    z = unit(basis[2])
    average_up = np.mean([-np.asarray(c['rotation'])[:, 1] for c in cameras], axis=0)
    if np.dot(z, average_up) < 0:
        z *= -1
    xy = (positions - origin) @ basis[:2].T
    design = np.column_stack((2*xy, np.ones(len(xy))))
    coefficients, *_ = np.linalg.lstsq(design, (xy*xy).sum(axis=1), rcond=None)
    ring_center = origin + coefficients[:2] @ basis[:2]
    radial = positions[0] - ring_center
    front = unit(radial - np.dot(radial, z)*z)
    return ring_center, z, front


def normalized_coordinates(xyz, center, z, front):
    """Align without changing checkpoints; normalize arbitrary SfM scale."""
    z, front = unit(z), unit(front - np.dot(front, z)*z)
    basis = np.column_stack((front, unit(np.cross(z, front)), z))
    local = (np.asarray(xyz) - center) @ basis
    low, high = np.quantile(local[:, 2], [0.01, 0.99])
    height = float(high - low)
    if height <= 1e-10:
        raise ValueError('Degenerate Gaussian height')
    midpoint = (low + high)/2
    center = np.asarray(center) + midpoint*z
    local[:, 2] -= midpoint
    return local / height, center, height


def image_pairs():
    """Join on source identity, never on an assumed enumeration order."""
    pairs = {}
    for rid in ['R058', 'R082']:
        records = json.loads((COMPARISON/rid/'per_view.json').read_text())
        pairs[rid] = [(Path(r['source']), Path(r['render'])) for r in records]
    image_map = json.loads((RUNS['R089']/'test/ours_30000/image_map.json').read_text())
    lookup = {x['source']: x['file'] for x in image_map}
    expected = {p.name for p, _ in pairs['R082']}
    if set(lookup) != expected or len(lookup) != len(image_map):
        raise ValueError('R082/R089 held-out source identities differ or repeat')
    pairs['R089'] = [(SOURCE/p.name, RUNS['R089']/'test/ours_30000/renders'/lookup[p.name])
                     for p, _ in pairs['R082']]
    for rid, items in pairs.items():
        if len({p.name for p, _ in items}) != len(items):
            raise ValueError(f'Duplicate held-out source identities in {rid}')
    return pairs


def evaluate(report, pairs):
    lpips = CroppedLPIPS(torch.device('cuda'))
    records, targets = {}, {}
    for rid in ORDER:
        records[rid] = []
        media = report/'media'/rid
        media.mkdir(parents=True, exist_ok=True)
        for source, render_path in pairs[rid]:
            rgb, alpha = load_rgba(source)
            # Common white-composited 8-bit target, with original soft alpha.
            gt = np.round((rgb*alpha[...,None] + 1-alpha[...,None])*255)/255
            rendered = load_rgb(render_path)
            if rendered.shape != gt.shape:
                raise ValueError(f'Unaligned image shapes: {source} and {render_path}')
            crop = foreground_crop(alpha)
            mse = np.mean((rendered.astype(float)-gt.astype(float))**2)
            record = dict(source=source.name, source_path=str(source), render_path=str(render_path),
                          width=gt.shape[1], height=gt.shape[0],
                          full_psnr=float(-10*np.log10(mse)),
                          foreground_psnr=foreground_psnr(rendered, gt, alpha),
                          crop_ssim=cropped_ssim(rendered, gt, crop, torch.device('cuda')),
                          crop_lpips=lpips(rendered, gt, crop),
                          crop=[crop.x0,crop.y0,crop.x1,crop.y1],
                          target_sha256=hashlib.sha256(np.uint8(gt*255).tobytes()).hexdigest())
            records[rid].append(record)
            # Same display pipeline for all runs; unamplified absolute RGB error.
            panels = [gt, rendered, np.abs(rendered-gt)]
            strip = Image.fromarray(np.uint8(np.clip(np.concatenate(panels,axis=1),0,1)*255))
            strip.thumbnail((1440,480))
            strip.save(media/f'heldout_{source.stem}.jpg',quality=91)
            # Two detail crops use the same fractional image regions within each
            # native target. Session 03 is explicitly labelled as a separate target.
            for label, box in [('print',(0.35,0.30,0.68,0.60)),('lower_edge',(0.22,0.66,0.80,0.90))]:
                h,w=gt.shape[:2]; x0,y0,x1,y1=[round(v*s) for v,s in zip(box,(w,h,w,h))]
                detail = np.concatenate([p[y0:y1,x0:x1] for p in panels],axis=1)
                image=Image.fromarray(np.uint8(np.clip(detail,0,1)*255)); image.thumbnail((1440,480))
                image.save(media/f'{label}_{source.stem}.jpg',quality=93)
            print(f'{rid} {source.name}: foreground PSNR {record["foreground_psnr"]:.3f}',flush=True)
    for left,right in zip(records['R082'],records['R089']):
        if left['target_sha256'] != right['target_sha256']:
            raise ValueError('Session 04 targets do not match byte for byte')
    save_json(report/'per_view.json',records)
    del lpips
    torch.cuda.empty_cache()
    return records


def load_geometry(rid):
    from tools.inspection.render_orbit_report import load_cfg, camera_frame
    from scene.gaussian_model import GaussianModel

    path=RUNS[rid];checkpoint=path/'point_cloud/iteration_30000/point_cloud.ply'
    vertices=PlyData.read(str(checkpoint))['vertex']
    xyz=np.column_stack([vertices[n] for n in ('x','y','z')]).astype(float)
    opacity=1/(1+np.exp(-np.asarray(vertices['opacity'],dtype=float)))
    scales=np.exp(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).astype(float))
    if not np.isfinite(xyz).all() or not np.isfinite(scales).all():
        raise ValueError(f'Non-finite checkpoint: {checkpoint}')
    cfg=load_cfg(path)
    if rid=='R089':
        cameras=json.loads((path/'cameras.json').read_text())
        center,z,front=fit_camera_ring(cameras)
        # Shift the recovered ring axis to the model's vertical midpoint.
        model=GaussianModel(cfg.sh_degree,fixed_camera=False)
        model.load_ply(str(checkpoint))
        frame_method='Camera-ring plane/circle fit; vertical midpoint from Gaussian quantiles; no TAE.'
    else:
        model=GaussianModel(cfg.sh_degree,fixed_camera=True,
                            axis_mode=getattr(cfg,'axis_mode','free'),
                            axis_constraint_mode=getattr(cfg,'axis_constraint_mode','clamp'),
                            axis_tilt_min_deg=getattr(cfg,'axis_tilt_min_deg',0.),
                            axis_tilt_max_deg=getattr(cfg,'axis_tilt_max_deg',90.),
                            axis_tilt_init_deg=getattr(cfg,'axis_tilt_init_deg',30.),
                            axis_side_init_deg=getattr(cfg,'axis_side_init_deg',0.),
                            axis_side_limit_deg=getattr(cfg,'axis_side_limit_deg',5.),
                            center_max_offset=getattr(cfg,'center_max_offset',0.25))
        # Read only camera metadata/images. Scene construction regenerates
        # random.ply even when loading a trained checkpoint, so avoid it here.
        from scene.colmap_loader import read_intrinsics_binary, read_intrinsics_text
        from scene.dataset_readers import readFixedCameras
        from scene.tae_calibration import load_tae_calibration
        source=Path(cfg.source_path)
        try:
            intrinsics=read_intrinsics_binary(str(source/'sparse/0/cameras.bin'))
        except FileNotFoundError:
            intrinsics=read_intrinsics_text(str(source/'sparse/0/cameras.txt'))
        image_dir=source/(cfg.images or 'images')
        image_names=sorted(p.name for p in image_dir.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg'])
        calibration=load_tae_calibration(str(source),image_names=image_names) if getattr(cfg,'leverage_TAE',False) else None
        cameras,_=readFixedCameras(cam_intrinsics=intrinsics,images_folder=str(image_dir),
                                  test_cam_names_list=image_names[::8],angle_noise_std=0.,
                                  rotation_direction=getattr(cfg,'rotation_direction',1),tae_calibration=calibration)
        model.load_ply(str(checkpoint))
        camera=cameras[0]
        template=SimpleNamespace(camera_center=torch.tensor(-np.asarray(camera.R)@np.asarray(camera.T)))
        _,center,_,z,front,_=camera_frame(model,template)
        frame_method='Saved RotGS axis/center and acquisition-camera direction; vertical midpoint from Gaussian quantiles.'
    coordinates,center,height=normalized_coordinates(xyz,center,z,front)
    stats=dict(count=len(xyz),height_model_units=height,
               opacity_median=float(np.median(opacity)),opacity_below_005=float(np.mean(opacity<.05)),
               scale_median_over_height=float(np.median(scales)/height),
               aspect_ratio_median=float(np.median(scales.max(axis=1)/scales.min(axis=1))),
               center=center.tolist(),vertical=unit(z).tolist(),front=unit(front).tolist(),
               frame_method=frame_method,checkpoint_sha256=sha256(checkpoint))
    model.fixed_camera=False
    return model,coordinates,opacity,scales/height,stats


def draw_diagnostics(report, geometries):
    figure,axes=plt.subplots(3,3,figsize=(12,11),constrained_layout=True)
    for row,rid in enumerate(ORDER):
        _,c,_,_,_=geometries[rid]
        r=np.hypot(c[:,0],c[:,1])
        for col,(x,y,xlabel,ylabel,extent) in enumerate([
            (c[:,0],c[:,1],'Front axis / height','Side axis / height',(-.8,.8,-.8,.8)),
            (c[:,0],c[:,2],'Front axis / height','Vertical / height',(-.8,.8,-.7,.7)),
            (r,c[:,2],'Radius / height','Vertical / height',(0,1,-.7,.7))]):
            ax=axes[row,col];a,b,d,e=extent
            hist,xe,ye=np.histogram2d(x,y,bins=100,range=((a,b),(d,e)))
            im=ax.imshow(np.ma.masked_equal(hist.T/len(c),0),origin='lower',extent=extent,
                         aspect='equal',cmap='magma',norm=LogNorm(vmin=1e-6,vmax=.02))
            ax.set(title=f'{rid} · {len(c):,} centers',xlabel=xlabel,ylabel=ylabel)
    figure.colorbar(im,ax=axes.ravel().tolist(),label='Fraction of all centers / bin (log)',shrink=.6)
    figure.savefig(report/'gaussian_centers.png',dpi=160);plt.close(figure)
    figure,axes=plt.subplots(1,3,figsize=(13,3.8),constrained_layout=True)
    for rid in ORDER:
        _,c,opacity,scales,stats=geometries[rid]
        axes[0].hist(opacity,bins=np.linspace(0,1,51),density=True,histtype='step',label=rid)
        axes[1].hist(np.log10(scales.ravel()),bins=np.linspace(-7,0,71),density=True,histtype='step',label=rid)
        axes[2].hist(np.log10(scales.max(axis=1)/scales.min(axis=1)),bins=np.linspace(0,6,61),density=True,histtype='step',label=rid)
    for ax,label in zip(axes,['Opacity','log10(scale / model height)','log10(max scale / min scale)']):
        ax.set(xlabel=label,ylabel='Density');ax.legend()
    figure.savefig(report/'gaussian_properties.png',dpi=160);plt.close(figure)


def render_previews(report,rid,model,stats,frames,size):
    from tools.inspection.render_orbit_report import virtual_camera, MediaWriters
    from gaussian_renderer import set_rasterizer,render
    center=np.asarray(stats['center']);z=np.asarray(stats['vertical']);front=np.asarray(stats['front'])
    side=unit(np.cross(z,front));distance=2.5*stats['height_model_units']
    fov=math.radians(35)
    template=SimpleNamespace(znear=max(.0001,distance/1000),zfar=distance*100,
                             FoVx=fov,FoVy=fov,image_width=size,image_height=size)
    media=report/'media'/rid;media.mkdir(parents=True,exist_ok=True)
    settings=dict(frames=frames,size=size,fov_deg=35,distance_over_height=2.5,
                  coordinate_frame=stats,renderer='RotGS rasterizer; stationary Gaussian SH; moving virtual camera; no exposure transform')
    cache=media/'orbit_settings.json'
    existing=json.loads(cache.read_text()) if cache.exists() else None
    trajectories={'orbit_30deg':30,'orbit_front':0,'orbit_top':85,'orbit_vertical':None}
    if existing==settings and all((media/f'{name}.{suffix}').exists()
       for name in trajectories for suffix in ['mp4','gif']):
        print('Reusing preview media',rid,flush=True);return settings
    pipeline=SimpleNamespace(debug=False)
    background=torch.ones(3,device='cuda')
    for name,elevation in trajectories.items():
        writers=MediaWriters(media/f'{name}.mp4',media/f'{name}.gif',media/f'{name}_poster.jpg',
                             fps=30,gif_fps=10,gif_width=360,source_size=(size,size))
        try:
            with torch.inference_mode():
                for i in range(frames):
                    theta=2*math.pi*i/frames
                    if elevation is None:
                        direction=math.cos(theta)*front+math.sin(theta)*z
                        up=-math.sin(theta)*front+math.cos(theta)*z
                    else:
                        el=math.radians(elevation)
                        direction=math.cos(el)*(math.cos(theta)*front+math.sin(theta)*side)+math.sin(el)*z
                        up=z
                    camera=virtual_camera(template,center+distance*direction,center,up)
                    raster=set_rasterizer(camera,model,pipeline,background)
                    rendered=render(raster,model)['render'].clamp(0,1)
                    rgb=rendered.mul(255).byte().permute(1,2,0).cpu().numpy()
                    writers.append(rgb)
        finally:
            writers.close()
        print('Rendered',rid,name,flush=True)
    save_json(cache,settings)
    return settings


def run_timing(rid):
    if rid=='R089':
        m=json.loads((RUNS[rid]/'run_metadata.json').read_text())
        return dict(seconds=m['seconds'],scope='Training wall time; excludes COLMAP/staging',
                    peak_allocated_gib=m['peak_allocated_bytes']/2**30,
                    peak_reserved_gib=m['peak_reserved_bytes']/2**30,gpu=m['gpu'])
    cfg=__import__('tools.inspection.render_orbit_report',fromlist=['load_cfg']).load_cfg(RUNS[rid])
    from tools.inspection.render_orbit_report import load_run_timing
    timing=load_run_timing(RUNS[rid],30000,cfg)
    # Historical loader can return datetime objects; keep provenance printable.
    return {k:v.isoformat() if isinstance(v,datetime) else v for k,v in timing.items()}


def write_report(report,records,stats,timing,manifest):
    means={rid:{metric:float(np.mean([r[metric] for r in rows])) for metric in
                ['full_psnr','foreground_psnr','crop_ssim','crop_lpips']} for rid,rows in records.items()}
    delta={metric:means['R089'][metric]-means['R082'][metric] for metric in means['R082']}
    summary=dict(means=means,paired_delta_R089_minus_R082=delta,gaussians=stats,timing=timing,
                 matched_runs=['R082','R089'],historical_reference='R058',views=10)
    save_json(report/'summary.json',summary)
    notes=(f'On the same ten Session 04 targets, R089 minus R082 is {delta["foreground_psnr"]:+.3f} dB foreground PSNR, '
           f'{delta["full_psnr"]:+.3f} dB full-frame PSNR, {delta["crop_ssim"]:+.5f} crop SSIM, and '
           f'{delta["crop_lpips"]:+.5f} crop LPIPS. Lower LPIPS is better. '
           f'R089 stores {stats["R089"]["count"]:,} Gaussians versus {stats["R082"]["count"]:,} for R082.')
    rows=''
    costs=''
    for rid in ORDER:
        s=means[rid];group='Session 04 shared target' if rid!='R058' else 'Session 03 native target'
        rows+=f'<tr><td><a href="#{rid}">{rid}</a></td><td>{group}</td><td>{s["full_psnr"]:.3f}</td><td>{s["foreground_psnr"]:.3f}</td><td>{s["crop_ssim"]:.5f}</td><td>{s["crop_lpips"]:.5f}</td><td>{stats[rid]["count"]:,}</td></tr>'
        t=timing[rid];seconds=t.get('seconds',t.get('duration_seconds'))
        duration=f'{seconds/60:.2f} min' if seconds is not None else 'Unavailable'
        allocated=f'{t["peak_allocated_gib"]:.2f} GiB' if 'peak_allocated_gib' in t else 'Not recovered'
        reserved=f'{t["peak_reserved_gib"]:.2f} GiB' if 'peak_reserved_gib' in t else 'Not recovered'
        scope=t.get('scope','W&B training runtime; excludes preprocessing')
        costs+=f'<tr><td>{rid}</td><td>{duration}</td><td>{allocated}</td><td>{reserved}</td><td>{html.escape(scope)}</td></tr>'
    sections=[]
    for rid in ORDER:
        videos=''.join(f'<figure><video controls loop muted playsinline preload="none" poster="media/{rid}/{name}_poster.jpg" src="media/{rid}/{name}.mp4"></video><figcaption>{label}</figcaption></figure>'
                      for name,label in [('orbit_30deg','30° orbit'),('orbit_front','Level orbit'),('orbit_top','85° top orbit'),('orbit_vertical','Vertical orbit')])
        sections.append(f'<section id="{rid}" data-video-section><div class="section-heading"><h2>{rid} — {LABELS[rid]}</h2><button class="section-play-toggle" type="button" aria-pressed="false" aria-label="Play or pause {rid} orbit videos">Play section</button></div><p><a href="{RUNS[rid]}">Model and original evidence</a> · 30,000 iterations · {stats[rid]["count"]:,} Gaussians.</p><div class="videos">{videos}</div><details><summary>Recorded cost and preview frame</summary><pre>{html.escape(json.dumps({"timing":timing[rid],"frame":stats[rid]},indent=2))}</pre></details></section>')
    sources=[p['source'] for p in records['R082']]
    options=''.join(f'<option value="{Path(s).stem}">{s} · nominal {int(Path(s).stem)*5}°</option>' for s in sources)
    images=''.join(f'<figure><img id="image-{rid}" src="media/{rid}/heldout_0000.jpg" alt="{rid}: target, render, absolute RGB error"><figcaption>{rid}: {LABELS[rid]}</figcaption></figure>' for rid in ORDER)
    detail=''.join(f'<figure><img id="detail-{rid}" src="media/{rid}/print_0000.jpg" alt="{rid}: print detail"><figcaption>{rid} native image crop</figcaption></figure>' for rid in ORDER)
    html_text=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E10 · Gaussian Splatting Pipeline Comparison</title>
<style>body{{font:16px/1.55 system-ui,sans-serif;background:#101720;color:#e8eff8;max-width:1500px;margin:2rem auto;padding:0 1rem}}a{{color:#82c8ff}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.6rem;text-align:left;border-bottom:1px solid #405064}}.scroll{{overflow:auto}}section{{margin:3rem 0}}.videos{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}}figure{{margin:.8rem 0}}img,video{{width:100%;display:block}}select{{margin:.5rem;padding:.5rem}}.notice{{background:#1d2c3e;padding:1rem;border-left:4px solid #82c8ff}}pre{{overflow:auto;font-size:.8rem}}figcaption{{font-size:.9rem;color:#bccadc}}nav a{{margin-right:1rem}}</style>
<style>.playback-toolbar,.section-heading{{display:flex;align-items:center;justify-content:space-between;gap:1rem;flex-wrap:wrap}}.playback-toolbar{{justify-content:flex-start;margin:1.5rem 0}}button{{font:inherit;font-weight:600;color:#fff;background:#174570;border:1px solid #71b8f0;border-radius:.5rem;padding:.6rem 1rem;cursor:pointer;white-space:nowrap}}button:hover{{background:#205988}}button:focus-visible{{outline:3px solid #fff;outline-offset:3px}}button[aria-pressed="true"]{{background:#68471a;border-color:#d49b45}}#playback-status{{color:#bccadc}}</style></head>
<body><h1>E10 — Gaussian Splatting Pipeline Comparison</h1><nav><a href="../../index.html">Report catalogue</a><a href="../../../experiments/E10_splatting_methods/study.md">Study</a><a href="../../../experiments/E10_splatting_methods/report.md">Narrative</a><a href="../../../RUN_LOG.md#r089">R089 ledger</a></nav>
<div class="playback-toolbar"><button id="play-all" type="button" aria-pressed="false">Play all</button><span id="playback-status" role="status" aria-live="polite"></span></div>
<p>30k checkpoints · ten held-out photometric views · reviewed 16 October 2026, Europe/Zurich.</p><p>{notes}</p>
<div class="notice">R082 and R089 share the same 1435 px Session 04 RGB/soft-alpha target. R058 uses its own 1777 px Session 03 capture and is a historical reference. Scores are mean per-view values from saved PNGs using one metric implementation. COLMAP used all views for geometry; ten views were held out from photometric training. Native rendering, initialization, losses and growth schedules differ. This is a single-seed comparison of complete pipelines.</div>
<h2>Saved-image evaluation</h2><div class="scroll"><table><thead><tr><th>Run</th><th>Target</th><th>Full PSNR ↑</th><th>FG PSNR ↑</th><th>Crop SSIM ↑</th><th>Crop LPIPS ↓</th><th>Gaussians</th></tr></thead><tbody>{rows}</tbody></table></div>
<p><a href="per_view.json">Per-view results</a> · <a href="summary.json">Summary, Gaussian statistics and cost</a> · <a href="manifest.json">Source hashes and protocols</a>. Native 3DGS metrics remain in the original model directory.</p>
<h2>Recorded training cost</h2><div class="scroll"><table><thead><tr><th>Run</th><th>Training duration</th><th>Peak allocated</th><th>Peak reserved</th><th>Timing scope</th></tr></thead><tbody>{costs}</tbody></table></div><p>R089 used an RTX 5090. Historical timing sources and available hardware records are retained in the summary. COLMAP and preparation are excluded; these durations do not establish an end-to-end speedup.</p>
<section><h2>Held-out views and details</h2><label>Source view<select id="view">{options}</select></label><p>Each strip shows target | saved render | unamplified absolute RGB error. Session 03 images share nominal angles but have different framing and lighting.</p>{images}<label>Detail region<select id="region"><option value="print">Print</option><option value="lower_edge">Lower edge</option></select></label>{detail}</section>
<section><h2>Gaussian distributions</h2><p>Frames use the saved RotGS axis or a COLMAP camera-ring fit. Coordinates and scales are divided by each model's 1–99% Gaussian-center height. This removes arbitrary SfM scale; it is a shape diagnostic, not physical geometry validation. All centers contribute to histogram denominators; displayed bounds can clip outliers.</p><a href="gaussian_centers.png"><img src="gaussian_centers.png" alt="Normalized center distributions with shared binning and density scale"></a><a href="gaussian_properties.png"><img src="gaussian_properties.png" alt="Opacity, normalized scale and aspect ratio distributions"></a></section>
<section><h2>Orbit previews</h2><p>All previews use the current RotGS inspection rasterizer, moving virtual cameras, stationary SH coefficients, white background, 35° field of view, and distance 2.5 times each model's normalized height. These previews are separate from the native saved renders used for scoring. Novel elevations are extrapolations.</p></section>{''.join(sections)}
<script>
const runs={json.dumps(ORDER)};
function update() {{
  const view=document.getElementById('view').value;
  const region=document.getElementById('region').value;
  for (const id of runs) {{
    document.getElementById('image-'+id).src='media/'+id+'/heldout_'+view+'.jpg';
    document.getElementById('detail-'+id).src='media/'+id+'/'+region+'_'+view+'.jpg';
  }}
}}
document.getElementById('view').addEventListener('change',update);
document.getElementById('region').addEventListener('change',update);

const playAll=document.getElementById('play-all');
const playbackStatus=document.getElementById('playback-status');
const allVideos=[...document.querySelectorAll('video')];
const videoSections=[...document.querySelectorAll('[data-video-section]')];
const isPlaying=video => !video.paused && !video.ended;
function syncPlaybackButtons() {{
  const anyPlaying=allVideos.some(isPlaying);
  playAll.textContent=anyPlaying ? 'Pause all' : 'Play all';
  playAll.setAttribute('aria-pressed',String(anyPlaying));
  for (const section of videoSections) {{
    const button=section.querySelector('.section-play-toggle');
    const playing=[...section.querySelectorAll('video')].some(isPlaying);
    button.textContent=playing ? 'Pause section' : 'Play section';
    button.setAttribute('aria-pressed',String(playing));
  }}
}}
function setPlayback(videos,shouldPlay) {{
  playbackStatus.textContent='';
  for (const video of videos) {{
    if (shouldPlay) {{
      video.play().catch(() => {{
        playbackStatus.textContent='Some videos could not start. Use their individual play controls.';
        syncPlaybackButtons();
      }});
    }} else {{
      video.pause();
    }}
  }}
  syncPlaybackButtons();
}}
playAll.addEventListener('click',() => setPlayback(allVideos,!allVideos.some(isPlaying)));
for (const section of videoSections) {{
  const button=section.querySelector('.section-play-toggle');
  const videos=[...section.querySelectorAll('video')];
  button.addEventListener('click',() => setPlayback(videos,!videos.some(isPlaying)));
}}
for (const video of allVideos) {{
  for (const event of ['play','pause','ended']) video.addEventListener(event,syncPlaybackButtons);
}}
syncPlaybackButtons();
</script></body></html>'''
    (report/'index.html').write_text(html_text)
    narrative=RESEARCH_ROOT/'experiments/E10_splatting_methods/report.md'
    generated='\n'.join(['<!-- BEGIN GENERATED RESULTS -->',notes,'',
        '| Run | Target | Full PSNR | Foreground PSNR | Crop SSIM | Crop LPIPS | Gaussians |',
        '|---|---|---:|---:|---:|---:|---:|']+
        [f'| {rid} | {"Session 04 shared" if rid!="R058" else "Session 03 native"} | {means[rid]["full_psnr"]:.3f} | {means[rid]["foreground_psnr"]:.3f} | {means[rid]["crop_ssim"]:.5f} | {means[rid]["crop_lpips"]:.5f} | {stats[rid]["count"]:,} |' for rid in ORDER]+
        ['<!-- END GENERATED RESULTS -->'])
    if narrative.exists():
        old=narrative.read_text();start=old.index('<!-- BEGIN GENERATED RESULTS -->');end=old.index('<!-- END GENERATED RESULTS -->')+len('<!-- END GENERATED RESULTS -->')
        narrative.write_text(old[:start]+generated+old[end:])
    else:
        narrative.write_text('# E10 — Comparison report\n\n[Study](study.md) · [Visual report](../../reports/experiments/E10_splatting_methods/index.html)\n\n'+generated+'\n\n## Interpretation and limits\n\nThese are saved-checkpoint, single-seed pipeline results. Session 04 scores use identical targets and metric code across native renderers. COLMAP geometry uses all views. R058 has a different Session 03 capture/preparation and is not part of the paired ranking. Normalized Gaussian distributions and novel-view previews do not establish physical geometry accuracy. Training-only timings exclude preprocessing and COLMAP; compare their recorded scope before drawing cost conclusions.\n\n## Your notes\n\n_Add observations and decisions here. This section is preserved when rebuilding._\n')
    return summary


def refresh_presentation(report):
    """Refresh page and chart layout from stored results and frame metadata."""
    records=json.loads((report/'per_view.json').read_text())
    summary=json.loads((report/'summary.json').read_text())
    manifest=json.loads((report/'manifest.json').read_text())
    geometries={}
    for rid in ORDER:
        stats=summary['gaussians'][rid]
        vertices=PlyData.read(str(RUNS[rid]/'point_cloud/iteration_30000/point_cloud.ply'))['vertex']
        xyz=np.column_stack([vertices[n] for n in ['x','y','z']]).astype(float)
        z=unit(stats['vertical'])
        front=np.asarray(stats['front'])
        front=unit(front-np.dot(front,z)*z)
        basis=np.column_stack((front,unit(np.cross(z,front)),z))
        height=stats['height_model_units']
        coordinates=(xyz-np.asarray(stats['center']))@basis/height
        opacity=1/(1+np.exp(-np.asarray(vertices['opacity'],dtype=float)))
        scales=np.exp(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).astype(float))/height
        geometries[rid]=(None,coordinates,opacity,scales,stats)
    draw_diagnostics(report,geometries)
    write_report(report,records,summary['gaussians'],summary['timing'],manifest)
    manifest['run_ids']=ORDER
    manifest['presentation']={
        'updated_at':datetime.now(timezone.utc).isoformat(),
        'builder_sha256':sha256(Path(__file__)),
        'order':'Descending run ID',
        'playback':'Global and per-section play/pause toggles',
    }
    save_json(report/'manifest.json',manifest)
    print('Updated presentation:',report/'index.html',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',type=Path,default=REPORTS_ROOT/'experiments/E10_splatting_methods')
    parser.add_argument('--frames',type=int,default=120)
    parser.add_argument('--size',type=int,default=720)
    parser.add_argument('--refresh-layout',action='store_true',
                        help='refresh HTML and charts from saved results; keep evaluation and videos')
    args=parser.parse_args()
    if args.frames<2 or args.size<32 or args.size%2:
        parser.error('frames must be >=2; size must be an even integer >=32')
    report=args.report_dir;report.mkdir(parents=True,exist_ok=True)
    if args.refresh_layout:
        refresh_presentation(report)
        return
    pairs=image_pairs()
    inputs={Path(__file__),REPO/'tools/evaluation/corrected_metrics.py',REPO/'utils/loss_utils.py',
            REPO/'gaussian_renderer/__init__.py',REPO/'tools/inspection/render_orbit_report.py',
            REPO/'scene/gaussian_model.py',REPO/'scene/dataset_readers.py',
            REPO/'scene/colmap_loader.py',REPO/'scene/tae_calibration.py',REPO/'utils/graphics_utils.py'}
    for rid,items in pairs.items():
        inputs.add(RUNS[rid]/'point_cloud/iteration_30000/point_cloud.ply')
        inputs.add(RUNS[rid]/'cfg_args')
        inputs.update((RUNS[rid]/'point_cloud/iteration_30000').glob('*.npy'))
        inputs.update((RUNS[rid]/'wandb').glob('run-*/files/wandb-summary.json'))
        if rid!='R089':
            from tools.inspection.render_orbit_report import load_cfg
            source=Path(load_cfg(RUNS[rid]).source_path)
            inputs.update((source/'sparse/0').glob('cameras.*'))
            inputs.update((source/'turntable_angle_estimation').glob('*.json'))
            inputs.update((source/'turntable_angle_estimation').glob('*.csv'))
        for source,render in items:inputs.update([source,render])
    inputs.update([RUNS['R089']/'cameras.json',RUNS['R089']/'run_metadata.json',
                   RUNS['R089']/'metric_provenance_30000.json',RUNS['R089']/'test/ours_30000/image_map.json'])
    hashes={str(p):sha256(p) for p in sorted(inputs)}
    old=json.loads((report/'manifest.json').read_text()) if (report/'manifest.json').exists() else {}
    media_complete=all((report/'media'/rid/f'{kind}_{p.stem}.jpg').exists()
                       for rid,items in pairs.items() for p,_ in items
                       for kind in ['heldout','print','lower_edge'])
    if old.get('input_sha256')==hashes and (report/'per_view.json').exists() and media_complete:
        records=json.loads((report/'per_view.json').read_text());print('Reusing verified saved-image evaluation',flush=True)
    else:
        records=evaluate(report,pairs)
    geometries={};stats={};timing={};preview={}
    for rid in ORDER:
        geometries[rid]=load_geometry(rid)
        stats[rid]=geometries[rid][-1];timing[rid]=run_timing(rid)
        preview[rid]=render_previews(report,rid,geometries[rid][0],stats[rid],args.frames,args.size)
    draw_diagnostics(report,geometries)
    manifest=dict(schema_version=1,study='E10',run_ids=ORDER,iteration=30000,
                  generated_at=datetime.now(timezone.utc).isoformat(),input_sha256=hashes,
                  image_protocol='8-bit saved RGB; white-composited original source RGB; original soft alpha; mean per-view dB; foreground crop alpha>0.001 plus 8px; SSIM; LPIPS-VGG v0.1 [-1,1]; no metric resizing',
                  geometry_protocol='All-view SfM; photometric training/test split only',
                  scope='R082/R089 matched Session 04 targets; R058 separate native Session 03 reference',
                  preview_settings=preview)
    write_report(report,records,stats,timing,manifest)
    save_json(report/'manifest.json',manifest)
    (report/'README.md').write_text('# E10 evidence bundle\n\n[Visual report](index.html) · [Study](../../../experiments/E10_splatting_methods/study.md)\n\nRebuild from `/workspaces/RotGS` with `.venv/bin/python tools/reports/build_splatting_comparison.py`. Source hashes, native/preview renderer distinctions, targets and frame settings are in `manifest.json`. Original models and native metric files are never edited. The narrative keeps an editable notes section.\n')
    print('Report written:',report/'index.html',flush=True)


if __name__=='__main__':
    main()
