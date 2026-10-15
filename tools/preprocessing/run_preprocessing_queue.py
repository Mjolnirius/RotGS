"""Queue Session_04 30-degree products using R082's preprocessing recipe.

Only current, complete, recipe-matched datasets are reused. Archived versions
remain untouched. Training options such as axis projection belong in train.py;
this queue prepares native RGBA images, camera intrinsics, and TAE calibration.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXPORT_ROOT = Path('/shared/datasets/3D_Scanning_MT/orbitvu/exports/Session_04')
OUTPUT_ROOT = Path('/shared/datasets/3D_Scanning_MT/cleaned/Session_04')
BASELINE = OUTPUT_ROOT / '4_Quoellfrisch_Top_30d_SAM31AC_1435/preprocessing_metadata.json'
LOG_ROOT = ROOT / 'logs/preprocess_queue/session_04_r082'
PREPROCESSOR = ROOT / 'tools/preprocessing/prepare_rotgs_sequence_SAM31.py'
GPU_POLL_SECONDS = 2
GPU_IDLE_REQUIRED_CHECKS = 5
GPU_IDLE_MAX_UTILIZATION = 5
MIN_FREE_BYTES = 10 * 1024**3
RETRY_COUNT = 2
RETRY_DELAY_SECONDS = 300

# Product-specific concepts from the earlier Session_04 preparation. Crop and
# mask settings stay matched to R082; the concept must describe each product.
PRODUCT_PROMPTS = {
    '4_Chips_Blau_30d': 'potato chips bag',
    '4_KaegiFret_Top_30d': 'chocolate bar package',
    '4_Leermond_30d': 'beer bottle',
    '4_Liquor_30d': 'liquor bottle',
    '4_Pepsi_Top_30d': 'beverage can',
    '4_Quoellfrisch_Down_30d': 'beverage can',
    '4_Quoellfrisch_Top_30d': 'beverage can',
    '4_Ramseier_30d': 'juice carton',
    '4_Wasser_Gelb_30d': 'water bottle',
    '4_Weisswein_30d': 'glass bottle',
}


def log(message):
    print(f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] {message}', flush=True)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_recipe(path):
    metadata = json.loads(path.read_text())
    segmentation = metadata['segmentation']
    crop = metadata['automatic_crop']
    tae = metadata['turntable_angle_estimation']
    if (metadata['crop_selection_method'] != 'sam31_automatic'
            or segmentation['inference_passes'] != 2
            or tae['status'] != 'completed'):
        raise ValueError('baseline must contain automatic two-pass SAM 3.1 preprocessing with completed TAE')
    recipe = {
        'cameras': Path(metadata['input_cameras_file']),
        'calibration': Path(metadata['undistortion']['calibration_file']),
        'checkpoint': Path(segmentation['checkpoint']),
        'crop_margin': crop['margin_pixels'],
        'segmentation_max_width': crop['working_size'][0],
        'mask_threshold': segmentation['mask_threshold'],
        'marker_size_mm': tae['marker_size_mm'],
    }
    recipe['cameras_sha256'] = sha256(recipe['cameras'])
    recipe['calibration_sha256'] = sha256(recipe['calibration'])
    if recipe['cameras_sha256'] != metadata['input_cameras_sha256']:
        raise ValueError('R082 input cameras.txt has changed since preprocessing')
    if recipe['calibration_sha256'] != metadata['undistortion']['calibration_sha256']:
        raise ValueError('R082 calibration has changed since preprocessing')
    return recipe


def load_tae_calibration(output, image_names):
    # Reuse training's strict CSV/image/geometry validation without importing
    # scene/__init__.py, which would load training's CUDA extensions in Python 3.12.
    module_name = '_preprocessing_queue_tae_calibration'
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, ROOT / 'scene/tae_calibration.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name].load_tae_calibration(output, image_names)


def png_names(folder):
    return {path.name for path in folder.iterdir() if path.is_file() and path.suffix.lower() == '.png'}


def dataset_problem(output, source, recipe, prompt):
    """Return why a candidate cannot be reused, or None for a complete match."""
    try:
        m = json.loads((output / 'preprocessing_metadata.json').read_text())
        s = m['segmentation']
        u = m['undistortion']
        crop = m['automatic_crop']
        tae = m['turntable_angle_estimation']
        checks = {
            'source': Path(m['input_folder']).resolve() == source.resolve(),
            'camera calibration': (m['input_cameras_sha256'] == recipe['cameras_sha256']
                                   and u['applied'] is True
                                   and u['calibration_sha256'] == recipe['calibration_sha256']),
            'automatic square crop': (m['crop_selection_method'] == 'sam31_automatic'
                                      and m['crop_mode'] == 'smallest_square'
                                      and crop['margin_pixels'] == recipe['crop_margin']
                                      and crop['working_size'][0] == recipe['segmentation_max_width']),
            'SAM 3.1 masks': (m['alpha_mode'] == 'sam31_product'
                             and m['overwrite_alpha_mask'] is True
                             and s['backend'] == 'sam3.1' and s['model'] == 'multiplex'
                             and Path(s['checkpoint']).resolve() == recipe['checkpoint'].resolve()
                             and s['text_prompt'] == prompt
                             and s['threshold_kind'] == 'probability'
                             and s['mask_threshold'] == recipe['mask_threshold']
                             and s['mode'] == 'per_frame_concept'
                             and s['input_view'] == 'cropped' and s['inference_passes'] == 2
                             and s['reused_crop_detection_masks'] is False
                             and s['source_alpha_used'] is False
                             and s['inference_batch_size'] == 1 and s['postprocess_batch_size'] == 1),
            'TAE': (tae['status'] == 'completed' and tae['marker_size_mm'] == recipe['marker_size_mm']),
            'native resolution': (m['output_size'] == m['crop_size']
                                  and m['output_size'][0] == m['output_size'][1]
                                  and s['working_size'][0] == min(recipe['segmentation_max_width'], m['output_size'][0])),
        }
        for name, matches in checks.items():
            if not matches:
                return f'{name} differs from the R082 recipe'
        rows = m['renamed_images']
        images = png_names(output / 'images')
        if (not images or len(rows) != m['image_count'] or len(images) != len(rows)
                or {row['source'] for row in rows} != png_names(source)
                or {row['output'] for row in rows} != images):
            return 'source/output image mapping is incomplete'
        for image_name in sorted(images):
            with Image.open(output / 'images' / image_name) as image:
                if image.mode != 'RGBA' or list(image.size) != m['output_size']:
                    return f'invalid RGBA dimensions: {image_name}'
                image.verify()
        cameras = (output / 'sparse/0/cameras.txt').read_text()
        camera_rows = [line.split() for line in cameras.splitlines() if line.strip() and not line.lstrip().startswith('#')]
        if not camera_rows or any(len(row) != 8 or row[1] != 'PINHOLE'
                                  or list(map(int, row[2:4])) != m['output_size'] for row in camera_rows):
            return 'prepared camera intrinsics are missing or have incorrect dimensions'
        for required in ('sam31_prompts.json', 'mask_review/index.html', 'crop_review/index.html', 'crop_review/detections.json'):
            if not (output / required).is_file():
                return f'missing {required}'
        load_tae_calibration(output, images)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        return f'incomplete or incompatible dataset: {exc}'
    return None


def completed_dataset(source, output_root, recipe, prompt):
    # Only immediate current outputs are considered. In particular, never reuse
    # Outdated_Preprocessing_Files or a manually cropped earlier generation.
    for output in sorted(output_root.glob(f'{source.name}_*')):
        if output.is_dir() and dataset_problem(output, source, recipe, prompt) is None:
            return output
    return None


def preprocessing_command(source, output, recipe, prompt):
    return [sys.executable, str(PREPROCESSOR), str(source),
            '--cameras', str(recipe['cameras']), '--calibration', str(recipe['calibration']),
            '--checkpoint', str(recipe['checkpoint']), '--device', 'cuda',
            '--text-prompt', prompt, '--crop-margin', str(recipe['crop_margin']),
            '--segmentation-max-width', str(recipe['segmentation_max_width']),
            '--mask-threshold', str(recipe['mask_threshold']),
            '--aruco-side-length-mm', str(recipe['marker_size_mm']), '--output', str(output)]


def gpu_snapshot(gpu, own_pgid=None):
    base = ['nvidia-smi', f'--id={gpu}', '--format=csv,noheader,nounits']
    stats = subprocess.check_output([*base, '--query-gpu=utilization.gpu,memory.used,memory.total'], text=True).strip()
    utilization = int(stats.split(',')[0].strip())
    processes = subprocess.check_output([*base, '--query-compute-apps=pid,process_name,used_gpu_memory'], text=True).strip()
    foreign = []
    for row in csv.reader(processes.splitlines()):
        pid, name, memory = (value.strip() for value in row)
        if memory.isdigit() and int(memory) == 0:
            continue
        if own_pgid is not None:
            try:
                if os.getpgid(int(pid)) == own_pgid:
                    continue
            except ProcessLookupError:
                continue
        foreign.append(f'{pid}:{name}:{memory}MiB')
    return utilization, foreign, f'gpu={stats} processes={processes or "none"}'


def record_gpu(log_root, label, detail):
    with (log_root / 'gpu-monitor.log').open('a') as handle:
        handle.write(f'[{datetime.now().astimezone().isoformat(timespec="seconds")}] job={label} {detail}\n')


def wait_for_gpu(gpu, log_root):
    idle = 0
    while idle < GPU_IDLE_REQUIRED_CHECKS:
        try:
            utilization, foreign, detail = gpu_snapshot(gpu)
            record_gpu(log_root, 'waiting', detail)
            idle = idle + 1 if utilization <= GPU_IDLE_MAX_UTILIZATION and not foreign else 0
            if idle:
                log(f'GPU {gpu} idle check {idle}/{GPU_IDLE_REQUIRED_CHECKS}.')
            else:
                log(f'Waiting for GPU {gpu}: utilization={utilization}%, foreign processes={foreign or "none"}.')
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            idle = 0
            log(f'Could not query GPU {gpu}; waiting: {exc}')
        if idle < GPU_IDLE_REQUIRED_CHECKS:
            time.sleep(GPU_POLL_SECONDS)


def wait_for_disk_space(output_root, log_root):
    paths = (output_root, log_root, Path(tempfile.gettempdir()))
    while any(shutil.disk_usage(path).free < MIN_FREE_BYTES for path in paths):
        log('Waiting for at least 10 GiB free on output, log, and temporary filesystems.')
        time.sleep(RETRY_DELAY_SECONDS)


def run_job(command, log_path, gpu, log_root):
    """Monitor an isolated job session; pause it while foreign CUDA work appears."""
    with log_path.open('a') as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                   start_new_session=True, cwd=ROOT)
        paused = False
        try:
            while process.poll() is None:
                try:
                    _, foreign, detail = gpu_snapshot(gpu, own_pgid=process.pid)
                    record_gpu(log_root, log_path.stem, detail)
                    if bool(foreign) != paused:
                        os.killpg(process.pid, signal.SIGSTOP if foreign else signal.SIGCONT)
                        paused = bool(foreign)
                        log(f'{"PAUSE" if paused else "RESUME"} {log_path.stem}: foreign CUDA processes={foreign or "none"}.')
                except ProcessLookupError:
                    break
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
                    record_gpu(log_root, log_path.stem, f'query-failed: {exc}')
                time.sleep(GPU_POLL_SECONDS)
            return process.wait()
        except BaseException:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGCONT)
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='check existing datasets and print pending commands without writing or using CUDA')
    parser.add_argument('--delay-minutes', type=int, default=0)
    parser.add_argument('--source-root', type=Path, default=EXPORT_ROOT)
    parser.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    parser.add_argument('--log-root', type=Path, default=LOG_ROOT)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args(argv)
    if args.delay_minutes < 0 or args.gpu < 0:
        parser.error('delay and GPU index must be non-negative')
    for option in ('source_root', 'output_root', 'baseline', 'log_root'):
        setattr(args, option, getattr(args, option).expanduser().resolve())
    recipe = load_recipe(args.baseline)
    sources = sorted(path for path in args.source_root.glob('*_30d') if path.is_dir())
    if not sources:
        parser.error(f'no *_30d product sequences in {args.source_root}')
    pending = []
    for source in sources:
        if source.name not in PRODUCT_PROMPTS:
            parser.error(f'add a product-specific text prompt for {source.name} to PRODUCT_PROMPTS')
        prompt = PRODUCT_PROMPTS[source.name]
        existing = completed_dataset(source, args.output_root, recipe, prompt)
        if existing:
            log(f'SKIP {source.name}: complete R082-compatible dataset at {existing}')
            continue
        archived = args.output_root / 'Outdated_Preprocessing_Files'
        if any(archived.glob(f'{source.name}_*')):
            log(f'{source.name}: archived preprocessing exists; preparing the current R082 recipe.')
        output = args.output_root / f'{source.name}_SAM31AC_TAE'
        command = preprocessing_command(source, output, recipe, prompt)
        pending.append((source, output, prompt, command))
        log(f'QUEUE {source.name} -> {output}')
        print(f'Command: {shlex.join(command)}', flush=True)
    log(f'{len(sources)} sequences: {len(sources) - len(pending)} already complete, {len(pending)} pending.')
    if args.dry_run or not pending:
        return 0
    if not args.output_root.is_dir() or not recipe['checkpoint'].is_file():
        parser.error('output root and local SAM 3.1 checkpoint must exist')
    if shutil.which('nvidia-smi') is None or importlib.util.find_spec('sam3') is None:
        parser.error('nvidia-smi and SAM 3.1 in .venv-sam31 are required')
    args.log_root.mkdir(parents=True, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    # Share the training queue's lock so the two queues cannot race to start.
    # Unlike train_queue.sh, wait for a running queue instead of exiting.
    with (ROOT / f'.train_queue_gpu{args.gpu}.lock').open('a') as lock:
        log(f'Waiting for the RotGS GPU {args.gpu} queue lock.')
        fcntl.flock(lock, fcntl.LOCK_EX)
        if args.delay_minutes:
            log(f'Waiting {args.delay_minutes} minute(s) before preprocessing.')
            time.sleep(args.delay_minutes * 60)
        for attempt in range(RETRY_COUNT + 1):
            failures = []
            for source, output, prompt, command in pending:
                existing = completed_dataset(source, args.output_root, recipe, prompt)
                if existing:
                    log(f'SKIP {source.name}: complete dataset at {existing}')
                    continue
                if output.exists():
                    log(f'FAILED {source.name}: existing incompatible/incomplete output is preserved: {output}')
                    failures.append((source, output, prompt, command))
                    continue
                wait_for_disk_space(args.output_root, args.log_root)
                wait_for_gpu(args.gpu, args.log_root)
                log_path = args.log_root / f'{source.name}.attempt-{attempt + 1}.log'
                log(f'START {source.name} (attempt {attempt + 1}); log: {log_path}')
                exit_code = run_job(command, log_path, args.gpu, args.log_root)
                problem = dataset_problem(output, source, recipe, prompt)
                if exit_code == 0 and problem is None:
                    log(f'DONE {source.name}: {output}')
                else:
                    log(f'FAILED {source.name} (exit {exit_code}): {problem or "see job log"}')
                    failures.append((source, output, prompt, command))
            pending = failures
            if not pending:
                log('Preprocessing queue finished successfully.')
                return 0
            if attempt < RETRY_COUNT:
                log(f'{len(pending)} incomplete job(s); retrying in {RETRY_DELAY_SECONDS} seconds.')
                time.sleep(RETRY_DELAY_SECONDS)
    log(f'Queue finished with {len(pending)} incomplete job(s) after {RETRY_COUNT} retries.')
    return 1


if __name__ == '__main__':
    def interrupt_queue(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt_queue)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log('Preprocessing queue interrupted.')
        raise SystemExit(130)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        log(f'ERROR: {exc}')
        raise SystemExit(1)
