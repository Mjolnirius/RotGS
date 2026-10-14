# SAM 3.1 product masks for RotGS

`prepare_rotgs_sequence_SAM31.py` provides a separate command and the callable
`prepare_rotgs_sequence_SAM31`. It uses Meta's SAM 3.1 multiplex checkpoint,
automatic crop discovery across every view, and automatic text-based masks.
The existing six-view crop editor is available with `--manual_crop`; an optional
mask box/click editor is available with `--interactive`. The result is a new
RGBA dataset with calibrated, cropped camera intrinsics, a crop review, and a mask review.

## Installation

Use a separate Python 3.12 environment. Do not reuse the Python 3.10 training
environment or the SAM 2.1 environment. Meta's current implementation requires
Python 3.12 or newer, PyTorch 2.7 or newer, and CUDA 12.6 or newer. This adapter
also checks bfloat16 GPU support. FlashAttention 3 and compilation are disabled.

Run from the repository root:

```bash
uv venv .venv-sam31 --python 3.12
uv pip install --python .venv-sam31/bin/python torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv-sam31/bin/python 'git+https://github.com/facebookresearch/sam3.git@2345a4ad109ac29c569da749c91d84f10dc08c40' opencv-python pillow 'numpy==1.26.4' tqdm scipy 'matplotlib<3.11' huggingface_hub einops psutil pycocotools 'setuptools<81'
```

Request access to [facebook/sam3.1](https://huggingface.co/facebook/sam3.1),
then authenticate and download the **SAM 3.1** checkpoint:

```bash
.venv-sam31/bin/hf auth login
.venv-sam31/bin/hf download facebook/sam3.1 sam3.1_multiplex.pt --local-dir checkpoints/sam31
```

The preprocessing command requires local weights and does not download weights.
The current SAM repository is needed for `build_sam3_predictor(version="sam3.1")`;
SAM 3 weights and SAM 2.1 weights cannot replace this checkpoint.
See the [official installation instructions](https://github.com/facebookresearch/sam3)
and [SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).

## Default automatic run (SAM31AC)

```bash
uv run --no-project --python .venv-sam31/bin/python \
  tools/preprocessing/prepare_rotgs_sequence_SAM31.py \
  /shared/datasets/3D_Scanning_MT/orbitvu/exports/Session_04/4_Quoellfrisch_Top_30d \
  --cameras /shared/datasets/3D_Scanning_MT/orbitvu/exports/Session_04/cameras.txt \
  --calibration /shared/datasets/3D_Scanning_MT/orbitvu/exports/Session_04/Calibration_04B/camera_calibration.npz \
  --checkpoint checkpoints/sam31/sam3.1_multiplex.pt \
  --text-prompt 'beverage can'
```

No crop editor opens by default. SAM 3.1 detects the text concept independently
on every full, undistorted view, preserving all components of the selected
object for crop discovery. Each mask's bounding box is mapped back to original
undistorted pixels, rounding outward. The union of every view's box determines
the smallest shared square. That square is enlarged by **10 native pixels on
all four sides**; this padding does not expand the alpha mask. Adjust it with
`--crop-margin PIXELS` (zero is allowed).

Crop discovery uses `--segmentation-max-width` as its working-width limit. It
tries the full text prompt, then its last word when different. It does not use
a full-image box fallback. If a view is missing, a detected object touches the
image boundary, or the square plus margin cannot fit within valid undistorted
pixels, preparation stops and suggests correcting the prompt or using
`--manual_crop`. Source alpha does not guide detection.

The default workflow uses **two SAM inference passes**:

1. Detect the product on every full undistorted view and calculate the shared square.
2. Segment every cropped view to create the final alpha masks.

The second pass uses RGB cropped directly from the native undistorted image,
limited by `--segmentation-max-width`; it does not reuse the first-pass masks.
The same text query now sees an object occupying more of SAM's internal input,
which improves available edge detail. For a 1435-pixel crop, the default 1600-pixel
limit retains all native crop pixels. Holes and disconnected parts of the selected
object are preserved. The discovery model/session closes before the final
segmentation model/session is created, keeping the two models from occupying
GPU memory together. RGB export uses the native undistorted source pixels.

Inspect `crop_review/index.html` for each view's magenta first-pass detection box
and green shared crop, and `mask_review/index.html` for final second-pass silhouettes.
Native detection boxes, first-pass working size, margin, and crop are saved in
`crop_review/detections.json` and preprocessing metadata. Segmentation metadata
records the cropped working size, two inference passes, and
`reused_crop_detection_masks: false`.

SAM 3.1's video defaults batch 16 neighboring views. This adapter bounds inference
and postprocessing batch sizes to **1**. The installed interactive `add_prompt`
implementation already disables batched grounding, so its 16-view configuration
alone does not explain the observed failure. Single-frame postprocessing reduces
output-buffer memory. Closing the segmenter exits both persistent predictor
and tracker autocast contexts in reverse order, releases the model, collects
reference cycles, and clears unused CUDA memory.

The installed SAM model internally resizes inputs to 1008 × 1008; raising
`--segmentation-max-width` alone does not increase that internal resolution.
The cropped second pass improves the object's share of those pixels. Final masks
remain binary; this adapter does not yet export continuous mask scores or add
edge feathering. Automatic cropping with manual mask prompts (`--interactive`,
`--box-only`, or `--prompts-json`) also uses a second SAM session on the crop.
`--manual_crop` or `--selection` skips automatic crop discovery and segments the
selected crop once.

## Manual crop and mask correction

Add **`--manual_crop`** to the command above to draw a crop and review the
six angles (0°, 60°, 120°, 180°, 240°, 300°) in the existing browser editor.
Open the printed URL, forwarding port 8765 in VS Code if necessary. The
automatic crop margin does not apply to manual selections.

To opt in to manual correction, add `--interactive`. After the model loads,
open the printed mask-editor URL. Draw boxes and add positive product
or negative background clicks. Preview each enabled anchor after its last edit,
then save and process the sequence. Use `--manual_crop --interactive` to draw
both crop and mask prompts.

Automatic mask detection on a manually selected/replayed crop requires
`--text-prompt`. It first detects with text alone and
uses the crop selection to rank objects. If no suitable object is detected, it
retries with the prompt's last word (`can` for `beverage can`), then text plus
the crop box, then the crop box alone. Masks occupying most of the crop
perimeter are rejected as likely background. These remain model
predictions, not rectangular masks. Box fallbacks are marked in the review
and stored as `detection_method` in per-image metadata. No clicks are required.
For manual crops with manual mask prompts, `--text-prompt` is optional. Automatic
crop discovery always requires it, including when `--interactive` or `--box-only`
is used. SAM 3.1 receives normalized XYWH boxes followed by separate point refinement
requests in pixel coordinates. If detection returns several objects, positive
click coverage, negative click exclusion, and box overlap select the product;
other initial objects are removed. Clicks correct only their specific anchor
view; other views are independently detected with text and the nearest
anchor's box as guidance. There is no temporal video propagation.

## Turntable angle estimation

TAE runs by default on the original uncropped source images with the original
camera calibration. ArUco side length defaults to **40 mm**, measured across the
black marker square. Override it with `--aruco-side-length-mm SIZE`, or disable
TAE explicitly with `--skip-tae`. Results are included under
`turntable_angle_estimation/` in the new dataset. The preprocessing metadata
records completion/skip status, marker size, source, calibration, and result path.
A TAE failure stops preparation before publishing the dataset.

## Diagnosing GPU failures without admin access

Add these options to the default command:

```bash
--diagnostics-dir /shared/datasets/3D_Scanning_MT/orbitvu/exports/Session_04/sam31_diagnostics \
--cuda-sync
```

Each invocation creates a unique run folder and prints its path. `--cuda-sync`
sets `CUDA_LAUNCH_BLOCKING=1` and `TORCH_SHOW_CPP_STACKTRACES=1` before importing
PyTorch, and synchronizes around each SAM request. It slows inference to help
attribute an asynchronous CUDA error to its request/operation. Use a new Python
process for this option. Without it, `--diagnostics-dir` logs the normal execution.

The folder contains:

- `console.log`: Python console output, warnings, and full Python traceback.
  Redirect launcher stdout/stderr separately to retain subprocess/native output too.
- `events.jsonl`: flushed phase/frame/request start and end events, exact prompts,
  fallback reasons, object IDs, mask sizes, and PyTorch allocated/reserved/peak memory.
- `telemetry.jsonl`: GPU memory, utilization, temperature, power and driver version,
  host/process RAM, and available container OOM counters, sampled about once a second.
  GPU query failures are logged while RAM monitoring continues.
- `last_frame.jpg`: the actual most recently requested inference input, retained
  after the temporary inference folder is removed; its SHA-256 is in the event log.
- `environment.json` and `implementation.json`: parameters, package/runtime versions,
  CUDA flags, and hashes of the installed SAM builder/model source.
- `status.json` and, on a caught error, `failure.json`: completion/failure state and
  the original error with frame/prompt context. Cleanup errors are recorded separately.

Events are flushed and synced to disk before requests. A SIGKILL cannot produce a
Python traceback; the status may remain `running`, with the last completed or
started operation available in the logs. Native fatal signals use Python's fault
handler. These diagnostics can narrow a reproducible input/operation failure and
show whether memory or temperature rises before GPU loss; they do not replace
host NVIDIA/Xid kernel logs or establish a hardware/driver cause on their own.

## Output and replay

The default automatic-crop output folder is `<sequence>_SAM31AC_WIDTH`;
manual or explicitly supplied crops use `<sequence>_und_rn_roi_sqr_SAM31_WIDTH`. It contains
`images/*.png`, updated `sparse/0/cameras.txt`, `preprocessing_metadata.json`,
`sam31_prompts.json`, and `mask_review/index.html`; automatic crops also include
`crop_review/index.html` and `crop_review/detections.json`. Existing output is never
overwritten. Original images remain intact; their alpha is replaced.
Native cropped RGB is retained for final export. Only temporary inference
frames use JPEG. Mask threshold is a **probability**, default 0.5; it must be
strictly between 0 and 1.

Use `--selection LEFT TOP RIGHT BOTTOM` to replay the crop in full-resolution
undistorted coordinates; it overrides crop discovery and requires no browser.
Use the saved `product_selection_left_top_right_bottom` for manual crops. For an
automatic crop, use the saved `crop_box_left_top_right_bottom` as `--selection`
to retain its square and margin exactly. Add `--prompts-json PATH` to replay saved anchors.
SAM 2.1 prompt files can be used when image names, crop, source size, and working
size match. `--box-only` skips the mask editor. `--anchor-interval` defaults to
18 frames, and `--segmentation-max-width` defaults to 1600.

Automatic mode resets detection on each image and permits IDs to differ between
images. Anchor overlap IoU is not evaluated in this mode. This avoids dependence
on video tracking memory and may require more inference work. It does not guarantee
perfect exclusion of all markers; review all silhouettes, especially box fallbacks.

Manual mode evaluates views from neighboring anchors and uses the nearest
anchor's prediction. All views reset model state; clicks are never reused on
another rotation. Inspect rim/base coverage, marker leakage, holes, and jumps
between anchor guidance boxes. Overlap IoU measures agreement between independent
predictions under neighboring guidance boxes, not accuracy. Connected component
filtering preserves holes; no automatic hole filling is added.

After crop discovery and optional angle estimation complete, recovery prompts
are saved beside the planned output before final masks and export, so they
survive subsequent segmentation, export, or publication failures. On successful
completion, the `<output-name>.prompts-<token>.json` recovery file is moved inside
the output folder, alongside `sam31_prompts.json`; no recovery copy remains in
the parent folder. CUDA model sessions are closed before export and
on preview/tracking failures. A compatibility shim drops the unsupported
`offload_state_to_cpu` argument in upstream versions whose multiplex loader
does not accept it; only video frames are requested to be offloaded.

## Validation

```bash
.venv-sam21/bin/python -m unittest tools.preprocessing.test_prepare_rotgs_sequence_SAM21 tools.preprocessing.test_prepare_rotgs_sequence_SAM31 tools.preprocessing.test_rotgs_sam31_diagnostics
```

These checks cover synthetic image export and mocked predictor requests,
including all-view automatic crop discovery, native-pixel margin, outward box
rounding, disconnected product parts, the manual crop flag, crop review output,
single-frame inference, separate cropped-view masks, and session cleanup on
first- and second-pass failures. All **30 regression tests pass**, including preservation of
the inference error when CUDA cleanup fails, synchronous error attribution,
GPU-loss telemetry, and log/input persistence after a worker is killed.
On 2026-10-06, the two-pass diagnostic run with `--cuda-sync` completed all
**73 views**, turntable angle estimation, and native RGBA export in about
248 seconds. Crop discovery used `beverage can` for 69 views and `can` for four;
final cropped-view segmentation used those prompts for 35 and 38 views respectively.
Every output is 1435 × 1435 pixels. The shared square includes every first-pass
detected box with at least 10 native pixels of margin, and no final mask touches
the crop boundary. Twelve representative rotations were visually reviewed.
The `0002.png` edge comparison shows the four-pixel stair steps replaced by
finer boundaries, with slightly more of the lower rim retained; its native RGB
pixels are identical to the previous output.

The full-run diagnostic bundle is at
`/shared/datasets/3D_Scanning_MT/orbitvu/exports/Session_04/sam31_diagnostics/sam31-uxiinlqj`.
It includes `summary.json`, `0002_edge_comparison.png`, `output_contact_sheet.jpg`,
and `launcher_console.log`. No GPU-query errors occurred and container OOM
counters remained zero. The GPU stayed online after processing.

That full run measured a peak of 11436 MiB GPU memory before the tracker autocast
cleanup fix. A subsequent focused full-view/cropped-view GPU check verified the
fix: allocated GPU memory fell to 9568256 bytes (about 9.1 MiB) after each model
closed, and the second pass did not increase the 6025882112-byte PyTorch peak.
`memory_release_check.json` records those measurements. These focused memory
measurements are separate from the full-sequence telemetry.

An earlier asynchronous run failed after 45 completed views with an unspecified
CUDA launch failure and GPU device loss. Synchronous CUDA changes execution
timing; the successful diagnostic run does **not** establish the cause of those
failures or verify that normal asynchronous execution is fixed. Host NVIDIA/Xid
logs would still be needed to classify the driver/device error.

The validated environment uses Python 3.12, PyTorch 2.10.0 with CUDA 12.8,
and an RTX 5090 with bfloat16 support. The SAM 3.1 checkpoint is local and readable
by PyTorch.

The earlier video adapter refined points before filling semantic frame caches,
so upstream partial propagation discarded uncached masks. Filling those caches
exposed another upstream tracker feature-shape error. Independent detection
avoids both temporal memory paths.
