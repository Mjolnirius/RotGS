# SAM 2.1 product masks for RotGS

`prepare_rotgs_sequence_SAM21.py` is a separate entry point and callable function.
The existing PNG/Lab preprocessing scripts are unchanged. Inputs are original PNG
exports with angle-named files covering 0..355 or 0..360 in 5-degree steps, plus
the session's cameras.txt and calibration. Run separately for each camera.

## Installation

Use a separate environment so installing SAM does not change the training environment.
From the RotGS repository, with the existing `.venv` already installed:

```bash
uv venv .venv-sam21 --python .venv/bin/python
.venv-sam21/bin/python -c 'import site; from pathlib import Path; Path(site.getsitepackages()[0], "rotgs-training.pth").write_text(str(Path(".venv/lib/python3.10/site-packages").resolve()) + "\n")'
SAM2_BUILD_CUDA=0 uv pip install --python .venv-sam21/bin/python --no-deps --no-build-isolation 'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4' hydra-core==1.3.2 omegaconf==2.3.0 iopath==0.1.10 antlr4-python3-runtime==4.9.3 portalocker==3.2.0
mkdir -p checkpoints/sam21
curl -L --fail https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -o checkpoints/sam21/sam2.1_hiera_large.pt
```

Official implementation/API: https://github.com/facebookresearch/sam2 .
The `.pth` file makes the existing training dependencies readable by this environment;
new packages are installed only into `.venv-sam21`. The explicit `--no-deps` prevents
upgrading PyTorch. These commands assume this repository's Python 3.10 training environment.
SAM's optional CUDA extension is unnecessary: this script disables automatic
hole filling and performs connected-component filtering with OpenCV.
Use `--model base_plus` with the matching `sam2.1_hiera_base_plus.pt` checkpoint
if desired. No weights are downloaded automatically by the script.

## Interactive run

```bash
.venv-sam21/bin/python tools/preprocessing/prepare_rotgs_sequence_SAM21.py \
  /shared/datasets/3D_Scanning_MT/orbitvu/exports/Session_04/4_Quoellfrisch_Top_30d \
  --cameras /shared/datasets/3D_Scanning_MT/orbitvu/exports/Session_04/cameras.txt \
  --checkpoint checkpoints/sam21/sam2.1_hiera_large.pt
```

1. Open the printed localhost URL (forward port 8765 in VS Code if needed).
   Select a crop containing the product throughout its rotation, with a small margin.
   This first browser step only selects the crop/output size.
2. Once the terminal prints that the SAM editor is ready, reopen/refresh the localhost
   root URL, or follow the new session URL printed in the terminal. Suggested
   anchors are every 18 frames (90 degrees), plus the last frame. Draw tighter
   product boxes as needed, place positive product clicks and negative marker clicks,
   then preview each enabled anchor. You may add or remove anchor frames.
3. Click **Save prompts and process sequence**. Inspect every image in the generated
   `mask_review/index.html`, especially the rim, base, and marker overlap boundaries.

Use `--aruco-side-length-mm SIZE` to also estimate angles from the uncropped original
images before segmentation. Original files remain intact. If this option is omitted,
angle estimation is not run or copied from another dataset.

Output defaults to a new sibling directory ending `_und_rn_roi_sqr_SAM21_WIDTH`.
Use `--output PATH` for a custom new directory; existing output is never overwritten.
Output contains `images/*.png`, updated `sparse/0/cameras.txt`, preprocessing metadata,
`sam21_prompts.json`, and the mask review page. Review flags are warnings, not proof
of accuracy. Inspect all frames before training.
Before tracking, a uniquely named `OUTPUT.prompts-XXXXXXXX.json` is also saved beside
the output directory. It survives failed or cancelled tracking so manual work can be
replayed with `--prompts-json`.

## Repeatable / unattended runs

`--selection LEFT TOP RIGHT BOTTOM` bypasses the crop editor. These coordinates are
in the **full-resolution undistorted image**. Use the saved metadata's
`product_selection_left_top_right_bottom` when reproducing a crop.
`--prompts-json previous_output/sam21_prompts.json` bypasses the SAM editor and replays
saved prompts. Supply both options for a fully unattended run. Saved source names,
crop, and coordinate size must match; mismatches are rejected. Use the same
`--segmentation-max-width` (default 1600). To correct masks, edit/add entries in the
JSON and rerun into a new output directory.

Each prompt entry has this form:

```json
{"frame_index": 18, "box": [350, 100, 1100, 1500],
 "points": [[700, 700], [750, 80]], "labels": [1, 0]}
```

Coordinates are in the **cropped working image** described by `coordinate_size`,
not full-resolution source pixels. Labels are 1 for product, 0 for background.
A box or at least one positive point is required. Keep the other saved JSON fields.

Alternatively, `--box-only` skips SAM interaction and seeds every suggested anchor
with the initial selection box. This is useful for experiments; the output still
requires visual review. `--anchor-interval` controls default anchor spacing only,
and does not modify a supplied prompts JSON.

## Mask and image behavior

- Existing alpha is discarded. It does not identify the product separately from markers.
- RGB is undistorted/cropped at native resolution and retained losslessly for export.
  SAM uses temporary high-quality JPEG copies because its official video loader
  accepts numbered JPEGs. JPEG RGB is never used for final output.
- Each anchor tracks independently toward adjacent anchors in both directions.
  At each frame, the nearest anchor's mask wins (ties prefer the earlier anchor).
  Overlap IoU below 0.90 is flagged for review. Switching anchors can cause seams;
  add corrected anchors where needed. There is no artificial 360-to-0 wrap.
- The largest connected component is retained on tracked frames. On an anchor,
  components containing positive clicks are preferred. This assumes one connected
  product silhouette; detached or disconnected product parts require care.
- Holes are not filled, boundaries are not eroded, and hidden surfaces are not generated.
  SAM logits are thresholded into membership masks; they are not physical opacity.
  Masks are mapped to the native crop before RGBA downsampling, which adds edge coverage.
  This is not a transparent/glass-product matting solution.
- RGB pixels removed in the original export cannot be recovered by SAM. Use original
  RGB exports if source alpha processing already destroyed product detail.
- Output publishes only after the complete run succeeds. Temporary intermediate files
  are removed on failure; the saved prompt recovery file remains. Allow temporary
  disk space for native-resolution RGB crops. Empty masks stop the run with a frame
  number; add a corrected anchor there. Broad boxes alone can lose the product.

This preprocessing does not change training, evaluation, or queued jobs.
