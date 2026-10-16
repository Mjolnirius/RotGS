# Corrected reconstruction evaluation

This tool evaluates saved RotGS test renders without modifying training code,
checkpoints, `results.json`, or the W&B export.

Its retained Session 03 inputs and outputs are in the [E04 analysis bundle](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/README.md). See the [project evaluation policy](../../docs/methods/evaluation.md) for comparisons across reports.

The source RGBA alpha is used as a soft mask for foreground PSNR.  Cropped SSIM
and LPIPS use the bounding box of `alpha > 0.001`, expanded by eight pixels and
clipped to the image.  They evaluate the complete crop and are not pixel-masked.
Saved RGB render and GT images are already on the same white background.

VGG LPIPS v0.1 is loaded from the repository's `lpipsPyTorch` implementation.
RGB `[0,1]` crops are explicitly transformed to `[-1,1]`, the range expected by
that implementation, and are evaluated one view at a time without resizing.

Run all approved stages with:

```bash
.venv/bin/python tools/evaluation/run_corrected_evaluation.py --stage all
```

Stages can also be run separately as `30k`, `render-25k`, `25k`, and
`relationships`.  The 25k render stage refuses to overwrite a nonempty target.
Every paired delta is defined as `30000 - 25000`.
