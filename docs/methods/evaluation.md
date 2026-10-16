# Evaluation and comparison policy

[Documentation home](../README.md) · [Experiment objectives](../experiments/README.md)

Use externally evaluated foreground PSNR as the main RGB measure within a fixed capture, held-out split, reference mask, resolution, checkpoint, and renderer. Retain full-frame PSNR, crop SSIM/LPIPS, and visual/silhouette evidence alongside it. The full timeline is not a single comparable leaderboard.

| Measurement | Definition and interpretation |
|---|---|
| Full-frame PSNR | RGB error over the complete white-composited image. Background coverage can affect cross-product rankings. |
| Native foreground PSNR | RGB squared error weighted by the original soft source alpha and normalized by foreground mass; average per-view PSNR with equal view weights. Different native masks can include different material. |
| Common-product PSNR | An explicitly aligned shared product target. The existing Session 04 top-can target uses R074's 1435 px RGB/alpha with checked crop offsets; this alignment is capture-specific. |
| Foreground-crop SSIM/LPIPS | The complete bounding crop of `alpha > 0.001`, expanded by eight pixels and clipped to the image. Background inside the crop remains included. |
| Corrected LPIPS | Bundled VGG LPIPS v0.1 with RGB mapped to `[-1, 1]`, without resizing the crops. Historical LPIPS preprocessing differs. |
| Float versus PNG | In-memory validation and evaluation of quantized saved PNGs are separate sources; keep their labels visible. |

Every comparison should name its dataset/capture/preparation, run IDs, iteration, held-out views, source revision, seed, image target, renderer and metric protocol. Ten views from one model are not ten training-seed repeats. Cleaner image boundaries do not establish physical geometry accuracy. A metric recalculation is not a training-loss ablation.

The Session 03 paired checkpoint analysis defines deltas as `30000 - 25000`. Positive PSNR/SSIM and negative LPIPS deltas indicate improvement. Its twelve completed captures include related top/down variants.

## Detailed protocols and evidence

- [Authoritative run-log comparison policy](/shared/datasets/3D_Scanning_MT/output_RotGS/RUN_LOG.md#psnr-comparison-policy)
- [Session 03 corrected methodology](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/corrected_evaluation_methodology.md)
- [October 7 saved-checkpoint evaluation](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/evaluations/2026-10-07_saved_checkpoints/README.md)
- [October 14 native PNG audit](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/evaluations/2026-10-14_native_psnr/README.md)
- [Shared top-can target](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/index.html)
- [Evaluation tool instructions](../../tools/evaluation/README.md)

This guide summarizes the existing protocols. Protocol changes need a new explicit version and must preserve historical results.
