# RotGS Session-03 training dynamics and foreground-aware evaluation

> Finalized Session 03 analysis of twelve completed captures. [Study overview](../E04_product_baseline.md) · [Original data, figures, and exports](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/README.md). Data filenames cited in the text refer to that evidence bundle. Reorganization on 15 October 2026 does not refresh this scientific analysis.

## 1. Scope and experimental overview

This report documents the finalized analysis of twelve completed Session-03 RotGS captures. Two failed runs remain in the provenance files but are excluded from every primary scientific comparison. Several captures are top/down variants of related product identities; the set is therefore not treated as twelve independent products.

The exported configurations agree on the effective training setup: 30,000 iterations, densification from iteration 500 to before 13,000 at 300-iteration intervals, five opacity resets, a maximum of 450,000 Gaussians, center warmup through iteration 2,000, axis and center optimization through 12,000, bounded global sweep correction of ±4°, white background, fixed camera, and disabled optical-flow supervision. Exact configuration values are retained in `run_configs.json`.

Claims are separated into four levels. **Observed** denotes values in the finalized data. **Implementation fact** denotes behavior verified against source. **Supported interpretation** requires agreement between data, configuration, and timing. **Hypothesis** denotes a plausible mechanism that this experiment did not isolate.

## 2. Data and implementation provenance

The complete, unsmoothed history in `timeseries_analysis.csv` is the time-series source of truth. `training_events.csv` supplies deterministic event reconstruction; the corrected-evaluation CSVs supply final quality results. Run identity is joined to local outputs through model path and exact local W&B run ID. All 240 recorded data-quality checks pass. All twelve corrected 30k evaluations pass validation, and corrected PNG foreground PSNR differs from the in-memory W&B value by at most 0.0045 dB.

The audited within-iteration order is forward/loss, backward, base metric capture, optional validation, topology/reset, post-topology metric capture, optimizer step, and W&B logging. Losses and `scene/gaussians` are therefore pre-topology; `scene/gaussians_after_update` and opacity statistics are post-topology or post-reset. Opacity statistics are sparse and event-triggered.

Scheduled spherical-harmonic increments were not render-effective in the audited single-camera path because the rasterizer captured degree zero once. Exposure optimization was instantiated and logged but disconnected from the verified render path. Neither schedule is used to explain loss or quality changes. The exact historical dirty training tree cannot be reconstructed; implementation claims are limited to mechanisms supported by retained source and run evidence.

| Event | Exact schedule | Interpretation |
|---|---:|---|
| Center warmup | 1–2,000 | Center fixed; first active iteration is 2,001 |
| Densification | 600, 900, …, 12,900 | 42 exact topology events |
| Opacity reset | 500, 3,000, 6,000, 9,000, 12,000 | Activated opacity capped at 0.01 after loss measurement |
| Axis/center stop | 12,000 | Axis and center cease updating |
| Topology freeze | 13,000 | No topology modification at or after this boundary |
| Validation/checkpoints | 5k, 10k, …, 30k | Six held-out-view evaluations |
| Residual LR terminal value | 15,000 | 2.5×10⁻⁵ and remains nonzero |

## 3. Training objective and loss behaviour

The total objective combines weighted image loss, flow loss, and center regularization. Flow is exactly zero in this set. The image objective contains foreground RGB, full-frame RGB, foreground-crop DSSIM, and alpha terms. Figure 1 shows every raw run faintly and an across-run median for orientation; it does not use W&B dashboard smoothing. Loss decreases throughout training with localized disturbances aligned with reset events.

![Figure 1 — Training convergence](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/report/figures/figure_01_training_convergence.png)

**Figure 1.** Raw, unsmoothed training histories. The black curve is the deterministic across-run median at each logged iteration. Red dotted lines mark opacity resets; gray lines mark phase boundaries. Peaks and event responses remain visible.

Each capture has 63 training cameras consumed deterministically. With logging every ten iterations, `gcd(63,10)=1`, so the same sampled training-view phase recurs after 63 observations, or 630 training iterations. Periodic variation should not automatically be interpreted as optimization instability, although view cycling need not explain all variance.

## 4. Gaussian population and densification dynamics

Final populations range from 119,486 to 447,520 Gaussians, a 3.745-fold range under shared settings. The two Chips runs finish near the configured maximum, but candidate truncation was not logged; they are described only as **near-cap**. Gaussian count characterizes adaptive representation capacity, not universal reconstruction quality.

![Figure 2 — Gaussian population and densification](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/report/figures/figure_02_gaussian_population.png)

**Figure 2.** Adaptive population growth and exact net changes at all 42 densification events. No smoothing or interpolation is applied to event values.

The final reset at 12,000 is followed by a large negative net update at 12,300 and regrowth at 12,600 and 12,900.

![Figure 3 — Final topology transition](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/report/figures/figure_03_final_topology_transition.png)

**Figure 3.** Final transition with exact post-update population, net topology change, and sparse post-event opacity observations. The temporal alignment supports the interpretation that the 12k reset **plausibly contributes** to the 12.3k pruning event. Logged net change does not separate cloning, splitting, removal of split parents, low-opacity pruning, or world-size pruning.

## 5. Opacity-reset dynamics

Each reset caps activated opacity at 0.01. Because loss is captured before reset and opacity statistics afterward, loss on the reset row describes the pre-reset render. The first affected render is subsequent and, under ten-iteration logging, the effect generally first appears at reset+10.

![Figure 4 — Opacity reset dynamics](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/report/figures/figure_04_opacity_reset_dynamics.png)

**Figure 4.** Synchronized raw observations. Opacity markers are sparse post-event statistics; loss is pre-topology. The reset directly explains the delayed alpha-loss disturbance and is a plausible major contributor to image-loss increases. Changing training views and coincident topology operations prevent attribution of the entire spike to reset alone.

Pruning stops after 12,900. Final PLY representations can consequently contain Gaussians whose opacity later falls below the former pruning threshold. Deleting them would require rerendering and reevaluation.

## 6. Rotation-axis, center, and sweep optimization

Axis and center parameters stabilize before the final endpoint and stop updating at 12,000. The final axis-side correction has mean +0.416° and sample SD 0.028° across runs. This is a shared systematic effect, not evidence for a unique physical origin; setup, calibration, crop/projection effects, initialization, coordinate conventions, mask asymmetry, and model coupling remain possible explanations.

![Figure 5 — Motion convergence](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/report/figures/figure_05_motion_convergence.png)

**Figure 5.** Raw axis-tilt, axis-side, and center-shift trajectories. Center optimization begins after 2k; axis and center stop at 12k.

Global sweep correction remains trainable after 12k. Its learning rate reaches 2.5×10⁻⁵ at 15k and remains nonzero. Leermond and Weisswein approach the −4° bound.

![Figure 6 — Sweep correction](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/report/figures/figure_06_sweep_correction.png)

**Figure 6.** Dataset-dependent sweep correction with both configured bounds. Saturation establishes that the bound is active for these runs; it does not identify the unconstrained optimum.

## 7. Reconstruction-quality convergence

Held-out-view foreground PSNR continues to rise after motion stops and topology freezes. Median improvements are +1.821 dB from 15k to 20k, +0.686 dB from 20k to 25k, and +0.182 dB from 25k to 30k. The median 15k–30k gain is +2.724 dB. This distinguishes topology formation from continued parameter refinement while demonstrating diminishing returns.

![Figure 7 — Reconstruction-quality convergence](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/report/figures/figure_07_quality_convergence.png)

**Figure 7.** Actual logged held-out-view checkpoints; connecting lines are visual guides only. The shaded interval has fixed topology.

Training evaluation uses five fixed training views; test evaluation uses ten held-out views. Their difference is a descriptive diagnostic rather than a rigorous generalization or overfitting estimate.

## 8. Foreground-aware reconstruction evaluation

Historical external PSNR, SSIM, and LPIPS operate on complete white-composited frames and are confounded by foreground coverage. Corrected foreground PSNR uses soft-alpha-normalized foreground MSE with equal-view averaging. Corrected SSIM and LPIPS use the complete crop defined by `alpha > 0.001`, fixed eight-pixel padding, and image-bound clipping; they are not pixel-masked. Corrected VGG LPIPS v0.1 receives RGB transformed to [−1,1] without resizing. Historical LPIPS differs in both spatial evaluation region and input preprocessing.

![Figure 8 — Evaluation-bias correction](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/report/figures/figure_08_evaluation_bias.png)

**Figure 8.** Descriptive coverage associations for N=12. Dashed lines are visual linear guides, not predictive models; no significance tests are used. Pearson/Spearman associations change from −0.8725/−0.9091 to −0.0721/+0.0140 for PSNR, from −0.6630/−0.8951 to −0.5107/−0.5944 for SSIM, and from +0.7434/+0.7552 to +0.4367/+0.3566 for LPIPS. Foreground correction essentially removes the PSNR relationship; moderate cropped-SSIM and LPIPS relationships remain.

### Corrected 30k metrics

| Capture | FG PSNR (dB) | Crop SSIM | Crop LPIPS | Coverage |
|---|---:|---:|---:|---:|
| Chips blue (down) | 25.927 | 0.9318 | 0.1316 | 0.152 |
| Chips blue | 27.136 | 0.9293 | 0.1191 | 0.184 |
| Kaegifret (down) | 25.889 | 0.9312 | 0.1094 | 0.115 |
| Kaegifret (top) | 28.743 | 0.9562 | 0.1268 | 0.135 |
| Leermond | 26.530 | 0.9183 | 0.2484 | 0.166 |
| Pepsi (down) | 25.802 | 0.8909 | 0.2216 | 0.288 |
| Pepsi | 26.101 | 0.9018 | 0.1929 | 0.292 |
| Quoellfrisch (top, 30°) | 27.978 | 0.9054 | 0.1873 | 0.298 |
| Ramseier Tetra | 27.209 | 0.9140 | 0.2482 | 0.261 |
| Schwamm (down) | 28.975 | 0.8868 | 0.2731 | 0.166 |
| Schwamm (top) | 27.813 | 0.8005 | 0.2748 | 0.263 |
| Weisswein | 26.763 | 0.8885 | 0.2678 | 0.201 |

## 9. Fixed-topology refinement: 25k versus 30k

Gaussian count is identical at 25k and 30k for every run because topology had already frozen. The comparison therefore measures continued optimization of the existing representation, not increased Gaussian capacity.

![Figure 9 — Fixed-topology refinement](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/report/figures/figure_09_fixed_topology_refinement.png)

**Figure 9.** Paired corrected metrics. Foreground PSNR improves in 11/12 runs (mean +0.1807 dB); cropped SSIM improves in 12/12 (mean +0.003259); cropped LPIPS improves in 12/12 (mean −0.005355, lower is better). The final 5k iterations provide small but highly consistent structural/perceptual gains without increasing model size.

## 10. Computational cost and rasterization behaviour

`performance/iteration_ms` is a CUDA-event timing interval from loop start through backward, not full wall-clock iteration time. It excludes validation, topology changes, optimizer stepping, checkpoint writing, W&B logging, and artifact handling. After 13k, population is exactly fixed while opacity evolves and the timing proxy falls by roughly one third from early to late fixed-topology windows.

![Figure 10 — Post-densification computational behaviour](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/report/figures/figure_10_post_densification_compute.png)

**Figure 10.** Gaussian counts are normalized by each run's final count. Opacity is shown only at sparse event-triggered observations. Timing uses deterministic 500-iteration medians normalized by each run's 13k–14k median; canonical raw data remain unchanged. Opacity-dependent early termination is a supported possible mechanism. Projected footprint, tile overlap, depth complexity, scale, and spatial distribution were not logged and remain competing explanations.

## 11. Cross-run observations and notable metric disagreements

The corrected metrics do not yield a single quality ordering. Schwamm top has 27.813 dB foreground PSNR but the lowest cropped SSIM (0.8005) and highest cropped LPIPS (0.2748). Schwamm down has the highest foreground PSNR (28.975 dB), but cropped SSIM is 0.8868 and LPIPS 0.2731. Kaegifret down has only 25.889 dB PSNR while achieving cropped SSIM 0.9312 and the lowest LPIPS, 0.1094. Pixel fidelity, structural similarity, and perceptual similarity capture different reconstruction properties.

Cross-run correlations remain exploratory: resolution, foreground coverage, texture, geometry, material, crop, and viewpoint vary together, and related captures are not independent. Gaussian count and runtime characterize representation and computation but are not universal quality measures.

## 12. Key findings

1. Shared settings yield a 3.745-fold range in representation size, demonstrating capture-dependent adaptation.
2. Topology formation ends at 12.9k, but foreground PSNR continues improving through 30k.
3. Opacity reset produces a reproducible disturbance with a verified measurement-order offset and plausibly contributes to the final prune/regrow sequence.
4. Axis and center stabilize early; the shared axis-side correction has no uniquely identified physical source.
5. Historical full-frame metrics are unsuitable as primary cross-capture quality metrics. Corrected foreground PSNR nearly removes the observed coverage association.
6. The 25k–30k results demonstrate diminishing returns alongside consistent SSIM and LPIPS improvement at unchanged model size.
7. CUDA forward/loss/backward cost changes under fixed topology, but current instrumentation cannot isolate opacity from footprint/tile/depth effects.

## 13. Limitations and unresolved questions

The study contains twelve completed captures, including related top/down variants. Cross-run relationships are exploratory and confounded by differing resolution, coverage, texture, geometry, material, crop, and viewpoint. The exact historical dirty patch is unavailable. Cropped SSIM and LPIPS retain moderate coverage relationships. Clone/split/prune decomposition and rasterization footprint/tile/depth metrics were not logged. The 25k–30k comparison supports diminishing returns for these runs but does not establish general endpoint interchangeability.

Chips blue, Chips blue down, Kaegifret top, and Quoellfrisch top 30° retain warnings that RGB under fully transparent source alpha may have been destroyed or premultiplied. Forty-eight overlays were generated, representatives from all four warned datasets were inspected, and no mask was repaired. The warning alone does not invalidate the reconstructions.

## 14. Appendix: W&B metric reference and secondary figures

The metric definitions and source mapping are provided in `metric_definitions.csv` and `code_metric_map.md`. Secondary vector figures reused by the LaTeX appendix are:

- `1b_loss_image.pdf` and `1c_weighted_components.pdf`: optimized image objective and its weighted components.
- `3a_visible_fraction.pdf` and `3b_visible_gaussians.pdf`: camera-dependent visibility diagnostics.
- `4b_opacity_min.pdf` and `4c_opacity_max.pdf`: sparse opacity extrema.
- `5e_axis_lr.pdf` and `5f_center_lr.pdf`: verified motion learning-rate schedules.
- `6b_test_fg_ssim.pdf`, `6c_train_fg_psnr.pdf`, and `6c_train_vs_test_fg_psnr.pdf`: secondary quality diagnostics.
- `7c_total_runtime.pdf`: total W&B runtime, which includes work excluded from the CUDA timing proxy.

Selected exploratory cross-run plots remain in `figures/` but are not promoted to main findings. Inactive detailed flow metrics, exposure LR, and nominal SH-degree events are not interpreted as effective rendering signals.
