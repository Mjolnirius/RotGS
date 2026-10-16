# R081 saturation and learning-rate audit

> R081 CPU audit dated 15 October 2026. [Study overview](../E08_capacity.md) · [Evidence bundle](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E08_capacity/R081_saturation_20261015). This is an audit of an existing run, not a new training experiment.

Audited 15 October 2026 on CPU using all six saved checkpoints and all 3,000 local W&B history samples. Training files and checkpoints were only read. Exact counts, schedules, provenance checks, and limitations are in [evidence.json](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E08_capacity/R081_saturation_20261015/evidence.json); trajectories are in [saturation.png](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E08_capacity/R081_saturation_20261015/saturation.png) and [saturation.pdf](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E08_capacity/R081_saturation_20261015/saturation.pdf).

R081 is still improving at 30k, but individual color and opacity parameters have entered regions with weak or zero gradients. Motion stops by schedule, and geometry continues becoming thinner.

| Component | Observation | Interpretation |
|---|---|---|
| Axis | Final tilt 28.506474°, permitted 28.066175–32.066175°. Side 0.176757°, permitted −1.792489–2.207511°. Logged boundary contacts only at 110–230; none after 2k. | No persistent bound saturation. Axis Jacobian has rank 2, as expected for a normalized direction. |
| Center | Maximum coordinate uses 15.46% of its ±0.05 bound; smallest tanh derivative 0.9761. | No significant tanh saturation. |
| Motion schedule | Axis and center rates become exactly zero at 12k. Final 25k–30k median per-view axis gradient norm is 16.83. | Flat curves are scheduled freezing; nonzero per-view gradients alone do not establish that later motion fitting helps. |
| Gaussian count | 223,120 of 450,000, last densification at 12,900. | Population stops by schedule, not cap. |
| Opacity | 13.19% >0.99; 5.974% >0.999; five exactly 1 in float32. 14,912 (6.683%) fall below 1/255. | High logits have small sigmoid derivatives. Below 1/255, no pixel contribution can pass the current renderer threshold because Gaussian falloff is at most one. These counts are not contribution weighted. |
| Color clamp | 14.865% of RGB channels are negative before clamping; 24.96% of Gaussians have at least one such channel. 7.30% have all three negative. | Renderer outputs zero for these channels and masks their direct SH-color gradients. Adam momentum can still move parameters temporarily. This is not a measurement of black pixels in the final image. |
| Higher-order SH | All higher coefficients and their Adam moments are exactly zero at every checkpoint, despite model degree 3. | Cached training rasterizer remains at degree 0. Fix its degree synchronization and complete SH rotation through degrees 2/3 before interpreting SH LR tuning. |
| Shape | Median largest/smallest scale rises 27.1 → 54.2 → 84.0 at 15k/25k/30k. 15.88% exceed 1,000 at 30k; minimum scale is 1.83e−9, with all parameter tensors finite. | Increasing anisotropy, not a hard scale bound or numerical underflow. Surface splats can legitimately be flat; these measurements do not prove physical shape error. |
| Late fitting | Logged held-out FG PSNR is 26.806 → 27.920 → 28.221 dB at 20k/25k/30k. Median same-Gaussian position change over 25k–30k is 0.000766 scene units. | Diminishing returns, with continuing geometry and appearance updates. |

Learning-rate recommendation: retain XYZ (base 1.6e−4→1.6e−6, actual 8e−4→8e−6 after spatial scale 5), rotation (1e−3), and the axis/center baseline initially. Fix the confirmed SH activation issue first, holding other settings fixed. The higher-order rate is already nonzero (1.25e−4); raising it cannot fix degree-0 rendering. Local angle corrections and sweep corrections are disabled, so their LR is irrelevant here.

After that controlled baseline, test a late scaling decay from 5e−3 to 1e−3 over 15k–30k. Separately test opacity decay from 2.5e−2 to 5e−3 over the same interval. These are conservative experiment values, not established optima. A lower feature rate can reduce future overshoot but cannot directly unstick channels already masked by the clamp. Keep feature LR 2.5e−3 for the first SH-fix comparison. A later small-rate motion refinement is a separate experiment; do not infer a need from gradient norms alone.

For the next fresh appearance baseline, the existing narrow projected-axis R082 reaches external FG PSNR 28.999 dB versus R081 28.217 dB under the same matched PNG evaluation. This supports retaining tight calibration bounds; it does not independently establish an optimal LR or accurate physical shape. Logged float validation above and external PNG results here are separate evaluation protocols.

Relevant source: `train.py:273–300` caches the rasterizer after SH degree changes; `gaussian_renderer/__init__.py:35` copies the model degree; `scene/gaussian_model.py:589–634` schedules motion; `utils/general_utils.py:73–75` returns zero at the motion horizon; renderer `forward.cu:70–75` and `backward.cu:29–34` clamp RGB and mask gradients. No retraining, hyperparameter edits, or model repairs were performed in this audit.
