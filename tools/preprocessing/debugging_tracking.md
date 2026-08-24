The evidence says 30k iterations are not the main limitation. With the current schedules, training is essentially plateaued by 15k. Simply running 60k with unchanged settings is unlikely to approach `chair_45`.

### Measured convergence

I rendered the ten held-out views at every saved checkpoint and calculated the metrics:

| Iteration | Gaussians | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---:|---:|---:|---:|---:|
| 7,000 | 2,998 | 10.41 | 0.6421 | 0.4335 |
| 15,000 | 28,882 | 21.54 | 0.7734 | 0.2700 |
| 30,000 | 28,882 | 21.76 | 0.7756 | 0.2599 |
| Chair 30,000 | 60,131 | 33.28 | 0.9700 | 0.0402 |

From 15k to 30k, another 15,000 iterations delivered only:

- `+0.22 dB` PSNR
- `+0.0022` SSIM
- `−0.0101` LPIPS
- No additional Gaussians
- Exactly zero Gaussian position movement

The detailed results are in [results.json](C:/Users/Thor/gitreps_SwissEP/RotGS/output/Quoellfrisch_02_top_cpy_rn_roi_sqr_PNGa_full/results.json) and [per_view.json](C:/Users/Thor/gitreps_SwissEP/RotGS/output/Quoellfrisch_02_top_cpy_rn_roi_sqr_PNGa_full/per_view.json).

### Most important problems

1. The learned rotation does not complete a full revolution.

Your first and last PNGs are byte-identical, but the learned effective rotation covers only approximately `316.4°`, not `360°`.

```text
Product residual endpoint change: +38.76°
Chair residual endpoint change:    −4.76°
```

For the chair, the residual almost perfectly cancels the injected `4.83°` endpoint noise. For your can, it changes the rotation speed substantially. That will smear labels and fine text because frames are assigned to incorrect angles.

Since your turntable angles are known exactly, I recommend:

- Disable synthetic angle noise.
- Disable the learned residual-angle predictor.
- Use the exact 0°, 5°, …, 360° angles.
- Keep the duplicated 360° image so the sequence closes periodically.

2. Only 6.44% of the initial point cloud is visible.

Your tight crop gives a roughly `14°` field of view:

```text
Initial points: 5,000
Initially visible: 322
Visible fraction: 6.44%
```

For `chair_45`:

```text
Initial points: 5,000
Initially visible: 2,694
Visible fraction: 53.88%
```

This explains why the chair already has 30,120 Gaussians at 7k while your model has only 2,998.

Rather than abandoning the tight crop, I recommend making random initialization frustum-aware: initialize the points inside the volume actually visible through the cropped camera, preferably using the alpha-mask dimensions to estimate the product volume.

3. Gaussian positions currently have an effective learning rate of zero.

All fixed-camera centers are identical, so camera-based scene normalization produces a radius of zero. The XYZ learning rate is multiplied by this radius in [gaussian_model.py](C:/Users/Thor/gitreps_SwissEP/RotGS/scene/gaussian_model.py:330).

Consequently:

- Existing points do not move.
- Spatial changes only occur when densification creates split points.
- Densification stops at 15k.
- After 15k, geometry is completely frozen.

For fixed-camera training, the spatial learning-rate scale should instead come from the camera distance or initialized product volume.

4. The model needs more spatial capacity.

The product occupies about `44.8%` of the image and contains fine printed text. The chair occupies only `19.6%`.

Approximate Gaussian density:

```text
Product: 0.22 Gaussians per foreground pixel
Chair:   1.05 Gaussians per foreground pixel
```

The chair therefore has approximately 4.8 times the Gaussian density relative to visible object area.

After fixing initialization and position learning, I would target roughly 60,000–100,000 Gaussians by:

- Extending densification from 15k to around 25k.
- Reducing `densify_grad_threshold` from `0.0002` to approximately `0.0001`.
- Extending total training to 45k.
- Monitoring VRAM and Gaussian count.

5. Foreground-weighted RGB loss would help fine text.

RotGS currently calculates RGB/SSIM loss over the complete white image. The alpha mask is used for optical flow, but not to focus RGB reconstruction on the product.

A useful loss would combine:

```text
80% foreground-weighted reconstruction loss
20% complete-image reconstruction loss
```

That would put more optimization pressure on text and label detail while preserving clean edges/background.

### Recommended next experiment

I would not run the existing configuration longer. I recommend a new controlled run with:

1. Exact periodic angles without learned residuals.
2. Frustum/mask-aware point initialization.
3. Nonzero XYZ learning rate for fixed-camera scenes.
4. Densification through 25k.
5. Total training of 45k.
6. Evaluation and saving at 5k, 10k, 15k, 20k, 25k, 35k, and 45k.
7. A target of at least 60k Gaussians.

A good stopping rule would be less than `0.2 dB` PSNR improvement and less than `0.005` LPIPS improvement over the previous 10k iterations. Your existing 15k→30k curve already meets that plateau condition.
