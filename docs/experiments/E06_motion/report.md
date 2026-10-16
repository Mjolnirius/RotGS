# RotGS implementation decisions and earlier top can baselines

> Detailed Session 04 assessment, including historical and superseded decisions. [Study overview](../E06_motion.md) · [Illustrated export and evidence](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/index.html). The completed axis decision is summarized in [E07](../E07_axis.md). Reorganization on 15 October 2026 does not rerun the analyses.

**Assessment date:** 14 October 2026, Europe/Zurich. **Coverage:** the ten recorded runs from 1–7 October, the completed R081/R082/R084 axis experiments and failed R083 attempt, plus the five highest-scoring earlier top-can runs from Session 03. **Purpose:** decide which preparation and motion choices to retain, which optimizations to investigate, and how to judge success.

The strongest evaluated clean Session 04 appearance candidate is now **R082: fixed measured TAE angles with projected axis optimization at the original ±0.5° width**. The completed fixed-calibration retry R084 is a strong simpler control. Both beat R078, and narrow projection beats wide R081 on aggregate image fidelity. Retain R078 as the historical control; one seed per setting has been evaluated.

The earlier PSNR leaders are genuine saved-image results, but their headline values describe the whole frame. **R025 reproduces at 32.500 dB full-frame and 27.359 dB foreground; R058 leads the five earlier models on foreground PSNR at 27.978 dB.** They already used non-SAM product masks. Their larger white-background regions account for a substantial full-frame bonus, while stronger foreground detail in some earlier models remains real. The absence of SAM does not invalidate a PSNR calculation, and SAM use alone does not establish a better reconstruction.

TAE should help when its measured motion and coordinate conventions agree with the reconstruction model. Your results support the complete TAE setup, but they do not yet separate the contribution of measured angles from calibrated axis and center geometry. Two concrete implementation findings deserve attention before broad parameter searches: **all six October 7 comparison models have zero higher-order spherical-harmonic coefficients**, and **all four original TAE top-can variants finish at both axis constraint limits**. The first limits learned appearance; the second identifies a pose optimization issue to investigate. Neither establishes the unique cause of blur or edge ripples.

Success targets below are proposed acceptance criteria, not predictions or success probabilities. Confidence describes the strength of the current evidence. No training was launched by this analysis. The axis follow-up renders saved endpoints and novel views, using the same reference images and scoring protocol; original historical sections retain their evidence limits.

## Completed axis ablation: boundary repair versus bound width

**Current decision: use projected ±0.5° (R082) as the appearance candidate, with fixed calibration (R084) as the control.** Widening to ±2° did not improve image fidelity on this seed.

| External 30k PNG / float-alpha edge metric | R078 legacy | R084 fixed | R082 projected ±0.5° | R081 projected ±2° |
|---|---:|---:|---:|---:|
| Native / common-product PSNR dB ↑ | 27.167 | 28.735 | **28.999** | 28.217 |
| Crop SSIM ↑ | 0.85307 | 0.88922 | **0.89250** | 0.87721 |
| Crop LPIPS ↓ | 0.24903 | 0.21022 | **0.20504** | 0.22647 |
| Lower-edge MAE px ↓ | 4.914 | 2.045 | 1.964 | **1.956** |
| Ripple RMS px ↓ | 1.075 | **0.406** | 0.416 | 0.425 |
| Sampled small-print SSIM ↑ | 0.83818 | 0.84206 | **0.89718** | 0.88700 |
| Stored Gaussians | 194,696 | 215,878 | 228,948 | 223,120 |

R082 versus R078 isolates the intended optimizer change at the same ±0.5° width: +1.832 dB foreground PSNR and 61.3% lower ripple; foreground PSNR, crop SSIM, LPIPS, edge MAE and ripple improve in all ten held-out views. R082 versus R081 isolates width under matching launch source hashes: narrow gains 0.782 dB foreground PSNR and wins image PSNR/SSIM/LPIPS in all ten views, while edge measurements are nearly tied. R084 gains 1.567 dB over R078, showing that fixed calibration already supports clean lower sides. R082's smaller 0.265 dB advantage over fixed wins foreground PSNR in seven views and needs seed confirmation.

**Bound interpretation:** R082 finishes at tilt 29.566781°, just 0.000606° above its lower limit, with rank-2 axis Jacobian and live gradients. It has 216 logged tilt-bound contacts with nonzero axis gradients, the last at iteration 11,230. This is a constraint-active solution, not R078's zero-gradient lock. Its side is +0.184176°. R084 stays at calibrated 30.066172°/+0.207511° by design. Center coordinates use under 16% of their bounds in all four runs. Later pose plateaus follow the 12k learning schedule.

![Four matched lower-edge comparisons](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/diagnostics/axis_ablation_20261014/lower_left_v03_comparison.jpg)

All 73 views per model were checked: improvements also occur in the 63 training views. All alternatives improve the held-out edge measurements at alpha 0.1/0.5/0.9. Fine-print tearing and novel-elevation shape distortion remain; higher-order SH coefficients are still zero. The next appearance experiment should repair SH activation and complete SH rotation while holding the axis setup fixed; do not interpret cleaner silhouettes as measured 3D accuracy.

R083 (`3qyvbyzq`) is the failed fixed-axis attempt at approximately 5k, exit code 1; it is excluded from 30k comparisons. R084 (`agabyrkn`) is the fresh completed retry, not a continuation. The three completed new runs have matching training/axis/calibration hashes; R078 remains a historical dirty-code control. Single-seed, paired-view evidence is not a substitute for repeated training seeds.

Evidence: [full interactive axis analysis](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/diagnostics/axis_ablation_20261014/index.html), [paired view differences](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/diagnostics/axis_ablation_20261014/paired_comparisons.json), [source hashes and motion histories](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/diagnostics/axis_ablation_20261014/evidence.json), and [all-view edge checks](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/diagnostics/axis_ablation_20261014/all_view_edges.json). The earlier R081-only update below is retained as historical context and is superseded by this completed comparison.

## Latest run update: R081 projected wide axis

The new run **R081 / `kg1lswrh`** started **14 October 2026 at 13:42:51 CEST**, saved its final PLY at **14:16:41 CEST**, and exited with code 0 at **14:16:42 CEST**. W&B runtime is **33m 50s**. It is a fresh seed-0 run using R078's 1435 px source, fixed measured angles, both angle corrections off, unchanged SH implementation, and densification until 13k. It changes the old forward clamp to post-update axis projection and widens tilt/side allowances from ±0.5° to ±2°. It does not isolate those two changes.

| External 30k PNG measure | R078 | R081 |
|---|---:|---:|
| Full-frame PSNR dB | 26.146 | 28.244 |
| Native / common-product PSNR dB | 27.167 | **28.217** |
| Foreground-crop SSIM | 0.85307 | **0.87721** |
| Foreground-crop LPIPS | 0.24903 | **0.22647** |
| Lower-edge mean absolute error px, alpha 0.5 | 4.914 | **1.956** |
| Lower-edge ripple RMS px, alpha 0.5 | 1.075 | **0.425** |
| Sampled small-print SSIM | 0.83818 | **0.88700** |
| Sampled barcode SSIM | 0.92522 | **0.95734** |
| Stored Gaussians | 194,696 | 223,120 |

Foreground PSNR gains **1.050 dB**, crop LPIPS improves **9.1%**, and the ripple proxy drops **60.5%**. PSNR and crop SSIM improve in all ten paired held-out views, and crop LPIPS improves in all ten. Edge error and ripple improve at alpha thresholds 0.1, 0.5 and 0.9. These views describe one trained model, not independent seed repeats.

The logged float endpoint is **28.302806 dB full-frame PSNR**, **28.220794 dB foreground PSNR**, and **0.877836 crop SSIM**. Keep these separate from the quantized-PNG scores above. The saved axis tilt is **28.506474°**, side **0.176757°**, and center shift **0.010469 scene units**, inside the effective tilt interval **28.066175–32.066175°** and side interval **−1.792489–2.207511°**. The axis changes across saved 5k/10k/15k checkpoints and is stable after the scheduled pose updates stop at 12k; it does not reproduce R078's old effective boundary lock.

![R078 and R081 matched edge close-ups](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/axis_R081_lower_left_v03.jpg)

The inspected lower-left boundary is much smoother; small print and barcode scores improve, although horizontal tearing remains visible in the print. Level views still show a pointed underside and raised lid, and top views retain smeared metal detail. **This is an appearance and silhouette improvement, not evidence of accurate 3D shape.** All higher-order SH coefficients remain exactly zero; the appearance limitation is still present.

**Initial R081-only decision (superseded above):** retain fixed TAE angles and use R081 as the candidate for matched repeats. Compare projected ±0.5° against projected ±2° and the legacy clamp with identical remaining settings, then repeat paired seeds before choosing bounds. This run meets the earlier 25% edge-improvement threshold on its one seed but does not satisfy the three-seed confirmation requirement. Keep SH changes as a separate experiment.

Evidence: [paired per-view scores](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/R081/per_view.json), [model/render hashes](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/R081/provenance.json), [configuration differences and source hashes](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/extension_R081.json), [saved log](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/runs/kg1lswrh/output.log), [W&B run](https://wandb.ai/3D-Scanning-MT/RotGS/runs/kg1lswrh), and [interactive comparison with videos](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/index.html#R081). A current render of R078 view 0 exactly matches its saved PNG, and current training/axis source hashes match R081's launch manifest. Historical dirty source patches remain incompletely preserved.

The six-variant sections below retain the October 7 comparison as historical evidence. Their best-result and bound statements apply to those six endpoints; this update supersedes their recommendation for the next candidate.

## The questions behind the experiments

The experiments pursue a clean, sharp, geometrically plausible product reconstruction. Specifically, they ask whether product segmentation removes marker clutter, whether an automatic crop preserves detail, whether physical motion calibration improves alignment, and whether additional angle learning helps after calibration. Ramseier and the down-facing can capture provide supporting examples on another product and another capture orientation.

The [earlier visual comparison](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/index.html) identifies the practical quality requirements: readable small print and barcode strokes, smooth lower edges, and plausible lid and base shape. A higher average image score answers only part of this objective.

## The ten October 1–7 runs and their roles

These are the ten timestamped local W&B training records selected for the original October 13 assessment, sorted by launch time. R081 is documented in the latest-run update above. All have saved 30,000-iteration endpoints. The dates below are CEST. Run identities are retained so each result can be traced to its configuration and model.

| Launch CEST | Run | Experiment role | Logged foreground PSNR | Logged foreground crop SSIM |
| --- | --- | --- | ---: | ---: |
| 1 Oct 11:03 | `o60bnz3d` | Earlier masks with TAE and both corrections | 21.127 | 0.8548 |
| 2 Oct 11:38 | `xrm5py3b` | SAM31 product isolation, 1650 px, TAE | 27.230 | 0.8563 |
| 6 Oct 13:39 | `v89ktriw` | Ramseier with SAM31 and TAE | 25.695 | 0.8682 |
| 7 Oct 12:15 | `kf3wv1dr` | Top SAM31 automatic crop, TAE, local and global | 27.038 | 0.8519 |
| 7 Oct 13:10 | `5q9tqv2u` | Down SAM31 automatic crop, TAE, local and global | 27.008 | 0.8394 |
| 7 Oct 15:13 | `w5cs4xiu` | Same top input, TAE, global correction only | 27.112 | 0.8526 |
| 7 Oct 15:52 | `78gbr4c6` | Same top input, TAE, local correction only | 27.024 | 0.8512 |
| 7 Oct 16:26 | `duiwndaw` | Same top input, TAE, both corrections disabled | 27.170 | 0.8537 |
| 7 Oct 17:03 | `x68bbab4` | Same top input, nominal angles, both corrections | 24.541 | 0.7897 |
| 7 Oct 17:35 | `pyds8296` | Same top input, nominal angles, global correction only | 24.049 | 0.7731 |

These native foreground scores use each prepared dataset's own alpha. The first three rows and the down capture are supporting comparisons, not matched tests of the same targets. In particular, the older masks include marker/background material. Source: [extracted scores](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/run_scores.csv), [complete run evidence](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/runs.json), and the copied configuration and W&B summary for each run under `evidence/runs/`.

## Five earlier top can baselines ranked by PSNR

The five earlier available single-camera top-can leaders, ranked by their historical full-frame held-out PSNR, are **R025, R058, R026, R027, and R024**. Each row represents its saved 30k endpoint. The initial 10k reference and multi-camera experiments do not outrank these five. The [selection record](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/ranking.json) lists the six available earlier single-camera candidates, including the lower-scoring 10k reference.

| Rank | Run and preparation | Historical full-frame PSNR | Independently verified full-frame PSNR | Verified foreground PSNR | Product coverage | Gaussians |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | R025 1600 px fresh densify10k | 32.500 | 32.500 | 27.359 | 26.1% | 167,805 |
| 2 | R058 1777 px undistorted densify13k | 32.115 | 32.115 | **27.978** | 29.8% | 297,363 |
| 3 | R026 1828 px crop densify10k | 31.461 | 31.461 | 26.728 | 28.0% | 148,432 |
| 4 | R027 1777 px undistorted densify10k | 31.351 | 31.351 | 27.037 | 29.8% | 149,793 |
| 5 | R024 1600 px continuation | 30.069 | 30.069 | 24.582 | 26.1% | 22,043 |

Coverage is mean soft alpha over the ten held-out frames. The foreground ranking changes to R058, R025, R027, R026, R024. This is why full-frame PSNR is useful for reproducing the historical leaderboard but foreground PSNR is preferable when selecting a product reconstruction. Sources: [earlier scores CSV](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/earlier_scores.csv), [audited runs and original paths](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/runs.json), and [original run log](/shared/datasets/3D_Scanning_MT/output_RotGS/RUN_LOG.md).

![Five earlier PSNR leaders with verified foreground scores](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/earlier/verified_scores.png)

*Figure 1. The old PSNR leaders and one recent Session 04 TAE reference, R074. All plotted values use independently checked saved PNGs. Each capture uses its own target and mask, so cross-session differences do not isolate a method's advantage. The PSNR axis is truncated.*

### Which part of the older PSNR can be trusted

**The reported arithmetic is reproducible.** An independent CPU calculation checked every saved render against its paired GT, using RGB in [0, 1], mean squared error over all pixels and channels, and an equal-weight average of per-view PSNR. All five run means and all fifty per-view historical scores agree within **0.000004 dB**. All fifty saved GT images exactly equal their corresponding prepared RGBA source composited on white. The sources match indices **0, 8, …, 72**, the expected ten withheld images among 73 exposures. No missing pair, accidental wrong-source comparison, or mask-free foreground assumption was found in these retained evaluation artifacts.

That establishes the numerical scores of the **saved images**. It does not independently reproduce historical training, certify perfect source masks, or prove metrological 3D accuracy. The earlier comparison already checked that fresh saved-model renders of view zero exactly match the retained images for R024, R025, and R058; R026 and R027 received a saved-image audit here rather than a new model-rendering check. Sources: [all audited image pairs and hashes](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/per_view.json), [historical evaluator snapshot](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/source_code/metrics.py), [PSNR formula](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/source_code/utils/image_utils.py), and the [earlier comparison protocol](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/index.html).

The audit found no PSNR unit-range error: the historical evaluator loads [0, 1] RGB and uses the corresponding unit-range formula. The historical LPIPS normalization caveat is separate and does not explain or invalidate these PSNR values.

**SAM is not required to make PSNR valid.** The old preparations have real RGBA product masks, recorded as `alpha_mode=regenerated`, with Lab background thresholds of 8 for the 1600 px preparation and 5 for the other two. The non-SAM adapter detects foreground against crop-border color, retains a product component within the selected region, fills enclosed regions, and produces antialiased edges. The score audit uses the actual stored alpha, not an assumption about which segmentation tool produced it.

![Older masks and current SAM mask](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/earlier/mask_audit.png)

*Figure 2. Prepared nominal-zero targets with their alpha 0.5 boundary in red, above the stored alpha images. The five old targets have plausible product masks rather than opaque full-frame alpha. Can sizes differ because the native crops, output scaling, and sessions differ. Visual plausibility is not an independent mask-accuracy measurement.*

There is a genuine source caveat: all three earlier preparation metadata files warn that **100% of fully transparent source pixels contain zero or near-zero RGB**, consistent with destroyed or premultiplied RGB. Regenerating alpha cannot recover discarded RGB. This warning does not make the verified PSNR false, but a mask that excludes missing product pixels can make both training and evaluation easier. Foreground PSNR still measures fidelity to the prepared, masked target; it cannot certify that every real product edge or reflective region was retained. Compare masks against intact original RGB or independent annotations before making that stronger claim. Sources: copied [R025 metadata](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/R025/preprocessing_metadata.json), [R026 metadata](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/R026/preprocessing_metadata.json), [R027 metadata](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/R027/preprocessing_metadata.json), and [non-SAM adapter](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/source_code/tools/preprocessing/prepare_rotgs_sequence.py).

### Why the headline PSNR is much higher

**White background dilutes the whole-frame error.** R025 evaluates a frame with roughly **26.1% product coverage**, leaving about 73.9% background. Its full-frame score is **5.141 dB above its foreground score**. Across the five old models that bonus ranges from **4.137 to 5.487 dB**. R074 covers about **44.2%** of its tighter frame with the product and has larger background-weighted errors, so its full-frame score is actually **0.955 dB below foreground PSNR**.

For soft-alpha coverage `p`, the squared error decomposes as `MSE_full = p × MSE_foreground + (1 − p) × MSE_background`. If background error were zero, the whole-frame PSNR bonus would be `−10 log10(p)`. This gives the mechanism behind the old scores without treating background fit as product detail. In the audited images, the background-weighted component contributes on average **14.6% of total squared error for R025** and **63.3% for R074**. This component includes partially transparent edges as well as pixels outside the product; it is not a direct count of visible floating artifacts.

Using the same external PNG scoring protocol, **R025 versus R074** has a **6.420 dB full-frame gap**, but only a **0.324 dB foreground gap**. The difference between their full-versus-foreground offsets is **6.096 dB**, or about **95% of the headline gap**. This is an exact arithmetic decomposition of those means. It is not a causal estimate of how much advantage comes from background, because the capture, alpha, framing, geometry, and implementation are not controlled across sessions. Source: [score decomposition](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/score_decomposition.json) and [recent reference audit](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/reference_R074.json).

![Headline score decomposition and error allocation](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/earlier/score_domains.png)

*Figure 3. The apparent six-decibel advantage becomes much smaller on the product foreground. Error fractions are averaged per view with the actual stored soft alpha; foreground and background contributions sum to one.*

**Framing and target detail also changed.** R025's source square is 1891 px but its prepared output is 1600 px, a **0.846 linear scale factor**. R026 and R027 retain native 1828 and 1777 px squares; the recent 1435 px automatic crop is native as well. Downsampling changes fine-detail difficulty, while different calibration, undistortion, physical placement, lighting, and motion settings prevent a causal comparison of crop size or SAM alone. These are documented differences, not a measured attribution of their individual effects.

The retained earlier configurations also record a **±4° global sweep allowance**, a broad **0–90° tilt interval**, **±5° side allowance**, and **0.25 scene-unit center offset allowance**, compared with tightly calibrated TAE geometry and a **0.05** center allowance in the latest runs. Greater freedom can absorb image/model mismatch without preserving physical shape. These stored settings help define a useful controlled baseline, but they do not prove which freedom caused the old foreground gains; some configuration timestamps postdate the launches, as the run log documents.

### The earlier models also contain real quality gains

The foreground audit prevents dismissing the old models as a white-background trick. **R058 scores 27.978 dB on the product**, and its retained corrected crop SSIM and LPIPS are **0.9054 and 0.1873**. R025 has corrected crop SSIM **0.8854** and LPIPS **0.2102**. These use each Session 03 model's own target; they are appearance references rather than controlled evidence that its training recipe beats TAE on the Session 04 capture.

Two within-preparation comparisons are stronger. **R024 to R025** uses the same 1600 px target and improves foreground PSNR from **24.582 to 27.359 dB**, a **2.777 dB gain**, while Gaussian count rises from 22,043 to 167,805. The change combines a continuation versus fresh training and a different growth schedule. **R027 to R058** uses the same undistorted 1777 px target and improves foreground PSNR by **0.941 dB**, with 149,793 versus 297,363 Gaussians and a 10k versus 13k growth cutoff. Code and other recorded settings still limit attributing either change to densification alone, but those gains survive removing white-background dilution.

![Ground truth and saved renders for the five older leaders](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/earlier/earlier_visuals.png)

*Figure 4. Each old model against its own target at nominal 0°. Product crops are resized for display, not for scoring. The low-population continuation, R024, shows stronger blur and distortion despite a 30.069 dB whole-frame score.*

![Older label texture comparison](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/earlier/earlier_texture.png)

*Figure 5. Own-target label regions at nominal 40° show the retained detail differences. The HTML viewer also exposes three original-size comparison triptychs per earlier run.*

The SH audit adds a qualification to the optimization discussion: **R025, R058, R026, and R027 also have all higher-order SH coefficients equal to zero**. Their apparent advantage cannot be explained by those saved models having richer nonzero SH appearance. R024 is the exception, with nonzero higher-order coefficients, yet has the weakest foreground PSNR among the five. Correcting active SH degree remains an implementation priority, but it does not by itself explain the old-versus-new score gap. Source: `ply_audit` entries in [audited run evidence](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/runs.json).

The earlier novel-view report shows a pointed base and lid artifacts even for R025 and R058. Strong held-out image PSNR therefore remains compatible with implausible geometry at other elevations. The earlier leaderboard should be used to recover useful appearance and growth baselines, not to certify the best 3D shape.

![Earlier foreground leader at novel elevations](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/earlier/R058_elevation_sheet.jpg)

*Figure 6. Retained R058 diagnostic renders from the earlier comparison. These views have no photographed reference and are a plausibility check; they expose shape defects that its 32.115 dB full-frame and 27.978 dB foreground averages cannot quantify.*

For the next comparison, keep R025 and R058 as historical appearance references and test their recorded geometry and growth choices on the **same Session 04 prepared images, corrected code, and common evaluation masks** as the TAE variants. A useful pass criterion is at least **0.30 dB foreground PSNR or 5% crop LPIPS improvement**, with no increase beyond **10% in boundary error**, repeated across paired seeds. That would establish an improvement on the current capture. Simply comparing the old 32.5 dB leaderboard to a new 26 dB whole-frame score would not.

## The six variants on the same top capture

All six central runs use the same named SAM31AC 1435 px preparation, 73 images, a fixed camera, no optical flow, a 450,000 Gaussian cap, and densification ending at 13,000 iterations. Their seed initialization uses `safe_state`, which sets seed 0; deterministic CUDA execution is not enforced. The common command settings and variant switches are recorded in the [training queue](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/jobs/train_queue.sh).

| Variant | Local angle correction | Global sweep bound | Axis setup | Foreground PSNR | Foreground crop SSIM | Gaussians |
| --- | --- | --- | --- | ---: | ---: | ---: |
| TAE and both corrections | ±0.05° | ±0.15° | Calibrated and bounded | 27.038 | 0.8519 | 192,448 |
| TAE and global only | Off | ±0.15° | Calibrated and bounded | 27.112 | 0.8526 | 191,905 |
| TAE and local only | ±0.05° | Off | Calibrated and bounded | 27.024 | 0.8512 | 189,606 |
| TAE with fixed angles | Off | Off | Calibrated and bounded | **27.170** | **0.8537** | 194,696 |
| Nominal and both corrections | ±0.05° | ±1.2° | Default initialization and free axis | 24.541 | 0.7897 | 152,688 |
| Nominal and global only | Off | ±1.2° | Default initialization and free axis | 24.049 | 0.7731 | 139,686 |

“Fixed angles” still permits axis and center optimization. “Local off” still permits the global sweep correction when its bound is nonzero. In TAE mode the local model has eight interpolation intervals and nine learned knot values, with an endpoint envelope. The nominal branch has one interval and two knot values without that envelope. Their local correction parameterizations are therefore also different. Sources: [angle predictor](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/scene/residual_predictor.py), [trainer construction](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/train.py), and [TAE configuration](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/scene/tae_calibration.py).

![Foreground quality for six same-input variants](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/core_scores.png)

*Figure 7. Logged float-image validation means over ten held-out views at 30k. The axes are truncated to reveal differences. There are no independent seed repeats from which to estimate uncertainty.*

### What the scores support

**The complete TAE setup has substantially better foreground scores here.** Fixed TAE angles exceed the nominal run with both corrections by 2.629 dB PSNR and 0.0640 SSIM; they exceed the latest nominal run by 3.122 dB and 0.0806 SSIM. These are observed comparisons of complete setups.

**Additional corrections have not improved the best TAE result.** The four TAE variants span only 0.146 dB PSNR and 0.00250 SSIM. Fixed measured angles score highest. Removing local correction from the TAE run with global correction gains 0.074 dB; removing global correction from the no-local TAE run gains another 0.058 dB. These small single-run differences support simplifying the baseline, but do not establish that corrections are generally harmful.

**Local correction helps the nominal setup in this comparison.** It gains 0.493 dB and 0.01657 SSIM over nominal global-only. This is consistent with angle learning compensating for some mismatch, but the learned parameter is not an independently measured physical error.

### Why full-frame PSNR changes the apparent winner

The nominal run with both corrections has **26.302 dB full-frame PSNR**, above the fixed-TAE run's **26.214 dB**, while its foreground PSNR is **24.541 versus 27.170 dB**. A global image average can reward background fit while hiding worse product reconstruction. On this matched input, foreground measures should carry more weight for product detail; evaluate background leakage separately.

![Full-frame and foreground PSNR](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/metric_domains.png)

*Figure 8. The score domain changes the ranking even for identical framing. The plot uses logged metrics and a truncated axis.*

## What the images show

The following figures use the final saved validation triptychs for the same held-out frame `0000.png`, a nominal 0° capture. Ground truth is followed by the six variants in the order used in Figure 7. Each close-up uses identical native pixel coordinates. Images are arranged and scaled for display without sharpening or error amplification. Open the PNGs to inspect their full resolution.

![Matched held-out render comparison](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/matched_view00.png)

*Figure 9. The complete product silhouette and branding remain recognizable. The nominal variants show larger appearance and contour differences; the TAE variants still lose fine detail.*

![Matched metallic lid close-ups](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/lid_closeups.png)

*Figure 10. Metallic grooves, pull-tab boundaries, and the rim remain less faithful than the photograph. The images do not identify whether pose, appearance capacity, or Gaussian coverage is the dominant cause.*

![Matched label close-ups](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/label_closeups.png)

*Figure 11. Major lettering survives, while narrow strokes and fine label structure remain blurred. The TAE correction switches produce subtle changes compared with the difference between complete motion setups.*

![Matched lower edge close-ups](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/edge_closeups.png)

*Figure 12. Lower boundary irregularities remain in the TAE group, including the fixed-angle variant. Disabling both corrections does not automatically produce a smooth boundary in this sampled view. The nominal boundaries show fewer regular spurs here while their label texture is more distorted, so the foreground score advantage does not imply an advantage on every visual property.*

The earlier report also compares small print at nominal 120°, barcode detail at 240°, and lower edges over multiple views. Its retained figure shows repeated lower-side spurs after separately disabling local and global corrections:

![Previously evaluated lower edges](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/previous/new_s04_lower_left_v03.jpg)

*Figure 13. Retained earlier report: R072 is SAM31 1650 px; R074 is TAE with both corrections; R076 disables local correction; R077 disables global correction. These are separate saved-model renders and belong to the earlier report's PNG evaluation protocol.*

The same earlier comparison finds pointed bases and raised or smeared lid structure at novel elevations. Those findings apply to the models it inspected. The three newest variants have no equivalent novel-elevation evaluation in that report. A convincing render near the capture elevation does not establish faithful 3D shape, and novel-view images without photographed references provide a plausibility check rather than a geometric accuracy score.

![Earlier TAE model viewed from new elevations](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/previous/tae_elevation_sheet.jpg)

*Figure 14. Retained R074 model, equivalent to the TAE run with both corrections. At 20° elevation the render exposes a pointed extension below the base and a raised lid artifact; the 40° views look more plausible. These are diagnostic synthetic views with no matching reference photographs, and they do not describe the newest fixed-angle endpoint.*

## Why TAE should help and why its benefit is not isolated

The measured sweep is **359.133883°**, rather than 360°, with a maximum absolute step error of **0.067441°**. After subtracting a straight-line correction for the closure error, the remaining measured angle deviation reaches **0.123090°**, with RMS **0.061661°**. A single accumulated sweep parameter cannot represent that nonlinear variation. TAE supplies the measured sequence directly, which gives a physical reason to expect better alignment.

![Measured turntable motion](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/measured_motion.png)

*Figure 15. The upper panel compares measured angle departures, their straight-line closure component, and the latest nominal run's learned global correction. The lower panel isolates nonlinear measured variation. This variation is real within the estimator's fitted model; it is not an estimate of its error.*

The reported maximum local angle uncertainty is **0.011469°**, but it is a local fit-conditioning estimate. It assumes exact intrinsics and marker dimensions and excludes their systematic errors. The TAE summary explicitly states these assumptions. The ±0.05° local correction bound applies on top of already measured angles; it is not meant to reproduce the full nonlinear departure from nominal angles. Sources: [angle table](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/turntable_angles.csv), [physical fit summary](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/tae_summary.json), [derived angle statistics](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/angle_analysis.json).

The `--leverage_TAE` switch simultaneously changes several parts of reconstruction:

| Component | TAE branch | Nominal branch |
| --- | --- | --- |
| Base angles | Measured sequence | Evenly spaced nominal angles |
| Normalized time | Measured angle divided by measured sweep | Frame index divided by 72 |
| Axis initialization | Physical estimate near 30.066° tilt | Camera-up axis at 0° tilt |
| Axis constraints | Calibrated tilt and side, each with ±0.5° allowance | Free normalized axis |
| Center initialization | Physical table center transformed into scene coordinates | Back-projected product alpha centroid |
| Local model | Eight intervals, anchored endpoint envelope | One interval, unanchored endpoints |
| Global allowance in these runs | ±0.15° or disabled | ±1.2° |
| Correction prior | Enabled when TAE is used | Disabled |

The physical table center and the product alpha centroid can locate different points along or away from the relevant rotation axis. Rotation depends on the axis line and its coordinate transform, not on the apparent object centroid alone. Verify transforms with marker reprojection and synthetic projection checks before attributing a pose discrepancy to the turntable measurement.

The [TAE argument configurator](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/scene/tae_calibration.py) overwrites the axis setup, and the [trainer](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/train.py) injects the physical center only in that branch. The current CLI cannot isolate angles from physical center initialization simply by changing one flag. Independent angle-source and geometry-source choices are needed for the controlled comparison below; these are proposed implementation choices, not existing CLI flags.

## Concrete optimization findings

### Higher-order appearance is unused in all six saved models

The current trainer increments `active_sh_degree` every 1000 iterations, but constructs its fixed rasterizer once at the beginning. The renderer copies the active degree into `GaussianRasterizationSettings`; those settings form an immutable tuple and the render call does not refresh them. In a fresh run the model starts at degree zero, so this code path keeps the rasterizer at degree zero while the model's degree increases.

The saved PLY inspection provides independent artifact evidence: **every one of the six models has 45 higher-order SH fields per Gaussian, all exactly zero**. Their higher-order appearance representation has therefore not learned nonzero coefficients. This matches the rasterizer behavior and makes appearance optimization a concrete candidate, particularly for the reflective lid. It does not predict a particular score gain.

Sources: [trainer around degree updates and rasterizer creation](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/train.py), [renderer settings](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/gaussian_renderer/__init__.py), [rasterization interface](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/__init__.py), and `sh_audit` entries in [run evidence](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/runs.json). The source snapshot describes the current code; dirty historical working trees prevent a commit hash alone from identifying the exact historical source. The zero coefficient finding describes the actual saved models.

There is a second appearance limitation: `rotate_shs` rotates only the first-degree block; rotations for degrees two and three are commented out. Fix the active-degree behavior and verify full appearance rotation before evaluating degree-three training. A degree-one experiment is a useful intermediate comparison. Reloading a PLY sets its active degree to the maximum, so validation and reloaded rendering should explicitly agree on the intended degree and motion state. Source: [SH rotation and PLY loading](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/scene/gaussian_model.py).

### Calibrated axes repeatedly reach their allowed limits

All four same-input TAE variants finish at approximately **30.566179° tilt**, the configured upper limit near **30.566175°**. Their side angle is approximately **−0.292489°**, the lower limit obtained from **0.207511° initialization minus 0.5°**. Float rounding accounts for the tiny limit differences. The nominal variants end at **21.435° and 19.158° tilt**, while their learned sweeps reach approximately **−1.198°**, very close to the −1.2° limit.

![Motion endpoints and constraints](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/motion_endpoints.png)

*Figure 16. TAE endpoints share the constrained axis, whereas nominal axes and sweep corrections move substantially. Similar constrained endpoints do not prove that the physical estimate is incorrect.*

`get_axis` applies hard clamps to tilt and side derived from a raw vector. When a raw angle is strictly beyond a clamp boundary, its derivative through that clamp is zero. The effective saved axis does not reveal how far the raw parameter moved beyond the boundary, so a saturated endpoint alone does not prove that gradients stalled. Logging raw angles, effective angles, and gradients would resolve this.

Axis and center learning rates become zero from 12k, while Gaussian growth continues until the configured 13k cutoff and enabled angle correction learning continues afterward. This schedule may leave pose and representation out of step. A small pose refinement after growth is a controlled hypothesis to test, not an established fix. Sources: [axis constraints and schedules](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/scene/gaussian_model.py), [optimizer and growth loop](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/train.py), copied run settings and final motion files.

### Segmentation improved isolation more clearly than raw product fidelity

The earlier-to-SAM31 native foreground PSNR jump is about 6.10 dB, but the old foreground includes marker material and is not the same scoring domain. The earlier report aligns the product onto a common target and changes the interpretation:

| Preparation with TAE | Common-product PSNR | Common-crop SSIM | Common-crop LPIPS |
| --- | ---: | ---: | ---: |
| Earlier masks, R071 | 27.592 | 0.8333 | 0.2667 |
| SAM31 1650 px, R072 | 27.161 | 0.8555 | 0.2445 |
| SAM31AC 1435 px, R074 | 27.035 | 0.8512 | 0.2513 |

![Preparation on a common product target](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/assets/preparation_common_target.png)

*Figure 17. Existing aligned PNG evaluation, with one shared SAM31 product target. Higher PSNR and SSIM and lower LPIPS are better. These scores are kept separate from logged float-image metrics.*

SAM31 improves common-crop SSIM and LPIPS over the earlier masks while removing most marker clutter, although common-product PSNR is lower. The automatic crop reduces the saved population from 216,998 to 192,448 Gaussians, **11.3% fewer**, with a small loss in common-crop scores. The preparation comparison also changes code and framing, so neither the quality change nor the population reduction is attributable to crop alone. Sources: [earlier comparison metrics](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/previous_comparison/summary.json), [visual observations](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/previous_comparison/observations.json).

## Decisions and measurable success criteria

The criteria below are proposed practical thresholds for this project. They should be agreed before running each comparison and can be tightened if application requirements demand more. They are not published standards or estimated probabilities. Unless specified otherwise, compare the same images, shared product masks, code revision, seed, training budget, and rendering settings.

### Decision 1 Use fixed TAE angles as the provisional baseline

**Updated 14 October:** retain fixed measured angles with both corrections disabled. R082 projected ±0.5° is now the leading appearance candidate; fixed calibration R084 is the simpler control, R081 is the wide-axis comparison, and R078 is the historical control. **Original six-variant support: moderate.** Keep SAM31 product isolation, calibrated bounded geometry, measured angles, and both corrections disabled. This is the strongest recorded six-variant foreground result, and it removes two degrees of freedom without a demonstrated image-quality penalty. Axis and center remain optimized.

**Success score:** across at least three paired seeds, remain within **0.10 dB foreground PSNR**, **0.002 crop SSIM**, and **5% crop LPIPS** of the best correction-enabled variant, with no more than **10% increase in lower-edge error**. If the gap remains around 0.15 dB or below, collect more repeats before claiming a robust winner. The current 27.170 dB and 0.8537 are baseline observations, not a guaranteed target after code fixes.

### Decision 2 Correct and validate the appearance implementation

**Current support: high for the implementation finding; unknown for the resulting quality gain.** Refresh or rebuild rasterizer settings when the active SH degree changes, log the actual renderer degree, and verify the appearance rotation for every enabled degree. Compare degree zero, correctly rotated degree one, and correctly rotated degree three using fixed TAE angles and the same geometry.

**Correctness success score:** renderer degree equals model degree at every update and validation; a synthetic case with nonzero higher-order coefficients produces a nonzero response and gradient; object rotation and the equivalent transformed viewing direction agree; same-state training and reloaded renders differ by no more than **1e−5 mean absolute RGB** and **0.01 dB foreground PSNR**, allowing a documented numerical tolerance if the backend requires it.

**Quality success score:** accept extra appearance capacity if it gives at least **0.30 dB foreground PSNR** or **5% improvement in common-crop LPIPS**, consistently across paired seeds, while lower-edge error worsens by no more than **10%**. A correctness fix can succeed even if higher-degree appearance fails this quality gate. Re-establish the baseline after the fix before attributing later gains to TAE.

### Decision 3 Separate measured angles from the geometry setup

**Current support: high that the present comparison is confounded; the angle-only gain is unknown.** Implement independent selection of angles and geometry initialization. Run this matrix with all local and global corrections disabled:

| Test | Base angles | Geometry package | Question answered |
| --- | --- | --- | --- |
| A | Measured | Calibrated center and bounded calibrated axis | Physical baseline |
| B | Nominal | Exactly the same calibrated geometry as A | Angle-only effect relative to A |
| C | Measured | Alpha-centroid center and default free axis | Angle effect under default geometry |
| D | Nominal | Exactly the same default geometry as C | Default baseline |

Within A/B and C/D, keep all optimizer settings and view ordering equal. In a first angle-only test, also use the same index-based normalized time in both branches; when corrections are disabled this should have no effect, and makes later extensions easier to interpret. Run paired seeds 0, 1, and 2 at minimum; the current hard-coded seed requires explicit reproducibility support before doing this.

**Success score for preferring measured angles:** A exceeds B by **at least 0.30 dB foreground PSNR and 0.005 crop SSIM**, the mean PSNR difference is positive for every paired seed, and common-crop LPIPS and silhouette error do not worsen by more than **5%**. This is a practical acceptance threshold, not a significance test. If A and B tie while both outperform C/D, retain the measured physical setup but attribute the demonstrated benefit mainly to geometry. If the gains differ across geometry packages, report their interaction.

### Decision 4 Improve pose optimization before widening its bounds

**14 October result:** R081 tested projection and wider bounds together and substantially improved the edge measures. Separate their effects with matched projected narrow/wide runs and repeats; the historical recommendation below explains why the old boundary hit alone was insufficient evidence.

**Current support: moderate for investigation; low for any particular remedy.** First verify camera/world signs, the rotation axis line, center scaling, and angle direction using projected marker geometry. Log the unclamped tilt and side, their effective values, and gradient norms. With measured angles fixed, compare frozen physical axis/center against current bounded optimization, then vary axis and center separately.

If raw parameters exceed hard limits, compare a smooth bounded angular parameterization under the same physical allowances. If raw parameters remain inside the bounds, investigate the schedule instead. A small-rate pose refinement after densification is preferable to changing the pose schedule and bounds together. Widening bounds becomes informative only after the coordinate and optimizer checks pass; a boundary hit by itself does not justify it.

**Diagnostic success score:** complete raw/effective pose and gradient records; synthetic and marker projections agree under the chosen transform; no unexpected sign or scale discrepancy. **Quality success score:** reduce lower-edge ripple RMS or boundary error by **at least 25%**, lose no more than **0.10 dB foreground PSNR or 0.002 crop SSIM**, and reproduce the improvement across three seeds. Check alpha thresholds **0.1, 0.5, and 0.9** so a softer silhouette cannot masquerade as better geometry.

### Decision 5 Retain SAM31 isolation and choose crop size by a quality budget

**Current support: high for removing Session 04 marker clutter; moderate for the crop tradeoff.** Keep product-only masks. The earlier Session 03 masks already isolate the can without SAM, so this evidence does not establish SAM's superiority over every earlier segmentation method. Compare 1650 px and 1435 px preparation under the same corrected code and identical physical motion. Evaluate on an aligned common product target and identical fine-print regions, and record Gaussian count, peak GPU memory, and training time under the same machine conditions.

**Success score for choosing the smaller crop:** common-product PSNR loss no greater than **0.20 dB**, common-crop SSIM loss no greater than **0.005**, LPIPS increase no greater than **5%**, no new clipped product pixels, and at least **10% fewer Gaussians or lower peak GPU memory**. Existing aligned scores and population counts fit that approximate tradeoff, but a controlled rerun is needed to establish the crop's contribution. Prefer the larger crop if small-print readability is the dominant requirement and its advantage persists.

### Decision 6 Add geometric evidence if faithful shape is required

**Current support: high that single-elevation image scores are insufficient.** Capture reference views from additional elevations or obtain independent shape/dimension measurements. The top and down runs are separate reconstructions and do not yet establish consistency in a shared coordinate system. Test a common model against photographed elevations it did not train on.

**Success score:** as an initial project target, obtain **silhouette IoU of at least 0.98** at withheld elevations and **diameter and height error of at most 1%** against independent measurements, alongside the image-quality gates above. These proposed targets have no current measured score. Virtual novel views cannot satisfy them. Joint calibrated multi-camera reconstruction is a later implementation task because the current TAE branch explicitly rejects multi-camera use.

### Decision 7 Tune Gaussian growth only after appearance and pose checks

**Current support: low for a larger population as the next remedy.** The six models contain 139,686–194,696 Gaussians, well below the 450,000 cap. Raising the cap alone changes no currently binding limit. If corrected appearance and stable pose still leave fine detail unresolved, compare densification cutoffs of 13k and 18k while keeping the cap, final iteration, gradients, and losses fixed.

**Success score:** at least **5% improvement in fine-print LPIPS** or **25% reduction in boundary error**, with no foreground PSNR loss beyond **0.10 dB**, and no more than **25% increase in peak memory or training time**. Track surface-edge defects separately from texture improvements so additional Gaussians cannot pass by fitting sharper but implausible spurs.

## Suggested order of work

1. Preserve R078 as the historical control, R082 projected ±0.5° as the appearance candidate, R084 fixed calibration as the simpler control, R081 as the wide-axis comparison, and R025/R058 as historical appearance references. The completed axis comparisons favor the boundary repair over widening; confirm narrow versus fixed with matched seed repeats.
2. Verify and correct active SH degree and appearance rotation; establish rendering parity and a fresh baseline.
3. Run the angle-versus-geometry matrix with paired seeds.
4. Audit pose transforms, raw clamp behavior, and optimization timing; test one pose change at a time.
5. Revisit crop and growth choices under the corrected implementation.
6. Add reference elevations and physical measurements before selecting a method for geometric accuracy.

This order tests concrete implementation behavior before adding more flexibility to calibrated angles. TAE already encodes closure drift and nonuniform steps; the remaining texture and shape errors can arise from appearance capacity, pose handling, representation, supervision, and insufficient view coverage as well as from angle estimation.

## Evaluation rules and evidence limits

Use the same ten held-out source indices **0, 8, …, 72** for comparable capture tests. Frames 0 and 72 are near the seam and are both withheld; they are not training leakage, but are near-repeated viewpoints. The ten views are not ten independent training repeats. Report per-view differences and seed-level variability separately.

Foreground PSNR uses soft alpha weighting. SSIM is computed over the foreground bounding rectangle with eight pixels of padding, including its background; it is not pixel-masked SSIM. Corrected LPIPS uses RGB in **[−1, 1]** with a fixed network and crop protocol. Keep logged float-image scores separate from PNG-equivalent external evaluations; the earlier report observes small differences from quantization. Keep different native masks, crops, captures, sessions, and dirty code revisions out of direct causal rankings.

The newest variants have saved comparison images for three held-out views rather than a complete standalone per-view numerical evaluation here. Their LPIPS, silhouette IoU, physical dimension accuracy, and multi-seed variability are not established by these W&B summaries. The earlier report's edge and LPIPS scores belong only to the models and protocol listed there. Extending those evaluations is necessary before applying every proposed decision gate.

The larger TAE-versus-nominal foreground differences are observed. Their physical cause is not isolated. The small differences among TAE correction variants do not support a statistically established winner. The implementation findings identify missing behavior and useful tests; they do not establish that fixing either behavior will remove the pointed base, lid distortion, or all lower-edge ripples.

## Evidence and reproducibility

| Evidence | What it supports |
| --- | --- |
| [Run scores CSV](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/run_scores.csv) | Chronology, final logged metrics, settings, counts, and motion endpoints |
| [Earlier scores CSV](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/earlier_scores.csv) and [selection](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/ranking.json) | The five historical top-can PSNR leaders and verified foreground scores |
| [Earlier per-view audit](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/per_view.json) and [manifest](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/manifest.json) | Fifty original old GT/render/source pairs, ten recent reference pairs, hashes, and score reproduction |
| [Earlier run evidence](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/runs.json) | Original paths, configurations, masks, and PLY SH checks for all five leaders |
| [PSNR domain decomposition](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/earlier/score_decomposition.json) | The 6.420 dB full-frame versus 0.324 dB foreground gap between R025 and R074 |
| [Complete run evidence](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/runs.json) | Saved configurations, summaries, motion, and numerical PLY SH audits |
| [Source and asset manifest](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/manifest.json) | Original source paths, SHA-256 hashes, and copied evidence identities |
| [TAE angle table](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/turntable_angles.csv) and [summary](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/tae_summary.json) | Physical sweep, per-frame angles, uncertainties, and fit assumptions |
| [Derived motion statistics](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/angle_analysis.json) | Departure from a linear closure correction |
| [Earlier aligned comparison](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/previous_comparison/summary.json) | Common-product and crop metrics for preparation variants |
| [Earlier focus metrics](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/previous_comparison/focus_summary.json) and [observations](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/previous_comparison/observations.json) | Retained fine-print, lower-edge, and geometry findings |
| [Current trainer snapshot](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/train.py) | Rasterizer lifetime, degree updates, losses, and optimization timing |
| [Current Gaussian model snapshot](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/scene/gaussian_model.py) | Axis bounds, center transform, SH rotation, and reload behavior |
| [Current TAE adapter](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/scene/tae_calibration.py) | Coupled physical defaults and angle loading |
| [Current angle predictor](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/evidence/source_code/scene/residual_predictor.py) | Correction parameterization and priors |
| [Report builder](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/build_report.py) | Extraction, PLY audit, evidence copying, and figures without CUDA rendering |
| [Earlier audit script](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/audit_earlier_runs.py) | Independent CPU PSNR, source/mask checks, ranking, and five-run illustrations |

Individual scientific figures are available as PNG and PDF in `assets/` and `assets/earlier/`. The HTML report includes an interactive viewer for the six recent same-input variants and the five earlier leaders at source views 0000, 0008, and 0016. Earlier triptychs combine unchanged saved GT and render pixels with unamplified absolute RGB error. Cross-session images have different own targets and are labeled accordingly. The Markdown text remains editable in `report.md`. All new artifacts are confined to this implementation-decisions folder.
