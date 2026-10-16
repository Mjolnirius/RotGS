# E04 — Coverage across products

[All objectives](README.md) · [Project documentation](../README.md)

**Objective:** How does a common single-camera setup behave across product captures?  
**Status at review:** Twelve completed captures analyzed  
**Primary run entries:** R049-R065  
**Reused references:** R025  
**Reviewed:** 2026-10-15, Europe/Zurich. Historical objectives were reconstructed from the run log and reports; they are editable descriptions of intent.

## Motivation and design

Run a common single-camera queue over different product captures and study convergence, quality, topology, and computational cost. R049 is the earlier Ramseier local run. R050–R065 include the W&B queue, two failed W&B runs, immediate queue failures, and successful retries. The finalized scientific cohort contains twelve completed captures, including related top/down captures; it is not twelve independent products.

## Findings and decision

The final analysis uses source-alpha foreground PSNR and complete foreground-crop SSIM/LPIPS. Historical full-frame results are retained for methodological comparison. Gaussian counts are unchanged between the 25k and 30k checkpoints, so that paired refinement study describes fixed-topology optimization. Use the full report for quantitative claims and source limitations.

## Evidence and follow-up

- [Final scientific narrative](E04_product_baseline/report.md)
- [Analysis data and export bundle](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/README.md)
- [Evaluation policy](../methods/evaluation.md)
- [Session 03 orbit gallery](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/inspection/session_03/orbits/index.html)

E09 is the later transfer test of the R082 setup. Keep its results separate from this older fixed cohort.

## Your notes

Add observations, changed objectives, and decisions here. Preserve these notes when updating the study.
