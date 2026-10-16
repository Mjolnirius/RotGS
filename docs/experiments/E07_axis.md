# E07 — Axis optimization and edge artifacts

[All objectives](README.md) · [Project documentation](../README.md)

**Objective:** Are edge spurs related to boundary handling, axis-bound width, or calibration?  
**Status at review:** Provisional decision; seed repeats pending  
**Primary run entries:** R081-R084  
**Reused references:** R078  
**Reviewed:** 2026-10-15, Europe/Zurich. Historical objectives were reconstructed from the run log and reports; they are editable descriptions of intent.

## Motivation and design

Investigate blue spurs, scalloped lower edges, and axis behavior in R078. R081 repairs boundary handling and widens bounds to ±2° together. R082 uses projection at the original ±0.5° width; R083 is the failed fixed-axis attempt and R084 its fresh successful retry.

## Findings and decision

The completed matched report favors R082 as the appearance candidate, with R084 fixed calibration as the simpler control. R082 versus R078 tests the intended boundary repair at the same width; R082 versus R081 tests width under matching launch source hashes. The R081-only recommendation is superseded by the completed comparison. Results come from one seed; cleaner silhouettes do not establish physical 3D accuracy.

## Evidence and follow-up

- [Completed axis comparison](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/diagnostics/axis_ablation_20261014/index.html)
- [Spur investigation and R081 follow-up](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/diagnostics/tae_spurs_20261014/index.html)
- [Shared top-can evidence bundle](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/README.md)

Confirm projected narrow versus fixed with paired seeds. The appearance/SH correctness follow-up remains a separate proposed intervention. E09 tests transfer to more products.

## Your notes

Add observations, changed objectives, and decisions here. Preserve these notes when updating the study.
