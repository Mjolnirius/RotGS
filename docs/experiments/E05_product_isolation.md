# E05 — Product isolation and automatic preparation

[All objectives](README.md) · [Project documentation](../README.md)

**Objective:** Do SAM product masks and automatic crops remove clutter while retaining useful detail?  
**Status at review:** Compared; preparation confounds documented  
**Primary run entries:** R072-R075  
**Reused references:** R068, R071  
**Reviewed:** 2026-10-15, Europe/Zurich. Historical objectives were reconstructed from the run log and reports; they are editable descriptions of intent.

## Motivation and design

Remove turntable-marker/background material from the product target and automate a shared crop. R072 uses SAM31 product isolation on the top capture; R073 is Ramseier support; R074/R075 use automatic top/down crops. R068/R071 supply earlier-mask references. Dataset reviews stay with each prepared dataset.

## Findings and decision

Native foreground masks can include different material, so the raw foreground scores across preparations do not measure the same target. The top-can comparison aligns preparations to R074's RGB and soft product alpha. Ramseier and the down capture broaden inspection coverage but are not matched tests of the top-can intervention.

## Evidence and follow-up

- [Aligned preparation comparison](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/index.html)
- [Preparation and implementation assessment](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/index.html)
- [Current top-can crop review](/shared/datasets/3D_Scanning_MT/cleaned/Session_04/4_Quoellfrisch_Top_30d_SAM31AC_1435/crop_review/index.html)
- [Current top-can mask review](/shared/datasets/3D_Scanning_MT/cleaned/Session_04/4_Quoellfrisch_Top_30d_SAM31AC_1435/mask_review/index.html)
- [SAM31 instructions](../../tools/preprocessing/prepare_rotgs_sequence_SAM31.md)

Keep capture and preparation identity in future comparisons. The artifact catalogue discovers all current and archived dataset reviews without relocating them.

## Your notes

Add observations, changed objectives, and decisions here. Preserve these notes when updating the study.
