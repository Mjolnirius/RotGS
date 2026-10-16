# E06 — Measured motion and corrections

[All objectives](README.md) · [Project documentation](../README.md)

**Objective:** What changes with measured TAE angles and with local/global angle correction?  
**Status at review:** Compared; causal separation remains open  
**Primary run entries:** R066-R071, R076-R080  
**Reused references:** R074  
**Reviewed:** 2026-10-15, Europe/Zurich. Historical objectives were reconstructed from the run log and reports; they are editable descriptions of intent.

## Motivation and design

Compare nominal angles with physical turntable-angle estimates (TAE), and determine whether learned local residuals or global sweep correction remain useful. R066 is the earlier down-capture attempt. R067 is the failed baseline attempt and R068 its retained completed endpoint. R068–R071 compare earlier-mask branches; R074 and R076–R080 compare SAM31AC top-can correction variants.

## Findings and decision

Fixed measured angles with both corrections disabled are the recorded provisional baseline. The original TAE and nominal branches also change geometry setup, so the large observed gap is not an isolated angle-only effect. Small correction-variant gaps require repeated seeds. E07 separately tests axis optimization.

## Evidence and follow-up

- [Detailed implementation narrative](E06_motion/report.md)
- [Illustrated implementation report](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions/index.html)
- [Aligned comparisons](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/index.html)
- [Pose inspection](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/inspection/session_04/view_poses/index.html)

The proposed next causal test holds the geometry package fixed while switching nominal/measured angles, then repeats with the other geometry package. Success thresholds in the detailed report are proposals, not demonstrated outcomes.

## Your notes

Add observations, changed objectives, and decisions here. Preserve these notes when updating the study.
