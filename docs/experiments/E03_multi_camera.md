# E03 — Multiple camera elevations

[All objectives](README.md) · [Project documentation](../README.md)

**Objective:** Can 5°, 30°, and 60° captures produce a coherent shared reconstruction?  
**Status at review:** Exploratory branch; unresolved  
**Primary run entries:** R028-R048  
**Reused references:** None separately registered  
**Reviewed:** 2026-10-15, Europe/Zurich. Historical objectives were reconstructed from the run log and reports; they are editable descriptions of intent.

## Motivation and design

Combine the Quöllfrisch 5°, 30°, and 60° captures into one model to improve view coverage. The sequence includes native geometry, rigid prefusion, pose-only and pose/depth changes, silhouette supervision, bounded tilt, fresh geometry with imported fixed motion, a Gaussian cap, and later axis/phase probes.

## Findings and decision

R028 has configuration only. Short smoke tests verify execution or resuming; they do not establish reconstruction quality. The retained endpoints document weak fusion and several follow-up strategies. The history does not establish a final successful multi-camera solution. Preserve parent/continuation links and label lower-iteration evaluations explicitly.

## Evidence and follow-up

- [Multi-camera development discussion](../archive/conversations/B_Multi_Cam.md)
- [Additional multi-camera conversation](../archive/conversations/B_Multi_Cam%20copy.md)
- [Prefusion and trainer adaptation](../archive/conversations/C_Multi_Cam_PreFuse.md)
- [Recovered checkpoint evaluations](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/evaluations/2026-10-07_saved_checkpoints/README.md)
- [Run log](/shared/datasets/3D_Scanning_MT/output_RotGS/RUN_LOG.md#session-03)

Reopen this study with an explicit shared-pose/view-coverage criterion and a fixed matched baseline. It is separate from single-camera product coverage.

## Your notes

Add observations, changed objectives, and decisions here. Preserve these notes when updating the study.
