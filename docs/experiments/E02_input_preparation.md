# E02 — Input preparation and detail

[All objectives](README.md) · [Project documentation](../README.md)

**Objective:** How do crop, undistortion, resolution, continuation, and densification affect detail?  
**Status at review:** Historical comparisons retained  
**Primary run entries:** R023-R027  
**Reused references:** R058  
**Reviewed:** 2026-10-15, Europe/Zurich. Historical objectives were reconstructed from the run log and reports; they are editable descriptions of intent.

## Motivation and design

After the 540 px milestone, the single-camera can experiments investigated higher-resolution preparations and more detail. R023/R024 are the 10k parent and its 30k continuation. R025 restarts with longer densification; R026 changes the crop; R027 uses undistorted input. R058 is a later 13k-densification reference reused from E04.

## Findings and decision

A continuation and a fresh run with more growth answer different questions. Crop, image dimensions, masks, intrinsics, and growth schedules differ across this sequence. Retain the observed comparisons without assigning the entire gain to resolution or densification alone.

## Evidence and follow-up

- [Single-camera development discussion](../archive/conversations/A_Full_Res.md)
- [Earlier convergence note](../archive/notes/debugging_tracking.md)
- [Aligned top-can comparison](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison/index.html)
- [Run log](/shared/datasets/3D_Scanning_MT/output_RotGS/RUN_LOG.md#session-03)

Use a common scoring target when reusing these as appearance references. E08 revisits growth under the newer axis setup.

## Your notes

Add observations, changed objectives, and decisions here. Preserve these notes when updating the study.
