# E08 — Capacity and optimization limits

[All objectives](README.md) · [Project documentation](../README.md)

**Objective:** What stops improving, and does extending densification help?  
**Status at review:** Audit complete; matched external follow-up pending  
**Primary run entries:** R085  
**Reused references:** R081  
**Reviewed:** 2026-10-15, Europe/Zurich. Historical objectives were reconstructed from the run log and reports; they are editable descriptions of intent.

## Motivation and design

Distinguish scheduled freezing, hard caps, gradient saturation, and representation changes before tuning learning rates or extending training. The R081 CPU audit reads checkpoints and local history; it is an analysis, not another training run. R085 is a fresh R081-settings run extending densification from 13k to 20k.

## Findings and decision

The audit separates motion freezing by schedule from bound saturation, and growth stopping by schedule from the Gaussian cap. It also records opacity/color gradient limitations and increasingly thin Gaussians. These observations do not prove that higher learning rates or more motion fitting improve quality.

R085 reached 30k with float validation scores recorded in the ledger. A matched external saved-PNG comparison was pending in that snapshot. Keep those float results separate from R081's external metrics until the same protocol is available.

## Evidence and follow-up

- [Full R081 audit](E08_capacity/R081_audit.md)
- [Audit evidence and figures](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E08_capacity/R081_saturation_20261015/evidence.json)
- [R085 record](/shared/datasets/3D_Scanning_MT/output_RotGS/RUN_LOG.md)

Complete the matched evaluation before deciding on longer densification. Record quality and cost together; more stored Gaussians alone is not success.

## Your notes

Add observations, changed objectives, and decisions here. Preserve these notes when updating the study.
