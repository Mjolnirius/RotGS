# Experiment objectives

<!-- Generated from studies.json by tools/reports/manage_reports.py. -->
[Documentation home](../README.md) · [Run ledger](/shared/datasets/3D_Scanning_MT/output_RotGS/RUN_LOG.md) · [Visual catalogue](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/index.html)

Reviewed 2026-10-15 (Europe/Zurich). Objective labels were reconstructed from the surviving history and are editable in `studies.json` and the study documents.

A study describes a question. A session describes acquisition context. Each documented R ID has one primary study here and can be reused as a reference in other studies. Keep baseline, failure, retry, and continuation roles in the detailed run ledger.

| Study | Objective | Primary runs | Reused references | State at review |
|---|---|---|---|---|
| [E01 — First working reconstruction](E01_first_reconstruction.md) | What prevents a recognizable, stable first can reconstruction? | R001-R022 | — | Milestone achieved; historical |
| [E02 — Input preparation and detail](E02_input_preparation.md) | How do crop, undistortion, resolution, continuation, and densification affect detail? | R023-R027 | R058 | Historical comparisons retained |
| [E03 — Multiple camera elevations](E03_multi_camera.md) | Can 5°, 30°, and 60° captures produce a coherent shared reconstruction? | R028-R048 | — | Exploratory branch; unresolved |
| [E04 — Coverage across products](E04_product_baseline.md) | How does a common single-camera setup behave across product captures? | R049-R065 | R025 | Twelve completed captures analyzed |
| [E05 — Product isolation and automatic preparation](E05_product_isolation.md) | Do SAM product masks and automatic crops remove clutter while retaining useful detail? | R072-R075 | R068, R071 | Compared; preparation confounds documented |
| [E06 — Measured motion and corrections](E06_motion.md) | What changes with measured TAE angles and with local/global angle correction? | R066-R071, R076-R080 | R074 | Compared; causal separation remains open |
| [E07 — Axis optimization and edge artifacts](E07_axis.md) | Are edge spurs related to boundary handling, axis-bound width, or calibration? | R081-R084 | R078 | Provisional decision; seed repeats pending |
| [E08 — Capacity and optimization limits](E08_capacity.md) | What stops improving, and does extending densification help? | R085 | R081 | Audit complete; matched external follow-up pending |
| [E09 — Transfer the R082 setup to more products](E09_product_transfer.md) | Does the selected R082 setup transfer to Ramseier, KaegiFret, and Weisswein? | IDs pending | R082, R073, R054, R059, R065 | Queue defined; consult logs for current outcomes |

## What each study established

- **[E01](E01_first_reconstruction.md):** Input geometry, constrained motion, loss changes, and rotation-direction debugging led to the Nice Can milestone.
- **[E02](E02_input_preparation.md):** Separate continuation from fresh growth experiments; differing preparations limit direct attribution.
- **[E03](E03_multi_camera.md):** Native and prefused geometry, pose, depth, silhouette, and capacity probes survive; smoke checks are distinct from quality evidence.
- **[E04](E04_product_baseline.md):** The finalized Session 03 analysis supplies corrected quality, convergence, and cost evidence; failed attempts remain in the ledger.
- **[E05](E05_product_isolation.md):** Use the aligned common-product target for top-can preparations; down-can and Ramseier are supporting captures.
- **[E06](E06_motion.md):** Fixed measured angles are a useful provisional baseline; angle input and calibrated geometry are not fully isolated.
- **[E07](E07_axis.md):** R082 projected ±0.5° is the recorded appearance candidate; R084 fixed calibration is the control. Preserve failed R083.
- **[E08](E08_capacity.md):** The R081 audit separates scheduled freezing from saturation. R085 extends growth to 20k; the ledger has float endpoint scores.
- **[E09](E09_product_transfer.md):** The current queue uses SAM31AC inputs and projected ±0.5° with fixed TAE angles. New attempts still need stable ledger IDs.

## How the objectives developed

The first reconstruction (E01) led to more detail (E02), multiple elevations (E03), and product coverage (E04). Product isolation (E05) and measured-motion comparisons (E06) motivated the axis investigation (E07). The R081 audit and growth experiment (E08) revisit optimization limits. The current queue transfers the selected R082 setup to other products (E09). These are research threads; their runs and references can overlap in time.

Future appearance/SH correctness experiments remain proposals until they receive a registered study and actual run records. Historical results retain their source revision and metric protocol.

Use [the study template](_template.md) and [reporting workflow](../reporting.md) for new work. Edit objective metadata in [studies.json](studies.json), then rebuild navigation.
