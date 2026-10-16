# End-to-end pipeline guide

[Documentation home](README.md) · [Experiment objectives](experiments/README.md)

Use this page to navigate the workflow. The detailed [recorded pipeline report](pipeline/recorded_2026-10-13.md) is a **13 October 2026 snapshot using R080 / `pyds8296`**, not a statement of the latest or preferred run. Its evidence and original export are retained in the [pipeline bundle](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/pipeline/2026-10-13/README.md).

| Stage | What to retain | Instructions and inspection |
|---|---|---|
| Acquisition and camera calibration | Capture/session identity, camera setup, calibration input and outputs | [Recorded acquisition/calibration](pipeline/recorded_2026-10-13.md) and `tools/calibration/` |
| Dataset preparation | Original source identity, transformed intrinsics, crop and mask settings | [SAM31 preparation](../tools/preprocessing/prepare_rotgs_sequence_SAM31.md), [SAM21 preparation](../tools/preprocessing/prepare_rotgs_sequence_SAM21.md) |
| Physical motion calibration | Measured angles, axis/center conventions, uncertainty and input provenance | [Turntable-angle instructions](../tools/calibration/turntable_angles/README.md) |
| Training | Study ID, stable run/attempt ID, parent, actual saved configuration, source revision, logs | [Objectives](experiments/README.md), [training queue](../jobs/train_queue.sh), [run ledger](/shared/datasets/3D_Scanning_MT/output_RotGS/RUN_LOG.md) |
| Evaluation | Capture/preparation, checkpoint, held-out views, scoring target and protocol | [Evaluation policy](methods/evaluation.md), [evaluation tool](../tools/evaluation/README.md) |
| Visual inspection | Reproducible settings, media and manifests | [Report catalogue](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/index.html), `tools/inspection/render_orbit_report.py`, `tools/inspection/export_view_poses.py` |
| Decision and reporting | What the experiment established, limitations, next objective | [Study template](experiments/_template.md) and [reporting workflow](reporting.md) |

The selected axis setup and the evidence supporting it are recorded in [E07](experiments/E07_axis.md); its transfer to more products is [E09](experiments/E09_product_transfer.md). Use a run's saved settings to reproduce it. Current defaults, historical example settings, and the current candidate can differ.
