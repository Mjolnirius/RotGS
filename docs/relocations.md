# Report relocations

[Documentation home](README.md) · [Reporting workflow](reporting.md)

Consolidated on 15 October 2026. Canonical sources are under `docs/`; generated reports and evidence are under `output_RotGS/reports/`. New work should use the canonical paths.

Compatibility symlinks at retired locations resolve original citations and paths embedded in scientific provenance. They contain no duplicate evidence. Keep them until dependent historical records and tools have a replacement path-resolution strategy.

| Former location | Canonical location |
|---|---|
| `milestones` | [docs/milestones](/workspaces/RotGS/docs/milestones) |
| `Session Saves` | [docs/archive/conversations](/workspaces/RotGS/docs/archive/conversations) |
| `analysis/wandb_report` | [output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis) |
| `analysis/R081_saturation_20261015` | [output_RotGS/reports/experiments/E08_capacity/R081_saturation_20261015](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E08_capacity/R081_saturation_20261015) |
| `output_RotGS/comparison_top_20261007` | [output_RotGS/reports/experiments/E07_axis/comparison](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E07_axis/comparison) |
| `output_RotGS/session_04/implementation_decisions` | [output_RotGS/reports/experiments/E06_motion/implementation_decisions](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions) |
| `output_RotGS/evaluation_psnr_20261007` | [output_RotGS/reports/evaluations/2026-10-07_saved_checkpoints](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/evaluations/2026-10-07_saved_checkpoints) |
| `output_RotGS/evaluation_psnr_20261014` | [output_RotGS/reports/evaluations/2026-10-14_native_psnr](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/evaluations/2026-10-14_native_psnr) |
| `output_RotGS/session_03/render_report` | [output_RotGS/reports/inspection/session_03/orbits](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/inspection/session_03/orbits) |
| `output_RotGS/session_04/render_report` | [output_RotGS/reports/inspection/session_04/orbits](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/inspection/session_04/orbits) |
| `output_RotGS/session_04/render_report_gaussian_centers` | [output_RotGS/reports/inspection/session_04/gaussian_centers](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/inspection/session_04/gaussian_centers) |
| `output_RotGS/session_04/view_pose_report` | [output_RotGS/reports/inspection/session_04/view_poses](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/inspection/session_04/view_poses) |
| `tools/preprocessing/debugging_tracking.md` | [docs/archive/notes/debugging_tracking.md](/workspaces/RotGS/docs/archive/notes/debugging_tracking.md) |
| `Session Saves/D_current_pipeline_latest_run.md` | [docs/pipeline/recorded_2026-10-13.md](/workspaces/RotGS/docs/pipeline/recorded_2026-10-13.md) |
| `Session Saves/assets` | [output_RotGS/reports/pipeline/2026-10-13/assets](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/pipeline/2026-10-13/assets) |
| `Session Saves/D_current_pipeline_latest_run.zip` | [output_RotGS/reports/pipeline/2026-10-13/D_current_pipeline_latest_run.zip](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/pipeline/2026-10-13/D_current_pipeline_latest_run.zip) |
| `analysis/wandb_report/rotgs_wandb_analysis_final.md` | [docs/experiments/E04_product_baseline/report.md](/workspaces/RotGS/docs/experiments/E04_product_baseline/report.md) |
| `output_RotGS/session_04/implementation_decisions/report.md` | [docs/experiments/E06_motion/report.md](/workspaces/RotGS/docs/experiments/E06_motion/report.md) |
| `analysis/R081_saturation_20261015/README.md` | [docs/experiments/E08_capacity/R081_audit.md](/workspaces/RotGS/docs/experiments/E08_capacity/R081_audit.md) |
| `output_RotGS/build_run_log_table.py` | [tools/reports/build_run_log_table.py](/workspaces/RotGS/tools/reports/build_run_log_table.py) |
| `analysis/wandb_report/export_wandb_analysis.py` | [tools/reports/export_wandb_analysis.py](/workspaces/RotGS/tools/reports/export_wandb_analysis.py) |
| `analysis/wandb_report/generate_final_report_assets.py` | [tools/reports/generate_training_report.py](/workspaces/RotGS/tools/reports/generate_training_report.py) |
| `orbitvu/Milestone_01_CODEX.md` | [docs/archive/conversations/Milestone_01_CODEX.md](/workspaces/RotGS/docs/archive/conversations/Milestone_01_CODEX.md) |

| `output_RotGS/session_04/implementation_decisions/format_report.py` | [tools/reports/format_implementation_report.py](/workspaces/RotGS/tools/reports/format_implementation_report.py) |

[Durable migration inventory and original SHA-256 hashes](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/migration_2026-10-15.json) record the moves. Original raw conversations, copied evidence, numerical results, media, ZIPs, and PDFs retain their bytes. Live document navigation and generator paths were updated separately. Temporary pre-edit text backups are recorded in that inventory; source and export preservation does not depend on keeping `/tmp`.

The former `D_current_pipeline_latest_run.md` is now a dated recorded pipeline report. Its October 13 example is preserved; the maintained pipeline guide explains subsequent work. Dataset preparation reviews and model-output directories were not moved.
