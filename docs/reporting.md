# Keeping reports organized

[Documentation home](README.md) · [Study overview](experiments/README.md)

## Before a run

1. Give the question a stable `EXX` study ID or reuse an existing study. Copy [the template](experiments/_template.md) and register it in [studies.json](experiments/studies.json).
2. Record the baseline/control, intended change, fixed settings, comparison protocol, and success criteria. A reused control belongs in `reference_runs`.
3. Give the actual attempt a stable R ID in the run ledger. Keep existing IDs stable. Record the study ID in the entry, along with parent/attempt relationships and your notes. Launch evidence, rather than folder existence, establishes a run.

## After a run or analysis

1. Update the run ledger from saved evidence, preserving `Your comments`.
2. Put new report bundles in `output_RotGS/reports/experiments/EXX_topic/`; use `reports/evaluations/YYYY-MM-DD_topic/` for a dated cross-study evaluation and `reports/inspection/session_NN/` for shared galleries.
3. Keep one editable narrative source. Put maintained conclusions under `docs/`; identify generated HTML/PDF/ZIP as exports. Put figures, raw exports, copied sources, metrics, and manifests in the evidence bundle. Historical analysis-specific builders remain with their evidence; reusable tools belong in `tools/reports/`.
4. Update the study's finding, decision, limits, and follow-up. Label older decisions as superseded and link forward. Keep original evidence and raw conversations intact.
5. Register new report entry points in [report_catalog.json](report_catalog.json), then rebuild and check navigation:

```bash
python3 tools/reports/manage_reports.py --build --check
```

The command uses only the standard library. It rebuilds the objective overview and offline report catalogue, checks their local links, checks the registered study documents and migrated HTML pages, verifies run-ID coverage against the main timeline, and discovers current and archived crop/mask reviews. It does not train, render models, evaluate checkpoints, or refresh scientific conclusions.

## Destinations and generators

- The run ledger remains `output_RotGS/RUN_LOG.md`; rebuild its sortable view with `python3 tools/reports/build_run_log_table.py`.
- Rebuild the implementation HTML from its versioned narrative with `python3 tools/reports/format_implementation_report.py`.
- Orbit generation defaults to `OUTPUT_ROOT/reports/inspection/SESSION/orbits`. `--report-dir` remains available for another registered gallery. Pose export takes an explicit `--output-dir`; use the same session inspection directory.
- The corrected Session 03 evaluator targets `reports/experiments/E04_product_baseline/wandb_analysis`.
- `tools/reports/export_wandb_analysis.py` and `tools/reports/generate_training_report.py` retain their scientific roles. They are separate from rebuilding navigation; consult the [analysis bundle](/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E04_product_baseline/wandb_analysis/README.md).

Use relative links within a bundle and the mounted absolute paths between the repository and dataset workspaces. Keep original manifests, copied source snapshots, and export archives unchanged. The [relocation map](relocations.md) explains compatibility links and the migration inventory.

Dataset reviews remain beside datasets, including archived preparations. HTML interfaces in `tools/preprocessing/` stay with the tools that load them.
