# Reporting tools

Research sources now live in `/shared/datasets/3D_Scanning_MT/research`. Follow the shared [reporting workflow](/shared/datasets/3D_Scanning_MT/research/reporting.md) to register runs, studies, and evidence.

From the RotGS repository:

```bash
python3 tools/reports/manage_reports.py --build --check
python3 tools/reports/build_run_log_table.py
```

`SCANNING_DATA_ROOT` selects the shared dataset workspace. `SCANNING_RESEARCH_ROOT` selects another research directory; the default is `SCANNING_DATA_ROOT/research`. The navigation tool also accepts `--research-root`.

Orbit reports default to `research/reports/inspection/SESSION/orbits` for the shared RotGS output root. `--report-dir` selects an explicit destination. Cross-session and cross-method comparisons should use an experiment report directory.
