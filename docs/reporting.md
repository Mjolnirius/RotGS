# Reporting tools

Research sources now live in `/shared/datasets/3D_Scanning_MT/research`. Follow the shared [reporting workflow](/shared/datasets/3D_Scanning_MT/research/reporting.md) to register runs, studies, and evidence.

From the RotGS repository:

```bash
python3 tools/reports/manage_reports.py --build --check
python3 tools/reports/build_run_log_table.py
```

`SCANNING_DATA_ROOT` selects the shared dataset workspace. `SCANNING_RESEARCH_ROOT` selects another research directory; the default is `SCANNING_DATA_ROOT/research`. The navigation tool also accepts `--research-root`.

Regenerate the E10 comparison with the existing CUDA environment:

```bash
.venv/bin/python tools/reports/build_splatting_comparison.py
```

For changes to the page or chart layout, add `--refresh-layout` to reuse the saved evaluation and videos.

The builder evaluates saved held-out RGB pairs, draws Gaussian diagnostics, and renders orbit previews. It records source hashes and settings in the report manifest. The imported 3DGS model and its original metric files remain intact; preview renders have a separate renderer label.

Orbit reports default to `research/reports/inspection/SESSION/orbits` for the shared RotGS output root. `--report-dir` selects an explicit destination. Cross-session and cross-method comparisons should use an experiment report directory.
