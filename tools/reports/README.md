# Report navigation and maintenance

From the repository root:

```bash
python3 tools/reports/manage_reports.py --build --check
python3 tools/reports/build_run_log_table.py
python3 tools/reports/format_implementation_report.py
```

`manage_reports.py` builds navigation from `docs/experiments/studies.json` and `docs/report_catalog.json` and checks local references and documented run coverage. Use `--check` alone for read-only validation. See [the reporting workflow](../../docs/reporting.md).

The formatter rebuilds the implementation HTML from its canonical Markdown source. The W&B export and final scientific figure generator remain separate commands. Navigation maintenance never invokes them or changes scientific evidence. The default shared output root is `/shared/datasets/3D_Scanning_MT/output_RotGS`.
