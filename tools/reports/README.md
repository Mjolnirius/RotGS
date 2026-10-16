# Report navigation and maintenance

From the repository root:

```bash
python3 tools/reports/manage_reports.py --build --check
python3 tools/reports/build_run_log_table.py
python3 tools/reports/format_implementation_report.py
```

`manage_reports.py` builds navigation from the shared research `experiments/studies.json` and `report_catalog.json` and checks local references and documented run coverage across method output roots. Use `--check` alone for read-only validation. Set `SCANNING_RESEARCH_ROOT` or pass `--research-root` to select another source directory. See [the reporting workflow](/shared/datasets/3D_Scanning_MT/research/reporting.md) and [tool commands](../../docs/reporting.md).

`build_splatting_comparison.py` builds the E10 report from R082/R089 matched saved images and the R058 historical reference, including Gaussian diagnostics and orbit previews. It reads trained checkpoints without invoking scene initialization or editing datasets.

The formatter rebuilds the implementation HTML from its canonical Markdown source. The W&B export and final scientific figure generator remain separate commands. Navigation maintenance never invokes them or changes scientific evidence. Research sources and generated reports default to `/shared/datasets/3D_Scanning_MT/research`; trained RotGS outputs remain under `output_RotGS`.
