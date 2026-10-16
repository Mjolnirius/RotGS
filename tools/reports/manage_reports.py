#!/usr/bin/env python3
"""Build report navigation and check links/run coverage without touching evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.reports.research_paths import DATA_ROOT, RESEARCH_ROOT

DOCS = RESEARCH_ROOT


def expand_runs(specifications):
    result = []
    for value in specifications:
        match = re.fullmatch(r"R(\d{3})(?:-R(\d{3}))?", value)
        if not match:
            raise ValueError(f"Invalid run ID/range: {value}")
        start, end = int(match[1]), int(match[2] or match[1])
        if start > end:
            raise ValueError(f"Reversed run range: {value}")
        result.extend(f"R{number:03d}" for number in range(start, end + 1))
    return result


def build(studies, catalog):
    lines = [
        "# Experiment objectives", "",
        "<!-- Generated from studies.json by tools/reports/manage_reports.py. -->",
        "[Documentation home](../README.md) · "
        f"[Run ledger]({DOCS}/RUN_LOG.md) · "
        f"[Visual catalogue]({DOCS}/reports/index.html)", "",
        f"Reviewed {studies['reviewed']} (Europe/Zurich). Objective labels were reconstructed "
        "from the surviving history and are editable in `studies.json` and the study documents.", "",
        "A study describes a question. A session describes acquisition context. Each documented "
        "R ID has one primary study here and can be reused as a reference in other studies. "
        "Keep baseline, failure, retry, and continuation roles in the detailed run ledger.", "",
        "| Study | Objective | Primary runs | Reused references | State at review |",
        "|---|---|---|---|---|",
    ]
    for s in studies["studies"]:
        primary = ", ".join(s["primary_runs"]) or "IDs pending"
        references = ", ".join(s["reference_runs"]) or "—"
        lines.append(f"| [{s['id']} — {s['title']}]({s['document']}) | {s['question']} | "
                     f"{primary} | {references} | {s['status']} |")
    lines += ["", "## What each study established", ""]
    for s in studies["studies"]:
        lines.append(f"- **[{s['id']}]({s['document']}):** {s['finding']}")
    lines += ["", "## How the objectives developed", "",
              "The first reconstruction (E01) led to more detail (E02), multiple elevations "
              "(E03), and product coverage (E04). Product isolation (E05) and measured-motion "
              "comparisons (E06) motivated the axis investigation (E07). The R081 audit and "
              "growth experiment (E08) revisit optimization limits. The current queue transfers "
              "the selected R082 setup to other products (E09). These are research threads; "
              "their runs and references can overlap in time.", "",
              "Future appearance/SH correctness experiments remain proposals until they receive "
              "a registered study and actual run records. Historical results retain their source "
              "revision and metric protocol.", "",
              "Use [the study template](_template.md) and [reporting workflow](../reporting.md) "
              "for new work. Edit objective metadata in [studies.json](studies.json), then rebuild navigation."]
    (DOCS / "experiments/README.md").write_text("\n".join(lines) + "\n")

    def link(label, path):
        return f'<a href="{html.escape(str(path), quote=True)}">{html.escape(label)}</a>'

    study_rows = []
    for s in studies["studies"]:
        doc = DOCS / "experiments" / s["document"]
        study_rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in [
            link(s["id"] + " — " + s["title"], doc), html.escape(s["question"]),
            html.escape(", ".join(s["primary_runs"]) or "IDs pending"), html.escape(s["status"]),
        ]) + "</tr>")
    report_rows = []
    for r in catalog["reports"]:
        report_rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in [
            link(r["title"], r["path"]), html.escape(r["purpose"]),
            html.escape(", ".join(r.get("studies", [])) or "Shared"), html.escape(r["scope"]),
        ]) + "</tr>")
    data = DATA_ROOT / "cleaned"
    reviews = sorted(data.glob("**/*_review/index.html"))
    review_rows = []
    for p in reviews:
        if p.parent.name not in {"mask_review", "crop_review"}:
            continue
        relative = p.relative_to(data)
        archived = "Outdated_Preprocessing_Files" in p.parts
        review_rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in [
            link(str(relative.parent.parent), p), html.escape(p.parent.name),
            "Archived preparation" if archived else "Current location",
        ]) + "</tr>")
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>3D scanning research and experiment objectives</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#182331;background:#fafbfd}}
a{{color:#125b96}}table{{border-collapse:collapse;width:100%;margin:1rem 0 2rem}}th,td{{text-align:left;vertical-align:top;padding:.65rem;border-bottom:1px solid #d9e0e8}}th{{background:#eaf0f7}}.table{{overflow:auto}}nav a{{margin-right:1rem}}code{{background:#eaf0f7;padding:.1rem .3rem}}</style></head>
<body><h1>3D scanning research and experiment objectives</h1>
<p>Navigation reviewed {html.escape(studies['reviewed'])}, Europe/Zurich. Scientific reports retain their own dates and scope.</p>
<nav>{link('Research home', DOCS/'README.md')}{link('Objective overview', DOCS/'experiments/README.md')}{link('Run ledger', DOCS/'RUN_LOG.md')}{link('Sortable top-can timeline', DOCS/'RUN_LOG.top_30d.html')}</nav>
<h2>Experiment objectives</h2><div class="table"><table><thead><tr><th>Study</th><th>Question</th><th>Primary run entries</th><th>State at review</th></tr></thead><tbody>{''.join(study_rows)}</tbody></table></div>
<h2>Reports and evidence</h2><p>Each report lists its actual coverage. A historical comparison and a proposed follow-up have different evidence status.</p>
<div class="table"><table><thead><tr><th>Report</th><th>Purpose</th><th>Studies</th><th>Coverage / date</th></tr></thead><tbody>{''.join(report_rows)}</tbody></table></div>
<h2>Dataset preparation reviews</h2><p>Discovered from the cleaned data tree. Reviews stay beside their datasets; archived preparations are labelled.</p>
<div class="table"><table><thead><tr><th>Prepared dataset</th><th>Review</th><th>Location</th></tr></thead><tbody>{''.join(review_rows)}</tbody></table></div>
<p>Maintain navigation with <code>python3 tools/reports/manage_reports.py --build --check</code>. See {link('reporting workflow', DOCS/'reporting.md')}.</p></body></html>
'''
    output = DOCS / "reports/index.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    print(f"Built objective overview and {output} ({len(review_rows)} dataset reviews).")


class HTMLLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(value for name, value in attrs if name in {"href", "src", "poster"} and value)


def local_target(source, value):
    value = value.strip().strip("<>")
    url = urlsplit(value)
    if url.scheme or url.netloc or not url.path or "{" in url.path:
        return None
    # IDE-style file citations may include a line number.
    path = re.sub(r":\d+(?:-\d+)?$", "", unquote(url.path))
    return source.parent / path


def links_in(path):
    text = path.read_text()
    if path.suffix == ".md":
        # Code examples are not document references.
        text = re.sub(r"^\s*(`{3,}|~{3,}).*?^\s*\1\s*$", "", text, flags=re.M | re.S)
        text = re.sub(r"`[^`\n]+`", "", text)
        values = re.findall(r"!?\[[^\]\n]*\]\((<[^>\n]+>|[^)\n]+)\)", text)
        values += re.findall(r"^\[[^\]\n]+\]:\s*(\S+)", text, flags=re.M)
    else:
        values = []
    parser = HTMLLinks()
    parser.feed(text)
    return values + parser.links


def check(studies, catalog):
    errors = []
    output = Path(catalog["output_root"])
    files = {p for p in DOCS.rglob("*.md") if not p.is_symlink()
             and not p.is_relative_to(DOCS/"archive")}
    files.add(DOCS/"RUN_LOG.md")
    files.update(p for p in (DOCS/"reports").rglob("*.html")
                 if not any(part in p.parts for part in ("evidence", "full_export", "chat_analysis_bundle")))
    files.update(p for p in (DATA_ROOT/"cleaned").glob("**/*_review/index.html")
                 if p.parent.name in {"mask_review", "crop_review"})
    references = 0
    for p in sorted(files):
        for value in links_in(p):
            target = local_target(p, value)
            if target is not None:
                references += 1
                if not target.exists():
                    errors.append(f"Broken link in {p}: {value}")
    for r in catalog["reports"]:
        if not Path(r["path"]).exists():
            errors.append(f"Missing catalogue report: {r['path']}")
    ledger = (DOCS/"RUN_LOG.md").read_text()
    section = ledger.split('<div class="rotgs-run-timeline">', 1)[1].split('</div>', 1)[0]
    documented = set(re.findall(r"^\| [~*]*(R\d{3})[~*]* \|", section, flags=re.M))
    assignments = Counter()
    identifiers = []
    for s in studies["studies"]:
        identifiers.append(s["id"])
        if not (DOCS/"experiments"/s["document"]).is_file():
            errors.append(f"Missing study document: {s['document']}")
        primary = expand_runs(s["primary_runs"])
        reused = expand_runs(s["reference_runs"])
        assignments.update(primary)
        for rid in set(primary + reused) - documented:
            errors.append(f"{s['id']} references undocumented {rid}")
    if len(identifiers) != len(set(identifiers)):
        errors.append("Duplicate study IDs")
    for r in catalog["reports"]:
        for identifier in set(r.get("studies", [])) - set(identifiers):
            errors.append(f"Unknown study {identifier} in {r['title']}")
    for rid in sorted(documented):
        if assignments[rid] != 1:
            errors.append(f"{rid} has {assignments[rid]} primary studies; expected one")
    for pair in catalog["relocations"]:
        old, new = Path(pair["old"]), Path(pair["new"])
        if not old.is_symlink() or old.resolve() != new.resolve() or not new.exists():
            errors.append(f"Invalid compatibility link: {old} -> {new}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"Checked {len(files)} documents/HTML pages and {references} local references; "
          f"all {len(documented)} documented run IDs have one primary objective.")
    roots = catalog.get("output_roots", {"RotGS": str(output)})
    unregistered = [p.parent.name for root in roots.values()
                    for p in Path(root).glob("**/cfg_args") if p.parent.name not in ledger]
    if unregistered:
        print("Output folders absent from the ledger snapshot (register actual attempts from evidence):")
        for name in sorted(unregistered):
            print("  " + name)
    return 0


def main():
    global DOCS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="rebuild objective overview and HTML catalogue")
    parser.add_argument("--check", action="store_true", help="read-only local-reference and objective-coverage checks")
    parser.add_argument("--research-root", type=Path, default=RESEARCH_ROOT,
                        help="shared research source directory (or SCANNING_RESEARCH_ROOT)")
    args = parser.parse_args()
    DOCS = args.research_root.expanduser().resolve()
    studies = json.loads((DOCS/"experiments/studies.json").read_text())
    catalog = json.loads((DOCS/"report_catalog.json").read_text())
    if args.build:
        build(studies, catalog)
    return check(studies, catalog) if args.check or not args.build else 0


if __name__ == "__main__":
    raise SystemExit(main())
