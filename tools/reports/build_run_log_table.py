#!/usr/bin/env python3
"""Build the offline sortable HTML view of RUN_LOG.md's filtered timeline.

Run again after editing the Markdown table. Only the sibling HTML is written.
"""
from pathlib import Path
import html
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.reports.research_paths import RESEARCH_ROOT
from tools.reports.ledger_tables import parse_timeline_table

ROOT = RESEARCH_ROOT
SOURCE = ROOT / 'RUN_LOG.md'
OUTPUT = ROOT / 'RUN_LOG.top_30d.html'
HEADING = '### Quoellfrisch top 30° runs'


def render_inline(text):
    pattern = r'(\[[^\]]+\]\((?:<[^>]+>|[^)]+)\)|`[^`]+`|~~.*?~~|\*\*.*?\*\*|<br\s*/?>)'
    parts = []
    for token in re.split(pattern, text):
        if not token:
            continue
        if token.startswith('~~'):
            parts.append('<del>' + render_inline(token[2:-2]) + '</del>')
        elif token.startswith('**'):
            parts.append('<strong>' + render_inline(token[2:-2]) + '</strong>')
        elif token.startswith('`'):
            parts.append('<code>' + html.escape(token[1:-1]) + '</code>')
        elif token.startswith('<br'):
            parts.append('<br>')
        elif token.startswith('['):
            link = re.fullmatch(r'\[([^\]]+)\]\((?:<([^>]+)>|([^)]+))\)', token)
            label, angled, bare = link.groups()
            target = angled or bare
            if re.match(r'(?i)\s*(javascript|data|vbscript):', target):
                raise ValueError('Unsupported link scheme')
            parts.append('<a href="' + html.escape(target, quote=True) + '">' + render_inline(label) + '</a>')
        else:
            parts.append(html.escape(token))
    return ''.join(parts)


def plain_text(text):
    text = re.sub(r'\[([^\]]+)\]\((?:<[^>]+>|[^)]+)\)', r'\1', text)
    text = re.sub(r'<br\s*/?>', ' ', text)
    return text.replace('~~', '').replace('**', '').replace('`', '').strip()


def sort_key(text, kind):
    value = plain_text(text)
    if kind == 'number':
        match = re.search(r'[+-]?\d[\d,]*(?:\.\d+)?', value)
        return match.group().replace(',', '') if match else ''
    if kind == 'date':
        match = re.search(r'\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?', value)
        return match.group() if match else ''
    return '' if value == '—' else value


def build():
    document = SOURCE.read_text()
    section = document.split(HEADING, 1)[1].split('</div>', 1)[0]
    headers, rows = parse_timeline_table(section)
    kinds = ['text', 'date', 'number', 'text', 'text', 'number', 'number', 'number', 'number', 'number', 'text', 'text']
    head = []
    for index, (label, kind) in enumerate(zip(headers, kinds)):
        safe = html.escape(label, quote=True)
        head.append(f'<th scope="col" aria-sort="none" data-kind="{kind}" data-label="{safe}"><button type="button" aria-label="Sort by {safe}, ascending">{safe}<span class="sort-indicator" aria-hidden="true">↕</span></button></th>')
    body = []
    for order, row in enumerate(rows):
        rendered = []
        for index, (value, kind) in enumerate(zip(row, kinds)):
            key = html.escape(sort_key(value, kind), quote=True)
            css = 'numeric' if kind == 'number' else 'reports' if index == 10 else ''
            rendered.append(f'<td class="{css}" data-sort-key="{key}">{render_inline(value)}</td>')
        body.append(f'<tr data-original-order="{order}">' + ''.join(rendered) + '</tr>')
    script = r'''
'use strict';
(() => {
  const table = document.getElementById('run-table');
  const body = table.querySelector('tbody');
  const headers = Array.from(table.querySelectorAll('thead th'));
  const originalRows = Array.from(body.rows);
  const status = document.getElementById('sort-status');
  const collator = new Intl.Collator('en', {numeric: true, sensitivity: 'base'});
  let activeColumn = -1;
  let ascending = true;

  function value(row, column) {
    const raw = row.cells[column].dataset.sortKey;
    if (raw === '') return null;
    return headers[column].dataset.kind === 'number' ? Number(raw) : raw;
  }

  function updateHeaders() {
    headers.forEach((header, index) => {
      const active = index === activeColumn;
      header.setAttribute('aria-sort', active ? (ascending ? 'ascending' : 'descending') : 'none');
      header.querySelector('.sort-indicator').textContent = active ? (ascending ? '▲' : '▼') : '↕';
      const next = active && ascending ? 'descending' : 'ascending';
      header.querySelector('button').setAttribute('aria-label', `Sort by ${header.dataset.label}, ${next}`);
    });
  }

  function sort(column) {
    ascending = activeColumn === column ? !ascending : true;
    activeColumn = column;
    const rows = [...originalRows].sort((a, b) => {
      const left = value(a, column);
      const right = value(b, column);
      // Missing entries stay last in either direction; ties retain timeline order.
      if (left === null && right !== null) return 1;
      if (right === null && left !== null) return -1;
      let comparison = 0;
      if (left !== null && right !== null) {
        comparison = headers[column].dataset.kind === 'number'
          ? left - right : collator.compare(left, right);
      }
      return (ascending ? comparison : -comparison)
        || Number(a.dataset.originalOrder) - Number(b.dataset.originalOrder);
    });
    body.append(...rows);
    updateHeaders();
    status.textContent = `${rows.length} runs · Sorted by ${headers[column].dataset.label}, ${ascending ? 'ascending' : 'descending'}`;
  }

  headers.forEach((header, column) => {
    header.querySelector('button').addEventListener('click', () => sort(column));
  });
  document.getElementById('reset-order').addEventListener('click', () => {
    activeColumn = -1;
    ascending = true;
    body.append(...originalRows);
    updateHeaders();
    status.textContent = `${originalRows.length} runs · Original timeline order`;
  });
})();
'''
    template = '''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quoellfrisch top 30° — sortable run timeline</title>
<style>
:root { color-scheme: light dark; --paper: #fff; --ink: #17302e; --muted: #566663; --line: #d8e2df; --accent: #17695e; --soft: #f1f6f4; }
* { box-sizing: border-box; }
body { margin: 0; padding: 28px; background: var(--paper); color: var(--ink); font: 15px/1.55 system-ui, sans-serif; }
main { max-width: 1920px; margin: auto; }
h1 { margin: 0 0 10px; font-size: clamp(24px, 3vw, 34px); }
p { margin: 0 0 16px; color: var(--muted); }
a { color: var(--accent); text-underline-offset: 3px; }
.toolbar { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; justify-content: space-between; margin: 18px 0 12px; }
button { font: inherit; color: inherit; cursor: pointer; }
#reset-order { padding: 7px 12px; background: var(--soft); border: 1px solid var(--line); border-radius: 6px; }
.table-wrap { max-height: 75vh; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 13px; }
th, td { vertical-align: top; padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; min-width: 110px; }
th { position: sticky; top: 0; z-index: 1; padding: 0; background: var(--soft); }
th button { width: 100%; min-height: 70px; padding: 11px 12px; background: transparent; border: 0; text-align: left; font-weight: 650; }
th button:hover { background: color-mix(in srgb, var(--accent) 12%, var(--soft)); }
button:focus-visible, .table-wrap:focus-visible { outline: 3px solid var(--accent); outline-offset: -3px; }
th[aria-sort="ascending"], th[aria-sort="descending"] { box-shadow: inset 0 -3px var(--accent); }
.sort-indicator { display: inline-block; margin-left: 6px; color: var(--accent); }
tbody tr:nth-child(even) { background: var(--soft); }
tbody tr:hover { background: color-mix(in srgb, var(--accent) 8%, var(--paper)); }
td.numeric { text-align: right; font-variant-numeric: tabular-nums; }
th:first-child, td:first-child { min-width: 75px; }
th:nth-child(2), td:nth-child(2) { min-width: 210px; }
th:nth-child(4), td:nth-child(4) { min-width: 300px; max-width: 360px; overflow-wrap: anywhere; }
th:last-child, td:last-child { min-width: 300px; max-width: 420px; }
.reports { min-width: 160px; }
.reports a { white-space: nowrap; }
.reports a[href*="reports/inspection/session_03/orbits/"] { color: #005fa8; }
.reports a[href*="reports/inspection/session_04/orbits/"] { color: #007767; }
.reports a[href*="gaussian_centers/"] { color: #7340ac; }
.reports a[href*="view_poses/"] { color: #985200; }
code { font: .9em ui-monospace, monospace; }
.sr-only { position: absolute; width: 1px; height: 1px; padding: 0; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }
footer { margin-top: 16px; font-size: 13px; color: var(--muted); }
@media (prefers-color-scheme: dark) {
  :root { --paper: #152321; --ink: #e4eeeb; --muted: #a7bbb5; --line: #3b5049; --accent: #76d6bd; --soft: #20332d; }
  .reports a[href*="reports/inspection/session_03/orbits/"] { color: #67b7ff; }
  .reports a[href*="reports/inspection/session_04/orbits/"] { color: #50d4bf; }
  .reports a[href*="gaussian_centers/"] { color: #c5a1ff; }
  .reports a[href*="view_poses/"] { color: #ffbd66; }
}
@media (max-width: 600px) { body { padding: 16px; } }
</style>
</head>
<body>
<main>
<h1>Quoellfrisch top 30° runs</h1>
<p>Click a column name to sort; click it again to reverse the order. Scores retain their evaluation labels and checkpoint qualifications.</p>
<div class="toolbar"><span id="sort-status" role="status" aria-live="polite">__COUNT__ runs · Original timeline order</span><button id="reset-order" type="button">Reset timeline order</button></div>
<div class="table-wrap" tabindex="0" role="region" aria-label="Scrollable Quoellfrisch run table">
<table id="run-table">
<caption class="sr-only">Quoellfrisch top 30-degree timeline. Activate column header buttons to sort ascending or descending.</caption>
<thead><tr>__HEAD__</tr></thead>
<tbody>__BODY__</tbody>
</table>
</div>
<noscript><p>Enable JavaScript to sort this table interactively.</p></noscript>
<footer>Source: <a href="RUN_LOG.md#quoellfrisch-top-30-runs">RUN_LOG.md</a>. Higher PSNR is better when the evaluation protocol and reference target match.</footer>
</main>
<script>__SCRIPT__</script>
</body>
</html>
'''
    output = template.replace('__COUNT__', str(len(rows))).replace('__HEAD__', '\n'.join(head)).replace('__BODY__', '\n'.join(body)).replace('__SCRIPT__', script)
    OUTPUT.write_text(output)
    print(f'Built {OUTPUT.name}: {len(rows)} rows, {len(headers)} sortable columns')


if __name__ == '__main__':
    build()
