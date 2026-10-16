"""Read one contiguous Markdown ledger table without hiding detached rows."""

import re


def parse_timeline_table(section):
    lines = section.splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().startswith('|')), None)
    if start is None:
        raise ValueError('Missing timeline table')
    end = start
    while end < len(lines) and lines[end].strip().startswith('|'):
        end += 1
    if any(line.strip().startswith('|') for line in lines[end:]):
        raise ValueError('Detached timeline rows: remove the blank line or text inside the table')
    cells = [[cell.strip() for cell in line.strip().split('|')[1:-1]]
             for line in lines[start:end]]
    if len(cells) < 2:
        raise ValueError('Missing timeline table separator')
    headers, separator, *rows = cells
    if len(headers) != 12 or len(separator) != 12 or any(len(row) != 12 for row in rows):
        raise ValueError('Unexpected timeline columns; expected 12')
    if not all(re.fullmatch(r':?-+:?', cell) for cell in separator):
        raise ValueError('Expected a Markdown table separator')
    return headers, rows
