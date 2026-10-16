#!/usr/bin/env python3
"""Export the local Markdown analysis to a self-contained offline HTML reader."""
from pathlib import Path
import html
import json
import os
import re
from urllib.parse import urlsplit, unquote

ROOT = Path('/shared/datasets/3D_Scanning_MT/output_RotGS/reports/experiments/E06_motion/implementation_decisions')
SOURCE = (ROOT / 'report.md').resolve()
markdown = (ROOT/'report.md').read_text()
runs = json.loads((ROOT/'evidence/runs.json').read_text())
core_order = ['kf3wv1dr','w5cs4xiu','78gbr4c6','duiwndaw','x68bbab4','pyds8296','kg1lswrh','nmj1jzpi','agabyrkn']
core = [{key:r[key] for key in ('id','label','comparisons')} for run_id in core_order
        for r in runs if r['id'] == run_id]
for r in core:
    r['label'] = 'Session 04 '+r['label']
earlier_path=ROOT/'evidence/earlier/runs.json'
if earlier_path.exists():
    earlier=json.loads(earlier_path.read_text())
    core += [{'id':r['run'],'label':'Session 03 '+r['run']+' '+r['label'],
              'comparisons':r['comparisons']} for r in earlier]

def document_url(value):
    url = urlsplit(value)
    if url.scheme or url.netloc or not url.path or value.startswith('#'):
        return value
    path = Path(unquote(url.path))
    if not path.is_absolute():
        path = SOURCE.parent / path
    if path.is_relative_to(Path('/shared/datasets/3D_Scanning_MT/output_RotGS')):
        target = os.path.relpath(path, ROOT)
    else:
        target = str(path)
    return target + ('?' + url.query if url.query else '') + ('#' + url.fragment if url.fragment else '')

def inline(value):
    placeholders = []
    def preserve(match):
        placeholders.append('<code>'+html.escape(match.group(1))+'</code>')
        return f'ZZCODE{len(placeholders)-1}ZZ'
    value = re.sub(r'`([^`]+)`',preserve,value)
    value = html.escape(value)
    value = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', lambda m: '<a href="' + html.escape(document_url(html.unescape(m[2])), quote=True) + '">' + m[1] + '</a>', value)
    value = re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',value)
    value = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<em>\1</em>',value)
    for i, replacement in enumerate(placeholders):
        value = value.replace(f'ZZCODE{i}ZZ',replacement)
    return value

def slug(text):
    return re.sub(r'[^a-z0-9]+','-',text.lower()).strip('-')

lines=markdown.splitlines()
parts=[];toc=[];i=0
while i<len(lines):
    line=lines[i]
    if not line.strip():i+=1;continue
    heading=re.match(r'^(#{1,3}) (.*)',line)
    if heading:
        level=len(heading.group(1));title=heading.group(2);anchor=slug(title)
        parts.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
        if level==2:toc.append((anchor,title))
        i+=1;continue
    if line.startswith('|'):
        table=[]
        while i<len(lines) and lines[i].startswith('|'):
            row=[c.strip() for c in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',c) for c in row):table.append(row)
            i+=1
        width=len(table[0]);rows=[]
        for index,row in enumerate(table):
            if len(row)!=width:raise ValueError('Inconsistent Markdown table columns')
            tag='th' if index==0 else 'td'
            rows.append('<tr>'+''.join(f'<{tag}>{inline(c)}</{tag}>' for c in row)+'</tr>')
        parts.append('<div class="table-wrap"><table><thead>'+rows[0]+'</thead><tbody>'+''.join(rows[1:])+'</tbody></table></div>')
        continue
    figure=re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
    if figure:
        alt,src=figure.groups()
        src=document_url(src)
        parts.append(f'<figure><a href="{html.escape(src)}"><img loading="lazy" src="{html.escape(src)}" alt="{html.escape(alt)}"></a><figcaption>Click to inspect the full figure.</figcaption></figure>')
        i+=1;continue
    if re.match(r'^\d+\. ',line):
        items=[]
        while i<len(lines) and re.match(r'^\d+\. ',lines[i]):
            items.append('<li>'+inline(re.sub(r'^\d+\. ','',lines[i]))+'</li>');i+=1
        parts.append('<ol>'+''.join(items)+'</ol>');continue
    paragraph=[line];i+=1
    while i<len(lines) and lines[i].strip():
        if lines[i].startswith(('#','|','![')) or re.match(r'^\d+\. ',lines[i]):break
        paragraph.append(lines[i]);i+=1
    text=' '.join(paragraph)
    css=' class="caption"' if text.startswith('*Figure ') else ''
    parts.append(f'<p{css}>{inline(text)}</p>')

toc_html=''.join(f'<a href="#{anchor}">{html.escape(title)}</a>' for anchor,title in toc)
options=''.join(f'<option value="{r["id"]}">{html.escape(r["label"])}</option>' for r in core)
viewer=f'''
<section class="viewer-section" id="interactive-validation-comparisons">
<h2>Interactive validation comparisons</h2>
<p>Nine Session 04 variants, including the new R081 projected wide-axis run, and five earlier Session 03 leaders. Each image contains its own ground truth, saved render, and absolute RGB error from left to right. Select two runs and the same nominal source view. Cross-session targets, framing, and lighting differ; these images are not pixel-aligned across sessions. Error values are not amplified.</p>
<div class="controls"><label>Variant A<select id="variant-a">{options}</select></label>
<label>Variant B<select id="variant-b">{options}</select></label>
<label>Held-out view<select id="view"><option value="0000">0000 nominal 0 degrees</option><option value="0008">0008 nominal 40 degrees</option><option value="0016">0016 nominal 80 degrees</option></select></label></div>
<div class="comparison-grid"><figure><a id="original-a"><img id="image-a" alt="Validation comparison for variant A"></a><figcaption id="caption-a"></figcaption></figure><figure><a id="original-b"><img id="image-b" alt="Validation comparison for variant B"></a><figcaption id="caption-b"></figcaption></figure></div>
</section>'''

css='''
:root{--ink:#17302e;--muted:#566663;--paper:#fff;--line:#d8e2df;--accent:#17695e;--warm:#a56f35;--soft:#f1f6f4}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#eef3f1;color:var(--ink);font:16px/1.7 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.topbar{padding:16px max(22px,calc((100vw - 1440px)/2));background:var(--ink);color:#fff;display:flex;gap:16px;justify-content:space-between;align-items:center;font-size:14px}.topbar a{color:#d6eee7;text-decoration:none}.topbar nav{display:flex;gap:18px;align-items:center}.topbar button{background:none;color:#d6eee7;border:1px solid #91b6ad;border-radius:6px;padding:5px 10px;cursor:pointer}
.layout{max-width:1440px;margin:0 auto;display:grid;grid-template-columns:240px minmax(0,1fr);gap:28px;padding:30px 24px 60px}aside{position:sticky;top:22px;align-self:start;max-height:calc(100vh - 44px);overflow:auto;font-size:13px}aside strong{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted)}aside a{display:block;color:var(--muted);text-decoration:none;padding:7px 10px;border-left:2px solid transparent}aside a:hover{border-color:var(--accent);background:#e1ebe6;color:var(--ink)}main{background:var(--paper);padding:38px 42px 55px;border:1px solid var(--line);border-radius:12px;min-width:0;box-shadow:0 8px 35px #17302e08}
h1{font-size:clamp(28px,3vw,42px);line-height:1.15;letter-spacing:-.025em;margin:0 0 24px;max-width:28ch}h2{font-size:25px;line-height:1.3;border-top:1px solid var(--line);padding-top:26px;margin:42px 0 18px;scroll-margin-top:22px}h3{font-size:20px;line-height:1.4;margin:30px 0 12px;scroll-margin-top:22px}p,ol{margin:14px 0;max-width:92ch}li{margin:9px 0}a{color:var(--accent);text-underline-offset:3px}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.88em;background:var(--soft);padding:1px 4px;border-radius:3px;overflow-wrap:anywhere}strong{font-weight:700}.caption{font-size:13px;line-height:1.6;color:var(--muted);margin-top:-3px}
.table-wrap{overflow:auto;margin:20px 0;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.5}th,td{padding:11px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line);min-width:90px}th{background:#eaf2ee;font-weight:650}tr:last-child td{border-bottom:none}tbody tr:nth-child(even){background:#f8faf9}figure{margin:26px 0 10px}figure img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}figcaption{font-size:12px;color:var(--muted);margin:5px 0}.quick{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:26px 0 30px}.quick article{padding:16px;border:1px solid var(--line);border-top:3px solid var(--accent);border-radius:7px;background:var(--soft)}.quick b{font-size:27px;display:block;line-height:1.3}.quick span{font-size:12px;color:var(--muted);display:block;margin-top:6px}.quick .warm{border-top-color:var(--warm)}.intro-note{font-size:12px;color:var(--muted);margin:6px 0 16px}.controls{display:grid;grid-template-columns:1fr 1fr .9fr;gap:14px;background:var(--soft);padding:16px;border-radius:8px}.controls label{font-size:12px;font-weight:650;min-width:0}select{display:block;width:100%;padding:9px;margin-top:5px;background:white;border:1px solid var(--line);border-radius:5px;font:inherit;color:var(--ink)}.comparison-grid{display:grid;grid-template-columns:1fr;gap:4px}.comparison-grid figcaption{font-size:13px}footer{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:18px;margin-top:32px}
@media(max-width:1000px){.layout{grid-template-columns:1fr;padding:18px 12px}aside{position:static;max-height:none}aside nav{display:flex;flex-wrap:wrap;gap:4px}aside a{border-left:0;padding:4px 8px;font-size:12px}main{padding:25px 20px}.quick{gap:8px}.controls{grid-template-columns:1fr}}
@media(max-width:600px){.quick{grid-template-columns:1fr}.topbar{align-items:flex-start;flex-direction:column}.topbar nav{flex-wrap:wrap}h1{font-size:31px}th,td{padding:8px}}
@media print{body{background:white;font-size:10pt}.topbar,aside,.controls,.viewer-section,.quick{display:none}.layout{display:block;padding:0;max-width:none}main{padding:0;border:none;box-shadow:none}h1{font-size:24pt}h2{font-size:17pt}h3{font-size:13pt}figure,table{break-inside:avoid}.table-wrap{overflow:visible}figure img{border:none}a{color:inherit}p{orphans:3;widows:3}}
'''
stats='''<div class="quick"><article><b>32.500 dB</b><span>R025 · independently verified full-frame PSNR</span></article><article><b>28.999 dB</b><span>R082 · external product foreground PSNR</span></article><article class="warm"><b>61.3% lower</b><span>R082 versus R078 · lower-edge ripple RMS</span></article></div><p class="intro-note">The older and newer captures have different targets and masks. Compare score domains before interpreting the gap. Completed axis ablations favor projected ±0.5°; fixed calibration is a strong control. Paired seed confirmation remains pending.</p>'''
body='\n'.join(parts)
first_heading_end=body.find('</h1>')+len('</h1>')
body=body[:first_heading_end]+stats+body[first_heading_end:]
target='<h2 id="why-tae-should-help-and-why-its-benefit-is-not-isolated">'
body=body.replace(target,viewer+target)
js='''
const runs=RUN_DATA;
const a=document.getElementById('variant-a'),b=document.getElementById('variant-b'),v=document.getElementById('view');
a.value='duiwndaw';b.value=runs.some(x=>x.id==='kg1lswrh')?'kg1lswrh':'duiwndaw';
function update(){for(const [side,select] of [['a',a],['b',b]]){const run=runs.find(x=>x.id===select.value);const entry=run.comparisons.find(x=>x.view===v.value);document.getElementById('image-'+side).src=entry.path;document.getElementById('original-'+side).href=entry.path;document.getElementById('caption-'+side).textContent=run.label+' · '+entry.caption;}}
for(const select of [a,b,v])select.addEventListener('change',update);update();
document.getElementById('print-report').addEventListener('click',()=>window.print());
'''.replace('RUN_DATA',json.dumps(core).replace('</','<\\/'))
document='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>RotGS implementation decisions and earlier top can baselines</title><style>CSS</style></head><body><header class="topbar"><span>RotGS · Sessions 03 and 04 · Evidence and implementation decisions</span><nav><a href="report.md">Editable Markdown</a><a href="run_scores.csv">Latest scores</a><a href="earlier_scores.csv">Earlier scores</a><a href="evidence/earlier/manifest.json">Earlier audit</a><button id="print-report" type="button">Print report</button></nav></header><div class="layout"><aside><strong>Report contents</strong><nav>TOC<a href="#interactive-validation-comparisons">Interactive comparisons</a></nav></aside><main>BODY<footer>Local report · 14 October 2026 · Figures and evidence retained in this folder. Original datasets, run outputs, and implementation source files were preserved.</footer></main></div><script>JS</script></body></html>'''
document=document.replace('CSS',css).replace('TOC',toc_html).replace('BODY',body).replace('JS',js)
document=document.replace('Local report · 14 October 2026','Local report · updated 14 October 2026')
(ROOT/'index.html').write_text(document)
print('Created',ROOT/'index.html')
