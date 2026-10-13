"""Automated cross-checks of the JRR manuscript build (run after build_jrr.sh)."""
import re, os, json, subprocess
H = os.path.dirname(os.path.abspath(__file__)); import shutil; PT = shutil.which('pdftotext') or 'pdftotext'
PI = shutil.which('pdfinfo') or 'pdfinfo'
res = {}
src = {f: open(os.path.join(H, 'sections', f + '.tex')).read() for f in ['abstract', 'sec1', 'sec2', 'sec3', 'sec4', 'sec5', 'sec6', 'declarations']}
alltext = '\n'.join(src.values())
cites = set(k.strip() for m in re.finditer(r'\\cite\{([^}]*)\}', alltext) for k in m.group(1).split(','))
bibs = set(re.findall(r'\\bibitem\{([^}]*)\}', open(os.path.join(H, 'references_jrr.tex')).read()))
res['cited_not_in_list'] = sorted(cites - bibs); res['listed_not_cited'] = sorted(bibs - cites); res['n_refs'] = len(bibs)
labels = set(re.findall(r'\\label\{([^}]*)\}', alltext)); refs = set(re.findall(r'\\ref\{([^}]*)\}', alltext))
res['refs_without_label'] = sorted(refs - labels); res['labels_never_referenced'] = sorted(labels - refs)
for t in ['main_named', 'main_anon', 'supplementary_named', 'supplementary_anon', 'titlepage']:
    log = open(os.path.join(H, t + '.build.log')).read()
    res[t] = {'pages': int(re.search(r'Pages:\s+(\d+)', subprocess.run([PI, os.path.join(H, t + '.pdf')], capture_output=True, text=True).stdout).group(1)),
              'undefined_refs_or_citations': len(re.findall(r'undefined', log)), 'overfull_hbox_gt_10pt': len([m for m in re.findall(r'Overfull \\hbox \(([\d.]+)pt', log) if float(m) > 10])}
txt = subprocess.run([PT, os.path.join(H, 'main_anon.pdf'), '-'], capture_output=True, text=True).stdout.lower()
res['anon_identifiers_found'] = [w for w in ['yi wang', 'lishui', 'jefflsxy', 'jeff-1980', 'github', 'orcid', 'v1.8-submission'] if w in txt]
txts = subprocess.run([PT, os.path.join(H, 'supplementary_anon.pdf'), '-'], capture_output=True, text=True).stdout.lower()
res['anon_supp_identifiers_found'] = [w for w in ['yi wang', 'lishui', 'jefflsxy', 'jeff-1980', 'github', 'orcid', 'v1.8-submission'] if w in txts]
mt = subprocess.run([PT, os.path.join(H, 'main_named.pdf'), '-'], capture_output=True, text=True).stdout
res['stray_macro_text_in_pdf'] = len(re.findall(r'\\J|\\N[A-Z]|\?\?|\bNone\b|\bnan\b', mt))
res['figures_in_main'] = len(re.findall(r'\\includegraphics', alltext)); res['tables_in_main'] = len(re.findall(r'\\begin\{table\}', alltext))
res['illustrations_total'] = res['figures_in_main'] + res['tables_in_main']
json.dump(res, open(os.path.join(H, 'consistency_check_jrr.json'), 'w'), indent=1); print(json.dumps(res, indent=1))
