"""Builds references_jrr.tex (SAGE Vancouver style, numbered by order of first citation) from refs_jrr.json and the \\cite keys in sections/*.tex."""
import json, re, os
H = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(H, 'refs_jrr.json')))['refs']
order = []
for f in ['abstract', 'sec1', 'sec2', 'sec3', 'sec4', 'sec5', 'sec6', 'declarations']:
    for m in re.finditer(r'\\cite\{([^}]*)\}', open(os.path.join(H, 'sections', f + '.tex')).read()):
        for k in m.group(1).split(','):
            k = k.strip()
            if k not in order: order.append(k)
missing = [k for k in order if k not in R]; assert not missing, missing
def esc(s): return s.replace('&', '\\&').replace('%', '\\%').replace('_', '\\_').replace('#', '\\#')
def initials(g): return ''.join(w[0] for w in re.split(r'[\s\-]+', g.replace('.', ' ')) if w)
def authors(a):
    names = []
    for fam, giv in a:
        names.append('et al.' if fam == 'et al.' else f"{fam} {initials(giv)}".strip())
    if names and names[-1] == 'et al.': return ', '.join(names[:-1]) + ', et al.'
    return ', '.join(names[:6]) + (', et al.' if len(names) > 6 else '')
lines = ['\\begin{thebibliography}{99}', '\\small', '\\setlength{\\itemsep}{0pt}']
for k in order:
    r = R[k]; au = authors(r['authors'])
    ven = r['journal']; vol = r['volume']; iss = r['issue']; pg = r['pages']
    if r['kind'] == 'preprint': tail = f"{ven}. {r['year']}; {pg}."
    else:
        tail = f"{ven} {r['year']}"
        if vol: tail += f"; {vol}"
        if iss: tail += f"({iss})"
        if pg: tail += f": {pg}"
        tail += '.'
    doi = f" DOI: {r['doi']}." if r['doi'] else ''
    lines.append(f"\\bibitem{{{k}}} {esc(au)}. {esc(r['title'].rstrip('.'))}. {esc(tail)}{esc(doi)}")
lines.append('\\end{thebibliography}')
open(os.path.join(H, 'references_jrr.tex'), 'w').write('\n'.join(lines) + '\n')
json.dump(order, open(os.path.join(H, 'refs_order.json'), 'w'))
print(len(order), 'references; first-citation order:', order)

rows = []
for i, k in enumerate(order, 1):
    r = R[k]; rows.append(f"{i} & {esc(k)} & {esc(r['verification'])} \\\\")
open(os.path.join(H, 'tables', 'jrr_supp_refcheck.tex'), 'w').write('\n'.join(rows) + '\n')
