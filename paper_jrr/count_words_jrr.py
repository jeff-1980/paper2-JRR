"""Approximate word counts of the JRR manuscript source, macros expanded. Math counts as one word per $...$ group.
Separate counts: abstract, body text per section (excluding captions, tables, footnote-style declarations), captions, declarations, references."""
import re, os, json
H = os.path.dirname(os.path.abspath(__file__))
macros = {}
for f in ['numbers.tex', 'numbers_jrr.tex']:
    for m in re.finditer(r'\\newcommand\{\\(\w+)\}\{(.*)\}\s*$', open(os.path.join(H, 'tables', f)).read(), re.M): macros[m.group(1)] = m.group(2)
def expand(t):
    for _ in range(3):
        t = re.sub(r'\\(' + '|'.join(sorted(macros, key=len, reverse=True)) + r')(?![A-Za-z])\s*(\{\})?', lambda m: macros[m.group(1)], t)
    return t
def words(t):
    t = expand(t)
    t = re.sub(r'\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}', '', t, flags=re.S)
    t = re.sub(r'\\cite\{[^}]*\}', '', t); t = re.sub(r'\\label\{[^}]*\}', '', t); t = re.sub(r'\\ref\{[^}]*\}', '1', t); t = re.sub(r'~\\ref\{[^}]*\}', ' 1', t)
    t = re.sub(r'\$[^$]*\$', ' M ', t)
    t = re.sub(r'\\(sub)*section\*?\{[^}]*\}', '', t)
    t = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', t); t = re.sub(r'[{}\\]', ' ', t); t = t.replace('--', ' ').replace('---', ' ')
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-%'’/]*", t))
out = {}
out['abstract'] = words(open(os.path.join(H, 'sections', 'abstract.tex')).read())
body = {}
for i, nm in enumerate(['1 Introduction', '2 Evaluation and maintenance-decision framework', '3 Interval reliability under sensor degradation', '4 Maintenance-trigger errors and cost sensitivity', '5 Discussion and limitations', '6 Conclusion'], 1):
    body[nm] = words(open(os.path.join(H, 'sections', f'sec{i}.tex')).read())
out['sections'] = body; out['body_total'] = sum(body.values())
caps = 0; ncap = 0
for i in range(1, 7):
    t = open(os.path.join(H, 'sections', f'sec{i}.tex')).read()
    for m in re.finditer(r'\\caption\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', t):
        caps += words(m.group(1)); ncap += 1
out['captions_words'] = caps; out['n_captions'] = ncap
out['declarations'] = words(open(os.path.join(H, 'sections', 'declarations.tex')).read())
refs = open(os.path.join(H, 'references_jrr.tex')).read(); out['references_words'] = words(re.sub(r'\\bibitem\{[^}]*\}', '', refs)); out['n_references'] = refs.count('\\bibitem')
t = ''.join(open(os.path.join(H, 'sections', f'sec{i}.tex')).read() for i in range(1, 7))
out['n_figures'] = len(re.findall(r'\\begin\{figure', t)); out['n_tables'] = len(re.findall(r'\\begin\{table', t))
out['abstract_plus_body'] = out['abstract'] + out['body_total']
out['abstract_body_captions'] = out['abstract'] + out['body_total'] + caps
json.dump(out, open(os.path.join(H, 'word_counts_jrr.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
