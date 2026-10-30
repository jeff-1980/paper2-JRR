"""Build the single flattened TIMC main document from manuscript/ (preamble + numbers + body, tables inlined).
Run: python make_timc_flat.py <manuscript dir> <out .tex>"""
import os, re, sys
M, OUT = sys.argv[1], sys.argv[2]
rd = lambda f: open(os.path.join(M, f)).read()
pre = rd('preamble.tex').replace('\\begin{document}', '')
pre = pre.replace('\\documentclass[11pt,a4paper]{article}', '\\documentclass[11pt,a4paper]{article}\n\\usepackage{lineno}')
body = rd('body.tex')
body = re.sub(r'\\input\{(rev_table_[a-z]+)\.tex\}', lambda m: rd(m.group(1) + '.tex').strip(), body)
assert '\\input{' not in body
body = body.replace('\\begin{document}\n\\onehalfspacing', '\\begin{document}\n\\singlespacing', 1)
kw = body.index('\\noindent\\textbf{Keywords:}'); end = body.index('\n', kw)
body = body[:end + 1] + '\\clearpage\n\\doublespacing\n\\linenumbers\n' + body[end + 1:]
a = body.index('\\section*{Statements and Declarations}'); b = body.index('\\begin{thebibliography}')
decl = r"""\singlespacing
\section*{Acknowledgements}
Not applicable.

\section*{Author contributions}
Yi Wang is the sole author and carried out the conceptualization, methodology, software, validation, formal analysis, investigation, data curation, visualization and writing.

\section*{Statements and declarations}
\subsection*{Ethical considerations}
Not applicable. The study uses only a publicly available simulated dataset and involves no human participants or animals.
\subsection*{Consent to participate}
Not applicable.
\subsection*{Consent for publication}
Not applicable.
\subsection*{Declaration of conflicting interests}
The author declared no potential conflicts of interest with respect to the research, authorship, and/or publication of this article.
\subsection*{Funding}
The author received no financial support for the research, authorship, and/or publication of this article.
\subsection*{Data availability}
The C-MAPSS data are available from the NASA Prognostics Center of Excellence. Code, trained checkpoints, result files and the scripts that generate every number, table and figure of this paper are available at \url{https://github.com/jeff-1980/paper2-JRR} (revision analyses in \texttt{code\_rev/} and \texttt{results\_rev/}; repository commit \texttt{6ccaeeb}). The analyses are inference on released checkpoints; training is not re-run. Supplementary Material S7 maps tables and figures to scripts and result files.
\subsection*{Use of artificial intelligence}
The AI assistant Claude (Anthropic) was used to write and run the analysis code, to check numbers and cross-references against result files, and to assist with restructuring and editing the text. The study design, the choice of perturbations and decision rules, and all interpretations were fixed by the author, who reviewed all outputs and takes full responsibility for the content. No AI tool is listed as an author.

"""
body = body[:a] + decl + body[b:]
src = pre + '\n' + rd('rev_numbers.tex') + '\n' + body
src = src.replace('figures/fig', 'fig')
open(OUT, 'w').write(src)
print('ok', len(src))
