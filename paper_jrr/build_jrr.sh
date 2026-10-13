#!/bin/bash
# Build all JRR PDFs. Requires: source ../../texenv.sh (local TeX wrapper) or a normal TeX Live.
set -e
cd "$(dirname "$0")"
python make_refs_jrr.py >/dev/null
for t in main_named main_anon titlepage cover_letter supplementary_named supplementary_anon; do
  [ -f $t.tex ] || continue
  pdflatex -interaction=nonstopmode -halt-on-error $t.tex >$t.build.log 2>&1 || { echo "FAILED $t"; tail -30 $t.build.log; exit 1; }
  pdflatex -interaction=nonstopmode -halt-on-error $t.tex >$t.build.log 2>&1
  pdflatex -interaction=nonstopmode -halt-on-error $t.tex >$t.build.log 2>&1
  echo "built $t.pdf  undefined-refs: $(grep -c 'undefined' $t.build.log)"
done
