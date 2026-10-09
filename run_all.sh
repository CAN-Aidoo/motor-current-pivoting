#!/usr/bin/env bash
# Reproduce every result and the paper from scratch (about 40 min on 2 CPU cores).
set -euo pipefail
for a in 0.075 0.15 0.3; do python exp_a.py "$a" 100; done
python analyze_ab.py 0.15
python figs_ab.py
python exp_c.py 0 2 40 & python exp_c.py 1 2 40 & wait
python analyze_c.py
cd paper && python gen_numbers.py && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "Done: paper/main.pdf"
