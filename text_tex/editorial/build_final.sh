#!/bin/sh
# Tectonic 0.16.9 (bundle v33) + Biber 2.17; biber must be on PATH.
set -eu
cd "$(dirname "$0")"
test -r 'Bib additions.bib'
tectonic -k --keep-logs COMPETITION_MANUSCRIPT_ASSEMBLED.tex
python3 count_pdf_characters.py
