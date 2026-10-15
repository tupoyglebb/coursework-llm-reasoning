#!/usr/bin/env python3
"""Lossless panel selection from approved PDFs; never estimates or redraws data.

Requires pypdf and Poppler pdftocairo. Cairo omits clipped text/graphics from
the exported PDF so invisible source panels cannot inflate PDF text counts.
The three unmodified source PDFs and their SHA256 are retained in figures/.
"""
from pathlib import Path
import subprocess
import tempfile
from pypdf import PdfReader, PdfWriter

ROOT = Path(__file__).resolve().parent / 'figures'
jobs = [
    ('05_balanced_stimulus_levels.pdf', '05_bolsen_panel.pdf', (0, 22, 394, 291.679375)),
    ('05_balanced_stimulus_levels.pdf', '05_corbacho_panel.pdf', (394, 22, 788.76, 291.679375)),
    ('04_corbacho_distributions.pdf', '04_corbacho_N0_A.pdf', (0, 22, 441, 506.599375)),
    ('04_corbacho_distributions.pdf', '04_corbacho_C_Nminus.pdf', (441, 22, 865, 506.599375)),
    ('04_corbacho_distributions.pdf', '04_corbacho_Nplus.pdf', (865, 22, 1074.978125, 506.599375)),
]
with tempfile.TemporaryDirectory() as work:
    for source, output, box in jobs:
        page = PdfReader(ROOT / source).pages[0]
        page.cropbox.lower_left = box[:2]
        page.cropbox.upper_right = box[2:]
        writer = PdfWriter()
        writer.add_page(page)
        cropped = Path(work) / output
        writer.write(cropped)
        subprocess.run(['pdftocairo', '-pdf', str(cropped), str(ROOT / output)], check=True)
