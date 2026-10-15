#!/usr/bin/env python3
"""Count ALL PDF text, including title, contents, tables, folios and references."""
import hashlib
import re
import subprocess
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PDF = ROOT / 'COMPETITION_MANUSCRIPT_ASSEMBLED.pdf'


def extract(*options):
    return subprocess.check_output(
        ['pdftotext', '-enc', 'UTF-8', *options, str(PDF), '-']
    ).decode('utf-8')


def normalized(text, form='NFKC'):
    # Layout whitespace is not an extra visible character per line/page.
    # One space separates words/lines/pages; preserve all non-whitespace glyphs.
    return re.sub(r'\s+', ' ', unicodedata.normalize(form, text)).strip()


raw = extract()
layout = extract('-layout')
primary = normalized(raw)
counts = {
    'PRIMARY: reading order, NFKC, whitespace collapsed': len(primary),
    'Alternative: reading order, NFC (ligatures remain single code points)': len(normalized(raw, 'NFC')),
    'Alternative: layout order, NFKC, whitespace collapsed': len(normalized(layout)),
    'Alternative: layout order, NFC, whitespace collapsed': len(normalized(layout, 'NFC')),
    'Raw extraction code points (includes line/page breaks; not visible-text count)': len(raw),
    'Alternative: NFKC with literal hyphen + linebreak removed between letters': len(normalized(re.sub(r'(?<=\w)-\r?\n(?=\w)', '', raw))),
}
version = subprocess.run(['pdftotext', '-v'], capture_output=True, text=True).stderr.splitlines()[0]
lines = [
    f'Количество символов с пробелами: {len(primary)}',
    f'Лимит 60000 соблюдён: {"да" if len(primary) <= 60000 else "нет"}',
    f'Превышение: {max(0, len(primary)-60000)}; запас: {max(0, 60000-len(primary))}',
    '',
    'PDF: ' + PDF.name,
    'SHA256: ' + hashlib.sha256(PDF.read_bytes()).hexdigest(),
    version,
    'Reproduce: python3 text_tex/editorial/count_pdf_characters.py',
    'Extraction: pdftotext -enc UTF-8 COMPETITION_MANUSCRIPT_ASSEMBLED.pdf -',
    'Alternative extraction adds -layout. No pages or text regions are excluded.',
    'Count: len(re.sub(r"\\s+", " ", unicodedata.normalize("NFKC", extracted)).strip()).',
    'Unit: Unicode code points including spaces. NFKC expands fi/fl ligatures.',
    'All runs of spaces/tabs/newlines/form feeds become one space; trim edges.',
    'Includes title, TOC (including visible leader dots), abstract, prose, tables,',
    'captions if any, AI declaration, references, visible page numbers and punctuation.',
    'Includes embedded figure titles, axis labels, tick labels, legends and sample sizes.',
    'No dehyphenation in primary count; alternate removes only literal hyphen at line end.',
    'Raw linebreaks/indentation are layout controls, so raw count is reported separately.',
    'These are reproducible PDF extraction counts, not a guarantee of a contest portal counter.',
    '',
    *[f'{label}: {value}' for label, value in counts.items()],
    '',
    'Per physical PDF page (NFKC, whitespace collapsed; total adds one space between pages):',
]
pages = raw.split('\f')
if not pages[-1].strip():
    pages.pop()
lines.extend(f'{i}: {len(normalized(page))}' for i, page in enumerate(pages, 1))
(ROOT / 'CHARACTER_COUNT.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
(ROOT / 'PDF_TEXT_EXTRACTED.txt').write_text(raw, encoding='utf-8')
print('\n'.join(lines[:3]))
