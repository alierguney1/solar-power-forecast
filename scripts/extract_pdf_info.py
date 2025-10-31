from PyPDF2 import PdfReader
from pathlib import Path
import re

pdf_path = Path('/workspaces/solar-power-forecast/paper.pdf')
reader = PdfReader(str(pdf_path))
text = ''
for page in reader.pages:
    try:
        text += page.extract_text() + '\n'
    except Exception:
        pass

print('PDF length (chars):', len(text))

patterns = [
    r'(?:station|plant|farm)\s*(\d)\D{0,20}(?:capacity|power)\D{0,10}(\d+(?:[\.,]\d+)?)\s*(?:MW|mw|megawatt)'
]

found = []
for pat in patterns:
    for m in re.finditer(pat, text, flags=re.IGNORECASE):
        station = m.group(1)
        cap = m.group(2).replace(',', '.')
        try:
            found.append((int(station), float(cap)))
        except Exception:
            pass

print('Detected capacities:', sorted(found))

for kw in ['capacity', 'panel', 'module', 'PV', 'Power (MW)', 'irradiance', 'CS6U-325P', 'HR-260P-18']:
    print(f"\n--- Context for '{kw}' ---")
    for line in text.splitlines():
        if kw.lower() in line.lower():
            print(line[:200])
