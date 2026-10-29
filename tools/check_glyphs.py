"""Verifica que ninguna figura de un notebook tenga glifos faltantes (cuadrados en vez de símbolos).

Uso: .venv/bin/python tools/check_glyphs.py notebooks/NB-03_numpy_algebra_lineal.ipynb
Re-ejecuta una copia en memoria con las advertencias activadas (los notebooks las silencian a propósito).
"""
import sys
import nbformat
from nbclient import NotebookClient

path = sys.argv[1]
nb = nbformat.read(path, 4)
for c in nb.cells:
    if c.cell_type == "code":
        c.outputs = []
        c.source = c.source.replace('warnings.filterwarnings("ignore")', 'warnings.filterwarnings("default")')
NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": "notebooks"}}).execute()

missing = set()
for c in nb.cells:
    for o in c.get("outputs", []) if c.cell_type == "code" else []:
        if o.output_type == "stream" and "Glyph" in o.get("text", ""):
            missing.update(l.strip() for l in o["text"].splitlines() if "Glyph" in l)
print("Glifos faltantes:" if missing else "OK: sin glifos faltantes")
for m in sorted(missing):
    print("  ", m[:150])
