"""Kit mínimo para construir y ejecutar notebooks según CLAUDE.md.

Uso: desde la raíz del proyecto, `.venv/bin/python tools/build_XX.py`
"""
import sys
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbclient import NotebookClient

NB_DIR = "/Users/andres/Desktop/ML-DataAnalisis/notebooks"
DATE = "2026-10-29"


class NB:
    def __init__(self, num: str, title: str):
        self.num, self.title = num, title
        self.cells = []
        # Celda de cabecera exigida por CLAUDE.md (instrucción 6)
        self.code(f"# NB-{num}: {title}\n# Generado para: Andrés Hormazábal — UDP\n"
                  f"# Parte del sistema de 25 notebooks de estudio ML/IA\n# Fecha de generación: {DATE}")

    def md(self, s):
        self.cells.append(new_markdown_cell(s.strip("\n")))

    def code(self, s, hidden=False):
        c = new_code_cell(s.strip("\n"))
        if hidden:  # celda colapsada: para soluciones de "Pruébalo tú mismo"
            c.metadata["jupyter"] = {"source_hidden": True}
            c.metadata["tags"] = ["solution"]
        self.cells.append(c)

    def note(self, s):
        lines = s.strip("\n").split("\n")
        self.md("> **NOTA:** " + lines[0] + "".join("\n> " + l for l in lines[1:]))

    def tryit(self, items):
        body = "\n\n".join(items)
        self.md("### 🔧 Pruébalo tú mismo\n\n" + body +
                "\n\n*(Las soluciones están al final de esta sección, en celdas colapsadas.)*")

    def solutions(self, title, code):
        self.md(f"#### ✅ Soluciones — {title}\n*Intenta primero por tu cuenta; expande la celda de abajo para comparar.*")
        self.code(code, hidden=True)

    def build(self, filename: str, execute: bool = True):
        nb = new_notebook(cells=self.cells)
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nb.metadata["language_info"] = {"name": "python"}
        path = f"{NB_DIR}/{filename}"
        if execute:
            client = NotebookClient(nb, timeout=900, kernel_name="python3",
                                    resources={"metadata": {"path": NB_DIR}}, allow_errors=False)
            try:
                client.execute()
            except Exception as e:
                # guardamos lo ejecutado hasta el fallo para depurar
                nbf.write(nb, path.replace(".ipynb", "_FAILED.ipynb"))
                print("FALLO:", str(e)[-2500:])
                sys.exit(1)
        nbf.write(nb, path)
        n_code = sum(c.cell_type == "code" for c in nb.cells)
        print(f"OK {filename}: {len(nb.cells)} celdas ({n_code} de código), ejecutado={execute}")
