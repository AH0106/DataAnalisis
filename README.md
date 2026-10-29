# ML-DataAnalisis — Notebooks de estudio ML / IA

Sistema de 25 notebooks de estudio (en español) para Ingeniería Civil en Informática y Telecomunicaciones, UDP.
Cubre IA clásica (AIMA), ML práctico (Géron), deep learning, producción y una línea de tesis sobre SNN + diagrama de ojo + predicción de BER.
El plan completo está en [CLAUDE.md](CLAUDE.md).

## Estado

| NB | Tema | Estado |
|----|------|--------|
| 01 | Python Data Science Stack | hecho (pendiente restilizar) |
| 02 | Pandas + MongoDB | hecho |
| 03 | NumPy avanzado y álgebra lineal | hecho |
| 04 | Búsqueda en espacios de estados | hecho |
| 05–25 | Resto del plan | pendiente |

`notebooks/Chapter-*.ipynb` son los notebooks base de *Hands-On ML* (Géron, 3ª ed.), que se irán reestructurando como NB-09 a NB-15.

## Uso

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/
```

NB-02 usa MongoDB real si hay un servidor en `localhost:27017`; si no, cae a `mongomock`.

## Estructura

```
notebooks/   notebooks generados (.ipynb, con salidas ya ejecutadas)
tools/       scripts que construyen y ejecutan los notebooks (nbkit.py, build_XX.py)
libros/      PDFs de referencia — local, no se versiona
```

Regenerar un notebook: `python tools/build_04.py` (desde la raíz).
