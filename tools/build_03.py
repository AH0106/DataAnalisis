import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbkit import NB

nb = NB("03", "NumPy Avanzado y Álgebra Lineal para ML")
M, C, NOTE = nb.md, nb.code, nb.note

# =====================================================================
# ENCABEZADO
# =====================================================================
M(r'''
# NB-03: NumPy Avanzado y Álgebra Lineal para ML
**Bloque:** 1 — Fundamentos de Datos y Cómputo
**Base bibliográfica:** VanderPlas, *Python Data Science Handbook* (cap. 2, NumPy) · Géron, *Hands-On ML* (cap. 4 y 8: mínimos cuadrados, gradiente, PCA) · AIMA 4.ª ed. (cap. 12–13: probabilidad; apéndice A: vectores, matrices y complejidad) · Goodfellow et al., *Deep Learning* (cap. 2–4: álgebra lineal, probabilidad, cómputo numérico)
**Prerequisitos:** NB-01 (arrays, broadcasting, `np.linalg` básico). Aquí no repetimos lo visto allí: profundizamos en la **geometría** y en el **cómputo numérico**.

## Objetivos de Aprendizaje
Al completar este notebook serás capaz de:
1. **Interpretar geométricamente** una matriz (como transformación), sus autovalores/autovectores y su SVD, y verificarlo con código.
2. **Anticipar problemas numéricos** (mal condicionamiento, desbordamiento, cancelación) y elegir el algoritmo estable (`lstsq`/QR/SVD en vez de la ecuación normal, `logsumexp` en vez de `exp`).
3. **Simular con NumPy** distribuciones, Monte Carlo, la regla de Bayes y el BER de un enlace BPSK, y compararlos con la teoría.
4. **Implementar** descenso de gradiente, verificación numérica de gradientes y Newton–Raphson, y explicar cómo el *número de condición* gobierna la velocidad de convergencia.
5. **Usar herramientas avanzadas de NumPy** (`einsum`, `strides`, `sliding_window_view`, FFT, `out=`) para escribir código rápido y preciso.

## Mapa de Conexiones
| Curso / Área | Relevancia |
|---|---|
| IA (CIT-2013) | Probabilidad y regla de Bayes (AIMA cap. 12–13), base de NB-08; búsqueda con costos y heurísticas usa norma y distancia (NB-04). |
| Big Data (CIT-2309) | Vectorización y SVD/PCA para reducir dimensionalidad antes de procesar a escala; similitud coseno en recuperación de información. |
| Comunicaciones Digitales | **BER de BPSK por Monte Carlo vs teoría**, convolución con la respuesta al impulso del canal, estimación de canal por mínimos cuadrados, ecualizador (Ejercicio 3). |
| Sistemas Distribuidos | Estabilidad numérica y reproducibilidad de cálculos en `float32`/`float64` (no asociatividad de la suma de flotantes). |
| Tesis SNN/Ojo | Covarianza + autovalores de las métricas de ojo (PCA, NB-14), ecualización lineal como línea base contra el ecualizador SNN, gradiente y gradient checking (backprop en NB-16). |
| Notebooks relacionados | NB-01 (base), NB-04, NB-08, NB-11 (gradiente y regresión), NB-14 (PCA), NB-16 (backpropagation) |

## Tabla de contenidos
1. Marco teórico: ¿por qué el álgebra lineal es el idioma del ML?
2. Configuración
3. Vectores y matrices: geometría (producto punto, normas, matrices como transformaciones, QR, condicionamiento)
4. Autovalores y autovectores (covarianza, PCA preview, iteración de potencias, PageRank)
5. SVD (compresión de imágenes, pseudoinversa, regularización)
6. Probabilidad con NumPy (muestreo, LLN/CLT, Monte Carlo, BER, Bayes, gaussiana multivariada, entropía y softmax estable)
7. Optimización numérica (gradiente, número de condición, gradient check, Newton–Raphson)
8. NumPy avanzado (`einsum`, strides, convolución/FFT, precisión, rendimiento)
9. 📋 Cheat Sheet
10. Conexiones con el mundo real
11. Ejercicios progresivos
12. Resumen y siguiente paso
''')

# =====================================================================
# TEORÍA
# =====================================================================
M(r'''
## 1. Marco teórico: ¿por qué el álgebra lineal es el idioma del ML?

### La intuición: datos como puntos, modelos como transformaciones
Cada muestra de un dataset (una casa, una imagen, un símbolo recibido) es un **vector**: un punto en un espacio de muchas dimensiones. Un modelo lineal es una **transformación** de ese espacio: estira, rota, proyecta. Casi todo lo que verás en este curso es una combinación de tres ideas:

```
   DATOS (vectores)          MODELO (matrices)             APRENDIZAJE (cálculo)
  ┌───────────────┐       ┌────────────────────┐        ┌──────────────────────────┐
  │ x ∈ ℝⁿ        │  ──►  │ y = W x + b        │  ──►   │ minimizar L(W) siguiendo │
  │ X ∈ ℝ^(m×n)   │       │ capas = W₃ W₂ W₁ x │        │ el gradiente  ∇L         │
  └───────────────┘       └────────────────────┘        └──────────────────────────┘
   geometría: ángulos,       descomposiciones:             optimización: GD, Newton,
   distancias, proyecciones   autovalores, SVD, QR          condicionamiento
```

> **Analogía:** una matriz es como un *filtro de cámara* aplicado a todas las fotos a la vez. Lo que te importa saber es qué le hace al espacio: ¿lo gira? ¿lo aplasta en una dirección? ¿pierde información (una dimensión colapsa)? Los autovalores y la SVD responden exactamente eso.

### Los tres conceptos que sostienen este notebook
**1. Producto punto = similitud y proyección.** Para $\mathbf a,\mathbf b\in\mathbb R^n$:
$$\mathbf a\cdot\mathbf b=\sum_i a_ib_i=\|\mathbf a\|\,\|\mathbf b\|\cos\theta \qquad\Rightarrow\qquad \text{proy}_{\mathbf b}\mathbf a=\frac{\mathbf a\cdot\mathbf b}{\mathbf b\cdot\mathbf b}\,\mathbf b$$

**2. Autovalores/autovectores = las direcciones "naturales" de una matriz.** $A\mathbf v=\lambda\mathbf v$: al aplicar $A$, el vector $\mathbf v$ solo se **estira** (por $\lambda$), no cambia de dirección. Para una matriz simétrica (como una covarianza), esas direcciones son **perpendiculares** y los $\lambda$ miden cuánto varía el dato en cada una (→ PCA).

**3. SVD = "girar, estirar, girar".** Toda matriz (incluso rectangular) se descompone como
$$A=U\Sigma V^\top$$
con $U,V$ ortogonales (giros/reflejos) y $\Sigma$ diagonal (estiramientos $\sigma_1\ge\sigma_2\ge\dots\ge 0$). Truncar a los $k$ mayores valores singulares da **la mejor aproximación de rango $k$** (Eckart–Young): es la base de PCA, compresión y sistemas de recomendación.

### Perspectiva AIMA vs Géron
| | AIMA (teoría) | Géron (práctica) |
|---|---|---|
| Probabilidad | axiomas, regla de Bayes, independencia condicional (cap. 12–13) | simulación, `scipy.stats`, clasificadores bayesianos |
| Optimización | búsqueda local, descenso de gradiente como búsqueda (cap. 4) | `SGDRegressor`, learning rate, escalado de features (cap. 4) |
| Álgebra lineal | apéndice A: notación | cap. 8: PCA vía SVD; cap. 4: ecuación normal |

### Dos ideas de cómputo numérico que verás todo el tiempo
- **Condicionamiento.** El *número de condición* $\kappa(A)=\sigma_{\max}/\sigma_{\min}$ mide cuánto se amplifican los errores al resolver $A\mathbf x=\mathbf b$. Regla práctica: pierdes $\log_{10}\kappa$ dígitos de precisión. Además $\kappa$ **gobierna la velocidad del descenso de gradiente**.
- **Los flotantes no son los reales.** Tienen precisión finita ($\epsilon\approx 2.2\times10^{-16}$ en `float64`; $\approx 1.2\times10^{-7}$ en `float32`), desbordan (`exp(1000)=inf`) y la suma **no es asociativa**. Escribir la fórmula "bonita" suele ser el camino al `NaN`.
''')

# =====================================================================
# CONFIG
# =====================================================================
M(r'''
## 2. Configuración
Importamos solo lo que usamos. Fíjate en que `scipy` aparece mucho más que en NB-01: NumPy te da los arrays y las operaciones básicas; SciPy te da los algoritmos numéricos serios (distribuciones, optimizadores, FFT, funciones especiales).
''')

C(r'''
# === Imports ===
# Estándar
import os, sys, time, math, warnings
warnings.filterwarnings("ignore")

# Datos y cómputo
import numpy as np
import scipy.linalg as sla
from scipy import stats, signal, special, optimize
from numpy.lib.stride_tricks import sliding_window_view

# Machine Learning (solo datasets incluidos en sklearn y una comparación de PCA)
from sklearn.datasets import load_iris, load_sample_image
from sklearn.decomposition import PCA

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# === Configuración ===
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)          # generador moderno; se pasa explícitamente a las funciones
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 11
sns.set_style("whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"      # trae subíndices y símbolos matemáticos (x₁, κ, ℓ₁) que la fuente por defecto no dibuja
np.set_printoptions(precision=4, suppress=True)   # salidas de arrays legibles, sin notación científica ruidosa

print("✅ Entorno listo")
print(f"numpy {np.__version__} | scipy {sla.__name__.split('.')[0]} {__import__('scipy').__version__}")
''')

C(r'''
def time_ms(fn, repeat=5):
    # El mínimo de varias corridas filtra el ruido del sistema operativo
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best * 1e3


def plot_transform(A, ax, title=None):
    # Dibuja cómo la matriz 2x2 A deforma el círculo unitario, el cuadrado unitario y la base canónica
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    square = np.array([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=float)
    ax.plot(*circle, "--", color="gray", lw=1, label="original")
    ax.plot(*square, "--", color="gray", lw=1)
    ax.plot(*(A @ circle), color="#2980b9", lw=2, label="transformado")
    ax.plot(*(A @ square), color="#2980b9", lw=2)
    for col, c in zip(np.eye(2).T, ["#c0392b", "#27ae60"]):                 # e1 y e2 → columnas de A
        ax.annotate("", xy=A @ col, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2))
    rank = np.linalg.matrix_rank(A)
    ax.set_title((title or "") + f"\ndet={np.linalg.det(A):.2f}, rango={rank}", fontsize=10)
    ax.set_aspect("equal"); ax.grid(True); ax.axhline(0, color="k", lw=.5); ax.axvline(0, color="k", lw=.5)
''')

# =====================================================================
# VECTORES Y MATRICES
# =====================================================================
M(r'''
## 3. Vectores y matrices: la geometría

### 3.1 Producto punto, ángulo y proyección
**Por qué importa:** el producto punto es la operación más repetida del ML. Cada neurona calcula $\mathbf w\cdot\mathbf x+b$; la atención de un Transformer es un producto punto entre consultas y claves (NB-20); la regresión lineal busca la **proyección** del target sobre el espacio de las features.

Veamos las tres lecturas del producto punto con un ejemplo en 2D: el valor, el ángulo entre los vectores, y la proyección de $\mathbf a$ sobre $\mathbf b$.
''')

C(r'''
a = np.array([3.0, 1.0])
b = np.array([1.0, 2.0])

dot = a @ b
cos_theta = dot / (np.linalg.norm(a) * np.linalg.norm(b))
angle = np.degrees(np.arccos(cos_theta))
proj = (a @ b) / (b @ b) * b                 # componente de a en la dirección de b
resid = a - proj                              # lo que queda: debe ser perpendicular a b

print(f"a·b = {dot:.2f} | cos θ = {cos_theta:.4f} | ángulo = {angle:.1f}°")
print(f"proyección de a sobre b = {proj} | residuo = {resid}")
print(f"¿residuo ⟂ b?  resid·b = {resid @ b:.2e}  (≈ 0)")

fig, ax = plt.subplots(figsize=(6, 5))
for v, c, lab in [(a, "#2980b9", "a"), (b, "#c0392b", "b"), (proj, "#27ae60", "proy_b(a)")]:
    ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=2.5))
    ax.text(*(v * 1.04), lab, color=c, fontsize=12)
ax.plot([proj[0], a[0]], [proj[1], a[1]], ":", color="gray", label="residuo (perpendicular a b)")
ax.set(xlim=(-0.5, 3.8), ylim=(-0.5, 2.8), title="Producto punto: ángulo y proyección", xlabel="x₁", ylabel="x₂")
ax.set_aspect("equal"); ax.grid(True); ax.legend(); plt.show()
''')

M(r'''
**Qué deberías ver:** `a` (azul), `b` (rojo) y la sombra de `a` sobre la recta de `b` (verde); el segmento punteado une la punta de `a` con su proyección formando un **ángulo recto** con `b`. Numéricamente, `resid·b` debe ser 0 (o, en general, del orden de $10^{-16}$: el redondeo impide garantizar un cero exacto).

### Normas: distintas formas de medir "tamaño"
Una **norma** mide la longitud de un vector. Las tres que más usarás:

| Norma | Fórmula | Dónde aparece |
|---|---|---|
| $\ell_2$ (euclídea) | $\sqrt{\sum x_i^2}$ | distancias, MSE, regularización Ridge |
| $\ell_1$ (Manhattan) | $\sum\lvert x_i\rvert$ | Lasso (promueve ceros → *sparsity*), distancia en una grilla (NB-04) |
| $\ell_\infty$ | $\max\lvert x_i\rvert$ | error en el peor caso, distancia de Chebyshev |

Para *ver* cómo cambian, dibujamos el conjunto de vectores con norma 1 ("bola unidad") en cada caso:
''')

C(r'''
fig, ax = plt.subplots(figsize=(5.5, 5.5))
angles = np.linspace(0, 2 * np.pi, 400)
dirs = np.vstack([np.cos(angles), np.sin(angles)])          # direcciones; las reescalamos para que cada una tenga norma 1
for p, lab, c in [(1, "ℓ₁ (rombo)", "#c0392b"), (2, "ℓ₂ (círculo)", "#2980b9"), (np.inf, "ℓ∞ (cuadrado)", "#27ae60")]:
    unit = dirs / np.linalg.norm(dirs, ord=p, axis=0)
    ax.plot(*unit, label=lab, color=c, lw=2)
ax.set(title="Bolas unidad ‖x‖ₚ = 1", xlabel="x₁", ylabel="x₂"); ax.set_aspect("equal"); ax.grid(True); ax.legend()
plt.show()
''')

M(r'''
**Qué deberías ver:** un rombo, un círculo y un cuadrado anidados. La esquina del rombo sobre los ejes es la razón geométrica por la que **Lasso** ($\ell_1$) produce coeficientes exactamente cero: al intersectar la región de error con el rombo, es muy probable tocar una esquina (NB-11).

### Similitud coseno
Cuando importa la **dirección** y no la magnitud (documentos de distinto largo, embeddings), se usa $\cos\theta=\frac{\mathbf a\cdot\mathbf b}{\|\mathbf a\|\|\mathbf b\|}$. Es 1 si apuntan igual, 0 si son perpendiculares, −1 si son opuestos.
''')

C(r'''
def cosine_similarity_matrix(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)      # normalizar filas: así el producto punto ES el coseno
    return Xn @ Xn.T

# Conteo de palabras ["red", "neurona", "señal", "base_de_datos"] para 4 "documentos"
docs = np.array([[5, 4, 0, 0],      # redes neuronales
                 [10, 8, 0, 0],     # lo mismo, pero el doble de largo
                 [3, 0, 6, 0],      # redes de comunicaciones
                 [0, 0, 0, 7]], dtype=float)   # bases de datos
S = cosine_similarity_matrix(docs)
print(S.round(2))
print("\nOjo: docs 0 y 1 tienen distancia euclídea", round(np.linalg.norm(docs[0] - docs[1]), 2), "pero similitud coseno", S[0, 1].round(2))
''')

M(r'''
**Qué deberías ver:** una matriz simétrica con unos en la diagonal. Los documentos 0 y 1 tienen coseno **exactamente 1** (misma dirección) aunque su distancia euclídea sea grande: el coseno ignora el largo. Esa es la razón de que los *embeddings* de texto se comparen con coseno.
''')

nb.tryit([
    "**3-1. Ángulo recto:** encuentra un vector `c` no nulo que sea perpendicular a `a = [3, 1]` y verifica `a @ c == 0`. ¿Cuántos vectores unitarios lo cumplen? (Pista: en 2D, intercambia componentes y cambia un signo.)",
    "**3-2. Proyección sobre un plano:** proyecta `v = [1, 2, 3]` sobre el plano generado por `e1 = [1,0,0]` y `e2 = [0,1,0]` (suma de proyecciones, porque son ortogonales). ¿Qué obtienes y por qué?",
    "**3-3. Normas:** calcula `np.linalg.norm(x, ord=p)` de `x = [3, -4, 0]` para `p = 1, 2, np.inf`. ¿Cuál es la mayor? ¿Siempre se cumple $\\|x\\|_\\infty \\le \\|x\\|_2 \\le \\|x\\|_1$?",
])
nb.solutions("3-1, 3-2 y 3-3", r'''
c = np.array([-a[1], a[0]])                                    # 3-1: rotar 90° → [-1, 3]
print("c =", c, "| a·c =", a @ c, "| hay exactamente 2 unitarios: ±c/‖c‖")
v3 = np.array([1.0, 2.0, 3.0]); e1, e2 = np.eye(3)[0], np.eye(3)[1]
p3 = (v3 @ e1) / (e1 @ e1) * e1 + (v3 @ e2) / (e2 @ e2) * e2     # 3-2: se obtiene [1, 2, 0]: "borra" la 3.ª coordenada
print("proyección sobre el plano xy:", p3)
x3 = np.array([3.0, -4.0, 0.0])                                 # 3-3
print({p: np.linalg.norm(x3, ord=p) for p in (1, 2, np.inf)}, "→ ℓ1 ≥ ℓ2 ≥ ℓ∞ siempre")
''')

M(r'''
### 3.2 Matrices como transformaciones
**Por qué:** si "una matriz es una función", ¿qué función es? Aplicada a un vector $\mathbf x$, $A\mathbf x$ lo transforma. Las **columnas** de $A$ te dicen adónde van los vectores base: $A\mathbf e_1$ es la 1.ª columna y $A\mathbf e_2$ la 2.ª. Todo lo demás se deduce por linealidad.

Dos números resumen la transformación:
- **Determinante:** el factor por el que cambia el **área** (negativo = hay reflexión; 0 = el espacio colapsa).
- **Rango:** la dimensión de lo que queda (rango 1 en 2D = todo cae sobre una recta; se pierde información de forma irreversible).

Dibujemos seis transformaciones típicas (flechas roja y verde = imágenes de $\mathbf e_1$ y $\mathbf e_2$):
''')

C(r'''
th = np.radians(30)
mats = {
    "Rotación 30°": np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]),
    "Escala diag(2, 0.5)": np.diag([2.0, 0.5]),
    "Cizalla (shear)": np.array([[1.0, 0.8], [0.0, 1.0]]),
    "Reflexión (eje x)": np.array([[1.0, 0.0], [0.0, -1.0]]),
    "Proyección sobre x (singular)": np.array([[1.0, 0.0], [0.0, 0.0]]),
    "General [[2,1],[1,1.5]]": np.array([[2.0, 1.0], [1.0, 1.5]]),
}
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
for ax, (name, A) in zip(axes.ravel(), mats.items()):
    plot_transform(A, ax, name)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
axes[0, 0].legend(loc="lower left", fontsize=8)
plt.tight_layout(); plt.show()
''')

M(r'''
**Qué deberías ver:**
- **Rotación:** el círculo no cambia y $\det=1$ (conserva áreas y ángulos).
- **Escala:** el círculo se vuelve una elipse; $\det=2\cdot 0.5=1$ (área igual, forma distinta).
- **Cizalla:** el cuadrado se inclina; $\det=1$.
- **Reflexión:** el espejo vuelve el $\det=-1$.
- **Proyección:** todo cae sobre una recta; $\det=0$ y **rango 1**: no existe $A^{-1}$ (no puedes recuperar el $x_2$ perdido).
- **General:** una elipse inclinada; sus ejes son justo lo que revelarán los autovalores y la SVD.

> **NOTA:** una matriz es invertible ⇔ $\det\neq0$ ⇔ rango completo ⇔ sus columnas son linealmente independientes. En la práctica "casi singular" ($\det$ minúsculo) es tan malo como singular: lo medimos con el número de condición (sección 3.4).
''')

nb.tryit([
    "**3-4. Componer transformaciones:** ¿qué hace `R @ S` (rotar después de escalar) frente a `S @ R`? Dibuja ambas con `plot_transform`. ¿Conmutan las matrices? ¿Y qué determinante tiene el producto?",
    "**3-5. Crear una cizalla distinta:** construye una matriz que mapee `e1 → [1, 0]` y `e2 → [-1, 1]`. Verifica que sus columnas son esas imágenes y que su determinante es 1.",
])
nb.solutions("3-4 y 3-5", r'''
R, S_ = mats["Rotación 30°"], mats["Escala diag(2, 0.5)"]
print("¿R@S == S@R?", np.allclose(R @ S_, S_ @ R), "| det(R@S) =", round(np.linalg.det(R @ S_), 3), "= det(R)·det(S)")
fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
plot_transform(R @ S_, axs[0], "R @ S (escalar y luego rotar)"); plot_transform(S_ @ R, axs[1], "S @ R (rotar y luego escalar)")
for ax in axs: ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.6, 2.6)
plt.show()                                      # las matrices NO conmutan: el orden de las transformaciones importa
A5 = np.column_stack([[1, 0], [-1, 1]])         # 3-5: las columnas son las imágenes de e1 y e2
print("A5 =\n", A5, "| det =", np.linalg.det(A5))
''')

M(r'''
### 3.3 Ortogonalidad, QR y mínimos cuadrados
**Por qué:** una base **ortonormal** (vectores perpendiculares de norma 1) es la mejor posible: los cálculos son estables y las proyecciones triviales ($Q^\top Q=I$, así que $Q^{-1}=Q^\top$). La **descomposición QR** convierte las columnas de cualquier matriz $A$ en una base ortonormal $Q$, guardando en $R$ (triangular superior) cómo reconstruir $A=QR$. El método clásico es **Gram–Schmidt**: a cada columna le restas sus proyecciones sobre las anteriores y normalizas.

Lo implementamos (versión "modificada", numéricamente más estable) y lo comparamos con `np.linalg.qr`:
''')

C(r'''
def gram_schmidt(A):
    # QR por Gram–Schmidt modificado: resta cada proyección del vector YA actualizado (más estable que el clásico)
    m, n = A.shape
    Q, R = np.zeros((m, n)), np.zeros((n, n))
    for j in range(n):
        v = A[:, j].astype(float).copy()
        for i in range(j):
            R[i, j] = Q[:, i] @ v
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R

X = rng.normal(size=(6, 3))
Q, R = gram_schmidt(X)
Q_np, R_np = np.linalg.qr(X)

print("A = QR            :", np.allclose(Q @ R, X))
print("QᵀQ = I           :", np.allclose(Q.T @ Q, np.eye(3)))
print("R triangular sup. :", np.allclose(R, np.triu(R)))
print("igual a np.qr (salvo signos):", np.allclose(np.abs(Q), np.abs(Q_np)))
''')

M(r'''
**Qué deberías ver:** cuatro `True`. Los signos de las columnas de $Q$ pueden diferir de NumPy (la descomposición es única salvo signos), por eso comparamos valores absolutos.

### Mínimos cuadrados = proyección
Cuando $A\mathbf x=\mathbf y$ no tiene solución exacta (más ecuaciones que incógnitas: lo normal en ML), buscamos el $\hat{\mathbf x}$ que minimice $\|A\mathbf x-\mathbf y\|_2$. La geometría lo explica todo: $A\hat{\mathbf x}$ es la **proyección de $\mathbf y$ sobre el espacio generado por las columnas de $A$**, y por eso el **residuo es perpendicular a cada columna**: $A^\top(\mathbf y-A\hat{\mathbf x})=\mathbf 0$ (esa es la *ecuación normal*, $A^\top A\hat{\mathbf x}=A^\top\mathbf y$).

Con QR se resuelve sin formar $A^\top A$: $R\hat{\mathbf x}=Q^\top\mathbf y$.
''')

C(r'''
n = 80
t = rng.uniform(-1, 1, n)
A_ls = np.column_stack([np.ones(n), t, t ** 2])               # modelo: y = c0 + c1 t + c2 t²
coef_true = np.array([1.0, -2.0, 3.0])
y = A_ls @ coef_true + rng.normal(0, 0.3, n)

Q, R = np.linalg.qr(A_ls)
x_qr = sla.solve_triangular(R, Q.T @ y)                      # QR: resolver un sistema triangular (barato y estable)
x_ne = np.linalg.solve(A_ls.T @ A_ls, A_ls.T @ y)            # ecuación normal
x_lstsq = np.linalg.lstsq(A_ls, y, rcond=None)[0]            # lo que usarías en la práctica
residual = y - A_ls @ x_qr

print("QR     :", x_qr, "\nNormal :", x_ne, "\nlstsq  :", x_lstsq, "\nreal   :", coef_true)
print("Residuo ⟂ columnas de A →  ‖Aᵀ r‖ =", f"{np.linalg.norm(A_ls.T @ residual):.2e}")
''')

M(r'''
**Qué deberías ver:** los tres métodos coinciden (con coeficientes cercanos a [1, −2, 3]) y la norma $\|A^\top r\|$ es prácticamente cero: la prueba numérica de que el residuo es perpendicular al espacio de las columnas. En este problema bien condicionado todo funciona. Veamos qué pasa cuando **no** lo está.

### 3.4 Número de condición: cuando resolver un sistema "bien" sale mal
El **número de condición** $\kappa(A)=\sigma_{\max}/\sigma_{\min}$ es el factor máximo por el que se amplifica un error relativo en los datos. Con `float64` ($\epsilon\approx10^{-16}$), el error relativo de la solución es del orden de $\kappa\cdot\epsilon$. Probemos con las **matrices de Hilbert**, un clásico patológicamente mal condicionado: resolvemos $H\mathbf x=\mathbf b$ con $\mathbf x$ conocido y medimos el error.
''')

C(r'''
rows = []
for n_ in range(2, 15):
    H = sla.hilbert(n_)
    x_true = np.ones(n_)
    x_hat = np.linalg.solve(H, H @ x_true)
    rows.append((n_, np.linalg.cond(H), np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)))
cond, err = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])

fig, ax = plt.subplots()
ax.loglog(cond, err, "o-", label="error relativo medido")
ax.loglog(cond, cond * np.finfo(float).eps, "--", color="gray", label="regla: κ · ε")
ax.set(title="Error al resolver H x = b vs número de condición (Hilbert)", xlabel="κ(H)", ylabel="error relativo de x")
ax.grid(True, which="both"); ax.legend(); plt.show()
print(f"n=14: κ = {cond[-1]:.1e} → error relativo ≈ {err[-1]:.1e} (¡casi sin dígitos correctos!)")
''')

M(r'''
**Qué deberías ver:** una nube de puntos que sube junto con la recta $\kappa\cdot\epsilon$: a mayor condicionamiento, mayor error; para $n=14$ el error relativo **supera 1** (¡el error es mayor que la propia solución!), es decir, el resultado es **basura** aunque `solve` no lance ninguna advertencia.

Esto explica por qué la **ecuación normal** es peligrosa: $\kappa(A^\top A)=\kappa(A)^2$. Con un ajuste polinomial de grado alto, $A$ ya viene mal condicionada y $A^\top A$ pierde el *doble* de dígitos.
''')

C(r'''
t = np.linspace(0, 1, 60)
print(f"{'grado':>5} {'κ(A)':>10} {'κ(AᵀA)':>12} {'error coef. normal':>20} {'error coef. lstsq':>19}")
for deg in (3, 6, 9, 12):
    Ad = np.vander(t, deg + 1, increasing=True)
    c_true_p = np.ones(deg + 1)                       # coeficientes CONOCIDOS y datos sin ruido: aislamos el error numérico
    y_exact = Ad @ c_true_p
    c_ne = np.linalg.solve(Ad.T @ Ad, Ad.T @ y_exact)
    c_ls = np.linalg.lstsq(Ad, y_exact, rcond=None)[0]
    rel_c = lambda c: np.linalg.norm(c - c_true_p) / np.linalg.norm(c_true_p)
    print(f"{deg:>5} {np.linalg.cond(Ad):>10.1e} {np.linalg.cond(Ad.T @ Ad):>12.1e} {rel_c(c_ne):>20.1e} {rel_c(c_ls):>19.1e}")
''')

M(r'''
**Qué deberías ver:** $\kappa(A^\top A)$ es aproximadamente el **cuadrado** de $\kappa(A)$, y el error en los **coeficientes** de la ecuación normal es varios órdenes de magnitud mayor que el de `lstsq` (que usa SVD) cuando el grado crece. Un detalle sutil: el *residuo* de ambos métodos puede ser casi igual (el ajuste "se ve bien"), pero los **coeficientes** difieren: una solución mal condicionada puede reproducir los datos y aun así tener parámetros sin sentido. **Conclusión práctica:** usa siempre `np.linalg.lstsq` o `scipy.linalg.lstsq` en lugar de `solve(AᵀA, Aᵀy)`; y recuerda que normalizar las features (NB-09) mejora el condicionamiento.

### ⚠️ Troubleshooting

**Error:** `numpy.linalg.LinAlgError: Singular matrix`
**Por qué:** `np.linalg.solve`/`inv` recibió una matriz de rango deficiente (columnas redundantes, por ejemplo dos features perfectamente correlacionadas).
**Solución:** elimina la redundancia, o usa `lstsq`/`pinv` (que dan la solución de mínima norma) o regulariza (Ridge, sección 5.3).

**Síntoma:** resultados "raros" **sin** error ni warning.
**Por qué:** matriz mal condicionada (ver arriba): NumPy no te avisa.
**Solución:** revisa `np.linalg.cond(A)` antes de confiar en una solución; si $\kappa>10^{8}$ desconfía.
''')

nb.tryit([
    "**3-6. Predice la pérdida de dígitos:** para `H = sla.hilbert(8)` calcula $\\log_{10}\\kappa$ y compáralo con el número de dígitos correctos que obtienes en `solve(H, H @ ones)` (usa `-np.log10(error)`).",
    "**3-7. Normalizar ayuda:** con `Ad = np.vander(t, 10, increasing=True)`, calcula `cond` antes y después de **escalar cada columna a norma 1**. ¿Cuánto mejora? ¿Por qué mejora tan poco aquí, y en qué caso ayudaría mucho más?",
])
nb.solutions("3-6 y 3-7", r'''
H8 = sla.hilbert(8); e8 = np.linalg.norm(np.linalg.solve(H8, H8 @ np.ones(8)) - np.ones(8)) / np.sqrt(8)
print(f"log10 κ = {np.log10(np.linalg.cond(H8)):.1f} → dígitos perdidos ≈ 16 - {-np.log10(e8):.1f} = {16 + np.log10(e8):.1f}")   # 3-6
Ad = np.vander(t, 10, increasing=True)
Ad_n = Ad / np.linalg.norm(Ad, axis=0)                      # 3-7: columnas de norma 1 (equilibrado)
print(f"κ antes = {np.linalg.cond(Ad):.1e} | κ después = {np.linalg.cond(Ad_n):.1e}")
print("Mejora poco: las columnas ya tienen escalas parecidas; el mal condicionamiento viene de la COLINEALIDAD (t⁹ ≈ t¹⁰), no de la escala.")
print("Escalar ayuda mucho cuando las columnas tienen magnitudes muy distintas (ver 7-2); para colinealidad hace falta otra base (Legendre) o regularizar.")
''')

M(r'''
### 📊 Resumen visual — vectores y matrices
```
 CONCEPTO              GEOMETRÍA                              CÓDIGO                         CUIDADO CON…
 ──────────────────────────────────────────────────────────────────────────────────────────────────────
 a·b                   proyección y ángulo                    a @ b                          dimensiones (1D vs 2D)
 ‖x‖₂ / ‖x‖₁ / ‖x‖∞    esfera / rombo / cuadrado              np.linalg.norm(x, ord=p)       ℓ1 → esquinas → sparsity
 A como función        columnas = imágenes de la base          A @ x                          el orden importa (A@B ≠ B@A)
 det(A), rango(A)      factor de área, dimensión que sobrevive np.linalg.det/matrix_rank     det≈0 ⇒ casi singular
 QR                    base ortonormal Q + triangular R        np.linalg.qr                   signos libres
 Mínimos cuadrados     proyectar y sobre col(A)                np.linalg.lstsq                NO usar solve(AᵀA, Aᵀy)
 κ(A)                  amplificación del error                 np.linalg.cond                 pierdes log10(κ) dígitos
```
''')

# =====================================================================
# AUTOVALORES
# =====================================================================
M(r'''
## 4. Autovalores y autovectores

### 4.1 La intuición geométrica
Cuando aplicas $A$ a un vector cualquiera, normalmente cambia de **dirección**. Pero casi siempre existen unas pocas direcciones especiales que **no se desvían**: solo se estiran o encogen. Esas son los **autovectores**, y el factor de estiramiento es el **autovalor**:
$$A\mathbf v=\lambda\mathbf v$$

> **Analogía:** imagina que estiras una hoja de goma sujetándola de dos lados opuestos. La mayoría de los puntos se desplazan en diagonal, pero los que estaban sobre la línea de estiramiento solo se alejan o se acercan a lo largo de esa línea. Esa línea es un autovector.

El resultado más importante para ML: si $A$ es **simétrica** (como una matriz de covarianza o $A^\top A$), sus autovectores son **perpendiculares** entre sí y la transformación es exactamente "estirar a lo largo de ejes perpendiculares" (**teorema espectral**: $A=Q\Lambda Q^\top$). Lo comprobamos dibujando cómo $A$ convierte el círculo en una elipse cuyos ejes son los autovectores:
''')

C(r'''
A_sym = np.array([[3.0, 1.0], [1.0, 2.0]])
lam, V = np.linalg.eigh(A_sym)             # eigh: para simétricas (más rápida, autovalores reales, V ortonormal)
print("autovalores λ :", lam)
print("autovectores (columnas):\n", V)
print("¿A v = λ v ?   ", all(np.allclose(A_sym @ V[:, i], lam[i] * V[:, i]) for i in range(2)))
print("¿V ortonormal? ", np.allclose(V.T @ V, np.eye(2)), "| ¿A = V Λ Vᵀ? ", np.allclose(V @ np.diag(lam) @ V.T, A_sym))

fig, ax = plt.subplots(figsize=(6, 6))
plot_transform(A_sym, ax, "A simétrica: el círculo → elipse")
for i, c in enumerate(["#8e44ad", "#d35400"]):
    ax.annotate("", xy=lam[i] * V[:, i], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=3))
    ax.text(*(lam[i] * V[:, i] * 1.08), f"λ={lam[i]:.2f}", color=c, fontsize=11)
ax.set_xlim(-3.8, 3.8); ax.set_ylim(-3.8, 3.8); plt.show()
''')

M(r'''
**Qué deberías ver:** el círculo gris se convierte en una elipse azul **inclinada**; las flechas morada y naranja (autovectores escalados por su $\lambda$) caen **exactamente sobre los ejes de la elipse**, con largo igual al semieje correspondiente. Los autovalores son ≈ 1.38 y 3.62.

> **NOTA:** usa `eigh` para matrices simétricas y `eig` para generales. `eig` puede devolver autovalores **complejos** (p. ej. una rotación pura no tiene ninguna dirección que se conserve), y sus autovectores no tienen por qué ser ortogonales. Además `eigh` los entrega **ordenados de menor a mayor**, mientras que la convención de PCA es de mayor a menor.
''')

C(r'''
lam_c, V_c = np.linalg.eig(mats["Rotación 30°"])
print("Autovalores de una rotación (complejos, sin dirección invariante real):", lam_c)
print("|λ| =", np.abs(lam_c), "→ una rotación no estira nada; solo gira.")
''')

M(r'''
### 4.2 Covarianza y PCA: los autovectores de los datos
**Por qué:** ¿en qué dirección "se estira" más una nube de datos? La **matriz de covarianza** $\Sigma$ describe cómo covarían las variables, y sus autovectores son los **ejes principales** de la nube; los autovalores son las **varianzas** a lo largo de esos ejes. Esa es la idea central de **PCA** (se desarrolla completa en NB-14).

Generamos una nube correlacionada y calculamos sus ejes:
''')

C(r'''
mu = np.array([2.0, 1.0])
Sigma_true = np.array([[3.0, 2.0], [2.0, 2.0]])
data = rng.multivariate_normal(mu, Sigma_true, size=600)

Xc = data - data.mean(axis=0)                     # centrar: PCA se hace sobre datos con media cero
Sigma = (Xc.T @ Xc) / (len(Xc) - 1)                # covarianza muestral; equivale a np.cov(data.T)
assert np.allclose(Sigma, np.cov(data.T))

lam_s, V_s = np.linalg.eigh(Sigma)
order = np.argsort(lam_s)[::-1]                    # de mayor a menor varianza
lam_s, V_s = lam_s[order], V_s[:, order]
scores = Xc @ V_s                                  # coordenadas de los datos en los ejes principales

print("autovalores de Σ:", lam_s, "| varianza explicada:", (lam_s / lam_s.sum()).round(3))
print("varianza de los datos proyectados:", scores.var(axis=0, ddof=1), "← igual a los autovalores")
print("covarianza entre componentes (≈ 0, decorrelacionadas):", np.cov(scores.T)[0, 1].round(6))
''')

C(r'''
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(*data.T, s=12, alpha=0.4, label="datos")
for i, c in enumerate(["#c0392b", "#27ae60"]):
    v = V_s[:, i] * 2 * np.sqrt(lam_s[i])              # 2 desviaciones estándar a lo largo del eje
    ax.annotate("", xy=mu + v, xytext=mu, arrowprops=dict(arrowstyle="->", color=c, lw=3))
    ax.text(*(mu + v * 1.1), f"PC{i+1}: {lam_s[i] / lam_s.sum():.0%}", color=c, fontsize=12)
ax.set(title="Ejes principales de una nube de puntos", xlabel="x₁", ylabel="x₂"); ax.set_aspect("equal"); ax.grid(True); ax.legend()
plt.show()
''')

M(r'''
**Qué deberías ver:** una nube alargada en diagonal con dos flechas: la roja (PC1) a lo largo del eje mayor, que explica cerca del **90 %** de la varianza, y la verde (PC2) perpendicular. Fíjate en tres verificaciones numéricas: la varianza de los datos proyectados **es** el autovalor, la covarianza entre componentes es ~0 (PCA **decorrelaciona**), y el número de componentes que necesitas depende de cuánta varianza quieres conservar.

### 4.3 Iteración de potencias y PageRank
Para matrices enormes (millones de filas) no se calcula la descomposición completa: se usan métodos iterativos. El más simple es la **iteración de potencias**: multiplicar repetidamente un vector aleatorio por $A$ y normalizar. Cada multiplicación amplifica la componente del autovector dominante, y tras unas iteraciones el vector **converge a él**.

$$\mathbf v_{k+1}=\frac{A\mathbf v_k}{\|A\mathbf v_k\|},\qquad \lambda\approx\mathbf v^\top A\mathbf v\qquad\text{(velocidad: }|\lambda_2/\lambda_1|^k\text{)}$$
''')

C(r'''
def power_iteration(A, iters=500, tol=1e-12, rng=rng):
    v = rng.normal(size=A.shape[0]); v /= np.linalg.norm(v)
    lam_old, hist = 0.0, []
    for k in range(iters):
        w = A @ v
        v = w / np.linalg.norm(w)
        lam = v @ A @ v                       # cociente de Rayleigh: estimación del autovalor
        hist.append(abs(lam - lam_old)); lam_old = lam
        if hist[-1] < tol:
            break
    return lam, v, hist

lam_pi, v_pi, hist = power_iteration(Sigma)
print(f"iteración de potencias: λ = {lam_pi:.6f} en {len(hist)} pasos | eigh: {lam_s[0]:.6f}")
print(f"cociente |λ2/λ1| = {lam_s[1]/lam_s[0]:.3f}  (cuanto más chico, más rápido converge)")
print("¿mismo eje que PC1?", np.allclose(np.abs(v_pi), np.abs(V_s[:, 0])))
''')

M(r'''
**Qué deberías ver:** convergencia en unas pocas iteraciones al mismo autovalor y eje que `eigh`. El ritmo lo fija $|\lambda_2/\lambda_1|$.

**PageRank** (el algoritmo original de Google) es exactamente esto aplicado a la web: la importancia de una página es el **autovector dominante** de la matriz de transición del "navegante aleatorio" (una cadena de Markov, ver NB-01 y NB-08). Con un factor de amortiguación $d=0.85$ (con probabilidad $1-d$ el navegante salta a una página al azar), la matriz es $G=dM+\frac{1-d}{n}\mathbf 1\mathbf 1^\top$.
''')

C(r'''
pages = ["Inicio", "Blog", "Docs", "Precios", "Contacto", "Foro"]
links = {0: [1, 2, 3], 1: [0, 2], 2: [0, 3], 3: [0, 4], 4: [0], 5: [0, 2]}       # a qué páginas enlaza cada una
n_p = len(pages)
M_web = np.zeros((n_p, n_p))
for src, dsts in links.items():
    M_web[dsts, src] = 1 / len(dsts)                     # columna src: probabilidad de ir de src a cada destino
assert np.allclose(M_web.sum(axis=0), 1)                 # columnas suman 1 (estocástica por columnas)

d = 0.85
G = d * M_web + (1 - d) / n_p * np.ones((n_p, n_p))      # el "salto aleatorio" hace G positiva → convergencia garantizada
pr = np.full(n_p, 1 / n_p)
for _ in range(100):
    pr = G @ pr                                          # distribución de probabilidad: ya suma 1, no hace falta normalizar

ev, evec = np.linalg.eig(G); pr_eig = np.real(evec[:, np.argmax(np.real(ev))]); pr_eig /= pr_eig.sum()
print("¿PageRank por potencias == por autovector?", np.allclose(pr, pr_eig))
for name, p_ in sorted(zip(pages, pr), key=lambda z: -z[1]):
    print(f"  {name:9s} {p_:.3f}")
''')

M(r'''
**Qué deberías ver:** "Inicio" (a la que casi todas enlazan) con el puntaje más alto, "Foro" (a la que nadie enlaza) con el más bajo. Dos mensajes: (1) el autovalor dominante de una matriz estocástica es 1 y su autovector es la **distribución estacionaria**; (2) un algoritmo de miles de millones de páginas es, en esencia, una iteración de potencias.
''')

nb.tryit([
    "**4-1. Mayor varianza:** usa `load_iris().data` (4 features). Calcula la covarianza y sus autovalores con `eigh`. ¿Qué fracción de la varianza explica el primer componente? ¿Cuántos componentes necesitas para el 95 %?",
    "**4-2. Autovalores y traza:** verifica que `suma(autovalores) == traza(Σ)` y que `producto(autovalores) == det(Σ)`. ¿Qué significan ambos en términos de varianza total y volumen?",
    "**4-3. Velocidad de potencias:** construye una matriz simétrica 2×2 con autovalores 10 y 9.9 (usa `Q diag(10, 9.9) Qᵀ`). ¿Cuántas iteraciones necesita `power_iteration`? Compara con autovalores 10 y 1.",
])
nb.solutions("4-1, 4-2 y 4-3", r'''
Xi = load_iris().data
Si = np.cov(Xi.T); li = np.sort(np.linalg.eigvalsh(Si))[::-1]
print("fracción por componente:", (li / li.sum()).round(4), "| acumulada:", np.cumsum(li / li.sum()).round(4))       # 4-1
print("componentes para 95%:", int(np.searchsorted(np.cumsum(li / li.sum()), 0.95) + 1))
print("traza == suma λ:", np.isclose(np.trace(Si), li.sum()), "| det == prod λ:", np.isclose(np.linalg.det(Si), li.prod()))   # 4-2
Qr = np.linalg.qr(rng.normal(size=(2, 2)))[0]
for pair in [(10, 9.9), (10, 1)]:                                                                                       # 4-3
    A_ = Qr @ np.diag(pair) @ Qr.T
    print(f"autovalores {pair}: {len(power_iteration(A_)[2])} iteraciones")
# traza = varianza total (se conserva al rotar los ejes); det = "volumen" de la nube. Con λ casi iguales la convergencia es lentísima.
''')

M(r'''
### 📊 Resumen visual — autovalores
```
        A v = λ v                      Σ = V Λ Vᵀ  (A simétrica)
   ┌─────────────────┐            ┌──────────────────────────────────────┐
   │ v: dirección que │            │ columnas de V: ejes principales       │
   │ solo se estira   │   ──►      │ λᵢ: varianza a lo largo de cada eje   │  ──► PCA (NB-14)
   │ λ: cuánto        │            │ suma λᵢ = traza  ·  prod λᵢ = det     │
   └─────────────────┘            └──────────────────────────────────────┘
 eigh (simétrica, ordenado ↑, ortonormal) · eig (general, puede ser complejo) · potencias (matrices enormes)
```
''')

# =====================================================================
# SVD
# =====================================================================
M(r'''
## 5. SVD: la descomposición universal

### 5.1 Intuición: "girar, estirar, girar"
Los autovalores solo existen para matrices **cuadradas** (y se portan bien solo para las simétricas). La SVD funciona para **cualquier** matriz $m\times n$:
$$A=U\,\Sigma\,V^\top\qquad U\in\mathbb R^{m\times m},\; V\in\mathbb R^{n\times n}\text{ ortogonales},\;\Sigma\text{ diagonal con }\sigma_1\ge\dots\ge0$$

Lee la transformación de derecha a izquierda: $V^\top$ **gira** el espacio de entrada, $\Sigma$ **estira** cada eje, $U$ **gira** el resultado. Es decir, toda matriz es una elipse que se obtiene estirando un círculo a lo largo de ejes perpendiculares; los $\sigma_i$ son los semiejes. La conexión con autovalores: los $\sigma_i^2$ son los autovalores de $A^\top A$ (y los $V$, sus autovectores).
''')

C(r'''
A2 = mats["General [[2,1],[1,1.5]]"] + np.array([[0.0, 0.5], [0.0, 0.0]])      # una matriz NO simétrica
U, s, Vt = np.linalg.svd(A2)
print("σ =", s, "| κ = σmax/σmin =", s[0] / s[-1])
print("A = U Σ Vᵀ :", np.allclose(U @ np.diag(s) @ Vt, A2))
print("σ² = autovalores de AᵀA :", np.allclose(np.sort(s ** 2), np.sort(np.linalg.eigvalsh(A2.T @ A2))))

fig, ax = plt.subplots(figsize=(6, 6))
plot_transform(A2, ax, "SVD: la elipse tiene semiejes σᵢ en direcciones uᵢ")
for i, c in enumerate(["#8e44ad", "#d35400"]):
    ax.annotate("", xy=s[i] * U[:, i], xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=c, lw=3))
    ax.text(*(s[i] * U[:, i] * 1.1), f"σ{i+1}={s[i]:.2f}", color=c, fontsize=11)
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); plt.show()
''')

M(r'''
**Qué deberías ver:** una elipse **inclinada** cuyos semiejes (flechas morada y naranja) miden $\sigma_1$ y $\sigma_2$ y apuntan en las direcciones $\mathbf u_1,\mathbf u_2$. Observa: aunque $A$ no es simétrica, la SVD siempre encuentra ejes **perpendiculares** de entrada ($V$) y de salida ($U$). El cociente $\sigma_1/\sigma_2$ es el número de condición: cuanto más aplastada la elipse, peor condicionada la matriz.

### 5.2 Compresión de imágenes con SVD
**Aplicación visual de Eckart–Young:** una imagen en escala de grises es una matriz de $m\times n$. Si conservamos solo los $k$ mayores valores singulares,
$$A_k=\sum_{i=1}^k\sigma_i\,\mathbf u_i\mathbf v_i^\top$$
es la **mejor aproximación posible de rango $k$**, y guardarla cuesta $k(m+n+1)$ números en vez de $mn$.
''')

C(r'''
img = load_sample_image("china.jpg").astype(float).mean(axis=2) / 255      # a gris, normalizada a [0,1]
m_, n_ = img.shape
img_mean = img.mean()
# Centramos (como en PCA): el brillo medio aporta casi toda la "energía" pero no es estructura que valga la pena comprimir
U_i, s_i, Vt_i = np.linalg.svd(img - img_mean, full_matrices=False)

def rank_k(k):
    return (U_i[:, :k] * s_i[:k]) @ Vt_i[:k] + img_mean          # U_k Σ_k V_kᵀ (+ la media), sin construir Σ completa

energy = np.cumsum(s_i ** 2) / np.sum(s_i ** 2)                   # fracción de "energía" (norma de Frobenius²) acumulada
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, k in zip(axes, [5, 20, 60]):
    ax.imshow(rank_k(k), cmap="gray", vmin=0, vmax=1); ax.axis("off")
    ax.set_title(f"k={k} | {k*(m_+n_+1)/(m_*n_):.1%} del tamaño\nenergía {energy[k-1]:.1%}")
axes[3].imshow(img, cmap="gray", vmin=0, vmax=1); axes[3].axis("off"); axes[3].set_title(f"original ({m_}×{n_})")
plt.tight_layout(); plt.show()
''')

C(r'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].semilogy(s_i, lw=2); axes[0].set(title="Valores singulares de la imagen (escala log)", xlabel="índice i", ylabel="σᵢ")
axes[1].plot(energy, lw=2)
for q in (0.9, 0.99):
    k_q = int(np.searchsorted(energy, q) + 1)
    axes[1].axhline(q, ls="--", color="gray"); axes[1].axvline(k_q, ls=":", color="gray")
    axes[1].annotate(f"{q:.0%} → k={k_q}", xy=(k_q, q), xytext=(k_q + 15, q - 0.08))
axes[1].set(title="Energía acumulada vs rango k", xlabel="k", ylabel="fracción de energía", xlim=(0, 200))
for ax in axes: ax.grid(True, which="both")
plt.tight_layout(); plt.show()
err_k = {k: np.linalg.norm(img - rank_k(k)) / np.linalg.norm(img) for k in (5, 20, 60)}
print("error relativo (Frobenius):", {k: f"{v:.2%}" for k, v in err_k.items()})
''')

M(r'''
**Qué deberías ver:** con $k=5$ solo se intuyen manchas grandes; con $k=20$ la escena ya es reconocible; con $k=60$ la imagen es muy fiel aunque con una textura de "grano" en las zonas lisas (cielo, agua), y aun así ya ocupa ~23 % del tamaño original: la compresión es real pero modesta. Los valores singulares **decaen rápido** (el eje está en escala log): unos pocos componentes concentran la mayor parte de la energía (cerca del 87 % con solo $k=5$, tras quitar el brillo medio). Esa es la razón de fondo por la que funciona PCA, la compresión y los sistemas de recomendación (matrices usuario–ítem de bajo rango).

> **NOTA (honestidad):** SVD *no* es cómo se comprimen las imágenes en la práctica (JPEG usa DCT por bloques, mucho más eficiente). Aquí es una ilustración de la **aproximación de bajo rango**, que sí es central en ML.

### 5.3 Pseudoinversa y regularización
Para una matriz mal condicionada, los $\sigma_i$ muy pequeños **amplifican el ruido**: la solución de mínimos cuadrados es $\hat{\mathbf x}=\sum_i\frac{\mathbf u_i^\top\mathbf b}{\sigma_i}\mathbf v_i$, y dividir por un $\sigma_i\approx10^{-12}$ dispara cualquier error de $\mathbf b$. Dos remedios, ambos naturales con SVD:

- **SVD truncada:** descartar los $\sigma_i$ pequeños.
- **Ridge (Tikhonov):** cambiar $1/\sigma_i$ por $\sigma_i/(\sigma_i^2+\lambda)$: los $\sigma$ grandes casi no cambian, los pequeños se **suprimen suavemente**. Es exactamente la regresión Ridge (NB-11).
''')

C(r'''
n_h = 12
H = sla.hilbert(n_h)
x_true = np.ones(n_h)
b_clean = H @ x_true
b_noisy = b_clean + rng.normal(0, 1e-8, n_h)             # ruido minúsculo en las mediciones

U_h, s_h, Vt_h = np.linalg.svd(H)
def solve_filtered(filter_fn):
    return Vt_h.T @ (filter_fn(s_h) * (U_h.T @ b_noisy))

x_ls = solve_filtered(lambda s: 1 / s)                                      # mínimos cuadrados "ingenuos"
x_trunc = solve_filtered(lambda s: np.where(s > 1e-6, 1 / s, 0.0))          # SVD truncada
lam_r = 1e-8
x_ridge = solve_filtered(lambda s: s / (s ** 2 + lam_r))                    # Ridge

rel = lambda x: np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
print(f"κ(H) = {s_h[0]/s_h[-1]:.1e} | ruido relativo en b = {np.linalg.norm(b_noisy-b_clean)/np.linalg.norm(b_clean):.1e}")
print(f"error  mínimos cuadrados: {rel(x_ls):.2e}\nerror  SVD truncada     : {rel(x_trunc):.2e}\nerror  Ridge (λ=1e-8)    : {rel(x_ridge):.2e}")
print("¿pinv coincide con la solución filtrada 1/σ?", np.allclose(np.linalg.pinv(H, rcond=0) @ b_noisy, x_ls, atol=1e-3 * np.abs(x_ls).max()))
''')

M(r'''
**Qué deberías ver:** un ruido relativo de ~$10^{-8}$ en $\mathbf b$ produce un error **enorme** (órdenes de magnitud mayor) en la solución ingenua, mientras que la SVD truncada y Ridge lo reducen a niveles razonables. Es el **compromiso sesgo–varianza** en estado puro: descartar información (introducir sesgo) a cambio de estabilidad (menor varianza). Lo retomas en NB-11.
''')

nb.tryit([
    "**5-1. Eckart–Young numérico:** verifica que el error de la mejor aproximación de rango $k$ en norma de Frobenius es $\\sqrt{\\sum_{i>k}\\sigma_i^2}$ (usa los `s_i` de la imagen para $k=20$).",
    "**5-2. Elegir k por energía:** escribe `choose_k(s, q)` que devuelva el menor $k$ con energía acumulada ≥ q. Calcula cuánto ahorro de almacenamiento da `q=0.95` en la imagen.",
    "**5-3. Barrido de λ:** grafica el error de Ridge en la sección 5.3 para `λ ∈ [1e-16, 1e-2]` (escala log). ¿Existe un valor óptimo? ¿Qué pasa con λ muy grande o muy pequeño?",
])
nb.solutions("5-1, 5-2 y 5-3", r'''
k = 20
print(f"error medido {np.linalg.norm(img - rank_k(k)):.4f} vs teoría {np.sqrt(np.sum(s_i[k:] ** 2)):.4f}")               # 5-1
def choose_k(s, q):
    return int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), q) + 1)
k95 = choose_k(s_i, 0.95); print(f"k(95%) = {k95} → almacenamiento {k95*(m_+n_+1)/(m_*n_):.1%} del original")            # 5-2
lams = np.logspace(-16, -2, 30)
errs = [rel(solve_filtered(lambda s, l=l: s / (s ** 2 + l))) for l in lams]                                            # 5-3
fig, ax = plt.subplots(); ax.loglog(lams, errs, "o-"); ax.axvline(lams[int(np.argmin(errs))], ls="--", color="gray", label="mejor λ")
ax.set(title="Error de Ridge vs λ (sesgo–varianza)", xlabel="λ", ylabel="error relativo"); ax.grid(True, which="both"); ax.legend(); plt.show()
print("λ muy pequeño → se amplifica el ruido (varianza alta); λ muy grande → solución sesgada hacia 0. El óptimo está en el medio.")
''')

M(r'''
### 📊 Resumen visual — SVD
```
   A  =  U     Σ      Vᵀ                 Usos:
  m×n   m×m  m×n    n×n                 ┌───────────────────────────────────────────────┐
        giro estira  giro                 │ rango y condicionamiento:  κ = σ₁/σᵣ          │
                                          │ mejor aprox. rango k:      A_k = Σᵢ≤k σᵢ uᵢ vᵢᵀ │
   σᵢ² = autovalores de AᵀA               │ PCA (sobre datos centrados) · pinv · Ridge     │
   V = autovectores de AᵀA                │ compresión y recomendación (matrices bajo rango)│
                                          └───────────────────────────────────────────────┘
```
''')

# =====================================================================
# PROBABILIDAD
# =====================================================================
M(r'''
## 6. Probabilidad con NumPy

**Por qué:** los datos reales son ruidosos, y los modelos de ML son, en el fondo, modelos probabilísticos. Con NumPy puedes **simular** un proceso aleatorio miles de veces y comparar con la teoría: es la mejor forma de *sentir* un resultado antes de demostrarlo (y una herramienta de verificación para el resto de la carrera). La probabilidad formal (axiomas, redes bayesianas) la verás en NB-08; aquí construimos la caja de herramientas de cómputo.

### 6.1 Muestreo y distribuciones
`rng` (un `Generator`) ofrece las distribuciones más comunes; `scipy.stats` aporta densidades (`pdf`), acumuladas (`cdf`), cuantiles (`ppf`) y ajuste. Comparamos histograma simulado y densidad teórica:
''')

C(r'''
N = 100_000
samples = {
    "Normal(μ=2, σ=1.5)": (rng.normal(2, 1.5, N), stats.norm(2, 1.5)),
    "Exponencial(λ=0.5)": (rng.exponential(2.0, N), stats.expon(scale=2.0)),         # NumPy usa la escala = 1/λ
    "Uniforme(0, 4)": (rng.uniform(0, 4, N), stats.uniform(0, 4)),
    "Poisson(λ=4) [discreta]": (rng.poisson(4, N), stats.poisson(4)),
}
fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
for ax, (name, (x, dist)) in zip(axes, samples.items()):
    if "Poisson" in name:
        ks = np.arange(0, 14)
        ax.bar(ks, np.bincount(x, minlength=14)[:14] / N, alpha=0.6, label="simulación")
        ax.plot(ks, dist.pmf(ks), "ro-", label="teoría (pmf)")
    else:
        ax.hist(x, bins=60, density=True, alpha=0.6, label="simulación")
        grid = np.linspace(x.min(), x.max(), 300); ax.plot(grid, dist.pdf(grid), "r", lw=2, label="teoría (pdf)")
    ax.set_title(name, fontsize=10); ax.grid(True); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
print({n: f"media sim={x.mean():.3f} | teórica={d.mean():.3f}" for n, (x, d) in samples.items()})
''')

M(r'''
**Qué deberías ver:** en las cuatro figuras, las barras (simulación) se pegan a la curva roja (teoría). Con $10^5$ muestras el acuerdo es casi perfecto; es la **ley de los grandes números** en acción.

> **NOTA:** `rng.exponential(scale)` recibe la **escala** $=1/\lambda$, no la tasa; confundirlas es un error clásico. Y usa siempre `rng = np.random.default_rng(seed)` en vez de `np.random.seed(...)` global: es el generador moderno, reproducible y recomendado por NumPy.

### 6.2 Ley de los grandes números y Teorema Central del Límite
- **LLN:** el promedio muestral converge al valor esperado al crecer $n$.
- **TCL:** la **suma/promedio** de muchas variables independientes (de casi cualquier distribución con varianza finita) se distribuye aproximadamente **normal**, con desviación $\sigma/\sqrt n$. Por eso la campana aparece en todas partes (el ruido térmico de un receptor es la suma de muchísimos efectos microscópicos, de ahí el ruido **gaussiano** de un canal AWGN).
''')

C(r'''
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
# LLN: promedio acumulado de tiradas de un dado (esperanza teórica = 3.5)
rolls = rng.integers(1, 7, 5000)
axes[0].plot(np.cumsum(rolls) / np.arange(1, len(rolls) + 1), lw=1.5, label="promedio acumulado")
axes[0].axhline(3.5, color="r", ls="--", label="E[X] = 3.5")
axes[0].set(title="LLN: el promedio converge", xlabel="nº de tiradas", ylabel="promedio"); axes[0].legend()

# TCL: promedios de n variables EXPONENCIALES (muy asimétricas) → campana
for n_, color in [(1, "#c0392b"), (30, "#27ae60")]:
    means = rng.exponential(1.0, (20_000, n_)).mean(axis=1)
    axes[1 if n_ == 1 else 2].hist(means, bins=60, density=True, color=color, alpha=0.6)
    axes[1 if n_ == 1 else 2].set_title(f"TCL: promedio de n={n_} exponenciales")
grid = np.linspace(0.4, 1.8, 200)
axes[2].plot(grid, stats.norm(1, 1 / np.sqrt(30)).pdf(grid), "k--", label="N(1, 1/√30)"); axes[2].legend()
for ax in axes: ax.grid(True)
plt.tight_layout(); plt.show()
''')

M(r'''
**Qué deberías ver:** (izquierda) el promedio de dados oscila al principio y se estabiliza en 3.5; (centro) con $n=1$ la distribución exponencial es muy asimétrica; (derecha) con $n=30$ el promedio ya es casi una campana, que coincide con la normal teórica de línea negra.

### 6.3 Monte Carlo: estimar con simulación
**Idea:** para calcular algo difícil (una integral, una probabilidad), **simulas** el experimento muchas veces y promedias. El error baja como $1/\sqrt N$ **sin importar la dimensión** (ventaja clave frente a integrar en una grilla). Ejemplo clásico: estimar $\pi$ lanzando puntos al azar en un cuadrado y contando cuántos caen en el círculo inscrito (la fracción es $\pi/4$).
''')

C(r'''
def estimate_pi(n, rng):
    xy = rng.uniform(-1, 1, size=(n, 2))
    return 4 * np.mean(np.sum(xy ** 2, axis=1) <= 1)       # vectorizado: sin bucles Python

Ns = np.logspace(2, 6, 9).astype(int)
errors = np.array([[abs(estimate_pi(n, rng) - np.pi) for _ in range(30)] for n in Ns])   # 30 repeticiones por tamaño

fig, ax = plt.subplots()
ax.loglog(Ns, errors.mean(axis=1), "o-", label="error medio (30 repeticiones)")
ax.loglog(Ns, 1.3 / np.sqrt(Ns), "--", color="gray", label="∝ 1/√N")
ax.set(title="Monte Carlo: el error decae como 1/√N", xlabel="N (muestras)", ylabel="|π̂ − π|"); ax.grid(True, which="both"); ax.legend()
plt.show()
print(f"π̂ con N=10⁶: {estimate_pi(1_000_000, rng):.5f}  (real: {np.pi:.5f})")
''')

M(r'''
**Qué deberías ver:** una recta de pendiente −1/2 en escala log-log: **para ganar un dígito decimal necesitas 100× más muestras**. Esa lentitud es el precio de Monte Carlo, y por eso se usa cuando no hay alternativa mejor.

#### Aplicación en Comunicaciones Digitales: BER de BPSK sobre AWGN
Un transmisor envía bits mapeados a símbolos $s=\pm\sqrt{E_b}$ (BPSK). El canal suma ruido gaussiano $n\sim\mathcal N(0,N_0/2)$ y el receptor decide por el **signo** de $r=s+n$. La teoría da
$$\text{BER}=Q\!\left(\sqrt{\tfrac{2E_b}{N_0}}\right)=\tfrac12\,\text{erfc}\!\left(\sqrt{\tfrac{E_b}{N_0}}\right)$$
Simulémoslo y comparemos con la fórmula. (Esto es exactamente un experimento de Monte Carlo sobre la ley del ruido gaussiano.)
''')

C(r'''
def simulate_bpsk_ber(ebn0_db, n_bits, rng):
    ebn0 = 10 ** (ebn0_db / 10)
    bits = rng.integers(0, 2, n_bits)
    s = 2 * bits - 1.0                                        # símbolos ±1 → Eb = 1
    sigma = np.sqrt(1 / (2 * ebn0))                           # σ² = N0/2 = 1/(2·Eb/N0)
    r = s + sigma * rng.normal(size=n_bits)
    return np.mean((r > 0) != bits.astype(bool))              # decisión por signo; BER = fracción de errores

def theory_ber_bpsk(ebn0_db):
    return 0.5 * special.erfc(np.sqrt(10 ** (np.asarray(ebn0_db) / 10)))

ebn0_grid = np.arange(0, 9)
ber_sim = np.array([simulate_bpsk_ber(e, 2_000_000, rng) for e in ebn0_grid])

fig, ax = plt.subplots()
ax.semilogy(ebn0_grid, ber_sim, "o", ms=8, label="simulación (2·10⁶ bits/punto)")
ax.semilogy(np.linspace(0, 8, 100), theory_ber_bpsk(np.linspace(0, 8, 100)), "r-", label="teoría ½·erfc(√(Eb/N0))")
ax.set(title="BER de BPSK en un canal AWGN", xlabel="Eb/N0 (dB)", ylabel="BER"); ax.grid(True, which="both"); ax.legend()
plt.show()
print("BER sim vs teoría:", [f"{e}dB: {s:.1e}/{theory_ber_bpsk(e):.1e}" for e, s in zip(ebn0_grid[[0, 4, 8]], ber_sim[[0, 4, 8]])])
''')

M(r'''
**Qué deberías ver:** la curva clásica "cascada" (*waterfall*): el BER cae casi **en línea recta en escala log** al subir $E_b/N_0$ (de ~$8\times10^{-2}$ a 0 dB hasta ~$2\times10^{-4}$ a 8 dB), y los puntos simulados caen sobre la curva teórica.

> **NOTA (precisión estadística):** para estimar un BER $p$ con error relativo del ~10 % necesitas observar unos **100 errores**, es decir, $\approx100/p$ bits. Para $p=10^{-6}$ son $10^8$ bits: ahí la simulación directa se vuelve costosa y se usan técnicas de *importance sampling*. Esto es relevante para tu tesis: predecir el BER con ML (a partir del diagrama de ojo) evita simular cientos de millones de bits.

### 6.4 Regla de Bayes: la paradoja del test positivo
AIMA cap. 12–13: $P(H\mid E)=\dfrac{P(E\mid H)\,P(H)}{P(E)}$. Con una enfermedad rara (1 %), un test con sensibilidad 95 % y especificidad 90 %, ¿cuál es la probabilidad de estar enfermo dado un test positivo? La intuición dice "~95 %"… pero Bayes dice otra cosa. Calculémoslo **exacto** y **simulado**:
''')

C(r'''
prev, sens, spec = 0.01, 0.95, 0.90

# Exacto (regla de Bayes con probabilidad total en el denominador)
p_pos = sens * prev + (1 - spec) * (1 - prev)
posterior = sens * prev / p_pos

# Simulación: 2 millones de personas
n_people = 2_000_000
sick = rng.random(n_people) < prev
positive = np.where(sick, rng.random(n_people) < sens, rng.random(n_people) < (1 - spec))
posterior_mc = sick[positive].mean()                        # entre los positivos, ¿qué fracción está enferma?

print(f"P(positivo) = {p_pos:.4f}")
print(f"P(enfermo | positivo)  exacto = {posterior:.4f} | simulado = {posterior_mc:.4f}")
print(f"Falsos positivos por cada verdadero positivo ≈ {(1 - posterior) / posterior:.1f}")
''')

M(r'''
**Qué deberías ver:** `P(enfermo | positivo) ≈ 0.088` (**menos del 9 %**, no 95 %) y la simulación coincide. La razón: hay tantos sanos que los falsos positivos (10 % de 99 %) superan con creces a los verdaderos positivos (95 % de 1 %). Es el mismo fenómeno que hace que, en detección de fraude o intrusos, **la tasa base importa más que la precisión del test** (conecta con precision/recall en NB-10).

### 6.5 Gaussiana multivariada, Cholesky y distancia de Mahalanobis
Para muestrear $\mathbf x\sim\mathcal N(\boldsymbol\mu,\Sigma)$: factoriza $\Sigma=LL^\top$ (**Cholesky**) y calcula $\mathbf x=\boldsymbol\mu+L\mathbf z$ con $\mathbf z\sim\mathcal N(0,I)$. La **distancia de Mahalanobis** $d^2=(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)$ mide distancia "en desviaciones estándar" **teniendo en cuenta la correlación**: dos puntos a igual distancia euclídea pueden ser uno normal y otro atípico. Bajo normalidad, $d^2\sim\chi^2_p$, lo que da un umbral natural para **detección de anomalías** (NB-15).
''')

C(r'''
mu_m = np.array([0.0, 0.0])
Sigma_m = np.array([[2.0, 1.6], [1.6, 2.0]])               # correlación fuerte (ρ = 0.8)
L = np.linalg.cholesky(Sigma_m)
Z = rng.normal(size=(5000, 2))
Xm = mu_m + Z @ L.T                                          # cada fila: μ + L z
print("covarianza muestral:\n", np.cov(Xm.T), "\n(objetivo Σ):\n", Sigma_m)

def mahalanobis_sq(X, mu, Sigma):
    D = X - mu
    return np.sum(D * np.linalg.solve(Sigma, D.T).T, axis=1)      # solve en vez de inv: más estable

d2 = mahalanobis_sq(Xm, mu_m, Sigma_m)
thr = stats.chi2.ppf(0.99, df=2)
test_pts = np.array([[2.0, 2.0], [2.0, -2.0]])             # ambos a igual distancia euclídea del origen (2.83)
print(f"\nUmbral χ²(2) al 99%: {thr:.2f} | fracción de puntos sobre el umbral: {(d2 > thr).mean():.3f} (≈ 0.01)")
print("d² de [2,2]:", mahalanobis_sq(test_pts, mu_m, Sigma_m).round(2), "← el primero es normal, el segundo es atípico")
''')

M(r'''
**Qué deberías ver:** la covarianza muestral ≈ Σ; ≈1 % de los puntos supera el umbral $\chi^2$ (como corresponde); y los dos puntos de prueba, **idénticos en distancia euclídea**, tienen $d^2$ muy distintos: `[2,2]` sigue la dirección de la correlación (normal) y `[2,−2]` va contra ella (atípico).

### 6.6 Entropía, KL y softmax estable
Tres funciones que te acompañarán en todo el ML (y que implementaremos "a mano" una vez para entenderlas):

$$H(p)=-\sum_i p_i\log p_i\quad\text{(incertidumbre)}\qquad D_{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}\qquad H(p,q)=H(p)+D_{KL}(p\|q)\;\text{(entropía cruzada)}$$

La **entropía cruzada** es la función de pérdida estándar de clasificación (NB-10, NB-16): minimizarla equivale a acercar la distribución predicha $q$ a la real $p$.
''')

C(r'''
def entropy(p):
    p = np.asarray(p, float)
    return -np.sum(special.xlogy(p, p))               # xlogy(0, 0) = 0: evita el NaN de 0·log(0)

def kl_div(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return np.sum(special.xlogy(p, p / q))

p = np.array([0.7, 0.2, 0.1, 0.0])                    # distribución "verdadera" (con una probabilidad exactamente 0)
q = np.array([0.4, 0.3, 0.2, 0.1])                    # lo que "predice" el modelo
ce = entropy(p) + kl_div(p, q)
print(f"H(p) = {entropy(p):.4f} | KL(p‖q) = {kl_div(p, q):.4f} | H(p,q) = {ce:.4f}")
print(f"Entropía cruzada directa: {-np.sum(special.xlogy(p, q)):.4f} (coincide)")
print(f"KL no es simétrica: KL(p‖q)={kl_div(p, q):.3f} vs KL(q‖p)={kl_div(q, p + 1e-12):.3f}")
print(f"Entropía máxima (uniforme de 4): {entropy(np.full(4, .25)):.4f} = ln 4 = {np.log(4):.4f}")
''')

M(r'''
**Qué deberías ver:** `H(p,q) = H(p) + KL(p‖q)` (la entropía cruzada siempre es ≥ la entropía; la diferencia es el KL, que nunca es negativo), KL **asimétrica**, y la entropía máxima para la distribución uniforme.

#### Softmax y `logsumexp`: el desborde clásico
La **softmax** convierte logits en probabilidades: $\text{softmax}(z)_i=e^{z_i}/\sum_j e^{z_j}$. Implementada literalmente, **desborda**: `np.exp(1000)` es `inf`, y `inf/inf` es `NaN`. La solución es restar el máximo (no cambia el resultado, porque se cancela en numerador y denominador).
''')

C(r'''
def softmax_naive(z):
    e = np.exp(z)
    return e / e.sum()

def softmax_stable(z):
    z = z - np.max(z, axis=-1, keepdims=True)         # el máximo pasa a 0 → exp ≤ 1, nunca desborda
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

z = np.array([1000.0, 1001.0, 1002.0])
print("ingenua :", softmax_naive(z))
print("estable :", softmax_stable(z))
print("scipy   :", special.softmax(z))

# Pérdida de entropía cruzada directamente desde logits (como hacen Keras/PyTorch con from_logits=True)
def cross_entropy_from_logits(z, y_idx):
    return special.logsumexp(z) - z[y_idx]            # -log softmax(z)[y] = logsumexp(z) - z[y], sin pasar por probabilidades

print(f"\nCE estable con logits grandes = {cross_entropy_from_logits(z, 2):.4f} (≈ 0.4076)")
print("CE ingenua:", -np.log(softmax_naive(z)[2]))
''')

M(r'''
**Qué deberías ver:** la versión ingenua da `[nan nan nan]` (con un `RuntimeWarning` que silenciamos arriba) y la estable da `[0.0900 0.2447 0.6652]`. Es también la razón por la que los *frameworks* ofrecen `from_logits=True`: calcular la pérdida directamente desde los logits con `logsumexp` es más estable que pasar por las probabilidades.

### ⚠️ Troubleshooting

**Síntoma:** `RuntimeWarning: overflow encountered in exp` / `invalid value encountered in divide` y `NaN` en la pérdida.
**Por qué:** `exp` de un número grande desborda; `inf/inf = NaN`. También `log(0) = -inf`.
**Solución:** usa `logsumexp`, `softmax` estable, `np.log1p`/`np.expm1` para valores pequeños y `xlogy` para $0\log0$; recorta probabilidades con `np.clip(p, 1e-12, 1)`.

**Síntoma:** resultados distintos en cada ejecución.
**Por qué:** no fijaste la semilla (o usaste `np.random` y `default_rng` mezclados).
**Solución:** un único `rng = np.random.default_rng(seed)` y pásalo a las funciones que lo necesiten.
''')

nb.tryit([
    "**6-1. Dado cargado:** simula 100.000 tiradas de un dado donde el 6 tiene probabilidad 0.25 y el resto reparten el resto por igual (`rng.choice(p=...)`). Compara el promedio con la esperanza teórica.",
    "**6-2. BER más alto:** extiende la simulación BPSK a 9–10 dB con 2·10⁶ bits. ¿Cuántos errores observas? ¿Cuántos bits necesitarías para ver 100 errores a 10 dB?",
    "**6-3. Cambiar la prevalencia:** repite el cálculo de Bayes con prevalencia 10 % y luego 50 %. ¿Cómo cambia el posterior? ¿Qué te dice sobre usar un test de screening en poblaciones de bajo riesgo?",
])
nb.solutions("6-1, 6-2 y 6-3", r'''
pm = np.array([0.15, 0.15, 0.15, 0.15, 0.15, 0.25])                              # 6-1
rolls_c = rng.choice(np.arange(1, 7), size=100_000, p=pm)
print(f"promedio = {rolls_c.mean():.3f} | E[X] = {np.sum(np.arange(1, 7) * pm):.3f}")
for e in (9, 10):                                                                  # 6-2
    b_ = simulate_bpsk_ber(e, 2_000_000, rng); print(f"{e} dB: BER sim = {b_:.1e} ({int(round(b_ * 2_000_000))} errores) | teoría {theory_ber_bpsk(e):.1e}")
print(f"bits para ~100 errores a 10 dB ≈ {100 / theory_ber_bpsk(10):.1e}")
for pv in (0.10, 0.50):                                                            # 6-3
    pp = sens * pv + (1 - spec) * (1 - pv); print(f"prevalencia {pv:.0%}: P(enfermo|+) = {sens * pv / pp:.3f}")
# Con prevalencia baja casi todos los positivos son falsos; con 50% el test sí es muy informativo.
''')

M(r'''
### 📊 Resumen visual — probabilidad en NumPy
```
 HERRAMIENTA                  PARA QUÉ                                   RECUERDA
 ──────────────────────────────────────────────────────────────────────────────────────────────
 rng = default_rng(seed)      muestreo reproducible                       exponential(scale=1/λ)
 scipy.stats.dist             pdf / cdf / ppf / fit                       frozen: stats.norm(mu, s)
 LLN / TCL                    promedios → valor esperado / campana         error ∝ σ/√n
 Monte Carlo                  estimar integrales y probabilidades          error ∝ 1/√N (100× muestras = +1 dígito)
 BER BPSK                     ½·erfc(√(Eb/N0))                             ~100 errores para 10% de error relativo
 Bayes                        posterior = verosimilitud·prior / evidencia  ¡la tasa base manda!
 Cholesky + Mahalanobis       muestrear gaussianas y detectar atípicos     d² ~ χ²(p) bajo normalidad
 H, KL, entropía cruzada      incertidumbre y pérdida de clasificación     xlogy y logsumexp evitan NaN
```
''')

# =====================================================================
# OPTIMIZACIÓN
# =====================================================================
M(r'''
## 7. Optimización numérica

### 7.1 Descenso de gradiente y el número de condición
**La intuición (la persona con los ojos vendados):** imagina que estás en una montaña con los ojos vendados y quieres llegar al valle. Solo puedes sentir la **pendiente bajo tus pies** (el gradiente). La estrategia: da un paso corto en la dirección que más baja, y repite. El tamaño del paso es la **tasa de aprendizaje** $\eta$.
$$\mathbf x_{k+1}=\mathbf x_k-\eta\,\nabla f(\mathbf x_k)$$

Para entenderlo bien, usamos la función cuadrática $f(\mathbf x)=\tfrac12\mathbf x^\top A\mathbf x-\mathbf b^\top\mathbf x$, cuyo gradiente es $A\mathbf x-\mathbf b$ y cuyo mínimo exacto es $A^{-1}\mathbf b$ (podemos comparar). Para una cuadrática, la velocidad de convergencia depende del **número de condición** $\kappa=\lambda_{\max}/\lambda_{\min}$ de $A$:
- un valle **redondo** ($\kappa\approx1$) → el gradiente apunta al mínimo → convergencia rápida;
- un valle **largo y estrecho** ($\kappa\gg1$) → el gradiente apunta a las paredes, no al fondo → trayectoria en zigzag y muy lenta.

Además hay un límite de estabilidad: si $\eta>2/\lambda_{\max}$ el algoritmo **diverge**.
''')

C(r'''
def gradient_descent(grad, x0, lr, n_iter):
    # Devuelve toda la trayectoria (n_iter+1 puntos) para poder graficarla
    path = [np.asarray(x0, float)]
    for _ in range(n_iter):
        path.append(path[-1] - lr * grad(path[-1]))
    return np.array(path)

def quad_problem(eigs, rotate=True):
    # Cuadrática con autovalores dados (opcionalmente rotada para que los ejes no coincidan con x1, x2)
    Q = np.array([[np.cos(.5), -np.sin(.5)], [np.sin(.5), np.cos(.5)]]) if rotate else np.eye(2)
    A = Q @ np.diag(eigs) @ Q.T
    b = np.array([1.0, 1.0])
    return A, b, np.linalg.solve(A, b)

def plot_quad(ax, A, b, path, title, xlim=(-1, 3.5), ylim=(-1, 3.5)):
    g = np.linspace(*xlim, 200), np.linspace(*ylim, 200)
    X1, X2 = np.meshgrid(*g)
    F = 0.5 * (A[0, 0] * X1 ** 2 + 2 * A[0, 1] * X1 * X2 + A[1, 1] * X2 ** 2) - b[0] * X1 - b[1] * X2
    ax.contour(X1, X2, F, levels=25, cmap="viridis", alpha=0.7)
    ax.plot(*path.T, "o-", color="#c0392b", ms=3, lw=1.2); ax.plot(*path[0], "ks"); ax.plot(*np.linalg.solve(A, b), "g*", ms=15)
    ax.set(title=title, xlim=xlim, ylim=ylim, xlabel="x₁", ylabel="x₂"); ax.set_aspect("equal")
''')

C(r'''
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
x0 = np.array([3.0, -0.5])
cases = [("κ = 2 (valle redondo)", [2.0, 1.0], 0.5, 30),
         ("κ = 50 (valle estrecho)", [5.0, 0.1], 0.35, 60),
         ("κ = 50 con η demasiado grande", [5.0, 0.1], 0.41, 20)]       # 2/λmax = 0.4 → 0.41 diverge
for ax, (name, eigs, lr, iters) in zip(axes, cases):
    A_, b_, xs = quad_problem(eigs)
    path = gradient_descent(lambda x: A_ @ x - b_, x0, lr, iters)
    plot_quad(ax, A_, b_, path, f"{name}\nη={lr} (límite 2/λmax={2/max(eigs):.2f}) | error final={np.linalg.norm(path[-1]-xs):.1e}",
              xlim=(-1.5, 4), ylim=(-3, 4))
plt.tight_layout(); plt.show()
''')

M(r'''
**Qué deberías ver:** (izquierda) una trayectoria que baja directo y llega a la estrella verde (el mínimo exacto); (centro) en el valle estrecho, un **zigzag** que avanza muy lento incluso después de 60 pasos; (derecha) con $\eta$ apenas por encima del límite, los pasos **crecen** en cada iteración y el algoritmo diverge (el error final es enorme).

La conclusión cuantitativa: con el mejor $\eta$ posible, el número de iteraciones necesarias crece **linealmente con $\kappa$**. Lo medimos:
''')

C(r'''
def iters_to_converge(eigs, tol=1e-6, max_iter=200_000):
    A_, b_, xs = quad_problem(eigs)
    lr = 2 / (max(eigs) + min(eigs))                    # η óptimo para una cuadrática
    x = np.array([3.0, -0.5])
    for k in range(max_iter):
        if np.linalg.norm(x - xs) < tol:
            return k
        x = x - lr * (A_ @ x - b_)
    return max_iter

kappas = np.array([1.5, 5, 10, 50, 100, 500])
its = np.array([iters_to_converge([k, 1.0]) for k in kappas])
fig, ax = plt.subplots()
ax.loglog(kappas, its, "o-", label="iteraciones medidas"); ax.loglog(kappas, 0.55 * kappas * np.log(1e6), "--", color="gray", label="≈ κ·ln(1/tol)")
ax.set(title="Iteraciones de GD hasta converger vs κ", xlabel="número de condición κ", ylabel="iteraciones"); ax.grid(True, which="both"); ax.legend(); plt.show()
print(dict(zip(kappas.tolist(), its.tolist())))
''')

M(r'''
**Qué deberías ver:** puntos que siguen casi paralelos a la recta $\propto\kappa$: **duplicar el condicionamiento duplica el costo**. Esta es la razón matemática de que **normalizar las features** (que equilibra las escalas y baja $\kappa$) acelere tanto el entrenamiento (NB-09, NB-11), y de que existan optimizadores adaptativos como Adam (NB-17) que "corrigen" las escalas por dirección.

### 7.2 Gradient checking: depurar el cálculo del gradiente
Cuando implementas backpropagation a mano (NB-16), un error de signo o un factor omitido **no produce un error**: la red simplemente no aprende. El antídoto es comparar tu gradiente analítico con uno **numérico** por diferencias finitas centrales:
$$\frac{\partial f}{\partial x_i}\approx\frac{f(\mathbf x+h\mathbf e_i)-f(\mathbf x-h\mathbf e_i)}{2h}\qquad(\text{error }O(h^2))$$
Lo probamos con la pérdida logística (regresión logística, NB-11), cuyo gradiente analítico conocemos.
''')

C(r'''
def numerical_gradient(f, w, h=1e-6):
    g = np.zeros_like(w)
    for i in range(len(w)):
        e = np.zeros_like(w); e[i] = h
        g[i] = (f(w + e) - f(w - e)) / (2 * h)       # diferencia central: mucho más precisa que la unilateral
    return g

Xlog = rng.normal(size=(200, 4)); ylog = np.where(Xlog @ np.array([1.0, -2.0, 0.5, 0.0]) + rng.normal(0, 0.5, 200) > 0, 1.0, -1.0)

def logistic_loss(w):                                  # etiquetas en {-1, +1}: L = mean log(1 + exp(-y wᵀx))
    return np.mean(np.logaddexp(0, -ylog * (Xlog @ w)))       # logaddexp(0, t) = log(1+e^t) sin desbordar

def logistic_grad(w):
    return -(Xlog * (ylog * special.expit(-ylog * (Xlog @ w)))[:, None]).mean(axis=0)   # expit = sigmoide estable

def logistic_grad_buggy(w):                            # error típico: olvidamos el signo menos
    return +(Xlog * (ylog * special.expit(-ylog * (Xlog @ w)))[:, None]).mean(axis=0)

w0 = rng.normal(size=4)
rel_err = lambda ga, gn: np.linalg.norm(ga - gn) / (np.linalg.norm(ga) + np.linalg.norm(gn))
gn = numerical_gradient(logistic_loss, w0)
print(f"gradiente correcto: error relativo = {rel_err(logistic_grad(w0), gn):.2e}   (regla: < 1e-6 → OK)")
print(f"gradiente con bug : error relativo = {rel_err(logistic_grad_buggy(w0), gn):.2e}   (≈ 1 → ¡hay un error!)")
''')

M(r'''
**Qué deberías ver:** un error relativo del orden de $10^{-9}$ o menor para el gradiente correcto, y cercano a **1** para el que tiene el signo invertido. Regla práctica: menos de $10^{-6}$ está bien; más de $10^{-3}$ casi seguro hay un bug. Úsalo **siempre** al implementar backprop desde cero (NB-16), y apágalo después (es $O(n)$ evaluaciones de la pérdida: carísimo).

### 7.3 Newton–Raphson
GD usa solo la pendiente (primer orden). **Newton** usa también la **curvatura** (la matriz Hessiana $H$) y modela la función localmente como una parábola, saltando directo a su mínimo.

- **Búsqueda de raíces** (1D): $x_{k+1}=x_k-\dfrac{f(x_k)}{f'(x_k)}$ (la recta tangente corta el eje).
- **Minimización** (nD): $\mathbf x_{k+1}=\mathbf x_k-H^{-1}\nabla f(\mathbf x_k)$ (se resuelve $H\Delta=-\nabla f$, no se invierte $H$).

Su virtud: **convergencia cuadrática** (los dígitos correctos se **duplican** en cada paso). Veámoslo calculando $\sqrt2$ como raíz de $f(x)=x^2-2$ y comparándolo con la bisección:
''')

C(r'''
def newton_root(f, df, x0, n=7):
    xs = [x0]
    for _ in range(n):
        xs.append(xs[-1] - f(xs[-1]) / df(xs[-1]))
    return np.array(xs)

def bisection(f, a, b, n=7):
    mids = []
    for _ in range(n):
        m = (a + b) / 2; mids.append(m)
        a, b = (a, m) if f(a) * f(m) <= 0 else (m, b)
    return np.array(mids)

fx, dfx = lambda x: x ** 2 - 2, lambda x: 2 * x
xs_newton = newton_root(fx, dfx, 1.0, n=6)
xs_bis = bisection(fx, 1.0, 2.0, n=7)
print(f"{'paso':>4} {'Newton':>20} {'error Newton':>14} {'error bisección':>16}")
for k in range(7):
    en = abs(xs_newton[k] - np.sqrt(2)) if k < len(xs_newton) else np.nan
    eb = abs(xs_bis[k] - np.sqrt(2)) if k < len(xs_bis) else np.nan
    print(f"{k:>4} {xs_newton[k] if k < len(xs_newton) else np.nan:>20.15f} {en:>14.2e} {eb:>16.2e}")
''')

M(r'''
**Qué deberías ver:** el error de Newton pasa por ~$10^{-1}\to10^{-3}\to10^{-6}\to10^{-12}\to$ precisión de máquina: **los exponentes se duplican** (convergencia cuadrática), mientras la bisección solo **reduce a la mitad el intervalo** en cada paso (convergencia lineal; su error puntual sube y baja, pero la cota se parte por 2).

Ahora Newton en varias dimensiones, sobre un problema difícil: la **función de Rosenbrock** $f(x,y)=(1-x)^2+100(y-x^2)^2$, un valle curvo y estrecho con mínimo en $(1,1)$, el banco de pruebas clásico de optimizadores. Comparamos GD, Newton (con una búsqueda de línea de retroceso para asegurar descenso) y BFGS (cuasi-Newton de SciPy).
''')

C(r'''
rosen = lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
rosen_grad = lambda p: np.array([-2 * (1 - p[0]) - 400 * p[0] * (p[1] - p[0] ** 2), 200 * (p[1] - p[0] ** 2)])
rosen_hess = lambda p: np.array([[2 - 400 * (p[1] - p[0] ** 2) + 800 * p[0] ** 2, -400 * p[0]], [-400 * p[0], 200.0]])

def newton_opt(f, g, H, x0, n=50, tol=1e-10):
    x = np.asarray(x0, float); path = [x.copy()]
    for _ in range(n):
        grad = g(x)
        if np.linalg.norm(grad) < tol:
            break
        step = np.linalg.solve(H(x), -grad)                 # resolver, no invertir
        if grad @ step >= 0:                                 # si H no es definida positiva, usar -grad (seguridad)
            step = -grad
        t = 1.0
        while f(x + t * step) > f(x) + 1e-4 * t * (grad @ step) and t > 1e-12:     # búsqueda de línea (Armijo)
            t *= 0.5
        x = x + t * step; path.append(x.copy())
    return np.array(path)

start = np.array([-1.2, 1.0])
path_gd = gradient_descent(rosen_grad, start, 1e-3, 5000)
path_newton = newton_opt(rosen, rosen_grad, rosen_hess, start)
res_bfgs = optimize.minimize(rosen, start, jac=rosen_grad, method="BFGS")

print(f"GD (5000 pasos, η=1e-3): x = {path_gd[-1].round(4)} | f = {rosen(path_gd[-1]):.2e}")
print(f"Newton ({len(path_newton)-1} pasos)        : x = {path_newton[-1].round(6)} | f = {rosen(path_newton[-1]):.2e}")
print(f"BFGS ({res_bfgs.nit} pasos)           : x = {res_bfgs.x.round(6)} | f = {res_bfgs.fun:.2e}")
''')

C(r'''
fig, ax = plt.subplots(figsize=(8, 5.5))
gx, gy = np.meshgrid(np.linspace(-1.6, 1.6, 300), np.linspace(-0.6, 1.8, 300))
ax.contour(gx, gy, (1 - gx) ** 2 + 100 * (gy - gx ** 2) ** 2, levels=np.logspace(-1, 3.5, 20), cmap="viridis", alpha=0.6)
ax.plot(*path_gd[::25].T, ".-", color="#c0392b", ms=4, lw=0.8, label=f"GD (5000 pasos)")
ax.plot(*path_newton.T, "o-", color="#2980b9", lw=2, label=f"Newton ({len(path_newton)-1} pasos)")
ax.plot(1, 1, "g*", ms=18, label="mínimo (1, 1)"); ax.plot(*start, "ks", label="inicio")
ax.set(title="Rosenbrock: GD vs Newton", xlabel="x", ylabel="y"); ax.grid(True); ax.legend(); plt.show()
''')

M(r'''
**Qué deberías ver:** GD (rojo) avanza lentísimo y todavía **no ha llegado** tras 5.000 pasos (se arrastra por el fondo curvo del valle); Newton (azul) y BFGS llegan al mínimo exacto en **decenas de pasos**. Newton aprovecha la curvatura: corrige la dirección según la forma del valle.

#### ¿Y por qué no usamos Newton en deep learning?
| | Gradiente descendente | Newton |
|---|---|---|
| Coste por paso | $O(n)$ | $O(n^2)$ memoria + $O(n^3)$ resolver (para $n=10^8$ parámetros: imposible) |
| Pasos necesarios | muchos | pocos (cuadrática cerca del óptimo) |
| Robustez | alta (funciona lejos del óptimo) | frágil: $H$ puede no ser definida positiva |
| Variantes prácticas | SGD, Momentum, **Adam** (NB-17) | **BFGS/L-BFGS** (cuasi-Newton; `scipy.optimize`, regresión logística en sklearn) |

> **NOTA (honestidad):** en problemas **pequeños y suaves** (regresión logística con pocas features) L-BFGS suele ser mejor que SGD, y de hecho es el solver por defecto de `LogisticRegression` en scikit-learn. En redes profundas con millones de parámetros y datos ruidosos, domina el gradiente estocástico.
''')

nb.tryit([
    "**7-1. Ajusta η:** en el caso κ = 50, prueba η = 0.05, 0.2 y 0.39. Grafica el error vs iteración (escala log) para las tres. ¿Cuál converge más rápido? ¿Coincide con $\\eta_{opt}=2/(\\lambda_{max}+\\lambda_{min})$?",
    "**7-2. Escalar features:** genera `X` con dos columnas, una en [0,1] y otra en [0,1000]; calcula $\\kappa(X^\\top X)$ antes y después de estandarizar (`(X-mean)/std`). Relaciónalo con la velocidad de GD.",
    "**7-3. Raíz cúbica con Newton:** usa `newton_root` para calcular $\\sqrt[3]{10}$ (raíz de $x^3-10$). ¿Cuántos pasos necesita para 12 dígitos?",
])
nb.solutions("7-1, 7-2 y 7-3", r'''
A_, b_, xs_ = quad_problem([5.0, 0.1])                                                    # 7-1
fig, ax = plt.subplots()
for lr in (0.05, 0.2, 0.39, 2 / 5.1):
    pth = gradient_descent(lambda x: A_ @ x - b_, x0, lr, 300)
    ax.semilogy(np.linalg.norm(pth - xs_, axis=1) + 1e-16, label=f"η={lr:.3f}")
ax.set(title="Error vs iteración (κ=50)", xlabel="iteración", ylabel="‖x - x*‖"); ax.grid(True, which="both"); ax.legend(); plt.show()
Xs = np.column_stack([rng.uniform(0, 1, 500), rng.uniform(0, 1000, 500)])                  # 7-2
Xz = (Xs - Xs.mean(0)) / Xs.std(0)
print(f"κ(XᵀX) sin escalar = {np.linalg.cond(Xs.T @ Xs):.1e} | estandarizado = {np.linalg.cond(Xz.T @ Xz):.1e}")
r3 = newton_root(lambda x: x ** 3 - 10, lambda x: 3 * x ** 2, 2.0, n=8)                    # 7-3
print("errores:", np.abs(r3 - 10 ** (1 / 3)))
print("→ en ~5 pasos se alcanza precisión de máquina (convergencia cuadrática)")
''')

M(r'''
### 📊 Resumen visual — optimización
```
   MÉTODO            USA                  COSTO/PASO    CONVERGENCIA           DEPENDE DE
 ─────────────────────────────────────────────────────────────────────────────────────────────
  Descenso grad.     ∇f                   O(n)          lineal (∝ κ·ln 1/ε)    κ = λmax/λmin, η < 2/λmax
  Newton             ∇f y Hessiana        O(n³)         cuadrática (cerca)     H def. positiva; línea de búsqueda
  BFGS / L-BFGS      ∇f (aprox. H)        O(n²)/O(n)    superlineal            problemas medianos y suaves
  SGD / Adam         ∇f en mini-lotes     O(batch)      ruidosa pero escalable  deep learning (NB-17)

   CHECKLIST ANTES DE ENTRENAR:  ✔ gradient check (<1e-6)   ✔ features escaladas (κ↓)   ✔ η < 2/λmax   ✔ pérdida sin NaN
```
''')

# =====================================================================
# NUMPY AVANZADO
# =====================================================================
M(r'''
## 8. NumPy avanzado

### 8.1 `einsum`: la notación de Einstein
**Por qué:** muchas operaciones de ML son sumas de productos con índices (`matmul`, trazas, productos por lotes, la atención de un Transformer). `np.einsum` las expresa con una **cadena de índices**: los índices repetidos se **suman**, los que aparecen a la derecha de `->` se **conservan**. Al principio parece críptico, pero evita transposiciones y temporales innecesarios.
''')

C(r'''
A_e, B_e = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
v1, v2 = rng.normal(size=4), rng.normal(size=4)

checks = {
    "producto matricial   'ij,jk->ik'": (np.einsum("ij,jk->ik", A_e, B_e), A_e @ B_e),
    "producto punto       'i,i->'    ": (np.einsum("i,i->", v1, v2), v1 @ v2),
    "producto externo     'i,j->ij'  ": (np.einsum("i,j->ij", v1, v2), np.outer(v1, v2)),
    "traza                'ii->'     ": (np.einsum("ii->", A_e[:3, :3]), np.trace(A_e[:3, :3])),
    "transpuesta          'ij->ji'   ": (np.einsum("ij->ji", A_e), A_e.T),
    "suma por columnas    'ij->j'    ": (np.einsum("ij->j", A_e), A_e.sum(axis=0)),
}
for name, (got, ref) in checks.items():
    print(name, "→", np.allclose(got, ref))

# Producto por lotes: 10 matrices (3x4) por 10 matrices (4x5) a la vez
Ab, Bb = rng.normal(size=(10, 3, 4)), rng.normal(size=(10, 4, 5))
print("lotes 'bij,bjk->bik' →", np.allclose(np.einsum("bij,bjk->bik", Ab, Bb), Ab @ Bb))
''')

M(r'''
**Qué deberías ver:** siete `True`. Lectura rápida de una cadena: en `'ij,jk->ik'` el índice `j` aparece en ambos operandos y **no** aparece a la derecha, luego se suma sobre él: es exactamente el producto matricial.

Un uso más cercano a IA Avanzada: las **puntuaciones de atención** (NB-20) $\text{scores}=QK^\top/\sqrt{d}$ para un lote de secuencias se escriben `'bqd,bkd->bqk'`:
''')

C(r'''
batch, seq, dim = 2, 5, 8
Qm, Km = rng.normal(size=(batch, seq, dim)), rng.normal(size=(batch, seq, dim))
scores = np.einsum("bqd,bkd->bqk", Qm, Km) / np.sqrt(dim)       # (lote, consulta, clave)
weights = special.softmax(scores, axis=-1)                       # cada consulta reparte atención entre las claves
print("scores:", scores.shape, "| ¿cada fila suma 1?", np.allclose(weights.sum(axis=-1), 1))
print("equivale a Q @ Kᵀ:", np.allclose(scores, (Qm @ Km.transpose(0, 2, 1)) / np.sqrt(dim)))
''')

M(r'''
**Qué deberías ver:** `scores: (2, 5, 5)`, y que cada fila de los pesos suma 1. Ya escribiste, sin saberlo, el corazón de un Transformer.

> **NOTA:** para operaciones con 3+ operandos, `np.einsum(..., optimize=True)` busca el mejor orden de contracción y puede acelerar órdenes de magnitud. Para un simple producto de dos matrices, `@` sigue siendo más rápido (usa BLAS).

### 8.2 Memoria, `strides` y *vistas*
Un array es un **bloque de memoria** más una *plantilla* (`shape` + `strides`: cuántos bytes saltar para avanzar una posición en cada eje). Muchas operaciones (transponer, rebanar, `reshape` sin copia) solo **cambian la plantilla**, sin mover datos. Eso las hace gratis, pero también significa que el orden de recorrido importa para la velocidad (la caché del CPU prefiere memoria contigua).
''')

C(r'''
a = np.arange(12).reshape(3, 4)
print("a      strides:", a.strides, "| C-contiguo:", a.flags["C_CONTIGUOUS"])
print("a.T    strides:", a.T.strides, "| ¿comparte memoria? ", np.shares_memory(a, a.T), "| F-contiguo:", a.T.flags["F_CONTIGUOUS"])

big = rng.normal(size=(4000, 4000))
big_T = big.T                                                    # vista: mismos datos, strides invertidos (se recorre "a saltos")
big_T_copy = np.ascontiguousarray(big.T)                         # copia contigua en memoria (cuesta una vez)

t_same = time_ms(lambda: big + big, 3)                           # ambos operandos se recorren en el mismo orden
t_mixed = time_ms(lambda: big + big_T, 3)                        # uno recorre filas y el otro columnas: la caché sufre
t_fixed = time_ms(lambda: big + big_T_copy, 3)                   # con la copia contigua se recupera la localidad
print(f"\nbig + big                      : {t_same:7.1f} ms")
print(f"big + big.T  (vista)           : {t_mixed:7.1f} ms   ← {t_mixed / t_same:.1f}× el tiempo: mismos datos, distinto orden de acceso")
print(f"big + ascontiguousarray(big.T) : {t_fixed:7.1f} ms   (sin contar el costo único de la copia)")
''')

M(r'''
**Qué deberías ver:** `a.T` comparte memoria con `a` (solo invirtió los strides: transponer es gratis). Pero **no todo es gratis**: sumar `big + big.T` recorre un operando por filas y el otro por columnas, y la caché del CPU lo penaliza (el factor exacto depende de tu máquina; lo normal es varias veces más tiempo que `big + big`). Si vas a usar esa vista muchas veces, pagar una copia contigua con `np.ascontiguousarray` suele compensar. *(Nota: `sum(axis=...)` no muestra esta diferencia porque NumPy la optimiza internamente para ambos ejes.)*

#### `sliding_window_view`: convolución y filtros sin bucles
Una **ventana deslizante** es una vista (sin copiar) de todos los segmentos consecutivos de longitud $k$ de una señal. Con ella, un **filtro FIR** (promedio móvil, convolución con la respuesta al impulso de un canal) es un producto matriz-vector. Es el vínculo directo con Comunicaciones Digitales: la salida de un canal con ISI es $y[n]=\sum_k h[k]\,x[n-k]$ (una convolución).
''')

C(r'''
x_sig = np.sin(2 * np.pi * 0.02 * np.arange(500)) + 0.4 * rng.normal(size=500)        # señal ruidosa
h_ma = np.ones(11) / 11                                                                # promedio móvil de 11 puntos

win = sliding_window_view(x_sig, 11)                       # forma (490, 11): cada fila = una ventana, SIN copiar datos
y_win = win @ h_ma[::-1]                                   # convolución = producto con h invertida (por definición)
y_conv = np.convolve(x_sig, h_ma, mode="valid")
y_fft = signal.fftconvolve(x_sig, h_ma, mode="valid")
print("ventana @ h == np.convolve == FFT:", np.allclose(y_win, y_conv), np.allclose(y_conv, y_fft))
print("win.base is x_sig (vista, no copia):", win.base is not None and np.shares_memory(win, x_sig))
''')

C(r'''
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_sig, color="lightgray", label="señal ruidosa")
ax.plot(np.arange(5, 495), y_conv, lw=2, label="promedio móvil (11 taps)")
ax.set(title="Filtro FIR de promedio móvil con sliding_window_view/np.convolve", xlabel="muestra n", ylabel="amplitud"); ax.grid(True); ax.legend()
plt.show()

# ¿Cuándo conviene FFT? Costo directo O(N·K) vs FFT O(N log N): gana cuando el filtro es largo
big_x = rng.normal(size=200_000)
rows = []
for K in (11, 101, 1001, 5001):
    hk = rng.normal(size=K)
    rows.append((K, time_ms(lambda: np.convolve(big_x, hk, "valid"), repeat=2), time_ms(lambda: signal.fftconvolve(big_x, hk, "valid"), repeat=2)))
print(f"{'taps':>6} {'np.convolve (ms)':>18} {'fftconvolve (ms)':>18}")
for K, t1, t2 in rows:
    print(f"{K:>6} {t1:>18.1f} {t2:>18.1f}")
''')

M(r'''
**Qué deberías ver:** la curva suave atravesando la señal ruidosa (el filtro elimina el ruido de alta frecuencia a costa de un pequeño retraso/recorte en los bordes por `mode="valid"`). En la tabla, `np.convolve` (costo $O(NK)$) gana con filtros cortos y la FFT (costo $O(N\log N)$) la supera ampliamente cuando el filtro tiene miles de coeficientes. En comunicaciones, un canal con ISI corta (3–10 taps) se simula con `np.convolve`; un canal de miles de taps (p. ej. acústico) con FFT.

### 8.3 Precisión numérica: los flotantes no son los reales
`float64` tiene ~16 dígitos; `float32` ~7. Tres trampas frecuentes y sus remedios:
''')

C(r'''
eps64, eps32 = np.finfo(np.float64).eps, np.finfo(np.float32).eps
print(f"ε float64 = {eps64:.2e} | ε float32 = {eps32:.2e}")
print("0.1 + 0.2 == 0.3 ?", 0.1 + 0.2 == 0.3, "| np.isclose:", np.isclose(0.1 + 0.2, 0.3), "| diferencia:", (0.1 + 0.2) - 0.3)

# 1) Cancelación catastrófica: (1 - cos x)/x² para x muy pequeño (valor real → 0.5)
x_small = 1e-8
naive = (1 - np.cos(x_small)) / x_small ** 2
stable = 2 * np.sin(x_small / 2) ** 2 / x_small ** 2          # identidad trigonométrica que evita restar números casi iguales
print(f"\n(1-cos x)/x² con x=1e-8 → ingenua: {naive}  | estable: {stable}  | real: 0.5")

# 2) log1p / expm1: precisión para argumentos pequeños
print(f"log(1+1e-12) = {np.log(1 + 1e-12):.6e} (impreciso) | log1p(1e-12) = {np.log1p(1e-12):.6e} (exacto: 1e-12)")
''')

C(r'''
# 3) Acumulación de errores en float32: sumar 10 millones de 0.1 (resultado exacto = 1.000.000)
v32 = np.full(10_000_000, 0.1, dtype=np.float32)
seq = np.cumsum(v32)[-1]                                       # cumsum acumula de forma secuencial → el error crece
pairwise = v32.sum()                                           # np.sum usa suma por pares → error mucho menor
exact = math.fsum(v32.astype(np.float64))                      # fsum: suma "exacta" de referencia
print(f"secuencial float32: {seq:,.1f} (error {abs(seq - exact):,.1f}) | por pares: {pairwise:,.1f} (error {abs(pairwise - exact):,.2f}) | fsum: {exact:,.1f}")
assert abs(pairwise - exact) < abs(seq - exact)
print("float32 también limita el tamaño: 16_777_217 como float32 =", np.float32(16_777_217), "(¡no representable!)")
''')

M(r'''
**Qué deberías ver:** (1) `0.1 + 0.2 != 0.3`, diferencia ~$5.5\times10^{-17}$: compara siempre con `np.isclose`; (2) la fórmula ingenua con $x=10^{-8}$ da **0** (¡todos los dígitos perdidos!) mientras la estable da 0.5; (3) la suma secuencial en `float32` se desvía de 1.000.000 en **decenas de miles** de unidades (un error de ~9 %), y la suma por pares de NumPy queda mucho más cerca. Esto último es relevante para **sistemas distribuidos**: como la suma de flotantes no es asociativa, el resultado de sumar en distinto orden (distintos nodos, distinto número de hilos) puede variar en los últimos dígitos, algo a tener presente al comparar resultados entre ejecuciones (reproducibilidad).

### 8.4 Rendimiento: lo que de verdad importa
Cuatro hábitos que dan la mayoría de las mejoras (mide antes de optimizar):
''')

C(r'''
n_big = 5_000_000
a_, b_, c_ = rng.normal(size=n_big), rng.normal(size=n_big), rng.normal(size=n_big)
tmp = np.empty(n_big)

def with_temporaries():  return a_ * b_ + c_                          # crea un temporal para a*b y otro para el resultado
def with_out():          np.multiply(a_, b_, out=tmp); np.add(tmp, c_, out=tmp); return tmp   # reutiliza un buffer

# np.vectorize NO vectoriza: es un bucle de Python con otra sintaxis
f_scalar = lambda v: v if v > 0 else 0.0
f_vec = np.vectorize(f_scalar)
small = a_[:200_000]

# float32 vs float64 en un producto matricial grande (la GPU y los modelos usan float32/float16)
M64 = rng.normal(size=(1500, 1500)); M32 = M64.astype(np.float32)

results = {
    "a*b+c (temporales)": time_ms(with_temporaries, 3),
    "a*b+c (out=, sin temporales)": time_ms(with_out, 3),
    "np.vectorize (200k elem.)": time_ms(lambda: f_vec(small), 2),
    "np.maximum(x, 0) (200k elem.)": time_ms(lambda: np.maximum(small, 0), 2),
    "matmul 1500² float64": time_ms(lambda: M64 @ M64, 3),
    "matmul 1500² float32": time_ms(lambda: M32 @ M32, 3),
}
for k, v in results.items():
    print(f"{k:34s} {v:9.2f} ms")
''')

M(r'''
**Qué deberías ver (los números exactos dependen de tu máquina):**
- `out=` suele ser algo más rápido que dejar que NumPy cree temporales (y usa menos memoria pico); la ganancia es modesta pero importa en arrays enormes y bucles internos.
- `np.vectorize` es **decenas o cientos de veces más lento** que la operación vectorizada real: su documentación lo dice, es solo *azúcar sintáctico* para un bucle.
- `float32` multiplica matrices típicamente **2–3× más rápido** que `float64` (la mitad de bytes y más SIMD por instrucción).

**Cuándo ir más allá de NumPy:** si tu cuello de botella es un bucle que no se puede vectorizar, **Numba** (`@njit`) compila Python a código máquina; **JAX/PyTorch** dan diferenciación automática y GPU con una API casi idéntica a NumPy (por eso lo aprendido aquí se transfiere a NB-16). Y si los datos no caben en RAM, Dask/Polars/Spark (ver NB-22).
''')

nb.tryit([
    "**8-1. einsum práctico:** escribe con `einsum` (a) la suma de todos los elementos de una matriz, (b) la norma de Frobenius al cuadrado, (c) la diagonal de `A @ B` **sin** calcular el producto completo. Verifica con NumPy.",
    "**8-2. Media móvil a mano:** implementa un promedio móvil de ventana `k` con `np.cumsum` (la diferencia de sumas acumuladas) y compáralo con `sliding_window_view(...).mean(axis=1)`. ¿Cuál es más rápido para `k` grande?",
    "**8-3. Varianza inestable:** calcula la varianza de `1e9 + rng.normal(size=10**6)` con la fórmula ingenua `mean(x²) - mean(x)²` y con `np.var`. ¿Qué observas? ¿Por qué?",
])
nb.solutions("8-1, 8-2 y 8-3", r'''
Ma, Mb = rng.normal(size=(5, 5)), rng.normal(size=(5, 5))                                           # 8-1
print(np.isclose(np.einsum("ij->", Ma), Ma.sum()), np.isclose(np.einsum("ij,ij->", Ma, Ma), np.linalg.norm(Ma) ** 2),
      np.allclose(np.einsum("ij,ji->i", Ma, Mb), np.diag(Ma @ Mb)))   # (c) es O(n²) en vez de O(n³)
sig = rng.normal(size=1_000_000); k_ = 5000                                                         # 8-2
def ma_cumsum(x, k):
    cs = np.cumsum(np.insert(x, 0, 0.0)); return (cs[k:] - cs[:-k]) / k
ref_ma = sliding_window_view(sig[:50_000], 500).mean(axis=1)
print("coinciden:", np.allclose(ma_cumsum(sig[:50_000], 500), ref_ma))
print(f"k={k_}: cumsum {time_ms(lambda: ma_cumsum(sig, k_), 2):.1f} ms | ventanas {time_ms(lambda: sliding_window_view(sig, k_).mean(axis=1), 1):.0f} ms (la ventana es O(N·k); cumsum es O(N))")
xv = 1e9 + rng.normal(size=10 ** 6)                                                                 # 8-3
print(f"ingenua: {np.mean(xv**2) - np.mean(xv)**2:.4f} | np.var: {np.var(xv):.4f} | real ≈ 1")
# La fórmula ingenua resta dos números ~1e18 que difieren en ~1: cancelación catastrófica (pierde todos los dígitos útiles).
''')

M(r'''
### 📊 Resumen visual — NumPy avanzado
```
 HERRAMIENTA            RESUELVE                                       RECUERDA
 ───────────────────────────────────────────────────────────────────────────────────────────────
 einsum                 sumas de productos con índices (matmul, atención)  índice repetido y ausente a la derecha → se suma
 strides / vistas       transponer/rebanar sin copiar                    modificar una vista modifica el original
 order='C'/'F'          localidad de memoria                             recorre el eje contiguo
 sliding_window_view    ventanas sin copiar (FIR, features locales)      costo O(N·k); FFT para k grande
 fftconvolve            convolución larga                                O(N log N)
 float32 / float64      memoria vs precisión                              ε=1.2e-7 / 2.2e-16; compara con isclose
 logsumexp, log1p, xlogy  estabilidad numérica                           evitan inf/NaN y cancelación
 out=, vectorizar       rendimiento                                      np.vectorize NO es vectorización
```
''')

# =====================================================================
# CHEAT SHEET
# =====================================================================
M(r'''
## 9. 📋 Cheat Sheet

| Concepto / Función | Qué hace | Cuándo usarlo | Sintaxis rápida |
|---|---|---|---|
| `a @ b`, `np.dot` | producto punto / matricial | siempre | `W @ x + b` |
| `np.linalg.norm` | norma de vector/matriz | distancias, regularización | `np.linalg.norm(x, ord=1)` · `axis=1` por filas |
| coseno | similitud por dirección | embeddings, texto | `Xn = X/np.linalg.norm(X,axis=1,keepdims=True); Xn@Xn.T` |
| `np.linalg.det`, `matrix_rank` | área/volumen y dimensión | ¿invertible? | `np.linalg.matrix_rank(A)` |
| `np.linalg.cond` | número de condición | diagnosticar inestabilidad | `np.linalg.cond(A)` (> 1e8: desconfía) |
| `np.linalg.solve` | resolver $A x=b$ | sistema cuadrado | `np.linalg.solve(A, b)` (nunca `inv(A)@b`) |
| `np.linalg.lstsq` | mínimos cuadrados | regresión, ajuste | `np.linalg.lstsq(X, y, rcond=None)[0]` |
| `np.linalg.qr` | descomposición QR | LS estable, ortonormalizar | `Q, R = np.linalg.qr(A)` |
| `np.linalg.eigh` / `eig` | autovalores (simétrica / general) | covarianza, PCA | `lam, V = np.linalg.eigh(Sigma)` (orden ascendente) |
| `np.linalg.svd` | SVD | rango, PCA, compresión | `U, s, Vt = np.linalg.svd(A, full_matrices=False)` |
| aprox. rango k | mejor aproximación de bajo rango | compresión | `(U[:,:k]*s[:k]) @ Vt[:k]` |
| `np.linalg.pinv` | pseudoinversa | sistemas rectangulares/singulares | `np.linalg.pinv(A) @ b` |
| `np.cov`, `np.corrcoef` | covarianza / correlación | features | `np.cov(X.T)` (variables en filas) |
| Cholesky | muestrear gaussianas | simulación | `L = np.linalg.cholesky(S); mu + Z @ L.T` |
| `rng = default_rng(s)` | generador reproducible | todo muestreo | `rng.normal(size=(n, d))` |
| `scipy.stats` | pdf/cdf/ppf/fit | probabilidad | `stats.norm(mu, s).cdf(x)` |
| `special.logsumexp`, `softmax` | log-suma-exp / softmax estables | clasificación, pérdidas | `special.softmax(z, axis=-1)` |
| `special.xlogy`, `expit` | $x\log y$ seguro / sigmoide | entropía, logística | `special.xlogy(p, p)` |
| gradient check | validar gradiente | al implementar backprop | `(f(w+h)-f(w-h))/(2h)` → error rel. < 1e-6 |
| `optimize.minimize` | optimizadores clásicos | problemas pequeños | `optimize.minimize(f, x0, jac=g, method="L-BFGS-B")` |
| `np.einsum` | contracciones con índices | lotes, atención | `np.einsum("bqd,bkd->bqk", Q, K)` |
| `sliding_window_view` | ventanas sin copia | FIR, features locales | `sliding_window_view(x, k) @ h[::-1]` |
| `signal.fftconvolve` | convolución rápida | filtros largos | `signal.fftconvolve(x, h, mode="valid")` |
| `np.isclose/allclose` | comparar flotantes | siempre (nunca `==`) | `np.allclose(a, b, atol=1e-8)` |
| `np.log1p`, `np.expm1` | exactas para argumentos pequeños | probabilidades pequeñas | `np.log1p(x)` |

**Regla de oro de estabilidad:** *si una fórmula resta números casi iguales, divide por algo casi cero, o hace `exp` de algo grande, reescríbela.*
''')

# =====================================================================
# MUNDO REAL
# =====================================================================
M(r'''
## 10. Conexiones con el Mundo Real

**En la industria.** El álgebra lineal numérica corre debajo de casi todo: motores de recomendación (factorización de matrices usuario–ítem ≈ SVD truncada), búsqueda semántica (similitud coseno entre *embeddings*), detección de anomalías (Mahalanobis), compresión y *denoising*, y el entrenamiento de todas las redes neuronales (multiplicaciones de matrices en GPU en `float32`/`bfloat16`). La estabilidad numérica no es un detalle académico: `softmax` ingenuas, `log(0)` y gradientes mal calculados son causas reales de *runs* de entrenamiento que mueren con `NaN`.

**Comunicaciones Digitales.** Todo este notebook es, en cierto sentido, el kit de herramientas de un ingeniero de comunicaciones: el **BER de BPSK** simulado en la sección 6.3 es la línea base de cualquier receptor; la convolución con la respuesta del canal modela la **ISI**; los **mínimos cuadrados** estiman el canal y diseñan ecualizadores (lo harás en el Ejercicio 3); la matriz de covarianza y la **transformada de Karhunen–Loève** (autovectores de la covarianza) son PCA en el dominio de señales; los métodos de Monte Carlo evalúan sistemas cuya teoría cerrada no existe.

**Sistemas Distribuidos.** Reproducibilidad numérica: la suma de flotantes no es asociativa, así que *reductions* distribuidas (all-reduce en entrenamiento multi-GPU) pueden diferir en los últimos bits según el orden. Por eso el entrenamiento distribuido exacto bit a bit exige cuidado explícito. Además, estimar con Monte Carlo en paralelo es *embarazosamente paralelo* (cada nodo simula con su propia semilla/`SeedSequence.spawn`).

**Tesis (SNN + diagrama de ojo + BER).**
- El **BER teórico y simulado** (sec. 6.3) define lo que tu modelo de ML debe aproximar; y la advertencia de las ~100 errores por punto explica **por qué** predecir el BER desde métricas del ojo vale la pena (evita simular $10^8$ bits).
- Las métricas de ojo de varias neuronas forman una matriz de características: su **covarianza y autovalores** (sec. 4.2) dirán cuántas dimensiones independientes tiene tu información (PCA, NB-14), y **Mahalanobis** (6.5) sirve para detectar simulaciones atípicas.
- El **ecualizador lineal por mínimos cuadrados** del Ejercicio 3 es la **línea base clásica** contra la que compararás tu ecualizador SNN.
- El **gradient checking** (7.2) es el seguro de vida cuando implementes *surrogate gradients* para SNN (NB-25): el gradiente de una neurona que dispara es una aproximación, y verificar la implementación numéricamente ahorra semanas de depuración.

**Contexto chileno / latinoamericano.** Ingeniería de telecomunicaciones (planificación de enlaces, evaluación de modulaciones con Monte Carlo), minería (PCA sobre sensores de planta para detección de fallas), astronomía (los grandes observatorios del norte de Chile procesan matrices enormes de imágenes con descomposiciones de bajo rango), y análisis financiero (matrices de covarianza de portafolios) usan exactamente estas herramientas.

**Limitaciones y alternativas.**
- NumPy corre en **una CPU** y en memoria; para matrices que no caben o necesitas GPU/autodiff, usa **PyTorch/JAX/CuPy** (misma idea, otra API; NB-16).
- Para matrices **dispersas** enormes (grafos, texto), `scipy.sparse` + `scipy.sparse.linalg.svds/eigsh` en vez de `np.linalg`.
- Newton y la SVD completa **no escalan** a millones de parámetros: se usan métodos iterativos/estocásticos y SVD aleatorizada (`sklearn.utils.extmath.randomized_svd`).
- Monte Carlo converge lento ($1/\sqrt N$): si hay estructura, usa *importance sampling* o cuasi-Monte Carlo.
''')

# =====================================================================
# EJERCICIOS
# =====================================================================
M(r'''
## 11. Ejercicios progresivos

Cada nivel tiene una celda **"Tu turno"** y, **después**, su solución completa en celdas separadas.

---
### Nivel 1 — Reproducir (guiado): PCA desde cero con autovalores y con SVD
**Objetivo:** reconstruir PCA con lo aprendido y comprobar que coincide con `sklearn.decomposition.PCA`. Usa el dataset **Iris** (`load_iris().data`, 150×4).

**Pasos:**
1. Centra los datos (resta la media de cada columna).
2. **Vía autovalores:** calcula la covarianza, sus autovectores (`eigh`) y ordénalos de mayor a menor varianza.
3. **Vía SVD:** aplica `np.linalg.svd` a la matriz centrada; recuerda que $\lambda_i=\sigma_i^2/(n-1)$ y que las filas de $V^\top$ son los componentes.
4. Proyecta a 2 componentes y verifica que ambos métodos coinciden con `PCA(2)` (salvo el signo de cada componente).
5. Grafica los datos proyectados coloreados por especie y reporta la varianza explicada.

> 💡 **Hints:** `PCA(n_components=2).fit(X)` expone `components_` y `explained_variance_`; compara con `np.abs(...)` para ignorar signos.
''')

C(r'''
# Tu turno (Nivel 1)
''')

M(r'''#### ✅ Solución — Nivel 1''')
C(r'''
iris = load_iris()
X_i, y_i = iris.data, iris.target
Xc_i = X_i - X_i.mean(axis=0)                                                    # 1) centrar

lam_e, V_e = np.linalg.eigh(np.cov(Xc_i.T))                                      # 2) vía autovalores
idx = np.argsort(lam_e)[::-1]; lam_e, V_e = lam_e[idx], V_e[:, idx]

U_s, s_s, Vt_s = np.linalg.svd(Xc_i, full_matrices=False)                        # 3) vía SVD
lam_svd = s_s ** 2 / (len(Xc_i) - 1)

sk = PCA(n_components=2).fit(X_i)
print("λ (eigh) :", lam_e.round(4)); print("λ (SVD)  :", lam_svd.round(4)); print("λ (sklearn, 2 comp.):", sk.explained_variance_.round(4))
assert np.allclose(lam_e, lam_svd) and np.allclose(lam_e[:2], sk.explained_variance_)
assert np.allclose(np.abs(V_e[:, :2].T), np.abs(sk.components_)) and np.allclose(np.abs(Vt_s[:2]), np.abs(sk.components_))

Z_i = Xc_i @ V_e[:, :2]                                                          # 4) proyección
assert np.allclose(np.abs(Z_i), np.abs(sk.transform(X_i)))
fig, ax = plt.subplots(figsize=(7, 5))
for c, name in enumerate(iris.target_names):
    ax.scatter(*Z_i[y_i == c].T, label=name, s=25)
ax.set(title=f"Iris en 2 componentes ({lam_e[:2].sum() / lam_e.sum():.1%} de la varianza)", xlabel=f"PC1 ({lam_e[0]/lam_e.sum():.1%})", ylabel=f"PC2 ({lam_e[1]/lam_e.sum():.1%})")
ax.grid(True); ax.legend(); plt.show()
print("✔ eigh, SVD y sklearn coinciden")
''')

M(r'''
**Qué deberías ver:** las tres rutas dan los mismos autovalores (~4.23, 0.24, 0.08, 0.02) y los mismos componentes (salvo signo). PC1 explica ≈ **92 %** de la varianza y la gráfica separa claramente *setosa* de las otras dos especies. Es la prueba de que PCA no es magia: es la SVD (o los autovectores de la covarianza) de los datos centrados. En NB-14 lo desarrollas completo (varianza explicada, reconstrucción, kernel PCA, t-SNE).

---
### Nivel 2 — Aplicar (problema nuevo): ajuste polinomial mal condicionado
**Problema:** quieres ajustar un polinomio a datos ruidosos de un sensor, probando grados de 1 a 15. Sospechas que resolver con la ecuación normal se vuelve inestable.

**Tu tarea (sin paso a paso):**
1. Genera 40 puntos $t\in[0,1]$ y, para cada grado de 1 a 15, la matriz de Vandermonde $A$ (base de potencias).
2. **Aísla el error numérico:** fija coeficientes verdaderos conocidos (`np.ones(grado+1)`), genera $y=A\,c_{true}$ **sin ruido** y recupera los coeficientes con (a) la **ecuación normal**, (b) **QR** y (c) **`lstsq`**. Reporta $\kappa(A)$, $\kappa(A^\top A)$ y el error relativo de los coeficientes.
3. Grafica error vs grado (escala log) y marca el primer grado donde la ecuación normal es ≥100× peor que `lstsq`.
4. Repite con la base de **Legendre** (`np.polynomial.legendre.legvander(2*t-1, grado)`) en lugar de potencias. ¿Qué pasa con $\kappa$ y con los errores?
5. Ahora con datos **ruidosos** ($y=\sin 2\pi t+\mathcal N(0,0.1^2)$), compara el error cuadrático medio de ajuste de los tres métodos en grado 12. ¿Se nota la diferencia? ¿Por qué el residuo puede engañarte?
6. Escribe 2–3 líneas con la lección práctica para ajustar modelos.
''')

C(r'''
# Tu turno (Nivel 2)
''')

M(r'''#### ✅ Solución — Nivel 2''')
C(r'''
rp = np.random.default_rng(3)
tp = np.sort(rp.uniform(0, 1, 40))
degrees = list(range(1, 16))

def solve_three(Av, y):
    out = {}
    try:
        out["normal"] = np.linalg.solve(Av.T @ Av, Av.T @ y)
    except np.linalg.LinAlgError:                       # κ(AᵀA) tan grande que la matriz es "singular" numéricamente
        out["normal"] = np.full(Av.shape[1], np.nan)
    Qp, Rp = np.linalg.qr(Av); out["QR"] = sla.solve_triangular(Rp, Qp.T @ y)
    out["lstsq"] = np.linalg.lstsq(Av, y, rcond=None)[0]
    return out

def coef_errors(basis_fn):
    errs = {"normal": [], "QR": [], "lstsq": []}; conds = []
    for deg in degrees:
        Av = basis_fn(deg); c_true = np.ones(deg + 1)
        sol = solve_three(Av, Av @ c_true)              # datos SIN ruido: todo error es numérico
        for k in errs:
            errs[k].append(np.linalg.norm(sol[k] - c_true) / np.linalg.norm(c_true))
        conds.append((np.linalg.cond(Av), np.linalg.cond(Av.T @ Av)))
    return errs, conds

pow_err, pow_cond = coef_errors(lambda d: np.vander(tp, d + 1, increasing=True))
leg_err, leg_cond = coef_errors(lambda d: np.polynomial.legendre.legvander(2 * tp - 1, d))

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
for ax, errs, name in [(axes[0], pow_err, "base de potencias"), (axes[1], leg_err, "base de Legendre")]:
    for k, c in zip(errs, ["#c0392b", "#2980b9", "#27ae60"]):
        ax.semilogy(degrees, np.maximum(errs[k], 1e-17), "o-", label=k, color=c, ms=5)
    ax.set(title=f"Error de coeficientes ({name})", xlabel="grado", ylabel="error relativo"); ax.grid(True, which="both"); ax.legend()
axes[2].semilogy(degrees, [c[0] for c in pow_cond], "o-", label="κ(A) potencias"); axes[2].semilogy(degrees, [c[1] for c in pow_cond], "s-", label="κ(AᵀA) potencias")
axes[2].semilogy(degrees, [c[0] for c in leg_cond], "^-", label="κ(A) Legendre")
axes[2].axhline(1 / np.finfo(float).eps, ls="--", color="gray", label="1/ε (sin dígitos)")
axes[2].set(title="Condicionamiento", xlabel="grado", ylabel="κ"); axes[2].grid(True, which="both"); axes[2].legend()
plt.tight_layout(); plt.show()

gap = next((d for d, a, b in zip(degrees, pow_err["normal"], pow_err["lstsq"]) if a > 100 * b), None)
print(f"Potencias: la ecuación normal es ≥100× peor que lstsq desde el grado {gap} | κ(A) grado 15: {pow_cond[-1][0]:.1e} (potencias) vs {leg_cond[-1][0]:.1e} (Legendre)")

# Paso 5: con datos ruidosos el RESIDUO apenas distingue los métodos, aunque los coeficientes sean muy distintos
yn = np.sin(2 * np.pi * tp) + rp.normal(0, 0.1, 40); A12 = np.vander(tp, 13, increasing=True)
sol_n = solve_three(A12, yn)
print("MSE de ajuste (grado 12):", {k: f"{np.mean((A12 @ c - yn) ** 2):.2e}" for k, c in sol_n.items()},
      "\n‖c_normal − c_lstsq‖/‖c_lstsq‖ =", f"{np.linalg.norm(sol_n['normal'] - sol_n['lstsq']) / np.linalg.norm(sol_n['lstsq']):.1e}")
assert gap is not None and leg_cond[-1][0] < 1e3 and pow_cond[-1][1] > 1e14
''')

M(r'''
**Qué deberías ver:** con la base de **potencias**, el error de los coeficientes de la ecuación normal **se dispara** (llega a ser del orden de la propia solución) mucho antes que el de QR/`lstsq`, porque $\kappa(A^\top A)\approx\kappa(A)^2$ supera $1/\epsilon$; con la base de **Legendre** (ortogonal en $[-1,1]$) $\kappa$ queda en decenas y los tres métodos funcionan: **la elección de la base importa tanto como el algoritmo**. En el paso 5 el MSE de ajuste de los tres métodos es casi idéntico aunque sus coeficientes difieran muchísimo: un residuo pequeño **no** garantiza coeficientes fiables (importa si vas a interpretarlos o extrapolar). Lección: usa `lstsq`/QR, escoge bases bien condicionadas y regulariza; y recuerda que un MSE de entrenamiento muy bajo con grado alto también puede ser **sobreajuste** (NB-11).

---
### Nivel 3 — Integrar (Comunicaciones Digitales + Tesis): estimación de canal y ecualizador por mínimos cuadrados
**Contexto:** transmites símbolos BPSK por un canal con **interferencia entre símbolos** (ISI): $y[n]=\sum_k h[k]\,x[n-k]+w[n]$. No conoces $h$. Este es el escenario de tu tesis, y el **ecualizador lineal MMSE/LS** es la línea base clásica contra la que se compararía un ecualizador SNN.

**Tu tarea:**
1. Simula un canal $h=[0.9,\,0.45,\,0.25]$ y ruido AWGN a $E_b/N_0$ = 12 dB (nota: usa $\sigma^2=\tfrac{\,\sum h^2\,}{2\,E_b/N_0}$ para fijar la SNR **en el receptor**).
2. **Estima el canal** con una secuencia de entrenamiento de 200 símbolos mediante mínimos cuadrados: arma la matriz de convolución con `sliding_window_view` y resuelve `lstsq`. Compara $\hat h$ con $h$.
3. **Diseña un ecualizador FIR** de 9 coeficientes que minimice $\sum(x[n-d]-\mathbf w^\top\mathbf y_n)^2$ sobre otra secuencia de entrenamiento (con retardo $d$ elegido por ti: prueba varios y quédate con el mejor).
4. Compara el **BER** (Monte Carlo con ≥ 200.000 bits) de: (a) detección directa sin ecualizar, (b) con tu ecualizador, (c) el límite de un canal sin ISI (teoría BPSK).
5. Conecta con la tesis: ¿qué limita a este ecualizador lineal y qué podría aportar uno no lineal (SNN)?

> 💡 **Hints:** la salida del canal es `np.convolve(x, h)[:len(x)]`; para el ecualizador, las filas de `sliding_window_view(y, L)` son las entradas $\mathbf y_n$ (recuerda alinear con el retardo $d$ para el objetivo).
''')

C(r'''
# Tu turno (Nivel 3)
''')

M(r'''#### ✅ Solución — Nivel 3''')
C(r'''
rc = np.random.default_rng(21)
h_true = np.array([0.9, 0.45, 0.25])
ebn0_db = 12.0
sigma2 = np.sum(h_true ** 2) / (2 * 10 ** (ebn0_db / 10))          # 1) potencia de ruido fijada por la SNR a la salida del canal

def channel(x, rng):
    return np.convolve(x, h_true)[:len(x)] + np.sqrt(sigma2) * rng.normal(size=len(x))

def bpsk(n, rng):
    return 2 * rng.integers(0, 2, n) - 1.0

# 2) Estimación del canal por mínimos cuadrados: y[n] = Σ_k h[k] x[n-k]  →  y = X h con X formada por ventanas de x
n_tr = 200
x_tr = bpsk(n_tr, rc); y_tr = channel(x_tr, rc)
n_taps = 3
x_pad = np.concatenate([np.zeros(n_taps - 1), x_tr])               # relleno inicial: el canal "arranca" con memoria vacía
X_win = sliding_window_view(x_pad, n_taps)[:, ::-1]                # cada fila: [x[n], x[n-1], x[n-2]]
h_hat = np.linalg.lstsq(X_win, y_tr, rcond=None)[0]
print("h real :", h_true, "\nh estim:", h_hat.round(3), f"| error relativo = {np.linalg.norm(h_hat - h_true)/np.linalg.norm(h_true):.3%}")
''')

C(r'''
# 3) Ecualizador FIR de L coeficientes: w minimiza Σ (x[n-d] - wᵀ y_n)², con y_n = [y[n], y[n-1], ..., y[n-L+1]]
L_eq = 9
def design_equalizer(delay, rng, n_train=20_000):
    x = bpsk(n_train, rng); y = channel(x, rng)
    Y = sliding_window_view(np.concatenate([np.zeros(L_eq - 1), y]), L_eq)[:, ::-1]     # filas: ventanas de y (más reciente primero)
    target = np.concatenate([np.zeros(delay), x])[:n_train]                              # x[n-d]
    w = np.linalg.lstsq(Y, target, rcond=None)[0]
    return w, np.mean((Y @ w - target) ** 2)

mse_by_delay = {d: design_equalizer(d, rc)[1] for d in range(0, 8)}
best_d = min(mse_by_delay, key=mse_by_delay.get)
w_eq, _ = design_equalizer(best_d, rc)
print("MSE por retardo:", {d: round(v, 4) for d, v in mse_by_delay.items()}, f"\n→ mejor retardo d = {best_d}")

# 4) BER Monte Carlo
def ber_compare(n_bits, rng):
    x = bpsk(n_bits, rng); y = channel(x, rng)
    ber_raw = np.mean(np.sign(y) != x)                                                  # (a) sin ecualizar
    Y = sliding_window_view(np.concatenate([np.zeros(L_eq - 1), y]), L_eq)[:, ::-1]
    z = Y @ w_eq
    x_del = np.concatenate([np.zeros(best_d), x])[:n_bits]
    ber_eq = np.mean(np.sign(z)[best_d:] != x_del[best_d:])                              # (b) ecualizado (ignora transitorio inicial)
    return ber_raw, ber_eq

n_ber = 2_000_000
ber_raw, ber_eq = ber_compare(n_ber, rc)
print(f"errores observados: sin ecualizar ≈ {int(round(ber_raw * n_ber))} | ecualizado ≈ {int(round(ber_eq * n_ber))} de {n_ber:,} bits")
ber_ideal = 0.5 * special.erfc(np.sqrt(10 ** (ebn0_db / 10)))                            # (c) canal sin ISI a la misma Eb/N0 (teoría)
print(f"\nBER sin ecualizar : {ber_raw:.2e}\nBER ecualizado    : {ber_eq:.2e}\nBER ideal (sin ISI): {ber_ideal:.2e}")
assert ber_eq < ber_raw / 2, "el ecualizador debería mejorar sustancialmente el BER"
''')

C(r'''
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
taps = np.arange(L_eq)
axes[0].stem(taps, w_eq); axes[0].set(title=f"Ecualizador FIR (L={L_eq}, retardo d={best_d})", xlabel="coeficiente", ylabel="valor"); axes[0].grid(True)
conv_total = np.convolve(h_true, w_eq)
axes[1].stem(np.arange(len(conv_total)), conv_total); axes[1].axvline(best_d, color="r", ls="--", label=f"posición deseada (d={best_d})")
axes[1].set(title="Respuesta combinada canal * ecualizador (ideal: un único pulso)", xlabel="n", ylabel="amplitud"); axes[1].grid(True); axes[1].legend()
plt.tight_layout(); plt.show()
''')

M(r'''
**Qué deberías ver:**
- **Estimación de canal:** $\hat h$ muy cercano a $[0.9,0.45,0.25]$ con error relativo de pocos %.
- **Retardo:** el MSE por retardo es casi **plano** para $d=0$–$4$ (el canal tiene su coeficiente dominante al inicio y el ecualizador es largo) y **empeora** para retardos mayores, cuando la información de $x[n-d]$ queda en el borde de la ventana. Por eso el "mejor" $d$ dentro de la zona plana es en gran parte ruido estadístico: cualquier $d$ pequeño sirve, pero elegir uno grande sí cuesta.
- **BER:** sin ecualizar el BER es del orden de $3\times10^{-2}$ (la ISI "come" el margen de ruido); ecualizado baja **más de tres órdenes de magnitud** (a ~$10^{-5}$, con solo unas decenas de errores observados, así que su valor exacto es incierto: ver la regla de ~100 errores de la sec. 6.3). Sigue **por encima** del límite sin ISI (~$10^{-8}$) porque el ecualizador lineal **amplifica ruido** y deja algo de ISI residual (el compromiso clásico *zero-forcing* vs *MMSE*).
- **Figura derecha:** la convolución $h * w$ debería parecerse a un **pulso único** en la posición $d$ (las colas residuales son ISI no eliminada).

**Conexión con la tesis:** un ecualizador **lineal** no puede resolver distorsiones **no lineales** (saturación de amplificadores, ópticas con detección cuadrática en un enlace óptico) ni explotar la estructura temporal con memoria rica. Ahí entra un ecualizador SNN (no lineal, con dinámica temporal): este ecualizador de mínimos cuadrados es la **línea base** (y el estimador de canal, parte del preprocesamiento) con la que justificarás cualquier mejora. Las **métricas de ojo** de la salida de cada neurona serán las *features* para predecir el BER (NB-25), y todo lo de esta sección (convolución, Monte Carlo, mínimos cuadrados) es el motor para generar esos datos.
''')

# =====================================================================
# CONCLUSIÓN
# =====================================================================
M(r'''
## 12. Resumen

1. Una **matriz es una transformación**: su determinante, rango, autovalores y SVD describen cómo deforma el espacio; el producto punto da ángulos y proyecciones, y **mínimos cuadrados es proyectar**.
2. Los **autovectores** de una covarianza son los ejes principales de los datos (PCA) y la **SVD** generaliza todo: da rango, condicionamiento, mejor aproximación de bajo rango y regularización (Ridge).
3. El **número de condición** manda en el cómputo numérico: determina cuántos dígitos pierdes (`lstsq`/QR > ecuación normal) y cuán rápido converge el descenso de gradiente (por eso se normalizan las features).
4. Con NumPy/SciPy simulas **probabilidad** (LLN, TCL, Monte Carlo, BER de BPSK, Bayes, Mahalanobis) y usas funciones **estables** (`logsumexp`, `log1p`, `xlogy`); *verifica gradientes* con diferencias finitas antes de confiar en ellos.
5. `einsum`, vistas/strides, `sliding_window_view` y FFT te dan código rápido y expresivo; y nunca compares flotantes con `==`.

## Siguiente paso
→ **Continúa en:** **NB-04 — Búsqueda en Espacios de Estados** (AIMA cap. 3–4): primer notebook de IA clásica; usa normas/distancias como heurísticas y la idea de descenso/ascenso local de la sección 7. Más adelante: **NB-08** (probabilidad, redes bayesianas y HMMs, retoma 6.4 y 4.3), **NB-11** (regresión, gradiente y regularización: retoma 3.3, 5.3 y 7.1–7.2) y **NB-14** (PCA completo: retoma 4.2 y 5.1).

## Para profundizar
- *VanderPlas, Python Data Science Handbook*, cap. 2: *Computation on Arrays: Broadcasting*, *Fancy Indexing*, *Structured Data*.
- *Géron, Hands-On ML 3.ª ed.*: cap. 4 (*Normal Equation*, *Gradient Descent* y el efecto de escalar las features) y cap. 8 (*PCA*, *SVD*).
- *AIMA 4.ª ed.*: cap. 12 (*Quantifying Uncertainty*, especialmente la regla de Bayes), cap. 13 (*Probabilistic Reasoning*) y apéndice A (*Notes on Languages and Algorithms*).
- *Goodfellow et al., Deep Learning*: cap. 2 (*Linear Algebra*), cap. 3 (*Probability and Information Theory*) y cap. 4 (*Numerical Computation*: overflow/underflow, condicionamiento, optimización basada en gradiente y métodos de segundo orden).
- *Trefethen & Bau, Numerical Linear Algebra* (lectura de referencia sobre SVD, QR y estabilidad).
- *Proakis & Salehi, Digital Communications* (BER de BPSK y ecualización lineal), para la parte de Comunicaciones Digitales.
''')

nb.build("NB-03_numpy_algebra_lineal.ipynb")
