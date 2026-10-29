import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbkit import NB

nb = NB("02", "Pandas + MongoDB para Big Data")
M, C, NOTE = nb.md, nb.code, nb.note

# =====================================================================
# ENCABEZADO
# =====================================================================
M(r'''
# NB-02: Pandas + MongoDB para Big Data
**Bloque:** 1 — Fundamentos de Datos y Cómputo
**Base bibliográfica:** VanderPlas, *Python Data Science Handbook* (cap. 3, pandas) · Kleppmann, *DDIA* (cap. 2: modelos de datos y lenguajes de consulta; cap. 5–6: replicación y particionamiento) · documentación oficial de MongoDB y PyMongo
**Prerequisitos:** NB-01 (DataFrames, `groupby`, `merge`, `pivot_table`, `.str`/`.dt`, `pipe`). Aquí **no** repetimos eso: vamos a lo que viene después.

## Objetivos de Aprendizaje
Al completar este notebook serás capaz de:
1. **Optimizar** un DataFrame (tipos, memoria) y elegir la forma más rápida de calcular algo (vectorizado vs `apply` vs bucle).
2. **Analizar series de tiempo** con ventanas móviles (`rolling`, `expanding`, `ewm`), `resample`, `merge_asof` y análisis de cohortes.
3. **Diseñar un pipeline ETL** reutilizable, idempotente y con validaciones y cuarentena de filas inválidas.
4. **Usar MongoDB desde Python** (PyMongo): CRUD, operadores de consulta, índices y *aggregation pipeline*.
5. **Decidir** cuándo modelar con documentos embebidos o referencias, y cuándo calcular en pandas o dentro de la base de datos.

## Mapa de Conexiones
| Curso / Área | Relevancia |
|---|---|
| IA (CIT-2013) | Preparar y versionar datos antes de entrenar; los datos limpios y bien tipados son el 80 % del trabajo (ver NB-09). |
| **Big Data (CIT-2309)** | **Referencia directa del proyecto**: ETL con pandas + almacenamiento en MongoDB + análisis con aggregation pipeline. |
| Comunicaciones Digitales | Series de tiempo con ventanas móviles (promedios, suavizado exponencial) ≈ filtros FIR/IIR sobre una señal. |
| Sistemas Distribuidos | MongoDB: réplicas, *sharding*, consistencia (Kleppmann cap. 5–6). Idempotencia en cargas = tolerancia a reintentos. |
| Tesis SNN/Ojo | Guardar cientos de simulaciones (métricas de ojo + BER + parámetros) como documentos de esquema flexible (Ejercicio 3). |
| Notebooks relacionados | NB-01 (base), NB-09 (proyecto ML end-to-end), NB-22 (pipelines y Airflow) |

## Tabla de contenidos
1. Marco teórico: ¿por qué pandas **y** MongoDB?
2. Configuración
3. Los datos: un e-commerce chileno simulado
4. pandas avanzado: memoria, MultiIndex, ventanas, series de tiempo, joins, cohortes, rendimiento
5. ETL: diseño de un pipeline robusto
6. MongoDB con PyMongo: CRUD, índices, aggregation pipeline, modelado
7. Pipeline completo pandas → MongoDB → pandas
8. 📋 Cheat Sheet
9. Conexiones con el mundo real
10. Ejercicios progresivos
11. Resumen y siguiente paso
''')

# =====================================================================
# MARCO TEÓRICO
# =====================================================================
M(r'''
## 1. Marco teórico: ¿por qué pandas **y** MongoDB?

### La intuición: una libreta y un archivador
Piensa en pandas como **la libreta de trabajo** sobre tu escritorio: rápida, flexible, perfecta para explorar y transformar, pero tiene que caber en tu escritorio (la RAM) y cuando cierras el cuaderno (el kernel), lo que no guardaste se pierde. MongoDB es **el archivador de la oficina**: guarda mucho más, lo comparten varias personas a la vez, sobrevive a que apagues el computador y sabe encontrar una carpeta sin que tengas que revisarlas todas (índices).

Un flujo típico de Big Data usa ambos: los datos crudos llegan, **pandas los limpia** (transformación), **MongoDB los almacena** (carga) y luego se consultan, ya sea con el *aggregation pipeline* dentro de la base de datos o trayendo un subconjunto a pandas para analizar o entrenar modelos.

```
  Fuentes             EXTRACT            TRANSFORM              LOAD              CONSUMO
 ┌────────┐        ┌──────────┐       ┌───────────────┐     ┌───────────┐     ┌──────────────┐
 │ CSV    │        │ pandas   │       │ limpiar       │     │ MongoDB   │     │ Aggregation  │
 │ APIs   │  ───►  │ read_*   │ ───►  │ validar       │ ──► │ (upsert)  │ ──► │ pandas / ML  │
 │ logs   │        │ chunks   │       │ enriquecer    │     │ + índices │     │ dashboards   │
 └────────┘        └──────────┘       └──────┬────────┘     └───────────┘     └──────────────┘
                                             ▼
                                      cuarentena (filas rechazadas + motivo)
```

### Modelo de documentos vs modelo relacional (Kleppmann, cap. 2)
En SQL cada pedido vive en varias tablas (`orders`, `order_items`, `customers`) y se reconstruye con `JOIN`. En MongoDB puedes guardar **un pedido completo como un único documento** (JSON/BSON) con sus ítems **embebidos**:

```json
{ "_id": "O-100", "customer": {"id": "C1", "region": "Biobío"},
  "items": [ {"sku": "SKU-7", "qty": 2, "unit_price": 9990}, {"sku": "SKU-3", "qty": 1, "unit_price": 4990} ],
  "total": 24970, "status": "entregado" }
```

| | Relacional (SQL) | Documentos (MongoDB) |
|---|---|---|
| Unidad | fila en una tabla normalizada | documento anidado (JSON/BSON) |
| Esquema | fijo, declarado de antemano | flexible (*schema-on-read*) |
| Relaciones | `JOIN` | embeber (localidad) o referenciar (`$lookup`) |
| Fortaleza | integridad, consultas ad-hoc complejas | leer una entidad completa de una sola vez, datos heterogéneos, escalado horizontal |
| Debilidad | joins caros a gran escala | duplicación de datos (*denormalización*), joins limitados |

**Regla de diseño práctica** — *embebe* cuando los datos se leen **juntos** y el hijo no tiene vida propia (ítems de un pedido); *referencia* cuando la entidad se comparte o crece sin límite (clientes, productos, o los comentarios de un post famoso).

### Escalar: réplicas y *sharding* (Kleppmann cap. 5–6, ver NB-22)
- **Replica set:** copias del mismo dato en varios nodos (alta disponibilidad: si cae el primario, se elige otro).
- **Sharding:** partir la colección entre nodos según una **shard key** (ej. `customer.id` con hash). Una mala shard key crea *hot spots* (un nodo recibe casi todo el tráfico).
- En este notebook trabajamos con un solo nodo; entender estas ideas importa porque condicionan cómo diseñas el documento y los índices.

> **Honestidad:** MongoDB no es "mejor" que SQL; es una herramienta distinta. Para datos muy relacionales con transacciones multi-tabla, PostgreSQL suele ser más cómodo. Para analítica sobre cientos de GB, ni pandas ni MongoDB son lo ideal: se usan Parquet + DuckDB/Spark (ver NB-22).
''')

# =====================================================================
# CONFIGURACIÓN
# =====================================================================
M(r'''
## 2. Configuración
Vamos a preparar el entorno. Dos detalles importantes:

1. **Conexión a MongoDB.** El helper `get_db()` intenta conectarse a un servidor MongoDB real (`MONGODB_URI` o `localhost:27017`). Si no hay ninguno, usa **`mongomock`**, un simulador en memoria que implementa la misma API de PyMongo. El código que escribas aquí funciona **idéntico** contra un MongoDB real: solo cambia el cliente.
2. Usamos `rng = np.random.default_rng(42)` para que los datos simulados sean idénticos en cada ejecución.
''')

C(r'''
# === Imports ===
# Estándar
import os, sys, time, json, logging, tempfile, warnings
from dataclasses import dataclass, field
from pathlib import Path
warnings.filterwarnings("ignore")

# Datos y cómputo
import numpy as np
import pandas as pd
from scipy.special import erfc   # solo para el Ejercicio 3 (relación SNR ↔ BER)

# Bases de datos
import bson                               # BSON: el formato binario de MongoDB (viene con pymongo)
import pymongo
from pymongo import MongoClient, ReplaceOne, UpdateOne
from pymongo.errors import DuplicateKeyError

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# === Configuración ===
SEED = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)      # generador moderno: se pasa explícitamente a las funciones
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 11
sns.set_style("whitegrid")
pd.set_option("display.max_columns", 30)
pd.set_option("display.precision", 2)

print("✅ Entorno listo")
print(f"pandas {pd.__version__} | numpy {np.__version__} | pymongo {pymongo.version}")
''')

C(r'''
def get_db(db_name: str = "nb02_ecommerce", uri: str | None = None):
    # Devuelve (db, backend). Intenta un servidor real; si no hay, cae a mongomock.
    uri = uri or os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=1500)
        client.admin.command("ping")          # fuerza la conexión: MongoClient es "perezoso"
        backend = f"MongoDB real ({uri})"
    except Exception:
        import mongomock
        client = mongomock.MongoClient()
        backend = "mongomock (simulador en memoria, misma API que PyMongo)"
    return client[db_name], backend

db, BACKEND = get_db()
for name in db.list_collection_names():      # partir siempre de cero hace el notebook reproducible
    db.drop_collection(name)
print("Backend:", BACKEND)
''')

NOTE(r'''En un proyecto real **nunca** escribas credenciales dentro del código. Usa variables de entorno (`MONGODB_URI`)
o un archivo `.env` fuera del control de versiones. Para MongoDB Atlas la URI tiene la forma
`mongodb+srv://usuario:clave@cluster.mongodb.net/`.''')

M(r'''
### ⚠️ Troubleshooting: la conexión

**Error:** `ServerSelectionTimeoutError: localhost:27017: [Errno 61] Connection refused`
**Por qué:** `MongoClient(...)` no se conecta al crearse; falla recién en la primera operación, y solo si no hay servidor escuchando.
**Solución:** levanta el servidor (`brew services start mongodb-community` en macOS, o `docker run -p 27017:27017 mongo`) o apunta `MONGODB_URI` a un clúster Atlas. Si solo quieres practicar, el fallback a `mongomock` ya lo resuelve.

**Error:** `TypeError: BulkOperationBuilder.add_replace() got an unexpected keyword argument 'sort'`
**Por qué:** `mongomock` (4.3) aún no soporta los argumentos nuevos que `pymongo ≥ 4.11` añadió a `ReplaceOne`/`UpdateOne`. Solo ocurre con el simulador, no con un servidor real.
**Solución:** `pip install "pymongo<4.11"` (este notebook se ejecutó con 4.9.2) o conéctate a un MongoDB real.
''')

# =====================================================================
# DATOS
# =====================================================================
M(r'''
## 3. Los datos: un e-commerce chileno simulado

Para practicar necesitamos un dataset **grande, relacional y sucio**, como los de verdad. Lo generamos de forma determinista con cuatro tablas: `customers`, `products`, `orders`, `order_items`. Incluimos campañas simuladas (tipo CyberDay y Black Friday) y problemas intencionales que tendremos que resolver en el ETL: pedidos duplicados, estados mal escritos, comunas nulas, cantidades inválidas y SKUs que no existen.

> **Importante:** los datos son **sintéticos**. Las cifras no representan a ninguna empresa real.

> **NOTA (por qué "solo" 6.000 pedidos):** el simulador `mongomock` resuelve cada *upsert* con una búsqueda lineal, así que cargar *n* documentos cuesta $O(n^2)$: con 20.000 pedidos la carga tardaba ~7 minutos. Un MongoDB real busca por `_id` con un índice ($O(\log n)$) y hace lo mismo en segundos. Si tienes un servidor real, sube `n_orders` a 200.000 sin problema: el código no cambia.
''')

C(r'''
REGIONS = {
    "Metropolitana": ["Santiago", "Providencia", "Las Condes", "Maipú", "Puente Alto", "Ñuñoa"],
    "Valparaíso": ["Valparaíso", "Viña del Mar", "Quilpué"],
    "Biobío": ["Concepción", "Talcahuano", "Los Ángeles"],
    "Antofagasta": ["Antofagasta", "Calama"],
    "Araucanía": ["Temuco", "Villarrica"],
}
REGION_P = [0.50, 0.17, 0.15, 0.09, 0.09]       # Santiago concentra gran parte de la demanda
CATEGORIES = ["Electrónica", "Hogar", "Deporte", "Moda", "Juguetes", "Libros"]
FIRST = ["Camila", "Matías", "Valentina", "Benjamín", "Sofía", "Vicente", "Javiera", "Tomás", "Fernanda", "Joaquín"]
LAST = ["González", "Muñoz", "Rojas", "Díaz", "Pérez", "Soto", "Contreras", "Silva", "Martínez", "Sepúlveda"]


def make_customers(rng, n):
    reg = rng.choice(list(REGIONS), n, p=REGION_P)
    comuna = [rng.choice(REGIONS[r]) for r in reg]
    ids = [f"C{i:05d}" for i in range(1, n + 1)]
    return pd.DataFrame({
        "customer_id": ids,
        "name": [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)],
        "email": [f"user{i}@correo.test" for i in range(1, n + 1)],
        "region": reg, "comuna": comuna,
        "signup_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D"),
    })


def make_products(rng, n):
    price = (rng.lognormal(9.6, 0.8, n) // 10 * 10).clip(1990, 899990).astype(int)   # precios en CLP
    return pd.DataFrame({"sku": [f"SKU-{i:04d}" for i in range(1, n + 1)],
                         "category": rng.choice(CATEGORIES, n),
                         "price_clp": price})
''')

M(r'''
Ahora los pedidos. Fíjate en cómo construimos la **estacionalidad**: la probabilidad de que un pedido caiga en cada día no es uniforme; hay picos en dos campañas (días con peso 5× y 6×) y una leve ola anual. Eso nos dará una serie de tiempo interesante para las ventanas móviles.
''')

C(r'''
def make_orders_and_items(rng, customers, products, n_orders):
    days = pd.date_range("2025-01-01", "2025-12-31")
    w = 1 + 0.25 * np.sin(2 * np.pi * (days.dayofyear.to_numpy() / 365))      # ola anual suave
    w = w * np.where(days.isin(pd.to_datetime(["2025-06-02", "2025-06-03", "2025-06-04"])), 5.0, 1.0)   # campaña "CyberDay"
    w = w * np.where(days == pd.Timestamp("2025-11-28"), 6.0, 1.0)                                        # campaña "Black Friday"
    day = rng.choice(days.values, n_orders, p=w / w.sum())
    ts = pd.to_datetime(day) + pd.to_timedelta(rng.integers(0, 86400, n_orders), unit="s")

    cw = rng.lognormal(0, 1, len(customers)); cw /= cw.sum()       # pocos clientes compran mucho (cola larga)
    orders = pd.DataFrame({
        "order_id": [f"O{i:06d}" for i in range(1, n_orders + 1)],
        "customer_id": rng.choice(customers["customer_id"], n_orders, p=cw),
        "order_ts": ts,
        "status": rng.choice(["entregado", "cancelado", "devuelto", "en_camino"], n_orders, p=[0.86, 0.06, 0.04, 0.04]),
        "payment": rng.choice(["tarjeta", "transferencia", "webpay"], n_orders, p=[0.45, 0.15, 0.40]),
    }).sort_values("order_ts", ignore_index=True)
    orders["order_id"] = [f"O{i:06d}" for i in range(1, n_orders + 1)]   # ids crecientes en el tiempo

    n_items = np.minimum(1 + rng.poisson(1.4, n_orders), 6)
    pw = rng.lognormal(0, 1, len(products)); pw /= pw.sum()          # productos populares y de nicho
    items = pd.DataFrame({
        "order_id": np.repeat(orders["order_id"].to_numpy(), n_items),
        "sku": rng.choice(products["sku"], n_items.sum(), p=pw),
        "qty": rng.integers(1, 4, n_items.sum()),
    })
    price = items["sku"].map(products.set_index("sku")["price_clp"])
    disc = rng.choice([0, 0, 0, 0.10, 0.20], len(items))
    items["unit_price"] = (price * (1 - disc) // 10 * 10).astype(int)
    return orders, items
''')

C(r'''
def corrupt(rng, customers, products, orders, items):
    # Introducimos a propósito los problemas de calidad típicos del mundo real
    customers = customers.copy()
    customers.loc[rng.random(len(customers)) < 0.02, "comuna"] = None                    # comunas faltantes

    orders = orders.copy()
    bad = rng.random(len(orders)) < 0.03
    orders.loc[bad, "status"] = rng.choice(["ENTREGADO", "entregado ", " Cancelado"], bad.sum())   # casing/espacios
    orders = pd.concat([orders, orders.sample(frac=0.01, random_state=1)], ignore_index=True)       # 1 % de duplicados

    items = items.copy()
    items.loc[rng.random(len(items)) < 0.003, "qty"] = rng.choice([0, -1], 1)[0]                   # cantidades inválidas
    ghost = rng.random(len(items)) < 0.002
    items.loc[ghost, "sku"] = "SKU-9999"                                                            # SKU inexistente
    return customers, products, orders, items


rng = np.random.default_rng(SEED)
customers_clean = make_customers(rng, 1500)
products_clean = make_products(rng, 150)
orders_clean, items_clean = make_orders_and_items(rng, customers_clean, products_clean, n_orders=6000)
customers_raw, products_raw, orders_raw, items_raw = corrupt(rng, customers_clean, products_clean, orders_clean, items_clean)

# "Landing zone": en un ETL real los datos llegan como archivos; así practicamos la fase Extract
LANDING = Path(tempfile.mkdtemp(prefix="nb02_landing_"))
for name, d in {"customers": customers_raw, "products": products_raw, "orders": orders_raw, "items": items_raw}.items():
    d.to_csv(LANDING / f"{name}.csv", index=False)
    print(f"{name:10s} → {len(d):>7,} filas  ({(LANDING / f'{name}.csv').stat().st_size / 1e6:.2f} MB)")
''')

M(r'''
**Qué deberías ver:** cuatro archivos CSV en una carpeta temporal: ~1.500 clientes, 150 productos, ~6.060 pedidos (6.000 + ~1 % duplicados) y ~14.000 ítems. Guardamos las versiones **sin corromper** (`*_clean`) solo para poder verificar más adelante que nuestro ETL recupera la verdad.

Echemos un vistazo rápido a lo que realmente llega (la tabla de pedidos con sus problemas):
''')

C(r'''
orders = pd.read_csv(LANDING / "orders.csv", parse_dates=["order_ts"])
items = pd.read_csv(LANDING / "items.csv")
customers = pd.read_csv(LANDING / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(LANDING / "products.csv")

print("Valores de status tal como llegan:", sorted(orders["status"].unique()))
print("order_id duplicados:", orders["order_id"].duplicated().sum())
print("items con qty <= 0:", (items["qty"] <= 0).sum(), "| items con SKU inexistente:", (~items["sku"].isin(products["sku"])).sum())
display(orders.head(3))
''')

# =====================================================================
# PANDAS AVANZADO
# =====================================================================
M(r'''
## 4. pandas avanzado

En NB-01 viste `groupby`, `merge`, `pivot_table` y `pipe`. Ahora vamos a las herramientas que marcan la diferencia con datos reales: **controlar la memoria**, **ventanas móviles**, **series de tiempo**, **joins por tiempo** y **cohortes**. Sigue el patrón *Explicar → Mostrar → Hacer → Reflexionar*.

Primero, necesitamos una tabla de trabajo limpia (después veremos un ETL formal). Por ahora hacemos lo mínimo: estados normalizados, sin duplicados, sin ítems inválidos.
''')

C(r'''
orders_w = (orders.assign(status=lambda d: d["status"].str.strip().str.lower())
                  .drop_duplicates("order_id"))
items_w = items[(items["qty"] > 0) & items["sku"].isin(products["sku"])].copy()
items_w["line_total"] = items_w["qty"] * items_w["unit_price"]

# Total por pedido (agregamos las líneas) y unión con la tabla de pedidos
order_totals = items_w.groupby("order_id").agg(total=("line_total", "sum"), n_items=("qty", "sum"))
sales = orders_w.merge(order_totals, on="order_id", how="inner").merge(customers[["customer_id", "region", "comuna"]], on="customer_id", how="left")
print(f"{len(sales):,} pedidos | ventas brutas: ${sales['total'].sum():,.0f} CLP")
display(sales.head(3))
''')

M(r'''
### 4.1 Memoria y tipos de datos
**Por qué importa:** pandas usa por defecto `int64` y `float64` (8 bytes por valor) aunque el dato solo necesite 1 o 2. Con millones de filas, eso significa gigas de RAM desperdiciados; y las columnas de texto repetido (`region`, `status`) son aún peores. En Big Data **la memoria es tu primer cuello de botella**, y *downcasting* + `category` es la mejora más barata.

La función siguiente reduce cada columna al tipo mínimo suficiente:
''')

C(r'''
def mem_mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1e6          # deep=True cuenta también el texto real


def optimize_dtypes(df: pd.DataFrame, cat_threshold: float = 0.5) -> pd.DataFrame:
    # Reduce memoria sin perder información. cat_threshold: proporción máxima de únicos para pasar a category
    out = df.copy()
    for col in out.select_dtypes("integer"):
        out[col] = pd.to_numeric(out[col], downcast="integer")      # int64 → int8/16/32 según el rango real
    for col in out.select_dtypes("floating"):
        out[col] = pd.to_numeric(out[col], downcast="float")        # float64 → float32 (cuidado con la precisión)
    for col in out.columns:
        if pd.api.types.is_string_dtype(out[col]) and out[col].nunique() / max(len(out), 1) < cat_threshold:
            out[col] = out[col].astype("category")                  # texto repetido → códigos enteros + diccionario
    return out

sales_opt = optimize_dtypes(sales)
print(f"Antes : {mem_mb(sales):.2f} MB\nDespués: {mem_mb(sales_opt):.2f} MB  ({1 - mem_mb(sales_opt)/mem_mb(sales):.0%} menos)")
display(pd.DataFrame({"antes": sales.dtypes.astype(str), "después": sales_opt.dtypes.astype(str)}))
''')

M(r'''
**Qué deberías ver:** una reducción de memoria grande y que `status`, `payment`, `region` y `comuna` pasaron a `category`, mientras que `total` y `n_items` se achicaron a enteros más pequeños. Dos cosas a tener presente:

> **NOTA:** `float32` tiene ~7 dígitos decimales de precisión. Para dinero en CLP (enteros) usa enteros; para ML con features continuas suele bastar `float32` (de hecho es lo que usan las GPUs).

### ⚠️ Troubleshooting

**Error:** `ValueError: Cannot setitem on a Categorical with a new category (...)`
**Por qué:** una columna `category` solo acepta valores de su diccionario de categorías; asignar uno nuevo falla.
**Solución:** agrega la categoría antes (`s.cat.add_categories(["otro"])`) o convierte temporalmente a `str`.
''')

nb.tryit([
    "**4-1. Un tipo de dato a la medida:** ¿cuál es el menor tipo entero que puede guardar `n_items`? Compruébalo con `sales['n_items'].max()` y `np.iinfo(np.int8)`.",
    "**4-2. ¿Siempre category?** Cambia `cat_threshold` a `0.0001` y a `1.0`. ¿Qué columnas cambian de tipo? ¿Por qué convertir una columna casi única (como `order_id`) a `category` sería un error?",
])
nb.solutions("4-1 y 4-2", r'''
print("max n_items:", sales["n_items"].max(), "| rango int8:", np.iinfo(np.int8))   # 4-1
for thr in (0.0001, 1.0):                                                           # 4-2
    t = optimize_dtypes(sales, cat_threshold=thr)
    print(f"umbral {thr}: categorías →", [c for c in t.columns if str(t[c].dtype) == "category"], f"| {mem_mb(t):.2f} MB")
# Con una columna casi única, el diccionario de categorías tiene tantas entradas como filas:
# ocupas MÁS memoria (códigos + diccionario) y no ganas nada.
''')

M(r'''
### 4.2 MultiIndex: varias dimensiones en un índice
Un `MultiIndex` es un índice jerárquico: sirve para representar tablas con **más de una clave** (región × mes). Se obtiene casi siempre como resultado de un `groupby` con dos claves, y se manipula con `unstack` (mover un nivel a columnas), `stack` (lo inverso) y `.xs` (cortar por un nivel).
''')

C(r'''
sales["month"] = sales["order_ts"].dt.to_period("M")
delivered = sales[sales["status"] == "entregado"]

rev = delivered.groupby(["region", "month"])["total"].sum()        # Series con MultiIndex (region, month)
print("Tipo de índice:", type(rev.index).__name__, "| niveles:", rev.index.names)

wide = rev.unstack("region")                      # el nivel 'region' pasa a columnas → tabla mes × región
display((wide / 1e6).round(1).head(4))
print("\nCorte transversal de una región con .xs:")
display((rev.xs("Biobío", level="region") / 1e6).round(1).head(3))
''')

M(r'''
**Qué deberías ver:** primero una tabla de ingresos (en millones de CLP) con un mes por fila y una región por columna (Metropolitana domina, como definimos en los datos); luego la serie mensual solo de Biobío.
''')

nb.tryit([
    "**4-3. Participación por región:** a partir de `wide`, calcula qué **porcentaje** del ingreso mensual aporta cada región (cada fila debe sumar 100 %). *Pista:* `wide.div(wide.sum(axis=1), axis=0)`.",
    "**4-4. De vuelta a largo:** usa `wide.stack()` para volver al formato original. ¿Qué diferencia hay con `rev` si alguna combinación no tuviera ventas?",
])
nb.solutions("4-3 y 4-4", r'''
share = wide.div(wide.sum(axis=1), axis=0)              # axis=0 alinea por filas: cada mes se divide por SU total
print(share.sum(axis=1).round(6).unique())              # todos 1.0 → verificación
display((share * 100).round(1).head(3))
back = wide.stack()      # stack descarta los NaN por defecto (dropna): una combinación sin ventas desaparece, en vez de quedar como 0 o NaN
print(back.shape, rev.shape)
''')

M(r'''
### 4.3 Ventanas móviles: `rolling`, `expanding`, `ewm`
**Por qué existen:** los datos diarios son ruidosos (un día malo no es una tendencia). Una **ventana móvil** promedia los últimos *k* valores y suaviza el ruido. Es exactamente un **filtro FIR** de promedio móvil en procesamiento de señales (Comunicaciones Digitales):

$$\text{MA}_t = \frac{1}{k}\sum_{i=0}^{k-1} x_{t-i} \qquad\qquad \text{EWM}_t = \alpha\, x_t + (1-\alpha)\,\text{EWM}_{t-1} \;\;(\text{filtro IIR de 1.er orden})$$

| Método | Qué hace | Cuándo |
|---|---|---|
| `rolling(k)` | ventana de tamaño fijo (los últimos *k*) | suavizar, volatilidad local |
| `expanding()` | desde el inicio hasta ahora (acumulativo) | máximo histórico, promedio acumulado |
| `ewm(span=k)` | pesos que decaen exponencialmente | reaccionar más rápido a datos recientes |

Primero construimos la serie diaria de ingresos (`resample("D")`), rellenando con 0 los días sin ventas.
''')

C(r'''
daily = (delivered.set_index("order_ts")["total"].resample("D").sum() / 1e6).rename("ingreso_MM")   # millones de CLP

feat = pd.DataFrame({"diario": daily})
feat["MA7"] = daily.rolling(7).mean()                       # los primeros 6 valores son NaN: no hay 7 datos aún
feat["MA7_centrado"] = daily.rolling(7, center=True).mean() # mira pasado Y futuro: solo para análisis, nunca para predecir
feat["EWM7"] = daily.ewm(span=7).mean()
feat["max_historico"] = daily.expanding().max()
display(feat.iloc[5:10])
''')

C(r'''
fig, ax = plt.subplots()
ax.plot(feat.index, feat["diario"], color="lightgray", label="diario (ruidoso)")
ax.plot(feat.index, feat["MA7"], label="media móvil 7 días", lw=2)
ax.plot(feat.index, feat["EWM7"], label="EWM (span=7)", lw=2, ls="--")
for d, txt in [("2025-06-03", "CyberDay (simulado)"), ("2025-11-28", "Black Friday (simulado)")]:
    ax.annotate(txt, xy=(pd.Timestamp(d), daily.loc[d]), xytext=(10, 10), textcoords="offset points", arrowprops=dict(arrowstyle="->", color="black"))
ax.set(title="Ingresos diarios y suavizado", xlabel="fecha", ylabel="ingreso (millones CLP)")
ax.legend(); ax.grid(True)
plt.show()
''')

M(r'''
**Qué deberías ver:** una nube gris irregular con dos picos muy marcados (las campañas), y encima dos curvas suaves. La media móvil de 7 días "reparte" cada pico sobre una semana (se ve como una meseta) y llega **con retraso** al pico; la EWM lo sigue más de cerca porque pondera más lo reciente.

> **NOTA:** `rolling(k)` da `NaN` en las primeras *k−1* posiciones. Si quieres valores desde el principio usa `min_periods=1`. Y `center=True` usa datos **del futuro**: es válido para describir el pasado, pero si construyes una *feature* para predecir produce **fuga de datos** (*data leakage*, ver NB-09).

### ⚠️ Troubleshooting

**Error:** `ValueError: window must be an integer 0 or greater` / resultados todo `NaN` al hacer `rolling("7D")`
**Por qué:** una ventana por tiempo (`"7D"`) exige un índice de tipo fecha **ordenado**; con índice numérico o desordenado falla o da NaN.
**Solución:** `df = df.sort_index()` y asegúrate de que el índice sea `DatetimeIndex` (`set_index("fecha")`).
''')

nb.tryit([
    "**4-5. Volatilidad:** calcula la desviación estándar móvil de 14 días (`rolling(14).std()`) y grafícala. ¿En qué fechas se dispara y por qué?",
    "**4-6. Ventana por tiempo:** reproduce `MA7` con `daily.rolling('7D').mean()`. ¿Coincide con `rolling(7)`? ¿Cuándo difieren (pista: días faltantes)?",
])
nb.solutions("4-5 y 4-6", r'''
vol = daily.rolling(14).std()
print("Pico de volatilidad en:", vol.idxmax().date(), "(14 días tras una campaña la desviación sube por el salto)")
ma_t = daily.rolling("7D").mean()
print("¿Idéntica a rolling(7) desde el día 7?", np.allclose(ma_t.iloc[6:], feat["MA7"].iloc[6:]))
# Coinciden aquí porque resample('D') no deja huecos. Con datos irregulares, '7D' promedia lo que haya en 7 días
# calendario; rolling(7) promediaría las últimas 7 FILAS aunque abarquen más tiempo.
''')

M(r'''
### 4.4 Series de tiempo: `shift`, `pct_change` y `merge_asof`
- `shift(n)` desplaza la serie *n* pasos: es la base de **features de rezago** (*lags*) para predecir.
- `pct_change(n)` calcula la variación porcentual respecto a *n* pasos atrás.
- `merge_asof` une dos tablas por **la fila más cercana en el tiempo** (no por igualdad). Es lo que necesitas cuando un evento (pedido a las 14:37) debe asociarse con el último dato conocido de otra serie (tipo de cambio del día).

Simulamos un tipo de cambio USD/CLP que solo se publica en días hábiles, y convertimos cada pedido a dólares con el valor **vigente al momento del pedido**:
''')

C(r'''
bdays = pd.bdate_range("2024-12-30", "2025-12-31")
usd = pd.DataFrame({"date": bdays, "usd_clp": 950 + np.cumsum(rng.normal(0, 4, len(bdays)))})   # camino aleatorio

# merge_asof EXIGE ambas tablas ordenadas por la clave. direction="backward": toma el último valor ANTERIOR o igual
s = sales.sort_values("order_ts")
s = pd.merge_asof(s, usd, left_on="order_ts", right_on="date", direction="backward")
s["total_usd"] = s["total"] / s["usd_clp"]
print(f"Sin valor de cambio asignado: {s['usd_clp'].isna().sum()} pedidos")
display(s[["order_id", "order_ts", "date", "usd_clp", "total", "total_usd"]].sample(4, random_state=3))
''')

M(r'''
**Qué deberías ver:** para cada pedido, la columna `date` es el día hábil **anterior o igual** a la fecha del pedido (un pedido de sábado toma el valor del viernes). Eso es justamente lo que busca `direction="backward"`.

> **NOTA:** Usar `direction="nearest"` o `"forward"` también compila, pero mezcla información del **futuro** en cada fila (otra forma de *data leakage*). En finanzas y ML, "lo que se sabía en ese momento" es lo único válido.
''')

nb.tryit([
    "**4-7. Crecimiento semanal:** con `daily.resample('W').sum().pct_change()` obtén la variación semanal de ingresos. ¿Cuál es la semana de mayor crecimiento? (Debe coincidir con una campaña.)",
    "**4-8. Lag feature:** crea `daily.shift(7)` (el ingreso de hace una semana) y calcula su correlación con `daily`. ¿Qué te dice sobre la estacionalidad semanal?",
])
nb.solutions("4-7 y 4-8", r'''
growth = daily.resample("W").sum().pct_change()
print("Semana de mayor crecimiento:", growth.idxmax().date(), f"({growth.max():+.0%})")
print("Correlación con el rezago de 7 días:", round(daily.corr(daily.shift(7)), 3),
      "→ baja: en estos datos simulados no hay patrón semanal, solo las campañas")
''')

M(r'''
### 4.5 Cohortes y segmentación RFM
Dos análisis clásicos de comercio electrónico que combinan todo lo anterior:

- **Cohorte de retención:** agrupar a los clientes por el **mes de su primera compra** y medir qué fracción vuelve a comprar *k* meses después.
- **RFM:** puntuar a cada cliente por **R**ecencia (¿hace cuánto compró?), **F**recuencia (¿cuántas veces?) y valor **M**onetario (¿cuánto gastó?). Sirve para segmentar (campeones, en riesgo…) y es un clásico caso de *feature engineering*.
''')

C(r'''
# --- Cohortes ---
buy = delivered[["customer_id", "order_ts"]].copy()
buy["order_month"] = buy["order_ts"].dt.to_period("M")
buy["cohort"] = buy.groupby("customer_id")["order_month"].transform("min")            # mes de la primera compra
buy["age"] = (buy["order_month"] - buy["cohort"]).apply(lambda x: x.n)               # meses desde la primera compra

cohort = buy.groupby(["cohort", "age"])["customer_id"].nunique().unstack("age")
retention = cohort.div(cohort[0], axis=0)         # columna 0 = tamaño de la cohorte (100 %)

fig, ax = plt.subplots(figsize=(10, 5.5))
sns.heatmap(retention.iloc[:, :9], annot=True, fmt=".0%", cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set(title="Retención por cohorte (fracción de clientes que vuelve a comprar)", xlabel="meses desde la primera compra", ylabel="cohorte")
plt.show()
''')

M(r'''
**Qué deberías ver:** un triángulo (las cohortes recientes tienen menos meses observados). La primera columna es 100 %. La retención **cae con los meses**: en las cohortes más antiguas parte en torno a 30–40 % el primer mes y baja hacia 10–20 %. Los huecos en blanco son meses en que ninguna persona de esa cohorte compró (o que aún no ocurren).

Ahora RFM con `qcut` (quintiles): cada dimensión recibe un puntaje 1–5.
''')

C(r'''
snapshot = delivered["order_ts"].max() + pd.Timedelta(days=1)       # "hoy" para medir la recencia
rfm = delivered.groupby("customer_id").agg(
    recency=("order_ts", lambda s: (snapshot - s.max()).days),
    frequency=("order_id", "nunique"),
    monetary=("total", "sum"),
)
# rank(method="first") desempata: qcut falla si hay muchos valores repetidos en los bordes de los cuantiles
rfm["R"] = pd.qcut(rfm["recency"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)   # menos días = mejor
rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["segment"] = np.select([(rfm.R >= 4) & (rfm.F >= 4), (rfm.R <= 2) & (rfm.F >= 3), rfm.R >= 4],
                           ["campeón", "en riesgo", "nuevo/reciente"], default="regular")
display(rfm["segment"].value_counts().to_frame("clientes").T)
display(rfm.groupby("segment")[["recency", "frequency", "monetary"]].mean().round(0))
''')

M(r'''
**Qué deberías ver:** hasta cuatro segmentos. Los "campeones" tienen recencia baja, frecuencia y gasto altos; los "en riesgo" compraron bastante en el pasado pero no recientemente: son la prioridad de una campaña de reactivación.

> **NOTA:** `rank(method="first")` antes de `qcut` es un truco práctico. Si muchos clientes tienen `frequency == 1`, los bordes de los cuantiles se repiten y `qcut` lanza `ValueError: Bin edges must be unique`.
''')

nb.tryit([
    "**4-9. Pareto:** ¿cuánto del ingreso total concentran los 20 % de clientes con mayor `monetary`?",
    "**4-10. Retención a 3 meses:** calcula el promedio de la columna `age == 3` de `retention` ponderado por el tamaño de cada cohorte.",
])
nb.solutions("4-9 y 4-10", r'''
top20 = rfm.sort_values("monetary", ascending=False).head(int(len(rfm) * 0.2))["monetary"].sum() / rfm["monetary"].sum()
print(f"El 20% de clientes concentra el {top20:.0%} del ingreso")               # 4-9
r3 = (retention[3] * cohort[0]).sum() / cohort.loc[retention[3].notna(), 0].sum()   # media ponderada por tamaño de cohorte
print(f"Retención media a 3 meses (ponderada): {r3:.1%}")                          # 4-10
''')

M(r'''
### 4.6 Rendimiento: ¿cómo calculo esto más rápido?
Mismo cálculo, cuatro formas. Medimos el **total por línea** (`qty * unit_price`) sobre 20.000 líneas de ítems:
''')

C(r'''
def time_ms(fn, repeat=3):
    best = np.inf
    for _ in range(repeat):                     # el mínimo de varias corridas filtra el ruido del sistema operativo
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return best * 1e3

sub = items_w.sample(20_000, replace=True, random_state=0).reset_index(drop=True)   # remuestreo: pocas filas reales, queremos 20.000 para medir

def with_iterrows():   return [r.qty * r.unit_price for _, r in sub.iterrows()]
def with_apply():      return sub.apply(lambda r: r["qty"] * r["unit_price"], axis=1)
def with_itertuples():  return [r.qty * r.unit_price for r in sub.itertuples()]
def with_vector():     return sub["qty"] * sub["unit_price"]

res = pd.Series({f.__name__.replace("with_", ""): time_ms(f) for f in [with_iterrows, with_apply, with_itertuples, with_vector]})
display((res.to_frame("ms").assign(x_mas_lento_que_vector=lambda d: d["ms"] / d.loc["vector", "ms"])).round(1))
''')

M(r'''
**Qué deberías ver:** `iterrows` y `apply(axis=1)` cientos o miles de veces más lentos que la versión vectorizada (en una corrida típica, `iterrows` ≈ 5.000× y `apply` ≈ 1.000×); `itertuples` en el medio. La razón: `iterrows` construye una `Series` de pandas **por cada fila** y `apply` ejecuta código Python por fila; la operación vectorizada ejecuta un solo bucle en C.

Reglas prácticas (en orden de preferencia):
1. **Vectoriza** (`a * b`, `np.where`, `.str`, `.dt`).
2. Si es una condición compleja: `np.select` / `np.where`.
3. Si no queda otra: `itertuples` (nunca `iterrows`).
4. Si el cuello de botella es real: pasar a Polars/DuckDB o compilar con Numba.

`query()` y `eval()` evalúan expresiones con `numexpr` (si está instalado), útiles en tablas grandes y para escribir filtros legibles:
''')

C(r'''
thr = 50_000
a = sales[(sales["total"] > thr) & (sales["status"] == "entregado") & (sales["payment"] == "webpay")]
b = sales.query("total > @thr and status == 'entregado' and payment == 'webpay'")   # @ referencia variables de Python
assert a.equals(b)
print(f"Ambos filtros devuelven {len(a):,} pedidos. query() es equivalente y más legible.")
''')

nb.tryit([
    "**4-11. Comprueba el crecimiento:** mide `with_vector` vs `with_apply` con 5.000, 20.000 y 40.000 filas. ¿Cómo crece cada tiempo?",
    "**4-12. np.where vs apply:** crea la columna `tramo` (`'alto'` si `total > 50_000`, si no `'bajo'`) con `np.where` y con `apply`. Compara tiempos.",
])
nb.solutions("4-11 y 4-12", r'''
for n in (5_000, 20_000, 40_000):
    s_ = items_w.sample(n, replace=True, random_state=0).reset_index(drop=True)
    t_v = time_ms(lambda: s_["qty"] * s_["unit_price"]); t_a = time_ms(lambda: s_.apply(lambda r: r["qty"] * r["unit_price"], axis=1), repeat=1)
    print(f"n={n:>6,}: vector={t_v:7.2f} ms | apply={t_a:8.1f} ms")
# apply crece ~linealmente con n (código Python por fila). El vectorizado es tan rápido que casi no cambia.
t1 = time_ms(lambda: np.where(sales["total"] > 50_000, "alto", "bajo"))
t2 = time_ms(lambda: sales["total"].apply(lambda v: "alto" if v > 50_000 else "bajo"))
print(f"np.where: {t1:.2f} ms | apply: {t2:.2f} ms")
''')

M(r'''
### 📊 Resumen visual — pandas avanzado
```
 PROBLEMA                          HERRAMIENTA                         CUIDADO CON…
 ───────────────────────────────────────────────────────────────────────────────────────────
 Se me acaba la RAM            →   downcast + category                 float32 pierde precisión
 Dos claves de agrupación      →   groupby([a,b]) + unstack/xs         stack() descarta NaN
 Suavizar ruido                →   rolling / ewm                       center=True usa el futuro
 Máximo/promedio acumulado     →   expanding                           —
 Unir por cercanía temporal    →   merge_asof (ordenar antes)          direction="nearest" filtra futuro
 Retención de clientes         →   cohorte + pivot + heatmap           cohortes recientes incompletas
 Segmentar clientes            →   RFM con qcut(rank(first))           bordes repetidos en qcut
 Mi código es lento            →   vectorizar > itertuples > apply     iterrows: nunca
```
''')

# =====================================================================
# ETL
# =====================================================================
M(r'''
## 5. ETL: diseño de un pipeline robusto

Un **ETL** (*Extract–Transform–Load*) no es "un script que limpia datos": es un proceso que debe correr **una y otra vez**, con datos nuevos, sin romperse ni duplicar información. Los principios que lo distinguen de un script de una sola vez:

| Principio | Qué significa | Cómo lo logramos aquí |
|---|---|---|
| **Idempotencia** | Ejecutarlo 2 veces da el mismo resultado que 1 | *upsert* por `_id` en vez de `insert` |
| **Validación explícita** | Los datos malos se detectan, no se asumen buenos | reglas con máscaras booleanas |
| **Cuarentena** | Lo rechazado no se pierde: se guarda con su motivo | DataFrame `quarantine` |
| **Observabilidad** | Saber cuántas filas entraron, salieron y por qué | reporte (`ETLReport`) + `logging` |
| **Separación de fases** | E, T y L son funciones independientes y testeables | tres funciones puras + un orquestador |

> **Conexión con Sistemas Distribuidos:** la idempotencia es lo que permite **reintentar** de forma segura cuando algo falla a mitad de camino (Kleppmann cap. 11; Airflow lo exige en NB-22).

### 5.1 Extract y reglas de validación
''')

C(r'''
@dataclass
class ETLReport:
    # Contadores y rechazos del pipeline: lo primero que miras cuando algo "se ve raro" en producción
    counts: dict = field(default_factory=dict)
    quarantine: list = field(default_factory=list)          # lista de DataFrames (entidad, clave, motivo)

    def reject(self, entity: str, keys: pd.Series, reason: str):
        if len(keys):
            self.quarantine.append(pd.DataFrame({"entity": entity, "key": keys.astype(str).to_numpy(), "reason": reason}))

    def quarantine_df(self) -> pd.DataFrame:
        return pd.concat(self.quarantine, ignore_index=True) if self.quarantine else pd.DataFrame(columns=["entity", "key", "reason"])


def extract(landing: Path) -> dict[str, pd.DataFrame]:
    # Tipos explícitos al leer: evita inferencias costosas y sorpresas (ej. IDs como números)
    return {
        "customers": pd.read_csv(landing / "customers.csv", parse_dates=["signup_date"]),
        "products": pd.read_csv(landing / "products.csv"),
        "orders": pd.read_csv(landing / "orders.csv", parse_dates=["order_ts"]),
        "items": pd.read_csv(landing / "items.csv", dtype={"qty": "int32", "unit_price": "int32"}),
    }
''')

M(r'''
### 5.2 Transform
La fase de transformación aplica las reglas **una a una** y registra cuántas filas descarta cada una. Fíjate en que cada paso es una función pequeña con una sola responsabilidad.
''')

C(r'''
def transform(raw: dict, rep: ETLReport) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cust, prod, orders, items = (raw[k].copy() for k in ("customers", "products", "orders", "items"))
    rep.counts["in_orders"], rep.counts["in_items"] = len(orders), len(items)

    # 1) Normalizar texto ANTES de comparar o deduplicar
    orders["status"] = orders["status"].str.strip().str.lower()
    cust["comuna"] = cust["comuna"].fillna("desconocida")

    # 2) Duplicados: conservamos la primera aparición y dejamos constancia
    dup = orders["order_id"].duplicated()
    rep.reject("orders", orders.loc[dup, "order_id"], "order_id duplicado")
    orders = orders[~dup]

    # 3) Ítems inválidos: cantidad no positiva o SKU inexistente
    bad_qty, bad_sku = items["qty"] <= 0, ~items["sku"].isin(prod["sku"])
    rep.reject("items", items.loc[bad_qty, "order_id"], "qty <= 0")
    rep.reject("items", items.loc[bad_sku & ~bad_qty, "order_id"], "sku inexistente")
    items = items[~(bad_qty | bad_sku)].copy()

    # 4) Un pedido sin ítems válidos o sin cliente conocido no se puede cargar
    no_items = ~orders["order_id"].isin(items["order_id"])
    no_cust = ~orders["customer_id"].isin(cust["customer_id"])
    rep.reject("orders", orders.loc[no_items, "order_id"], "sin ítems válidos")
    rep.reject("orders", orders.loc[no_cust, "order_id"], "cliente desconocido")
    orders = orders[~(no_items | no_cust)]
    items = items[items["order_id"].isin(orders["order_id"])]

    rep.counts["out_orders"], rep.counts["out_items"] = len(orders), len(items)
    return cust, prod, build_documents(orders, items, cust)
''')

M(r'''
Este es el paso de **modelado**: pasamos de tablas planas a **documentos**. Cada pedido embebe sus ítems y una *copia* (snapshot) de los datos del cliente que necesitamos al consultar. Esa **denormalización** evita un `JOIN` en cada lectura, a cambio de duplicar datos.

Hay un detalle técnico importante: PyMongo solo sabe serializar **tipos nativos de Python** (`int`, `float`, `str`, `datetime`…). Un `numpy.int64` o un `NaN` dentro de un documento causa problemas. Por eso convertimos explícitamente.
''')

C(r'''
def build_documents(orders: pd.DataFrame, items: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    # Une pedidos + ítems anidados + snapshot del cliente. Devuelve un DataFrame con una columna 'doc' (dict)
    items = items.assign(qty=items["qty"].astype(int), unit_price=items["unit_price"].astype(int))
    nested = (items.groupby("order_id")[["sku", "qty", "unit_price"]]
                   .apply(lambda g: g.to_dict("records"))        # lista de dicts por pedido
                   .rename("items"))
    totals = (items["qty"] * items["unit_price"]).groupby(items["order_id"]).sum().rename("total")
    o = (orders.join(nested, on="order_id").join(totals, on="order_id")
               .merge(cust[["customer_id", "region", "comuna"]], on="customer_id", how="left"))

    docs = [{
        "_id": r.order_id,                                       # clave natural como _id → hace el upsert idempotente
        "customer": {"id": r.customer_id, "region": r.region, "comuna": r.comuna},
        "ts": r.order_ts.to_pydatetime(),                        # pd.Timestamp → datetime nativo
        "status": r.status, "payment": r.payment,
        "items": r.items, "total": int(r.total),                 # int() evita numpy.int64
    } for r in o.itertuples(index=False)]
    return pd.DataFrame({"doc": docs})
''')

M(r'''
### 5.3 Load idempotente
`ReplaceOne(filtro, doc, upsert=True)` significa: "si existe un documento con este `_id`, **reemplázalo**; si no, **insértalo**". Ejecutarlo dos veces deja el mismo estado. Lo enviamos con `bulk_write` en lotes: una sola ida y vuelta a la red por lote en lugar de una por documento (en MongoDB real, es la diferencia entre segundos y minutos).
''')

C(r'''
def load(coll, docs: list[dict], batch_size: int = 5000, key: str = "_id") -> dict:
    stats = {"matched": 0, "upserted": 0, "modified": 0}
    for i in range(0, len(docs), batch_size):
        ops = [ReplaceOne({key: d[key]}, d, upsert=True) for d in docs[i:i + batch_size]]
        r = coll.bulk_write(ops, ordered=False)          # ordered=False: un fallo no frena el resto del lote
        stats["matched"] += r.matched_count; stats["upserted"] += r.upserted_count; stats["modified"] += r.modified_count
    return stats


def validate_bson(docs: list[dict], sample: int = 500) -> None:
    # Falla temprano (y con un mensaje claro) si algún documento no es serializable
    for d in docs[:sample]:
        bson.encode(d)
''')

M(r'''
### 5.4 El orquestador
Una función que une las tres fases, mide el tiempo y devuelve un reporte. En NB-22 este mismo esquema se convierte en un DAG de Airflow (cada función = una tarea).
''')

C(r'''
def to_records(df: pd.DataFrame, id_col: str) -> list[dict]:
    # DataFrame → lista de dicts nativos con _id. to_json → json.loads garantiza tipos Python (no numpy)
    d = df.copy()
    for c in d.select_dtypes("datetime"):
        d[c] = d[c].dt.strftime("%Y-%m-%d")
    return [{**r, "_id": r[id_col]} for r in json.loads(d.to_json(orient="records"))]


def run_etl(landing: Path, db, logger=None) -> ETLReport:
    log = (logger or logging.getLogger("etl")).info
    rep = ETLReport()
    t0 = time.perf_counter()
    raw = extract(landing);                                    log("extract ok")
    cust, prod, doc_df = transform(raw, rep);                  log("transform ok")
    docs = doc_df["doc"].tolist(); validate_bson(docs)
    rep.counts["orders_load"] = load(db["orders"], docs)
    rep.counts["customers_load"] = load(db["customers"], to_records(cust, "customer_id"))
    rep.counts["products_load"] = load(db["products"], to_records(prod, "sku"))
    rep.counts["seconds"] = round(time.perf_counter() - t0, 2)
    return rep
''')

M(r'''
Ejecutemos el pipeline **dos veces seguidas** y observemos la idempotencia: en la segunda corrida no deben aparecer documentos nuevos (`upserted = 0`) y el total en la colección debe ser el mismo.
''')

C(r'''
logging.basicConfig(level=logging.WARNING)          # en producción: INFO hacia un archivo/stdout del contenedor
rep1 = run_etl(LANDING, db)
n_after_1 = db["orders"].count_documents({})
rep2 = run_etl(LANDING, db)
n_after_2 = db["orders"].count_documents({})

print("Corrida 1:", rep1.counts["orders_load"])
print("Corrida 2:", rep2.counts["orders_load"])
print(f"Pedidos en la colección → tras corrida 1: {n_after_1:,} | tras corrida 2: {n_after_2:,}")
assert n_after_1 == n_after_2, "¡el pipeline no es idempotente!"
print(f"Filas de entrada → salida: pedidos {rep1.counts['in_orders']:,} → {rep1.counts['out_orders']:,} | ítems {rep1.counts['in_items']:,} → {rep1.counts['out_items']:,}")
display(rep1.quarantine_df().groupby(["entity", "reason"]).size().to_frame("filas"))
''')

M(r'''
**Qué deberías ver:** en la corrida 1 todos los pedidos son `upserted`; en la 2 son `matched` (ya existían) y `upserted = 0`. El conteo no cambia. La tabla de cuarentena resume **qué se descartó y por qué**: ~60 duplicados, unos pocos ítems con `qty <= 0` y con SKU inexistente. Ese reporte es lo que enseñarías al dueño de los datos para que arregle el origen.

> **NOTA:** Una cuarentena silenciosa es peligrosa. En producción, el pipeline debería **alertar** si el porcentaje rechazado supera un umbral (p. ej. >5 %): suele indicar que cambió el formato del origen.

### ⚠️ Troubleshooting

**Error:** `bson.errors.InvalidDocument: cannot encode object: 5, of type: <class 'numpy.int64'>`
**Por qué:** PyMongo no sabe serializar tipos de NumPy. Aparece al insertar filas sacadas directamente de un DataFrame.
**Solución:** convierte con `int(x)` / `float(x)`, o pasa por `json.loads(df.to_json(orient="records"))`. Mira cómo lo detecta `validate_bson` antes de cargar:
''')

C(r'''
try:
    bson.encode({"qty": np.int64(3)})
except bson.errors.InvalidDocument as e:
    print("Error esperado →", str(e)[:80])

print("NaN sí se serializa (como float), pero es una trampa:", bson.decode(bson.encode({"x": float("nan")})))
print("Convención sana: usar None (→ null) o dejar el campo ausente; así $exists y $ne:null funcionan como esperas.")
''')

nb.tryit([
    "**5-1. Una regla nueva:** agrega a `transform` la regla \"el pedido tiene fecha futura (`order_ts` > 2025-12-31)\" y verifica que el ETL sigue siendo idempotente. ¿En qué paso debe ir la regla?",
    "**5-2. Umbral de alerta:** escribe `check_quality(rep, max_reject=0.05)` que lance `ValueError` si más del 5 % de los pedidos de entrada fue rechazado. Pruébala con `rep1`.",
])
nb.solutions("5-1 y 5-2", r'''
# 5-2: la función de alerta (la regla 5-1 es análoga a "cliente desconocido": máscara + rep.reject + filtrar)
def check_quality(rep: ETLReport, max_reject: float = 0.05) -> None:
    q = rep.quarantine_df()
    rejected_orders = q.query("entity == 'orders'")["key"].nunique()
    frac = rejected_orders / rep.counts["in_orders"]
    if frac > max_reject:
        raise ValueError(f"Calidad insuficiente: {frac:.1%} de pedidos rechazados (umbral {max_reject:.0%})")
    print(f"Calidad OK: {frac:.2%} de pedidos rechazados")

check_quality(rep1)
try:
    check_quality(rep1, max_reject=0.001)
except ValueError as e:
    print("Alerta esperada →", e)
# Para 5-1: la regla va después de normalizar y ANTES de construir documentos, igual que las demás (cada regla
# solo mira los datos ya limpiados por las anteriores).
''')

# =====================================================================
# MONGODB
# =====================================================================
M(r'''
## 6. MongoDB con PyMongo

Los datos ya están cargados. Ahora aprendemos a **trabajar** con ellos. Vocabulario mínimo:

| SQL | MongoDB | Ejemplo |
|---|---|---|
| base de datos | *database* | `db = client["nb02_ecommerce"]` |
| tabla | *collection* | `db["orders"]` |
| fila | *document* (dict/JSON → BSON) | `{"_id": "O000001", ...}` |
| columna | *field* | `"total"` |
| `SELECT … WHERE` | `find(filtro, proyección)` | `find({"total": {"$gt": 1000}})` |
| `GROUP BY` | `aggregate([...])` | pipeline de etapas |
| índice | *index* | `create_index(...)` |

Cada documento tiene un `_id` único e inmutable. Si no lo defines, MongoDB crea un `ObjectId`. Nosotros usamos el `order_id` natural, lo que nos dio la idempotencia.

### 6.1 Leer: `find`, operadores y proyección
Los filtros son **documentos**, y los operadores empiezan con `$`: `$gt`, `$gte`, `$lt`, `$in`, `$and`, `$or`, `$regex`, `$exists`, `$elemMatch`…
''')

C(r'''
orders_c = db["orders"]
print("Un documento completo:")
print(json.dumps(orders_c.find_one({"_id": "O000001"}), default=str, indent=2, ensure_ascii=False)[:700], "...")
''')

C(r'''
# Filtro con varios operadores + proyección (solo los campos que necesito) + orden + límite
q = {"status": "entregado", "payment": {"$in": ["webpay", "tarjeta"]}, "total": {"$gte": 200_000}, "customer.region": "Biobío"}
proj = {"total": 1, "customer.comuna": 1, "ts": 1}                 # _id viene por defecto
top = list(orders_c.find(q, proj).sort("total", -1).limit(3))
for d in top:
    print(d["_id"], d["customer"]["comuna"], f"${d['total']:,}", d["ts"].date())
print("Total que cumple el filtro:", orders_c.count_documents(q))
''')

M(r'''
**Qué deberías ver:** los 3 pedidos de mayor monto en Biobío pagados con webpay o tarjeta y de al menos $200.000, y cuántos pedidos cumplen el filtro en total. Con la notación de punto (`"customer.region"`) se consultan campos **dentro** de subdocumentos.

Los operadores sobre **arreglos** son una de las fortalezas del modelo de documentos. `$elemMatch` exige que **un mismo elemento** del arreglo cumpla todas las condiciones:
''')

C(r'''
# Pedidos que contienen una línea con qty >= 3 Y precio unitario >= 100.000 (ambas condiciones en la MISMA línea)
f = {"items": {"$elemMatch": {"qty": {"$gte": 3}, "unit_price": {"$gte": 100_000}}}}
print("Con $elemMatch :", orders_c.count_documents(f))

# Sin $elemMatch las condiciones pueden cumplirlas ÍTEMS DISTINTOS: resultado más grande y distinto
g = {"items.qty": {"$gte": 3}, "items.unit_price": {"$gte": 100_000}}
print("Sin $elemMatch :", orders_c.count_documents(g), "(incluye pedidos donde un ítem tiene qty>=3 y OTRO es caro)")
print("Regex en comuna ('^Vi'):", orders_c.count_documents({"customer.comuna": {"$regex": "^Vi"}}))
''')

nb.tryit([
    "**6-1. Cancelaciones en Antofagasta:** cuenta los pedidos `cancelado` de la región Antofagasta pagados por `transferencia`. Verifica el resultado con pandas sobre `sales`.",
    "**6-2. Proyección y orden:** lista los 5 pedidos **más antiguos** (campos `ts` y `total` solamente). ¿Qué pasa si pones `sort('ts', 1)` vs `sort('ts', -1)`?",
])
nb.solutions("6-1 y 6-2", r'''
n_mongo = orders_c.count_documents({"status": "cancelado", "customer.region": "Antofagasta", "payment": "transferencia"})
n_pd = len(sales.query("status == 'cancelado' and region == 'Antofagasta' and payment == 'transferencia'"))
print(n_mongo, n_pd); assert n_mongo == n_pd                                   # 6-1
for d in orders_c.find({}, {"ts": 1, "total": 1}).sort("ts", 1).limit(5):      # 6-2: 1 = ascendente (más antiguo primero)
    print(d)
''')

M(r'''
### 6.2 Crear, actualizar y borrar (CRUD)
La **C**, **U** y **D** del CRUD. Lo más importante son los **operadores de actualización**, que modifican campos *sin reemplazar* el documento entero (y por eso son atómicos a nivel de documento):

| Operador | Efecto |
|---|---|
| `$set` / `$unset` | fija / elimina un campo |
| `$inc` | suma (o resta) un número, de forma atómica |
| `$push` / `$addToSet` | agrega a un arreglo / agrega solo si no existe |
| `$pull` | quita elementos de un arreglo |
| `upsert=True` | inserta si no existe |

Practiquemos en una colección aparte para no tocar nuestros pedidos:
''')

C(r'''
lab = db["lab"]
lab.insert_one({"_id": "p1", "name": "Camila", "points": 10, "tags": ["nuevo"]})
lab.insert_many([{"_id": "p2", "name": "Matías", "points": 5, "tags": []}, {"_id": "p3", "name": "Sofía", "points": 0, "tags": ["vip"]}])

lab.update_one({"_id": "p1"}, {"$inc": {"points": 5}, "$push": {"tags": "frecuente"}, "$set": {"city": "Temuco"}})
lab.update_many({"points": {"$lt": 10}}, {"$addToSet": {"tags": "bajo"}})
lab.update_one({"_id": "p9"}, {"$set": {"name": "Nuevo"}}, upsert=True)         # no existe → se crea
lab.update_one({"_id": "p3"}, {"$unset": {"points": ""}})                        # elimina el campo
res = lab.delete_many({"tags": "bajo"})

print("Borrados:", res.deleted_count)
for d in lab.find().sort("_id"):
    print(d)
''')

M(r'''
**Qué deberías ver:** `Borrados: 2` y solo **dos documentos** al final: `p1` (con 15 puntos, tags `["nuevo", "frecuente"]` y `city`) y `p9` (creado por el upsert). `p2` y `p3` tenían `points < 10`, recibieron el tag `"bajo"` con `$addToSet` y por eso `delete_many` los eliminó (a `p3` además se le había quitado el campo `points`).

> **NOTA:** `update_one` con un filtro que coincide con **varios** documentos modifica solo el **primero**. Es una de las causas más comunes de "actualicé y no cambió todo". Para todos: `update_many`.

### ⚠️ Troubleshooting

**Error:** `DuplicateKeyError: E11000 duplicate key error ... dup key: { _id: "p1" }`
**Por qué:** `insert_one` con un `_id` que ya existe.
**Solución:** usa `ReplaceOne(..., upsert=True)` / `update_one(..., upsert=True)` si quieres "crear o actualizar", o captura la excepción:
''')

C(r'''
try:
    lab.insert_one({"_id": "p1", "name": "Duplicado"})
except DuplicateKeyError:
    print("DuplicateKeyError capturada: el _id 'p1' ya existe → usar upsert o elegir otro _id")
''')

nb.tryit([
    "**6-3. Atómico:** usa `$inc` para sumar 100 puntos a todos los documentos de `lab` que aún tengan el campo `points` (`{'points': {'$exists': True}}`). ¿Por qué `$inc` es más seguro que leer, sumar en Python y escribir?",
    "**6-4. Contador con upsert:** escribe una función `track(coll, key)` que incremente un contador `hits` del documento `key` y lo cree en 0+1 si no existe, en **una sola operación**. Llámala 3 veces.",
])
nb.solutions("6-3 y 6-4", r'''
r = lab.update_many({"points": {"$exists": True}}, {"$inc": {"points": 100}})
print("Modificados:", r.modified_count, "| la operación es atómica: dos procesos concurrentes no pierden incrementos,")
print("en cambio 'leer → sumar → escribir' tiene una condición de carrera (lost update, ver NB-22).")

def track(coll, key):
    coll.update_one({"_id": key}, {"$inc": {"hits": 1}}, upsert=True)

for _ in range(3): track(lab, "page:/home")
print(lab.find_one({"_id": "page:/home"}))
''')

M(r'''
### 6.3 Índices: por qué una consulta tarda 1 ms o 10 s
**Intuición:** sin índice, para encontrar los pedidos de un cliente MongoDB debe leer **toda la colección** (*collection scan*, `COLLSCAN`), como buscar un nombre en una lista desordenada. Un **índice** es una estructura ordenada (B-tree) que apunta a los documentos: la búsqueda pasa de $O(n)$ a $O(\log n)$ (*index scan*, `IXSCAN`).

```
 Sin índice (COLLSCAN)                 Con índice sobre customer.id (IXSCAN)
 ┌─────────────────────────┐           ┌────────────────┐        ┌──────────────────┐
 │ doc1  doc2  doc3 ... docN│  N lecturas│ C00001 ──► doc7 │  ~log N │ solo los docs que │
 │  ✗     ✗     ✓  ...  ✗  │           │ C00002 ──► doc3 │ lecturas│ coinciden         │
 └─────────────────────────┘           └────────────────┘        └──────────────────┘
```

**Reglas de oro:**
- Indexa los campos por los que **filtras u ordenas** con frecuencia.
- En un índice **compuesto** el orden importa: *igualdad → orden → rango* (regla ESR). Un índice `(a, b)` sirve para consultas por `a` y por `a+b`, pero **no** por `b` solo.
- Cada índice **cuesta**: ocupa memoria y hace más lentas las escrituras. No indexes todo.
- Un índice `unique=True` además hace **cumplir** una restricción (como `UNIQUE` en SQL).
''')

C(r'''
orders_c.create_index("customer.id")                                                    # simple
orders_c.create_index([("customer.region", 1), ("status", 1), ("ts", -1)], name="region_status_ts")   # compuesto (ESR)
db["customers"].create_index("email", unique=True)                                      # restricción de unicidad
print("Índices de orders:", list(orders_c.index_information()))

try:
    db["customers"].insert_one({"_id": "C-nuevo", "email": "user1@correo.test"})        # email ya existente
except DuplicateKeyError:
    print("El índice único rechazó un email repetido ✔")

try:
    plan = orders_c.find({"customer.id": "C00001"}).explain()
    print("Plan ganador:", plan["queryPlanner"]["winningPlan"])
except Exception:
    print("explain() no está disponible en mongomock. En MongoDB real verás 'IXSCAN' (con índice) o 'COLLSCAN' (sin él).")
''')

M(r'''
> **NOTA:** `explain("executionStats")` en un MongoDB real te dice cuántos documentos examinó (`totalDocsExamined`) frente a cuántos devolvió (`nReturned`). Una buena consulta tiene ambos números casi iguales; si examina 1.000.000 para devolver 10, falta un índice.

### 6.4 Aggregation Pipeline: la potencia de MongoDB
Un *pipeline* es una **lista de etapas**; los documentos fluyen de una etapa a la siguiente, como una tubería. Piensa en `pandas.groupby` pero ejecutado **dentro de la base de datos**, cerca de los datos (no transfieres millones de filas a tu notebook).

| Etapa | Equivale a (SQL / pandas) | Función |
|---|---|---|
| `$match` | `WHERE` / filtro booleano | filtra (¡ponlo lo más **temprano** posible: usa índices!) |
| `$group` | `GROUP BY` / `groupby().agg` | agrupa y acumula (`$sum`, `$avg`, `$push`…) |
| `$project` / `$addFields` | `SELECT` / `assign` | elige o calcula campos |
| `$unwind` | `explode` | un documento por elemento de un arreglo |
| `$lookup` | `JOIN` / `merge` | une con otra colección |
| `$sort`, `$limit` | `ORDER BY`, `LIMIT` | ordena y recorta |
| `$bucket` | `pd.cut` + groupby | histogramas por tramos |

**Ejemplo 1 — Ingresos por región** (solo pedidos entregados). Y para aprender de verdad, **comprobamos que pandas da el mismo resultado**:
''')

C(r'''
pipeline = [
    {"$match": {"status": "entregado"}},
    {"$group": {"_id": "$customer.region", "pedidos": {"$sum": 1}, "ingreso": {"$sum": "$total"}, "ticket_medio": {"$avg": "$total"}}},
    {"$sort": {"ingreso": -1}},
]
mongo_rev = pd.DataFrame(orders_c.aggregate(pipeline)).rename(columns={"_id": "region"}).set_index("region")
display(mongo_rev.round(0))

# Verificación cruzada con pandas (misma pregunta, otra herramienta)
pd_rev = delivered.groupby("region")["total"].agg(pedidos="size", ingreso="sum", ticket_medio="mean")
assert (mongo_rev["ingreso"].sort_index() == pd_rev["ingreso"].sort_index()).all()
print("✔ MongoDB y pandas coinciden")
''')

M(r'''
**Qué deberías ver:** una tabla de 5 regiones ordenada por ingreso, con Metropolitana primero (50 % de los clientes), y la confirmación de que ambos cálculos son idénticos.

**Ejemplo 2 — Ingresos por mes** con `$dateToString`, y **Ejemplo 3 — Top 5 productos** usando `$unwind` (para "desarmar" el arreglo de ítems) y `$lookup` (para traer la categoría desde la colección `products`).
''')

C(r'''
monthly = list(orders_c.aggregate([
    {"$match": {"status": "entregado"}},
    {"$group": {"_id": {"$dateToString": {"format": "%Y-%m", "date": "$ts"}}, "ingreso": {"$sum": "$total"}}},
    {"$sort": {"_id": 1}},
]))
print("Mensual (primeros 3):", [(m["_id"], f"${m['ingreso']/1e6:.1f}M") for m in monthly[:3]])

top_products = list(orders_c.aggregate([
    {"$match": {"status": "entregado"}},
    {"$unwind": "$items"},                                                            # 1 doc por línea de pedido
    {"$group": {"_id": "$items.sku", "unidades": {"$sum": "$items.qty"},
                "ingreso": {"$sum": {"$multiply": ["$items.qty", "$items.unit_price"]}}}},
    {"$sort": {"ingreso": -1}}, {"$limit": 5},
    {"$lookup": {"from": "products", "localField": "_id", "foreignField": "_id", "as": "prod"}},
    {"$unwind": "$prod"},
    {"$project": {"sku": "$_id", "categoria": "$prod.category", "unidades": 1, "ingreso": 1, "_id": 0}},
]))
display(pd.DataFrame(top_products))
''')

M(r'''
**Qué deberías ver:** los 5 productos que más ingreso generan, con su categoría traída por `$lookup`. Observa el costo del `$lookup`: es un *join* y, en colecciones **shardeadas**, un join es caro. Esa es una razón para denormalizar (copiar la categoría dentro del ítem) si esa consulta es crítica. Nota que aquí el `$lookup` se hace **después** del `$limit 5`: une 5 documentos, no miles.

**Ejemplo 4 — Histograma con `$bucket`** (tramos de monto del pedido):
''')

C(r'''
buckets = list(orders_c.aggregate([
    {"$match": {"status": "entregado"}},
    {"$bucket": {"groupBy": "$total", "boundaries": [0, 50_000, 100_000, 200_000, 500_000, 10_000_000],
                 "default": "otros", "output": {"pedidos": {"$sum": 1}, "ingreso": {"$sum": "$total"}}}},
]))
bk = pd.DataFrame(buckets).set_index("_id")
labels = {0: "<50k", 50_000: "50-100k", 100_000: "100-200k", 200_000: "200-500k", 500_000: ">500k"}
bk.index = [labels.get(i, str(i)) for i in bk.index]      # el _id de cada tramo es su límite inferior
display(bk)

fig, ax = plt.subplots()
ax.bar(bk.index, bk["pedidos"], color="#2980b9")
ax.set(title="Pedidos por tramo de monto (calculado dentro de MongoDB)", xlabel="monto del pedido (CLP)", ylabel="nº de pedidos")
ax.grid(True, axis="y"); plt.show()
''')

M(r'''
### ⚠️ Troubleshooting — aggregation

**Error:** `OperationFailure: Exceeded memory limit for $group, but didn't allow external sorting`
**Por qué:** una etapa (`$group`, `$sort`) excede los 100 MB de RAM permitidos por etapa.
**Solución:** filtra antes con `$match` (reduce datos), agrega un índice para el `$sort`, o pasa `allowDiskUse=True` a `aggregate()`.

**Error:** el resultado de `$group` sale con `_id` en vez del nombre que quieres.
**Por qué:** en `$group` la clave de agrupación **siempre** se llama `_id`.
**Solución:** renómbrala después (`$project` o `.rename(columns={"_id": ...})` en pandas, como hicimos arriba).
''')

nb.tryit([
    "**6-5. Ticket medio por método de pago:** con un pipeline, calcula el monto promedio de pedidos entregados por `payment`. Verifica con pandas.",
    "**6-6. Pipeline en dos pasos:** obtén, por cliente, el número de pedidos y el gasto total (`$group`), y luego cuenta cuántos clientes tienen **3 o más** pedidos (`$match` después del `$group`, luego `$count`).",
    "**6-7. Orden de etapas:** ejecuta el pipeline del Ejemplo 1 poniendo `$match` **al final** en vez de al principio. ¿Da el mismo resultado? ¿Por qué en MongoDB real sería más lento?",
])
nb.solutions("6-5, 6-6 y 6-7", r'''
r = pd.DataFrame(orders_c.aggregate([{"$match": {"status": "entregado"}},
                                     {"$group": {"_id": "$payment", "ticket": {"$avg": "$total"}}}])).set_index("_id")["ticket"].sort_index()
assert np.allclose(r, delivered.groupby("payment")["total"].mean().sort_index()); print(r.round(0).to_dict())   # 6-5

n = list(orders_c.aggregate([{"$group": {"_id": "$customer.id", "pedidos": {"$sum": 1}, "gasto": {"$sum": "$total"}}},
                             {"$match": {"pedidos": {"$gte": 3}}}, {"$count": "clientes_3_o_mas"}]))
print(n)                                                                                                      # 6-6

late = list(orders_c.aggregate([{"$group": {"_id": {"r": "$customer.region", "s": "$status"}, "ingreso": {"$sum": "$total"}}},
                                {"$match": {"_id.s": "entregado"}}]))
print(len(late), "grupos (el resultado es equivalente) — pero agrupó TODO antes de filtrar: más memoria y sin usar índices.")   # 6-7
''')

M(r'''
### 6.5 Modelado: ¿embeber o referenciar?
Ya usamos **ambos** enfoques en este notebook. Esta tabla resume cuándo elegir cada uno (Kleppmann, cap. 2 y la guía de modelado de MongoDB):

| Situación | Conviene | Por qué |
|---|---|---|
| Ítems de un pedido | **Embeber** | se leen siempre juntos, y el ítem no existe sin su pedido |
| Datos del cliente en el pedido | **Embeber una copia (snapshot)** de lo necesario | el pedido refleja la dirección *de ese día*; evita `$lookup` |
| Catálogo de productos | **Referenciar** | se comparte entre millones de pedidos y se actualiza (precio) en un solo lugar |
| Relación 1:muchos **sin cota** (ej. eventos de un sensor) | **Referenciar** (o *bucket pattern*) | un documento crece sin límite y MongoDB limita cada documento a **16 MB** |
| Datos de lectura frecuente + escritura rara | **Denormalizar** | optimiza la consulta común |
| Datos que cambian y deben ser consistentes en todos lados | **Normalizar** (referencia) | un solo lugar para actualizar |

> **Honestidad:** la denormalización mueve el costo de la **lectura** a la **escritura** (si cambia la comuna de un cliente, los pedidos antiguos conservan la vieja: puede ser lo deseado, o un bug). Elige conscientemente.

### 6.6 De Mongo a pandas, y de vuelta
Un cursor de MongoDB es un iterable de `dict`. `pd.DataFrame(list(cursor))` lo convierte en tabla; para documentos **anidados** usamos `pd.json_normalize`, que "aplana" los subdocumentos con puntos (`customer.region`).
''')

C(r'''
cursor = orders_c.find({"status": "entregado", "customer.region": "Valparaíso"}, {"customer": 1, "total": 1, "ts": 1, "items": 1})
flat = pd.json_normalize(list(cursor), sep="_")           # customer.region → customer_region
flat["n_lines"] = flat["items"].str.len()                  # los arreglos quedan como listas dentro de la celda
print(flat.shape); display(flat.drop(columns=["items"]).head(3))
print("\nColumnas con nombres aplanados:", [c for c in flat.columns if c.startswith("customer")])

# Para explotar el arreglo de ítems a formato largo (una fila por línea): explode + json_normalize
lines = flat[["_id", "items"]].explode("items").reset_index(drop=True)
lines = pd.concat([lines["_id"], pd.json_normalize(lines["items"].tolist())], axis=1)
display(lines.head(3))
''')

M(r'''
**Qué deberías ver:** primero una tabla con una fila por pedido y columnas `customer_id`, `customer_region`, `customer_comuna`, etc.; luego una tabla en formato largo con una fila por **línea de pedido** (`_id`, `sku`, `qty`, `unit_price`).

> **NOTA (límite de pandas y estrategia):** `list(cursor)` carga **todo** en RAM. Con colecciones grandes, filtra y proyecta en MongoDB (solo lo que necesitas) o itera con `cursor.batch_size(10_000)` procesando por trozos, igual que `read_csv(chunksize=...)` en NB-01.

### 📊 Resumen visual — MongoDB
```
 ┌────────── CONSULTA SIMPLE ──────────┐   ┌──────────── ANÁLISIS ────────────┐
 │ find(filtro, proyección)            │   │ aggregate([ $match → $group →    │
 │   .sort().limit()                   │   │   $project → $sort → $limit ])   │
 │ operadores: $gt $in $regex $elemMatch│   │ $unwind (arreglos) · $lookup (join)│
 └─────────────────────────────────────┘   └──────────────────────────────────┘
 ┌──────── ESCRITURA ─────────┐  ┌──────── RENDIMIENTO ────────┐  ┌──── DISEÑO ────┐
 │ insert · update ($set $inc │  │ índices: simple, compuesto  │  │ embeber: se lee │
 │ $push) · replace · delete  │  │ (ESR), unique · explain     │  │ junto · referen-│
 │ upsert = idempotencia      │  │ $match primero              │  │ ciar: compartido│
 └────────────────────────────┘  └─────────────────────────────┘  └────────────────┘
```
''')

# =====================================================================
# PIPELINE COMPLETO
# =====================================================================
M(r'''
## 7. Pipeline completo: pandas → MongoDB → pandas

Cerramos el círculo. Con los datos **ya cargados por el ETL**, respondemos una pregunta de negocio en dos pasos: (1) MongoDB hace la agregación pesada; (2) pandas hace el análisis final y el gráfico.

**Pregunta:** *¿Qué categorías de producto generan más ingreso en cada región?*

Esta pregunta necesita dos colecciones (`orders` y `products`): un buen caso para `$lookup`. Aquí **elegimos** dónde hacer cada cosa: la agregación grande dentro de Mongo (devuelve ~30 filas, no ~14.000) y el pivoteo/visualización en pandas.
''')

C(r'''
cat_region = list(orders_c.aggregate([
    {"$match": {"status": "entregado"}},
    {"$unwind": "$items"},
    # Agrupamos ANTES del $lookup: así el join se hace sobre ≤ 5 regiones × 150 SKUs, no sobre cada línea de pedido
    {"$group": {"_id": {"region": "$customer.region", "sku": "$items.sku"},
                "ingreso": {"$sum": {"$multiply": ["$items.qty", "$items.unit_price"]}}}},
    {"$lookup": {"from": "products", "localField": "_id.sku", "foreignField": "_id", "as": "p"}},
    {"$unwind": "$p"},
    {"$group": {"_id": {"region": "$_id.region", "categoria": "$p.category"}, "ingreso": {"$sum": "$ingreso"}}},
]))
cr = pd.DataFrame([{**d["_id"], "ingreso": d["ingreso"]} for d in cat_region])
piv = cr.pivot_table(index="region", columns="categoria", values="ingreso", aggfunc="sum")
share = piv.div(piv.sum(axis=1), axis=0)           # participación de cada categoría dentro de su región

fig, ax = plt.subplots(figsize=(10, 4.5))
sns.heatmap(share, annot=True, fmt=".0%", cmap="YlGnBu", ax=ax)
ax.set(title="Participación de cada categoría en el ingreso de su región", xlabel="categoría", ylabel="región")
plt.show()

# Verificación independiente con pandas puro sobre los datos limpios
chk = (items_w.merge(delivered[["order_id", "region"]], on="order_id").merge(products[["sku", "category"]], on="sku")
              .assign(rev=lambda d: d["qty"] * d["unit_price"]).pivot_table(index="region", columns="category", values="rev", aggfunc="sum"))
assert np.allclose(piv.sort_index().to_numpy(), chk.sort_index().to_numpy())
print("✔ Resultado idéntico calculando en MongoDB y en pandas")
''')

M(r'''
**Qué deberías ver:** una tabla de calor 5 regiones × 6 categorías donde cada fila suma 100 %. Como los datos son sintéticos, las participaciones serán parecidas entre regiones; en datos reales las diferencias regionales son justamente lo que busca el análisis.

### ¿Cuándo calcular en Mongo y cuándo en pandas?
| Calcula en **MongoDB** (aggregation) cuando… | Calcula en **pandas** cuando… |
|---|---|
| el dato es grande y el resultado es pequeño | el resultado cabe en RAM y vas a iterar/explorar |
| hay un índice que acelera el `$match` | necesitas funciones que Mongo no tiene (ventanas, `merge_asof`, ML) |
| varias aplicaciones comparten el cálculo | es un análisis único / ad-hoc |
| no quieres mover datos por la red | quieres graficar o modelar |
''')

# =====================================================================
# CHEAT SHEET
# =====================================================================
M(r'''
## 8. 📋 Cheat Sheet

### pandas avanzado
| Concepto / Función | Qué hace | Cuándo usarlo | Sintaxis rápida |
|---|---|---|---|
| `pd.to_numeric(downcast=)` | reduce el tamaño de enteros/floats | memoria alta | `pd.to_numeric(s, downcast="integer")` |
| `astype("category")` | texto repetido → códigos | columnas con pocos valores únicos | `df["region"].astype("category")` |
| `memory_usage(deep=True)` | memoria real en bytes | medir antes/después | `df.memory_usage(deep=True).sum()/1e6` |
| `unstack` / `stack` | nivel del índice ↔ columnas | tabla ancha desde `groupby([a,b])` | `s.unstack("region")` |
| `.xs(clave, level=)` | corte transversal de un MultiIndex | seleccionar un nivel | `s.xs("Biobío", level="region")` |
| `rolling(k)` | ventana móvil de tamaño fijo | suavizar, volatilidad | `s.rolling(7).mean()` |
| `rolling("7D")` | ventana por tiempo | datos irregulares | `s.rolling("7D").mean()` |
| `expanding()` | acumulado desde el inicio | máximo/promedio histórico | `s.expanding().max()` |
| `ewm(span=)` | media con decaimiento exponencial | reaccionar a lo reciente | `s.ewm(span=7).mean()` |
| `resample("W")` | re-muestrear por tiempo | agregar a otra frecuencia | `s.resample("W").sum()` |
| `shift(n)` / `pct_change()` | rezago / variación % | features de lag | `s.shift(7)` |
| `merge_asof` | join por el valor más cercano | unir eventos a series de tiempo | `pd.merge_asof(a, b, on="t", direction="backward")` |
| `pd.qcut(rank(first))` | cuantiles robustos | RFM, scoring | `pd.qcut(s.rank(method="first"), 5, labels=False)` |
| `query("a > @x")` | filtro legible | condiciones compuestas | `df.query("a > @x and b == 'k'")` |
| `itertuples()` | iterar filas (rápido) | si no se puede vectorizar | `for r in df.itertuples(): r.col` |

### MongoDB / PyMongo
| Concepto / Función | Qué hace | Cuándo usarlo | Sintaxis rápida |
|---|---|---|---|
| `MongoClient(uri)` | conecta al servidor | inicio | `db = MongoClient(uri)["mi_db"]` |
| `insert_many` | inserta varios docs | cargas simples | `coll.insert_many(docs)` |
| `bulk_write([ReplaceOne(...)])` | operaciones en lote | ETL idempotente | `ReplaceOne({"_id": i}, doc, upsert=True)` |
| `find(filtro, proy)` | consulta | lectura | `coll.find({"a": {"$gt": 5}}, {"a": 1})` |
| `$elemMatch` | condición sobre un elemento de arreglo | arreglos de subdocumentos | `{"items": {"$elemMatch": {"qty": {"$gte": 3}}}}` |
| `update_one/many` | modifica campos | actualizaciones parciales | `coll.update_many(f, {"$inc": {"n": 1}})` |
| `$set $unset $inc $push $addToSet $pull` | operadores de actualización | siempre que no reemplaces todo | `{"$push": {"tags": "x"}}` |
| `create_index` | crea índice | filtros/orden frecuentes | `coll.create_index([("a", 1), ("b", -1)])` |
| `explain()` | plan de ejecución | depurar lentitud | `coll.find(f).explain()["queryPlanner"]` |
| `aggregate([...])` | pipeline de etapas | análisis en la BD | `coll.aggregate([{"$match": …}, {"$group": …}])` |
| `$unwind` / `$lookup` | desarmar arreglo / join | análisis por línea / unir colecciones | `{"$unwind": "$items"}` |
| `pd.json_normalize` | aplana documentos anidados | Mongo → pandas | `pd.json_normalize(list(cur), sep="_")` |

### Patrón ETL en 6 líneas
```python
raw = extract(landing)                       # leer con tipos explícitos
clean, rejected = transform(raw, report)     # reglas + cuarentena (con motivo)
validate_bson(docs)                          # fallar temprano
load(coll, docs)                             # bulk upsert por _id → idempotente
check_quality(report, max_reject=0.05)       # alertar si algo cambió en el origen
```
''')

# =====================================================================
# MUNDO REAL
# =====================================================================
M(r'''
## 9. Conexiones con el Mundo Real

**En la industria.** El patrón *extraer con pandas → validar → cargar a una base de documentos → consultar con agregaciones* es el esqueleto de muchísimos sistemas de datos: catálogos de e-commerce, perfiles de usuario, registros de eventos, internet de las cosas. MongoDB suele elegirse cuando los datos son heterogéneos o cambian de forma con frecuencia (esquema flexible) y se necesita escalar horizontalmente; para analítica pesada sobre históricos, el estándar es el formato columnar (Parquet) con motores como Spark o DuckDB.

**Big Data (CIT-2309).** Este notebook es una plantilla para el proyecto: `extract` (CSV/APIs) → `transform` (reglas y cuarentena) → `load` (MongoDB) → `aggregate` (análisis). Lo que **debes poder justificar** en una presentación: por qué embebiste o referenciaste cada entidad, qué índices creaste y cómo lo verificaste con `explain`, y por qué tu carga es idempotente.

**Sistemas Distribuidos.** Tres ideas de este notebook reaparecen en NB-22: (1) **idempotencia** = tolerancia a reintentos, (2) **shard key** como decisión que condiciona la escalabilidad, (3) **replica sets** y consistencia: por defecto lees del primario (consistente); leer de réplicas es más rápido pero puede devolver datos levemente desactualizados.

**Telecomunicaciones.** Los operadores móviles registran millones de eventos por segundo (CDRs, métricas por celda). Un diseño típico guarda eventos crudos con un *bucket pattern* (un documento por celda y hora con un arreglo de mediciones) y consulta agregados con ventanas móviles, exactamente la idea de la sección 4.3.

**Tesis (SNN + diagrama de ojo + BER).** Cada simulación de tu ecualizador produce un registro con **parámetros** (número de neuronas, umbral, ruido), **métricas de ojo** por neurona (apertura, ancho, jitter) y el **BER** resultante. Los experimentos añaden campos con el tiempo: es el caso de uso ideal para un esquema flexible y consultas por rango de SNR. Lo practicas en el Ejercicio 3; en NB-25 se usa para el pipeline experimental.

**Contexto chileno / latinoamericano.** Retail y comercio electrónico (campañas como CyberDay), banca/fintech (segmentación y detección de fraude con RFM y ventanas de comportamiento), minería (series de sensores por equipo) y salud pública (registros clínicos de esquema variable) comparten estos patrones. Para practicar con datos reales, los portales de datos abiertos del Estado y las series del Banco Central son un buen punto de partida.

**Limitaciones y alternativas.**
- pandas + MongoDB **no** es una arquitectura de streaming: para ingestión continua se usan colas (Kafka) y procesadores de flujo (ver NB-22).
- Sin transacciones multi-documento (disponibles, pero con costo), invariantes entre colecciones son responsabilidad de la aplicación.
- Si tus consultas son principalmente analíticas y columnares, MongoDB no es lo más eficiente: **Parquet + DuckDB/Polars** suele ser mucho mejor.
- `mongomock` sirve para aprender y testear, pero no reproduce rendimiento, `explain` ni algunas etapas avanzadas: valida en un MongoDB real antes de producción.
''')

# =====================================================================
# EJERCICIOS
# =====================================================================
M(r'''
## 10. Ejercicios progresivos

Cada nivel tiene una celda **"Tu turno"** y, **después**, su solución completa en celdas separadas. Resiste la tentación de abrirla primero: lo que cuesta es lo que se aprende.

---
### Nivel 1 — Reproducir (guiado): KPI semanal en MongoDB
**Objetivo:** calcular una media móvil de ingresos semanales y guardarla en una colección `weekly_kpis`, con una carga idempotente.

**Pasos:**
1. Con `delivered`, calcula el ingreso por semana (`resample("W")`) en millones de CLP.
2. Agrega la media móvil de 4 semanas (`rolling(4)`) y la variación porcentual (`pct_change`).
3. Convierte cada fila en un documento con `_id` = fecha de fin de semana en texto (`"2025-01-05"`), y tipos nativos (ojo con `NaN` → `None`).
4. Cárgala con `load(...)` **dos veces** y verifica que el conteo no cambia.
5. Lee de vuelta con `find` y confirma que los valores coinciden.

> 💡 **Hints:** `.dt.strftime("%Y-%m-%d")` para el id; `float(x) if pd.notna(x) else None`; reutiliza la función `load` de la sección 5.3.
''')

C(r'''
# Tu turno (Nivel 1)
''')

M(r'''#### ✅ Solución — Nivel 1''')
C(r'''
weekly = (delivered.set_index("order_ts")["total"].resample("W").sum() / 1e6).rename("ingreso_MM").to_frame()
weekly["ma4"] = weekly["ingreso_MM"].rolling(4).mean()
weekly["var_pct"] = weekly["ingreso_MM"].pct_change()

def nn(x):   # NaN → None y numpy → float nativo (BSON no digiere numpy.float64 en todos los contextos)
    return float(x) if pd.notna(x) else None

weekly_docs = [{"_id": ts.strftime("%Y-%m-%d"), "ingreso_MM": nn(r.ingreso_MM), "ma4": nn(r.ma4), "var_pct": nn(r.var_pct)}
               for ts, r in weekly.iterrows()]
validate_bson(weekly_docs)

wk = db["weekly_kpis"]
load(wk, weekly_docs); n1 = wk.count_documents({})
load(wk, weekly_docs); n2 = wk.count_documents({})
assert n1 == n2 == len(weekly_docs)

back = pd.DataFrame(wk.find().sort("_id", 1)).set_index("_id")
assert np.allclose(back["ingreso_MM"], weekly["ingreso_MM"])
print(f"✔ {n2} semanas cargadas (idempotente). Ejemplo:", back.iloc[4].to_dict())
''')

M(r'''
---
### Nivel 2 — Aplicar (problema nuevo): segmentos RFM dentro de MongoDB
**Problema:** el equipo de marketing quiere ver el segmento RFM de cada cliente **dentro de la colección `customers`** para consultarlo desde su aplicación.

**Tu tarea (sin paso a paso):**
1. Usando el DataFrame `rfm` de la sección 4.5, escribe el campo `segment` en cada documento de `customers` con **operaciones en lote** (`bulk_write` + `UpdateOne` con `$set`). Los clientes sin compras entregadas deben quedar con `segment = "sin_compras"`.
2. Con un *aggregation pipeline* calcula, por segmento, el número de clientes y el **ingreso total** de pedidos entregados (necesitarás un `$lookup`; *pista:* agrupa primero los pedidos por cliente para que el join sea pequeño).
3. Verifica que tu resultado coincide con el cálculo en pandas.
4. Crea un índice sobre `segment` y explica en una línea cuándo serviría.
''')

C(r'''
# Tu turno (Nivel 2)
''')

M(r'''#### ✅ Solución — Nivel 2''')
C(r'''
cust_c = db["customers"]
seg = rfm["segment"].to_dict()
ops = [UpdateOne({"_id": cid}, {"$set": {"segment": seg.get(cid, "sin_compras")}})        # $set: solo toca este campo
       for cid in customers["customer_id"]]
r = cust_c.bulk_write(ops, ordered=False)
print("Clientes actualizados:", r.matched_count)

pipe = [
    {"$match": {"status": "entregado"}},
    {"$group": {"_id": "$customer.id", "ingreso": {"$sum": "$total"}}},                 # primero: 1 fila por cliente
    {"$lookup": {"from": "customers", "localField": "_id", "foreignField": "_id", "as": "c"}},   # join pequeño
    {"$unwind": "$c"},
    {"$group": {"_id": "$c.segment", "clientes": {"$sum": 1}, "ingreso": {"$sum": "$ingreso"}}},
    {"$sort": {"ingreso": -1}},
]
mongo_seg = pd.DataFrame(orders_c.aggregate(pipe)).set_index("_id")
display(mongo_seg)
print("Clientes sin compras entregadas:", cust_c.count_documents({"segment": "sin_compras"}))

ref = (rfm.groupby("segment").agg(clientes=("monetary", "size"), ingreso=("monetary", "sum")))
for s_ in ref.index:
    assert mongo_seg.loc[s_, "ingreso"] == ref.loc[s_, "ingreso"] and mongo_seg.loc[s_, "clientes"] == ref.loc[s_, "clientes"]
print("✔ Coincide con pandas")

cust_c.create_index("segment")
# Sirve cuando la aplicación consulta "dame los clientes en riesgo" con frecuencia (filtro por segment).
# Con solo 5 valores distintos tiene baja selectividad: ayuda más combinado, p. ej. (segment, region).
''')

M(r'''
---
### Nivel 3 — Integrar (cross-curso + Tesis): registro de experimentos SNN/BER
**Contexto:** en tu tesis vas a correr cientos de simulaciones de un ecualizador SNN. Cada una produce parámetros, métricas de ojo por neurona y un BER. Quieres guardarlas en MongoDB y consultarlas.

**Tu tarea:**
1. Genera **300 experimentos sintéticos**. Para cada uno: `params` (`n_neurons` ∈ {16, 32, 64}, `threshold`, `snr_db` ∈ [4, 16]), `eye` con `height`, `width`, `jitter_rms` (relacionados con el SNR más ruido) y `ber` (usa $\text{BER}\approx\tfrac12\,\text{erfc}\!\big(\sqrt{\text{SNR}_{lin}/2}\big)$, con ruido multiplicativo). Algunos experimentos deben tener un campo extra `notes` (esquema flexible).
2. Cárgalos con `insert_many`.
3. Con un pipeline, calcula el **BER medio por tramo de SNR** (`$bucket`) y por `n_neurons`.
4. Trae a pandas los datos aplanados (`json_normalize`) y calcula la correlación de `eye.height` con $\log_{10}(\text{BER})$. Grafícala.

> ⚠️ **Los datos son sintéticos**: ilustran el flujo de trabajo, no resultados reales de tesis. Se retoma en NB-25.
''')

C(r'''
# Tu turno (Nivel 3)
''')

M(r'''#### ✅ Solución — Nivel 3''')
C(r'''
rg = np.random.default_rng(11)
N = 300
snr_db = rg.uniform(4, 16, N)
snr_lin = 10 ** (snr_db / 10)
ber = np.clip(0.5 * erfc(np.sqrt(snr_lin / 2)) * rg.lognormal(0, 0.25, N), 1e-9, 0.5)   # ruido multiplicativo
height = np.clip(0.12 * snr_db + rg.normal(0, 0.15, N), 0.05, None)                     # apertura crece con el SNR
exp_docs = [{
    "_id": f"exp-{i:04d}",
    "params": {"n_neurons": int(rg.choice([16, 32, 64])), "threshold": round(float(rg.uniform(0.3, 1.0)), 3), "snr_db": round(float(snr_db[i]), 2)},
    "eye": {"height": round(float(height[i]), 4), "width": round(float(np.clip(0.5 + 0.02 * snr_db[i] + rg.normal(0, 0.03), 0.1, 1)), 4),
            "jitter_rms": round(float(np.clip(0.2 - 0.008 * snr_db[i] + rg.normal(0, 0.01), 0.01, None)), 4)},
    "ber": float(ber[i]),
    **({"notes": "re-run con semilla distinta"} if i % 10 == 0 else {}),        # campo extra solo en algunos: esquema flexible
} for i in range(N)]

exps = db["experiments"]
exps.insert_many(exp_docs)
print("Experimentos:", exps.count_documents({}), "| con 'notes':", exps.count_documents({"notes": {"$exists": True}}))

res = list(exps.aggregate([
    {"$bucket": {"groupBy": "$params.snr_db", "boundaries": [4, 8, 12, 16.01], "output": {"n": {"$sum": 1}, "ber_medio": {"$avg": "$ber"}}}},
]))
display(pd.DataFrame(res).rename(columns={"_id": "snr_desde"}).set_index("snr_desde"))
by_n = pd.DataFrame(exps.aggregate([{"$group": {"_id": "$params.n_neurons", "ber_medio": {"$avg": "$ber"}, "n": {"$sum": 1}}}, {"$sort": {"_id": 1}}]))
display(by_n)

df_exp = pd.json_normalize(list(exps.find()), sep="_")
df_exp["log10_ber"] = np.log10(df_exp["ber"])
r_ = df_exp["eye_height"].corr(df_exp["log10_ber"])
fig, ax = plt.subplots()
sc = ax.scatter(df_exp["eye_height"], df_exp["log10_ber"], c=df_exp["params_snr_db"], cmap="viridis", s=25)
fig.colorbar(sc, label="SNR (dB)")
ax.set(title=f"Apertura del ojo vs. BER (sintético) — correlación = {r_:.2f}", xlabel="altura del ojo (u.a.)", ylabel="log10(BER)")
ax.grid(True); plt.show()
print(f"Correlación (altura del ojo, log10 BER) = {r_:.3f} → negativa: ojo más abierto ⇒ menor BER")
assert r_ < -0.8
''')

M(r'''
**Qué deberías ver:** un BER medio que **cae varios órdenes de magnitud** al subir el SNR, y una nube de puntos con pendiente claramente negativa: ojo más abierto ⇒ menor BER. Es justo la relación que tu tesis querría **aprender** con ML en lugar de calcularla con una fórmula (NB-25). Observa que el *esquema flexible* te dejó añadir `notes` solo a algunos experimentos sin migrar nada.
''')

# =====================================================================
# CONCLUSIÓN
# =====================================================================
M(r'''
## 11. Resumen

1. Con **tipos adecuados** (`downcast`, `category`) y **vectorización** pandas rinde mucho más; `iterrows`/`apply(axis=1)` son los grandes enemigos.
2. Las **ventanas** (`rolling`, `ewm`, `expanding`), `resample` y `merge_asof` cubren el análisis de series de tiempo y cohortes, cuidando no filtrar información del futuro.
3. Un **ETL** serio es idempotente (upsert), valida con reglas explícitas, manda lo rechazado a cuarentena y reporta métricas.
4. **MongoDB** guarda documentos: se consulta con `find`, se actualiza con operadores atómicos (`$set`, `$inc`, `$push`), se acelera con **índices** (regla ESR) y se analiza con el **aggregation pipeline**.
5. **Embeber** lo que se lee junto, **referenciar** lo compartido; calcula en Mongo lo grande-y-pequeño-de-resultado, y en pandas lo exploratorio.

## Siguiente paso
→ **Continúa en:** **NB-03 — NumPy Avanzado y Álgebra Lineal para ML** (la base matemática de PCA, regresión y backpropagation). Más adelante: **NB-09** (proyecto ML end-to-end, usa este mismo flujo de datos) y **NB-22** (convertir este ETL en un DAG de Airflow).

## Para profundizar
- *VanderPlas, Python Data Science Handbook*, cap. 3: *Hierarchical Indexing*, *Combining Datasets*, *Aggregation and Grouping*, *High-Performance Pandas: eval() and query()*, *Working with Time Series*.
- *Kleppmann, DDIA*, cap. 2 (*Relational Model Versus Document Model*), cap. 3 (*Indexes*), cap. 5–6 (réplicas y particionamiento) y cap. 11 (idempotencia en procesamiento).
- Documentación de MongoDB: *Data Modeling Patterns*, *Aggregation Pipeline Quick Reference* e *Indexing Strategies* (regla ESR).
- *Data Pipelines with Apache Airflow*, cap. 1–3: cómo convertir `extract/transform/load` en tareas de un DAG.
''')

nb.build("NB-02_pandas_mongodb.ipynb")
