# TFM — Predicción del bienestar laboral a partir de reseñas de Glassdoor

**Trabajo de Fin de Máster — UNED**  
Uso de características NLP extraídas de reseñas de Glassdoor para predecir valoraciones de bienestar laboral, con experimentos de generalización entre mercados (S&P 500 → FTSE 100).

---

## Qué contiene este repositorio

Todo el código necesario para reproducir los experimentos del TFM, desde la recogida de datos hasta el entrenamiento de modelos y la evaluación cross-market. Lo único que no está incluido son los archivos de datos grandes (ver [Qué no está en el repo](#qué-no-está-en-el-repo)).

---

## Requisitos

```bash
pip install pandas numpy scikit-learn xgboost lightgbm sentence-transformers nrclex empath nltk matplotlib openpyxl google-cloud-storage google-cloud-aiplatform
```

Para entrenamiento en **Google Cloud Vertex AI**:
```bash
gcloud auth application-default login
export GCP_PROJECT="tu-proyecto"
export GCP_BUCKET="tu-bucket"
```

---

## Estructura de carpetas

```
repo/
├── notebooks/                      # Pipeline principal, paso a paso
├── scripts/                        # Versiones .py independientes del entrenamiento
├── vertex/                         # Jobs de entrenamiento en Google Cloud Vertex AI
├── ftse100/                        # Experimento cross-market con FTSE 100
├── 02_other_inputs_preprocessing/  # Datos de empresa, estado y bolsa
├── analysis/                       # Scripts y outputs de análisis de correlación
├── results/                        # Resultados: PDFs, importancia de features, Excel
└── docs/                           # Papers de referencia y guía de ejecución
```

---

## Pipeline

### Paso 0 — Scraping de Glassdoor *(externo, no reproducible desde cero)*

Las reseñas fueron scrapeadas por Camilo Pinzón Castellanos (UNED) usando `notebooks/Glassdoor web scraper II.ipynb`. El output es un conjunto de CSVs por empresa en `01_outputs_p1/01_Reviews/` y embeddings RoBERTa en `01_outputs_p1/02_Embeddings/`.

Cada CSV ya contiene las siguientes features NLP pre-computadas:

| Grupo de features | Columnas |
|---|---|
| Densidades POS (NLTK) | `summary_ld_*`, `pros_ld_*`, `cons_ld_*` |
| EmoLex (NRCLex) | `pros_emolex_*`, `cons_emolex_*` |
| Sentimiento NLTK (SIA) | `pros_nltk_sia`, `cons_nltk_sia` |
| Empath | `pros_empath_*`, `cons_empath_*` |
| Similitud ODI (RoBERTa) | `pros_sim_ODI_*`, `cons_sim_ODI_*` |
| Similitud JDI (RoBERTa) | `pros_sim_JDI_*`, `cons_sim_JDI_*` |
| Vocabulario ODI | `ODI_related_vocab_pros`, `ODI_related_vocab_cons` |

> **Nota:** Estos datos (~16 GB) no están en el repo. Contacta con los autores para obtenerlos o vuelve a ejecutar el scraper (requiere cuenta de Glassdoor y tiempo de cómputo considerable).

---

### Paso 1 — Datos bursátiles

**Notebook:** `notebooks/Obtain stock data.ipynb`  
Descarga precios históricos de las empresas del S&P 500 mediante la API de `yfinance`. El output va a `02_other_inputs_preprocessing/`.

---

### Paso 2 — Añadir features contextuales

**Notebook:** `notebooks/Bring data from P1 and add necessary features.ipynb`  
Lee los CSVs de reseñas de `01_outputs_p1/01_Reviews/` y los enriquece con datos de empresa, estado y bolsa de `02_other_inputs_preprocessing/`.

**Inputs necesarios:**
- `01_outputs_p1/01_Reviews/*.csv` (del Paso 0)
- `02_other_inputs_preprocessing/01_State and company data/` ✅ incluido en el repo

---

### Paso 3 — Preprocesamiento de datos

**Notebook:** `notebooks/Data preprocessing.ipynb`  
Lee la configuración de `config_file.json`. Aplica división train/test, imputación de valores ausentes, tratamiento de outliers, ingeniería de features (target encoding, one-hot encoding, features de fecha) y StandardScaler. El output va a `03_outputs_data_preprocessing/output_data_preprocessing_<timestamp>/`.

> **Importante:** El timestamp generado (ej. `20240322_105534`) hay que copiarlo en la variable `EXECUTION_TO_USE` de los scripts `scripts/run_*.py` antes de ejecutarlos.

---

### Paso 4 — Entrenamiento de modelos (local)

Dos opciones equivalentes:

**Opción A — Notebook:** `notebooks/Model building.ipynb`  
Interactivo, con gráficas. Lee desde `03_outputs_data_preprocessing/`.

**Opción B — Scripts:** `scripts/run_*.py`  
Un script por algoritmo. Ejecutar desde la raíz del repo:
```bash
python scripts/run_xgboost.py
python scripts/run_random_forest.py
python scripts/run_elasticnet.py
python scripts/run_knn.py
python scripts/run_decision_tree.py
```

> Antes de ejecutar, cambiar `EXECUTION_TO_USE` en cada script al timestamp de tu ejecución del Paso 3.

Los grupos de features (1–7) y las configuraciones de PCA están definidos dentro de cada script. El output va a `05_outputs_modeling/`.

---

### Paso 4 (nube) — Google Cloud Vertex AI

Para entrenamiento paralelo a gran escala. Requiere un proyecto GCP configurado.

```bash
# Subir datos preprocesados a GCS
python vertex/upload_data.py

# Lanzar todos los jobs
python vertex/submit_all.py

# O lanzar grupos individuales
python vertex/submit_grupo1.py
python vertex/submit_grupo2.py
# ...
```

Los paquetes de entrenamiento de cada grupo están en `vertex/grupo*_vertex/`.

---

### Paso 5 — Experimentos cross-market

Los CSVs del FTSE 100 ya contienen todas las features NLP pre-computadas (EmoLex, Empath, NLTK SIA, similitudes ODI/JDI basadas en RoBERTa, densidades POS). Ver [Datos FTSE 100](#datos-ftse-100-no-incluidos).

**Experimento B — FTSE100 → FTSE100 (baseline dentro del mercado UK):**
```bash
python scripts/run_cross_B_ftse_ftse.py
```

**Experimento C — S&P500 → FTSE100 (generalización cross-market):**
```bash
python scripts/run_cross_C_sp500_ftse.py
```

Generación del informe de resultados:
```bash
python scripts/generate_crosstraining_report.py
```

---

## Qué no está en el repo

### Features NLP de las reseñas SP500 (`01_outputs_p1/`)

Los CSVs por empresa con todas las features pre-computadas no están incluidos (~16 GB). Son la entrada directa al Paso 2. Contacta con los autores o vuelve a ejecutar el scraper.

### Datos preprocesados (`03_outputs_data_preprocessing/`)

Se generan en el Paso 3. Vuelve a ejecutar `notebooks/Data preprocessing.ipynb`.

### Outputs de modelos (`05_outputs_modeling/`)

Se generan en el Paso 4. Vuelve a ejecutar los scripts de entrenamiento.

### Datos FTSE 100 (`ftse100/data/`)

Los dos CSVs del FTSE 100 superan el límite de 100 MB de GitHub:
- `dataframes_Train_scrapeadas.csv` — 557 MB
- `dataframes_Test_scrapeadas.csv` — 140 MB

Para reproducir los experimentos cross-market, coloca estos archivos en `ftse100/data/` antes de ejecutar los scripts. Ver `ftse100/data/README.md`.

---

## Scripts de preprocesamiento FTSE 100

Si necesitas recomputar las features desde datos crudos del FTSE 100:

```
ftse100/
├── compute_emolex_ftse100.py   # Recomputa features EmoLex desde reseñas crudas
├── merge_ftse100.py            # Une los archivos de reseñas por empresa
├── normalize_ftse100.py        # Normaliza y estandariza los datos FTSE100
└── analisis_uk_ftse100.py      # Análisis exploratorio del dataset FTSE100
```

---

## Dashboard — NVIDIA Employee Satisfaction Predictor

Demo en tiempo real construida sobre el mejor modelo (XGBoost, Feature Group 1) que predice la satisfacción de empleados de NVIDIA a partir de una reseña nueva.

```
Dashboard/
├── fastapi/
│   ├── main.py              # API REST: recibe reseña → calcula features → predice
│   ├── start.sh             # Script de arranque
│   ├── requirements.txt     # Dependencias Python
│   └── scaler.pkl           # Scaler entrenado
├── Modelo FG-1/
│   └── dict_*.pkl           # Modelo XGBoost entrenado (11 MB)
├── n8n_workflow.json        # Flujo n8n: Tally → FastAPI → gold.csv
├── Dashboard TFM final.twb  # Dashboard Tableau (conectado a gold.xlsx)
├── gold.xlsx                # Log de predicciones (fuente de datos de Tableau)
├── NVDA_gold.csv            # Predicciones batch sobre todas las reviews de NVDA
├── batch_predict_nvda.py    # Predicción batch sobre el dataset preprocesado
├── generate_keywords.py     # Genera word cloud para Tableau desde gold.xlsx
└── generate_pdf.py          # Genera el PDF de resumen del producto
```

### Arrancar la API localmente

```bash
cd Dashboard/fastapi
pip install -r requirements.txt
bash start.sh
# → http://localhost:8000/predict
```

**Requisito:** los embeddings JDI/ODI deben estar en `01_outputs_p1/02_Embeddings/` relativo a la raíz del repo.

### Flujo completo

```
Tally (formulario) → n8n → POST /predict → FastAPI
  → POS tags + EmoLex + VADER + Empath + JDI/ODI sim (RoBERTa all-roberta-large-v1)
  → XGBoost → "Alto" / "Bajo"
  → gold.xlsx ← Tableau dashboard
```

Para el flujo n8n: importar `n8n_workflow.json` en n8n y configurar la variable de entorno `GOLD_CSV_PATH` con la ruta absoluta a `Dashboard/gold.csv` en tu máquina.

---

## Grupos de features

| Grupo | Descripción |
|---|---|
| 1 | Solo longitud de texto |
| 2 | Grupo 1 + POS + EmoLex + VADER + ODI vocab + Empath + geográficas + empresa + bolsa |
| 3 | Grupo 2 sin features geográficas/empresa/bolsa (NLP puro) |
| 4 | Grupo 3 + similitud JDI/ODI (RoBERTa) |
| 5 | Grupo 2 + similitud JDI/ODI (set completo) |
| 6 | Grupo 5 sin POS/EmoLex/VADER/Empath/ODI vocab (solo similitud semántica) |
| 7 | Grupo 5 sin similitud JDI/ODI |

Cada grupo se prueba con 3 configuraciones de PCA: sin PCA, 1 componente/grupo, 5 componentes/grupo.

---

## Resultados pre-computados

En `results/`:
- `Cross_Training_Results.pdf` — resultados de los experimentos cross-market
- `Feature dictionary and execution planner.xlsx` — definición de grupos de features y registro de ejecuciones
- `feature_importance_5_6.png/csv` — importancia de features para los grupos 5 y 6

---

## Archivo de configuración

`config_file.json` controla el pipeline de preprocesamiento: rutas de entrada, proporciones train/test, umbrales de outliers, fecha de referencia y parámetros de ingeniería de features. Editar antes del Paso 3 si es necesario.

---

## Seguridad

Todas las credenciales de GCP se leen de variables de entorno:
```bash
export GCP_PROJECT="tu-proyecto"
export GCP_BUCKET="tu-bucket"
```

No hay credenciales hardcodeadas en el código. La carpeta `.claude/` (contexto de sesión local) está excluida del repo mediante `.gitignore`.
