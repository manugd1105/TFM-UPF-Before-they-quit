"""
batch_predict_nvda.py
---------------------
Extrae las reviews de NVDA del CSV de preprocesamiento (ya escaladas),
carga el modelo XGBoost, calibra las predicciones y guarda el gold CSV.

NOTA: El modelo fue guardado con sklearn 1.7.2 pero se ejecuta con 1.4.2.
Esto provoca que el base_score de XGBoost (≈media del target) no se restaure
correctamente, desplazando las predicciones ~3.2 unidades. Se corrige con
una calibración lineal (regresión sobre 50k muestras del training data).

Uso:
    python batch_predict_nvda.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

# ── Rutas ────────────────────────────────────────────────────────────────────
# BASE_DIR apunta a la raíz del repo (un nivel arriba de Dashboard/)
BASE_DIR    = Path(__file__).resolve().parent.parent
MODEL_PATH  = BASE_DIR / "Dashboard/Modelo FG-1/dict_20260614_144022.pkl"
# NOTA: ajusta EXECUTION_TO_USE al timestamp de tu ejecución de preprocesamiento
EXECUTION_TO_USE = os.environ.get("EXECUTION_TO_USE", "20240322_105534")
PREPROC_DIR = BASE_DIR / f"03_outputs_data_preprocessing/output_data_preprocessing_{EXECUTION_TO_USE}"
TRAIN_CSV   = PREPROC_DIR / f"dataframes_Train_{EXECUTION_TO_USE}.csv"
TEST_CSV    = PREPROC_DIR / f"dataframes_Test_{EXECUTION_TO_USE}.csv"
OUTPUT_CSV  = BASE_DIR / "Dashboard/NVDA_gold.csv"

COMPANY        = "NVDA"
CAL_NROWS      = 50_000   # filas de training para estimar calibración
CAL_THRESHOLD  = 4.0      # umbral Alto / Bajo

# ── Carga modelo ──────────────────────────────────────────────────────────────
print("Cargando modelo...")
with open(MODEL_PATH, "rb") as f:
    _pkl = pickle.load(f)

MODEL         = _pkl["Random search object"].best_estimator_
FEATURE_ORDER = list(MODEL.feature_names_in_)
print(f"Modelo cargado — {len(FEATURE_ORDER)} features")

# ── Calibración lineal sobre training data ────────────────────────────────────
# Corrige el desplazamiento causado por el base_score incorrecto al cargar el pkl
print(f"Calculando calibración desde {CAL_NROWS} filas de training...")
train_cal = pd.read_csv(TRAIN_CSV, nrows=CAL_NROWS)
for col in FEATURE_ORDER:
    if col not in train_cal.columns:
        train_cal[col] = 0.0

X_cal  = train_cal[FEATURE_ORDER].astype(float).values
y_cal  = train_cal["overall_rating"].values
p_cal  = MODEL.predict(X_cal)
cal_b, cal_a = np.polyfit(p_cal, y_cal, 1)   # y = a*pred + b
print(f"Calibración: y = {cal_a:.4f} * raw_pred + {cal_b:.4f}")

# ── Carga y filtra datos preprocesados de NVDA ────────────────────────────────
print("Cargando datos preprocesados de NVDA...")
nvda_train = pd.read_csv(TRAIN_CSV)[lambda d: d["company"] == COMPANY]
nvda_test  = pd.read_csv(TEST_CSV)[lambda d: d["company"]  == COMPANY]
df = pd.concat([nvda_train, nvda_test], ignore_index=True)
print(f"{len(df)} reviews de {COMPANY} ({len(nvda_train)} train + {len(nvda_test)} test)")

for col in FEATURE_ORDER:
    if col not in df.columns:
        df[col] = 0.0

X = df[FEATURE_ORDER].astype(float).values

# ── Predicción + calibración ──────────────────────────────────────────────────
print("Prediciendo...")
raw_preds   = MODEL.predict(X).astype(float)
predictions = np.clip(cal_a * raw_preds + cal_b, 1.0, 5.0)

y_real = df["overall_rating"].values
rmse   = np.sqrt(mean_squared_error(y_real, predictions))
r2     = r2_score(y_real, predictions)
print(f"Predicciones — min={predictions.min():.3f} | max={predictions.max():.3f} | media={predictions.mean():.3f}")
print(f"vs real       — RMSE={rmse:.3f}  R²={r2:.3f}")

# ── Añadir al DataFrame ───────────────────────────────────────────────────────
df["predicted_rating"]       = predictions
df["predicted_satisfaction"] = np.where(predictions >= CAL_THRESHOLD, "Alto", "Bajo")
df["confidence"]             = np.clip(np.abs(predictions - CAL_THRESHOLD) / 2.0, 0, 1).round(3)

# ── Guardar gold CSV ──────────────────────────────────────────────────────────
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nGold CSV guardado en: {OUTPUT_CSV}")
print(f"   Filas: {len(df)} | Columnas: {len(df.columns)}")
print(f"\nDistribucion satisfaccion:")
print(df["predicted_satisfaction"].value_counts())
print(f"\nPrimeras 5 predicciones:")
print(df[["review_id", "overall_rating", "predicted_rating", "predicted_satisfaction", "confidence"]].head())
