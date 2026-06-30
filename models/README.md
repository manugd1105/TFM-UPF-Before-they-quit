# Top 10 trained models

Modelos entrenados en Google Cloud Vertex AI, ordenados por MAPE test (menor = mejor).
Ranking según la tabla de resultados del TFM (Capítulo 5).
Cada archivo `.pkl` es un diccionario Python con el objeto `RandomizedSearchCV` y métricas.

| Rank | File | Model | Exec ID | PCA | Feature group | MAPE train | MAPE test |
|------|------|-------|---------|-----|---------------|------------|-----------|
| 1 | `01_xgboost_exec6-4_grupo6_MAPE0.2549.pkl` | XGBoost | 6-4 | None | Grupo 6 | 0.1946 | 0.2549 |
| 2 | `02_xgboost_exec5-6_grupo5_MAPE0.2550.pkl` | XGBoost | 5-6 | None | Grupo 5 | 0.2113 | 0.2550 |
| 3 | `03_xgboost_exec4-21_grupo4_MAPE0.2578.pkl` | XGBoost | 4-21 | None | Grupo 4 | 0.2169 | 0.2578 |
| 4 | `04_xgboost_exec6-3_grupo6_MAPE0.2582.pkl` | XGBoost | 6-3 | None | Grupo 6 | 0.2020 | 0.2582 |
| 5 | `05_xgboost_exec3-21_grupo3_MAPE0.2741.pkl` | XGBoost | 3-21 | None | Grupo 3 | 0.2386 | 0.2741 |
| 6 | `06_xgboost_exec2-17_grupo2_MAPE0.2769.pkl` | XGBoost | 2-17 | 5_dim | Grupo 2 | 0.2519 | 0.2769 |
| 7 | `07_xgboost_exec4-23_grupo4_MAPE0.2807.pkl` | XGBoost | 4-23 | 5_dim | Grupo 4 | 0.2584 | 0.2807 |
| 8 | `08_xgboost_exec2-18_grupo2_MAPE0.2873.pkl` | XGBoost | 2-18 | 1_per_5 | Grupo 2 | 0.2355 | 0.2873 |
| 9 | `09_xgboost_exec3-23_grupo3_MAPE0.2949.pkl` | XGBoost | 3-23 | 5_dim | Grupo 3 | 0.2735 | 0.2949 |
| 10 | `10_config_20260518_093753.json` | Random Forest | 4-13 | None | Grupo 4 | 0.1081 | 0.2925 |

> **Nota rank 10:** El pkl de la ejecución 4-13 (Random Forest) no está incluido porque pesa varios gigas. En su lugar se incluye el `config_20260518_093753.json` con la configuración completa del modelo (hiperparámetros del mejor estimador, métricas, etc.).

## How to load a model

```python
import pickle

with open("models/01_xgboost_exec6-4_grupo6_MAPE0.2549.pkl", "rb") as f:
    d = pickle.load(f)

model = d["Random search object"].best_estimator_
# model is a fitted XGBoost regressor
```

## Feature groups reference

| Group | Description |
|-------|-------------|
| 2 | Length + POS + EmoLex + VADER + Empath + ODI vocab + geographic + company + stock |
| 3 | Group 2 without geographic/company/stock (NLP only) |
| 4 | Group 3 + JDI/ODI cosine similarity |
| 5 | Group 2 + JDI/ODI cosine similarity (full feature set) |
| 6 | Group 5 without POS/EmoLex/VADER/Empath/ODI vocab (similarity-only NLP) |
