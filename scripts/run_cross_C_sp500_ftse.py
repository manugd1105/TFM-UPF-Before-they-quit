"""
run_cross_C_sp500_ftse.py
─────────────────────────
Experimento C: Train SP500 → Test FTSE100
¿Generaliza el modelo entrenado en empresas US al mercado UK?

Feature set: intersección de columnas numéricas no-metadata entre SP500 y FTSE100
             (= group5 sin features de estado US). Ambos datasets ya están
             estandarizados independientemente (~N(0,1)) con sus propios scalers.
Model: XGBoost, n_iter=100, CV=5.
"""

import os, json, pickle, warnings
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from xgboost import XGBRegressor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
# Ejecutar desde la raíz del repo.
# SP500_TRAIN/TEST: ajusta EXECUTION_TO_USE al timestamp de tu ejecución de Data preprocessing.
# FTSE_TEST: coloca los CSVs en ftse100/data/ (ver ftse100/data/README.md).
EXECUTION_TO_USE = "20240322_105534"  # ← CAMBIAR al timestamp de tu ejecución de preprocessing
SP500_TRAIN = f"./03_outputs_data_preprocessing/output_data_preprocessing_{EXECUTION_TO_USE}/dataframes_Train_{EXECUTION_TO_USE}.csv"
FTSE_TEST   = "./ftse100/data/dataframes_Test_scrapeadas.csv"
# También evaluamos en SP500 test para ver la degradación por eliminar state features
SP500_TEST  = f"./03_outputs_data_preprocessing/output_data_preprocessing_{EXECUTION_TO_USE}/dataframes_Test_{EXECUTION_TO_USE}.csv"

OUTPUT_DIR = "./05_outputs_modeling/cross_training/C_sp500_ftse"

N_ITER      = 100
N_FOLDS     = 5
RANDOM_SEED = 8

# ── Columnas a excluir ────────────────────────────────────────────────────────
META_COLS = {
    "review_id", "summary", "date", "job_title", "overall_rating",
    "overall_rating_cat", "pros", "cons", "author_location", "company",
    "Company", "Ticker", "Sector", "Headquarters Location", "Unnamed: 0",
    # SP500 metadata extra
    "author_location_filledhq", "author_location_filledhq_state",
    "Acronym", "State name", "Headquarters state",
    # Sub-ratings Glassdoor (no son features del modelo original)
    "wl_balance", "culture_values", "diversity_inclusion",
    "career_opportunities", "compensation_benefits", "senior_management",
}

# ── Hiperparámetros ────────────────────────────────────────────────────────────
PARAM_GRID = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 4, 5, 6, 7, 9],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2, 0.3],
    "subsample":        [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma":            [0, 0.1, 0.2],
}

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("Experimento C: SP500 (train) → FTSE100 (test)")
    print("=" * 60)

    # Cargar datos
    print("Cargando SP500 train...")
    sp500_train = pd.read_csv(SP500_TRAIN, low_memory=False)
    print(f"  SP500 train: {len(sp500_train):,} filas")

    print("Cargando FTSE100 test...")
    ftse_test = pd.read_csv(FTSE_TEST, low_memory=False)
    print(f"  FTSE100 test: {len(ftse_test):,} filas")

    print("Cargando SP500 test (para evaluación secundaria)...")
    sp500_test = pd.read_csv(SP500_TEST, low_memory=False)
    print(f"  SP500 test: {len(sp500_test):,} filas")

    # Determinar features comunes: numéricas, no-metadata, presentes en SP500 train Y FTSE100 test
    sp500_num = set(sp500_train.select_dtypes(include="number").columns) - META_COLS
    ftse_num  = set(ftse_test.select_dtypes(include="number").columns)   - META_COLS

    feat_cols = sorted(sp500_num & ftse_num)
    print(f"\n  Features comunes (group5 sin estado): {len(feat_cols)}")

    # Comprobar que SP500 test también las tiene
    missing_sp500_test = [c for c in feat_cols if c not in sp500_test.columns]
    if missing_sp500_test:
        print(f"  AVISO: {len(missing_sp500_test)} features faltan en SP500 test: {missing_sp500_test}")
        feat_cols = [c for c in feat_cols if c in sp500_test.columns]

    # Preparar matrices
    train_x    = sp500_train[feat_cols].fillna(0)
    train_y    = sp500_train["overall_rating"]
    ftse_x     = ftse_test[feat_cols].fillna(0)
    ftse_y     = ftse_test["overall_rating"]
    sp500_x    = sp500_test[feat_cols].fillna(0)
    sp500_y    = sp500_test["overall_rating"]

    print(f"  Target SP500 train: {train_y.value_counts().sort_index().to_dict()}")
    print(f"  Target FTSE test:   {ftse_y.value_counts().sort_index().to_dict()}")

    # Entrenar
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0,
    )
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_GRID,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=N_FOLDS,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbose=1,
    )

    print(f"\nEntrenando XGBoost ({N_ITER} iter × {N_FOLDS} folds) sobre SP500...")
    t0 = datetime.now()
    search.fit(train_x, train_y)
    train_time = str(datetime.now() - t0)
    print(f"Entrenamiento completado en {train_time}")

    # Predicciones
    train_pred    = search.best_estimator_.predict(train_x)
    ftse_pred     = search.best_estimator_.predict(ftse_x)
    sp500_pred    = search.best_estimator_.predict(sp500_x)

    # Métricas
    def compute_metrics(real, pred, label):
        mape = metrics.mean_absolute_percentage_error(real, pred)
        mse  = metrics.mean_squared_error(real, pred)
        mae  = metrics.mean_absolute_error(real, pred)
        print(f"  [{label}] MAPE={mape:.4f} | MSE={mse:.4f} | MAE={mae:.4f}")
        return {"mape": mape, "mse": mse, "mae": mae}

    print(f"\n{'─'*50}")
    m_train     = compute_metrics(train_y, train_pred, "SP500 train  (in-sample)")
    m_sp500     = compute_metrics(sp500_y, sp500_pred, "SP500 test   (same dist)")
    m_ftse      = compute_metrics(ftse_y,  ftse_pred,  "FTSE100 test (cross-mkt)")
    print(f"  Best params: {search.best_params_}")
    print(f"{'─'*50}")

    # Referencia: baseline SP500 group5 completo (exec 7-4)
    print(f"\n  [Referencia 7-4] MAPE_test SP500 group5 = 0.2549 (con state features)")

    # Guardar config
    config = {
        "experiment": "C_sp500_ftse",
        "train_data":      SP500_TRAIN,
        "test_data_ftse":  FTSE_TEST,
        "test_data_sp500": SP500_TEST,
        "n_features": len(feat_cols),
        "feature_cols": feat_cols,
        "n_iter": N_ITER,
        "n_folds": N_FOLDS,
        "train_rows":    len(sp500_train),
        "ftse_test_rows": len(ftse_test),
        "sp500_test_rows": len(sp500_test),
        "train_time": train_time,
        "best_params": search.best_params_,
        "metrics_sp500_train": m_train,
        "metrics_sp500_test":  m_sp500,
        "metrics_ftse_test":   m_ftse,
    }
    with open(os.path.join(OUTPUT_DIR, f"config_{ts}.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Guardar predicciones
    pd.DataFrame({"review_id": sp500_train["review_id"], "real": train_y.values, "pred": train_pred})\
        .to_csv(os.path.join(OUTPUT_DIR, f"sp500_train_preds_{ts}.csv"), index=False)
    pd.DataFrame({"review_id": sp500_test["review_id"], "real": sp500_y.values, "pred": sp500_pred})\
        .to_csv(os.path.join(OUTPUT_DIR, f"sp500_test_preds_{ts}.csv"), index=False)
    pd.DataFrame({"review_id": ftse_test["review_id"],  "real": ftse_y.values,  "pred": ftse_pred})\
        .to_csv(os.path.join(OUTPUT_DIR, f"ftse_test_preds_{ts}.csv"),  index=False)

    # CV results
    pd.DataFrame(search.cv_results_)\
        .to_csv(os.path.join(OUTPUT_DIR, f"cv_results_{ts}.csv"), index=False)

    # Feature importances
    fi = pd.DataFrame({
        "feature": feat_cols,
        "importance": search.best_estimator_.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(OUTPUT_DIR, f"feature_importance_{ts}.csv"), index=False)
    print(f"\nTop 10 features:\n{fi.head(10).to_string(index=False)}")

    # Plots
    for real, pred, name in [
        (train_y, train_pred, "SP500_train"),
        (sp500_y, sp500_pred, "SP500_test"),
        (ftse_y,  ftse_pred,  "FTSE100_test"),
    ]:
        plt.figure(figsize=(6, 5))
        plt.scatter(real, pred, alpha=0.2, s=2)
        plt.xlabel("Real rating"); plt.ylabel("Predicted rating")
        plt.title(f"C SP500→FTSE | {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_scatter_{ts}.png"), dpi=100)
        plt.close()

    # Pickle del modelo
    with open(os.path.join(OUTPUT_DIR, f"model_{ts}.pkl"), "wb") as f:
        pickle.dump(search, f)

    # Resumen
    summary = (f"{ts}; Exp=C_sp500_ftse; n_features={len(feat_cols)}; "
               f"train_rows={len(sp500_train)}; "
               f"train_time={train_time}; "
               f"MAPE_sp500_train={m_train['mape']:.4f}; "
               f"MAPE_sp500_test={m_sp500['mape']:.4f}; "
               f"MAPE_ftse_test={m_ftse['mape']:.4f}; "
               f"MSE_ftse_test={m_ftse['mse']:.4f}\n")
    with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "a") as f:
        f.write(summary)

    print(f"\nOutputs guardados en: {OUTPUT_DIR}")
    print("Experimento C completado.")


if __name__ == "__main__":
    main()
