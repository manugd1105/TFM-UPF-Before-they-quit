"""
run_cross_B_ftse_ftse.py
─────────────────────────
Experimento B: Train FTSE100 → Test FTSE100
Baseline del modelo dentro del mercado UK.

Feature set: group5_no_state (todos los features de group5 excepto los de estado US,
             que no existen en FTSE100). Se determina dinámicamente por intersección
             de columnas disponibles con la definición original de group5.
Model: XGBoost, n_iter=100, CV=5.
"""

import os, json, pickle, math, warnings
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
# Ejecutar desde la raíz del repo. Los CSVs de FTSE100 deben colocarse manualmente
# en ftse100/data/ (no están en el repo por superar el límite de 100 MB de GitHub).
# Ver ftse100/data/README.md para instrucciones.
FTSE_TRAIN = "./ftse100/data/dataframes_Train_scrapeadas.csv"
FTSE_TEST  = "./ftse100/data/dataframes_Test_scrapeadas.csv"
OUTPUT_DIR = "./05_outputs_modeling/cross_training/B_ftse_ftse"

N_ITER      = 100
N_FOLDS     = 5
RANDOM_SEED = 8

# ── Columnas a excluir (metadata, target, texto) ───────────────────────────────
META_COLS = {
    "review_id", "summary", "date", "job_title", "overall_rating",
    "overall_rating_cat", "pros", "cons", "author_location", "company",
    "Company", "Ticker", "Sector", "Headquarters Location", "Unnamed: 0",
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
    print("Experimento B: FTSE100 → FTSE100")
    print("=" * 60)

    # Cargar datos
    print("Cargando datos FTSE100...")
    train = pd.read_csv(FTSE_TRAIN, low_memory=False)
    test  = pd.read_csv(FTSE_TEST,  low_memory=False)
    print(f"  Train: {len(train):,} filas | Test: {len(test):,} filas")

    # Seleccionar features: numéricas, no-metadata, presentes en ambos
    feat_cols = [
        c for c in train.select_dtypes(include="number").columns
        if c not in META_COLS and c in test.columns
    ]
    print(f"  Features seleccionadas: {len(feat_cols)}")

    train_x = train[feat_cols].fillna(0)
    train_y = train["overall_rating"]
    test_x  = test[feat_cols].fillna(0)
    test_y  = test["overall_rating"]

    print(f"  Target distribución train: {train_y.value_counts().sort_index().to_dict()}")
    print(f"  Target distribución test:  {test_y.value_counts().sort_index().to_dict()}")

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

    print(f"\nEntrenando XGBoost ({N_ITER} iter × {N_FOLDS} folds)...")
    t0 = datetime.now()
    search.fit(train_x, train_y)
    train_time = str(datetime.now() - t0)
    print(f"Entrenamiento completado en {train_time}")

    # Métricas
    train_pred = search.best_estimator_.predict(train_x)
    test_pred  = search.best_estimator_.predict(test_x)

    mape_train = metrics.mean_absolute_percentage_error(train_y, train_pred)
    mape_test  = metrics.mean_absolute_percentage_error(test_y,  test_pred)
    mse_train  = metrics.mean_squared_error(train_y, train_pred)
    mse_test   = metrics.mean_squared_error(test_y,  test_pred)
    mae_train  = metrics.mean_absolute_error(train_y, train_pred)
    mae_test   = metrics.mean_absolute_error(test_y,  test_pred)

    print(f"\n{'─'*40}")
    print(f"MAPE  train: {mape_train:.4f}  |  test: {mape_test:.4f}")
    print(f"MSE   train: {mse_train:.4f}  |  test: {mse_test:.4f}")
    print(f"MAE   train: {mae_train:.4f}  |  test: {mae_test:.4f}")
    print(f"Best params: {search.best_params_}")
    print(f"{'─'*40}")

    # Guardar config
    config = {
        "experiment": "B_ftse_ftse",
        "train_data": FTSE_TRAIN,
        "test_data":  FTSE_TEST,
        "n_features": len(feat_cols),
        "feature_cols": feat_cols,
        "n_iter": N_ITER,
        "n_folds": N_FOLDS,
        "train_rows": len(train),
        "test_rows":  len(test),
        "train_time": train_time,
        "best_params": search.best_params_,
        "mape_train": mape_train, "mape_test": mape_test,
        "mse_train":  mse_train,  "mse_test":  mse_test,
        "mae_train":  mae_train,  "mae_test":  mae_test,
    }
    with open(os.path.join(OUTPUT_DIR, f"config_{ts}.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Guardar predicciones
    pd.DataFrame({"review_id": train["review_id"], "real": train_y.values, "pred": train_pred})\
        .to_csv(os.path.join(OUTPUT_DIR, f"train_preds_{ts}.csv"), index=False)
    pd.DataFrame({"review_id": test["review_id"],  "real": test_y.values,  "pred": test_pred})\
        .to_csv(os.path.join(OUTPUT_DIR, f"test_preds_{ts}.csv"),  index=False)

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
    for real, pred, name in [(train_y, train_pred, "Train"), (test_y, test_pred, "Test")]:
        plt.figure(figsize=(6, 5))
        plt.scatter(real, pred, alpha=0.2, s=2)
        plt.xlabel("Real rating"); plt.ylabel("Predicted rating")
        plt.title(f"B FTSE→FTSE | {name} | MAPE={mape_test:.4f}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_scatter_{ts}.png"), dpi=100)
        plt.close()

    # Pickle del modelo
    with open(os.path.join(OUTPUT_DIR, f"model_{ts}.pkl"), "wb") as f:
        pickle.dump(search, f)

    # Resumen en model_summary.txt
    summary = (f"{ts}; Exp=B_ftse_ftse; n_features={len(feat_cols)}; "
               f"train_rows={len(train)}; test_rows={len(test)}; "
               f"train_time={train_time}; "
               f"MAPE_train={mape_train:.4f}; MAPE_test={mape_test:.4f}; "
               f"MSE_train={mse_train:.4f}; MSE_test={mse_test:.4f}\n")
    with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "a") as f:
        f.write(summary)

    print(f"\nOutputs guardados en: {OUTPUT_DIR}")
    print("Experimento B completado.")


if __name__ == "__main__":
    main()
