"""
trainer/task_cross.py
─────────────────────
Vertex AI Custom Job — Experimentos Cross-training B y C.

  B-1  XGBoost  FTSE100 train → FTSE100 test         n_iter=200
  C-1  XGBoost  SP500 train (no state) → FTSE100 test + SP500 test  n_iter=200

Feature set: FEATURES_NO_STATE = group5 completo menos las 75 features de estado US.
             Estas features están en FTSE100 pero no en el dataset UK.
"""

import os
import json
import pickle
import warnings
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
import xgboost as xgb
from google.cloud import storage

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
BUCKET_NAME   = os.environ.get("GCP_BUCKET", "CHANGE_ME")
OUTPUT_PREFIX = "cross_training"

SP500_EXEC_ID = "20240322_105534"
SP500_TRAIN   = f"datos_entrada/dataframes_Train_{SP500_EXEC_ID}.csv"
SP500_TEST    = f"datos_entrada/dataframes_Test_{SP500_EXEC_ID}.csv"
FTSE_TRAIN    = "datos_entrada/dataframes_Train_ftse100.csv"
FTSE_TEST     = "datos_entrada/dataframes_Test_ftse100.csv"

N_ITER      = 200
N_FOLDS     = 5
N_JOBS      = 16
RANDOM_SEED = 8

# ─── Hyperparameter grid (igual que exec 7-4) ─────────────────────────────────
PARAM_GRID = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 5, 7, 9],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma":            [0, 0.1, 0.5],
}

# ─── Feature set: group5 MINUS state features ─────────────────────────────────
# State features son los 75 que existen en SP500 pero NO en FTSE100.
# Se determinan dinámicamente como la intersección de columnas numéricas
# no-metadata presentes en FTSE100 (que es el subconjunto común).
META_COLS = {
    "review_id", "summary", "date", "job_title", "overall_rating",
    "overall_rating_cat", "pros", "cons", "author_location", "company",
    "Company", "Ticker", "Sector", "Headquarters Location", "Unnamed: 0",
    "author_location_filledhq", "author_location_filledhq_state",
    "Acronym", "State name", "Headquarters state",
    "wl_balance", "culture_values", "diversity_inclusion",
    "career_opportunities", "compensation_benefits", "senior_management",
}

def get_cross_features(sp500_df, ftse_df):
    """Features numéricas comunes entre SP500 y FTSE100, excluyendo metadata."""
    sp500_num = set(sp500_df.select_dtypes(include="number").columns) - META_COLS
    ftse_num  = set(ftse_df.select_dtypes(include="number").columns)  - META_COLS
    common    = sorted(sp500_num & ftse_num)
    print(f"  Features comunes (group5 sin estado): {len(common)}")
    return common

# ─── GCS helpers ──────────────────────────────────────────────────────────────
_gcs_client = None

def gcs():
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client

def read_csv_gcs(path):
    return pd.read_csv(f"gs://{BUCKET_NAME}/{path}", low_memory=False)

def upload_file(local_path, gcs_path):
    gcs().bucket(BUCKET_NAME).blob(gcs_path).upload_from_filename(local_path)
    print(f"    -> gs://{BUCKET_NAME}/{gcs_path}")

def upload_string(content, gcs_path):
    gcs().bucket(BUCKET_NAME).blob(gcs_path).upload_from_string(content)

def already_done(exec_id):
    prefix = f"{OUTPUT_PREFIX}/{exec_id.replace('-','_')}/Dashboard output"
    return any(True for _ in gcs().bucket(BUCKET_NAME).list_blobs(prefix=prefix))

# ─── Save helpers ─────────────────────────────────────────────────────────────
def save_and_upload_results(tmp, gcs_folder, ts, exec_id, model_type,
                             train_raw, train_y, train_pred,
                             test_sets,   # list of (label, df, y_real, y_pred)
                             search, feat_cols, n_iter, train_time,
                             mape_train, mse_train):
    """
    Guarda todos los artifacts en tmp/ y los sube a GCS.
    test_sets: lista de tuplas (label, df_raw, y_real, y_pred)
    """
    def lp(f): return os.path.join(tmp, f)
    def gp(f): return f"{gcs_folder}/{f}"

    # Predicciones train
    pd.concat([train_raw[["review_id"]], train_y.reset_index(drop=True)], axis=1)\
        .to_csv(lp(f"Train real_{ts}.csv"), index=False)
    pd.concat([train_raw[["review_id"]], pd.Series(train_pred, name=0)], axis=1)\
        .to_csv(lp(f"Train pred_{ts}.csv"), index=False)
    upload_file(lp(f"Train real_{ts}.csv"), gp(f"Train real_{ts}.csv"))
    upload_file(lp(f"Train pred_{ts}.csv"), gp(f"Train pred_{ts}.csv"))

    # Predicciones de cada test set
    for label, df_raw, y_real, y_pred in test_sets:
        pd.concat([df_raw[["review_id"]], y_real.reset_index(drop=True)], axis=1)\
            .to_csv(lp(f"Test real_{label}_{ts}.csv"), index=False)
        pd.concat([df_raw[["review_id"]], pd.Series(y_pred, name=0)], axis=1)\
            .to_csv(lp(f"Test pred_{label}_{ts}.csv"), index=False)
        upload_file(lp(f"Test real_{label}_{ts}.csv"), gp(f"Test real_{label}_{ts}.csv"))
        upload_file(lp(f"Test pred_{label}_{ts}.csv"), gp(f"Test pred_{label}_{ts}.csv"))

    # Scatter plots
    plt.figure(); plt.scatter(train_y, train_pred, alpha=0.3, s=1)
    plt.xlabel("Real"); plt.ylabel("Predicted"); plt.title("Training comparison")
    plt.savefig(lp(f"Training_comparison_{ts}.png")); plt.close()
    upload_file(lp(f"Training_comparison_{ts}.png"), gp(f"Training_comparison_{ts}.png"))

    for label, _, y_real, y_pred in test_sets:
        plt.figure(); plt.scatter(y_real, y_pred, alpha=0.3, s=1)
        plt.xlabel("Real"); plt.ylabel("Predicted"); plt.title(f"Test {label}")
        plt.savefig(lp(f"Testing_{label}_comparison_{ts}.png")); plt.close()
        upload_file(lp(f"Testing_{label}_comparison_{ts}.png"),
                    gp(f"Testing_{label}_comparison_{ts}.png"))

    # CV building summary
    pd.DataFrame(search.cv_results_)\
        .to_csv(lp(f"Building summary_{ts}.csv"), index=False)
    upload_file(lp(f"Building summary_{ts}.csv"), gp(f"Building summary_{ts}.csv"))

    # Config
    test_metrics = {}
    for label, _, y_real, y_pred in test_sets:
        test_metrics[label] = {
            "mape": metrics.mean_absolute_percentage_error(y_real, y_pred),
            "mse":  metrics.mean_squared_error(y_real, y_pred),
            "mae":  metrics.mean_absolute_error(y_real, y_pred),
        }
    cfg = {
        "execution_id": exec_id, "model_type": model_type,
        "n_iter": n_iter, "n_folds": N_FOLDS,
        "n_features": len(feat_cols), "feature_cols": feat_cols,
        "best_params": search.best_params_,
        "mape_train": mape_train, "mse_train": mse_train,
        "test_metrics": test_metrics,
    }
    with open(lp(f"config_{ts}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    upload_file(lp(f"config_{ts}.json"), gp(f"config_{ts}.json"))

    # Feature importance
    fi = pd.DataFrame({
        "feature": feat_cols,
        "importance": search.best_estimator_.feature_importances_,
    }).sort_values("importance", ascending=False)
    fi.to_csv(lp(f"feature_importance_{ts}.csv"), index=False)
    upload_file(lp(f"feature_importance_{ts}.csv"), gp(f"feature_importance_{ts}.csv"))

    # Dashboard Excel
    lst_params = list(search.cv_results_["params"][0].keys())
    deepdive = pd.DataFrame({
        col: search.cv_results_[col]
        for col in ["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time",
                    "mean_test_score", "std_test_score", "rank_test_score"] +
                   [f"split{s}_test_score" for s in range(N_FOLDS)]
    })
    deepdive["Execution name"] = ts
    hp_df = pd.DataFrame({k: [d[k] for d in search.cv_results_["params"]] for k in lst_params})
    for i, hp in enumerate(lst_params):
        deepdive[f"param_{i}_name"] = hp
        deepdive[f"param_{i}"] = hp_df[hp]

    # Primary test metrics (first test set)
    primary_label, _, y_real_p, y_pred_p = test_sets[0]
    mape_test_p = test_metrics[primary_label]["mape"]
    mse_test_p  = test_metrics[primary_label]["mse"]

    xls_name = f"Dashboard output_{ts}.xlsx"
    with pd.ExcelWriter(lp(xls_name), engine="xlsxwriter") as writer:
        pd.DataFrame({
            "Execution name": [ts], "Execution ID": [exec_id],
            "Model type": [model_type], "PCA": ["No"],
            "Number of search iterations": [n_iter], "Number of folds": [N_FOLDS],
            "Train time": [train_time],
            "Metric 1": ["MAPE"], "Metric 1 train": [mape_train],
            "Metric 1 test (primary)": [mape_test_p],
            "Metric 2": ["MSE"],  "Metric 2 train": [mse_train],
            "Metric 2 test (primary)": [mse_test_p],
            **{f"MAPE_{lbl}": [test_metrics[lbl]["mape"]] for lbl, *_ in test_sets},
            **{f"MSE_{lbl}":  [test_metrics[lbl]["mse"]]  for lbl, *_ in test_sets},
        }).to_excel(writer, sheet_name="General", index=False)

        train_real_df = pd.concat([train_raw[["review_id"]], train_y.reset_index(drop=True)], axis=1)
        train_pred_df = pd.concat([train_raw[["review_id"]], pd.Series(train_pred, name=0)], axis=1)
        pd.merge(train_real_df, train_pred_df, on="review_id").rename(
            columns={"overall_rating": f"Real_{ts}", 0: f"Pred_{ts}"}
        ).to_excel(writer, sheet_name="Train predictions", index=False)

        for label, df_raw, y_real, y_pred in test_sets:
            test_real_df = pd.concat([df_raw[["review_id"]], y_real.reset_index(drop=True)], axis=1)
            test_pred_df = pd.concat([df_raw[["review_id"]], pd.Series(y_pred, name=0)], axis=1)
            pd.merge(test_real_df, test_pred_df, on="review_id").rename(
                columns={"overall_rating": f"Real_{ts}", 0: f"Pred_{ts}"}
            ).to_excel(writer, sheet_name=f"Test {label}"[:31], index=False)

        deepdive.to_excel(writer, sheet_name="Execution deepdive", index=False)

    upload_file(lp(xls_name), gp(xls_name))

    # Pickle
    with open(lp(f"dict_{ts}.pkl"), "wb") as f:
        pickle.dump({"Random search object": search}, f)
    upload_file(lp(f"dict_{ts}.pkl"), gp(f"dict_{ts}.pkl"))

    # Per-execution summary
    summary_parts = [f"{ts}; Execution ID: {exec_id}; Model type: {model_type}; "
                     f"Train time: {train_time}; MAPE train: {mape_train}; "
                     f"MSE train: {mse_train}"]
    for label, _, y_real, y_pred in test_sets:
        summary_parts.append(f"MAPE_{label}: {test_metrics[label]['mape']:.4f}; "
                             f"MSE_{label}: {test_metrics[label]['mse']:.4f}")
    summary_line = "; ".join(summary_parts)

    with open(lp("model_summary.txt"), "w") as f:
        f.write(summary_line + "\n")
    upload_file(lp("model_summary.txt"), gp("model_summary.txt"))

    return summary_line, test_metrics


# ─── Run experiment B: FTSE → FTSE ───────────────────────────────────────────
def run_B(ftse_train, ftse_test, feat_cols):
    exec_id = "B-1"
    print(f"\n{'='*60}")
    print(f"START: {exec_id} | XGBoost | FTSE train → FTSE test")
    print(f"{'='*60}")

    train_x = ftse_train[feat_cols].fillna(0)
    train_y = ftse_train["overall_rating"]
    test_x  = ftse_test[feat_cols].fillna(0)
    test_y  = ftse_test["overall_rating"]
    print(f"  Train: {len(train_x):,} rows | Test: {len(test_x):,} rows")

    model = xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=0,
    )
    search = RandomizedSearchCV(
        estimator=model, param_distributions=PARAM_GRID,
        n_iter=N_ITER, scoring="neg_root_mean_squared_error",
        cv=N_FOLDS, random_state=RANDOM_SEED, n_jobs=1, verbose=1,
    )

    print(f"  Training ({N_ITER} iter × {N_FOLDS} folds)...")
    t0 = datetime.now()
    search.fit(train_x, train_y)
    train_time = str(datetime.now() - t0)
    print(f"  Done in {train_time}")

    train_pred = search.best_estimator_.predict(train_x)
    test_pred  = search.best_estimator_.predict(test_x)

    mape_train = metrics.mean_absolute_percentage_error(train_y, train_pred)
    mse_train  = metrics.mean_squared_error(train_y, train_pred)
    mape_test  = metrics.mean_absolute_percentage_error(test_y, test_pred)
    mse_test   = metrics.mean_squared_error(test_y, test_pred)

    print(f"  MAPE train={mape_train:.4f} | MAPE test={mape_test:.4f}")
    print(f"  MSE  train={mse_train:.4f}  | MSE  test={mse_test:.4f}")
    print(f"  Best params: {search.best_params_}")

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_folder = f"{OUTPUT_PREFIX}/B_1"

    with tempfile.TemporaryDirectory() as tmp:
        summary_line, _ = save_and_upload_results(
            tmp=tmp, gcs_folder=gcs_folder, ts=ts,
            exec_id=exec_id, model_type="xgboost",
            train_raw=ftse_train, train_y=train_y, train_pred=train_pred,
            test_sets=[("FTSE100", ftse_test, test_y, test_pred)],
            search=search, feat_cols=feat_cols, n_iter=N_ITER,
            train_time=train_time, mape_train=mape_train, mse_train=mse_train,
        )

    return summary_line, mape_test, mse_test


# ─── Run experiment C: SP500 → FTSE COMPLETO (+ SP500 secondary) ─────────────
def run_C(sp500_train, sp500_test, ftse_full, feat_cols):
    """
    Entrenamos en SP500 train (285k) y evaluamos en:
      - FTSE100 COMPLETO (train+test, 135k): ninguna fila fue usada para entrenar,
        así que todo el dataset FTSE100 es válido como test en este escenario.
      - SP500 test (71k): para cuantificar la degradación por eliminar state features.
    """
    exec_id = "C-1"
    print(f"\n{'='*60}")
    print(f"START: {exec_id} | XGBoost | SP500 train → FTSE100 completo (135k) + SP500 test")
    print(f"{'='*60}")

    train_x    = sp500_train[feat_cols].fillna(0)
    train_y    = sp500_train["overall_rating"]
    ftse_x     = ftse_full[feat_cols].fillna(0)
    ftse_y     = ftse_full["overall_rating"]
    sp500_x    = sp500_test[feat_cols].fillna(0)
    sp500_y    = sp500_test["overall_rating"]

    print(f"  SP500 train: {len(train_x):,} | SP500 test: {len(sp500_x):,} | FTSE FULL: {len(ftse_x):,}")

    model = xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=0,
    )
    search = RandomizedSearchCV(
        estimator=model, param_distributions=PARAM_GRID,
        n_iter=N_ITER, scoring="neg_root_mean_squared_error",
        cv=N_FOLDS, random_state=RANDOM_SEED, n_jobs=1, verbose=1,
    )

    print(f"  Training ({N_ITER} iter × {N_FOLDS} folds) on SP500...")
    t0 = datetime.now()
    search.fit(train_x, train_y)
    train_time = str(datetime.now() - t0)
    print(f"  Done in {train_time}")

    train_pred = search.best_estimator_.predict(train_x)
    ftse_pred  = search.best_estimator_.predict(ftse_x)
    sp500_pred = search.best_estimator_.predict(sp500_x)

    mape_train     = metrics.mean_absolute_percentage_error(train_y, train_pred)
    mse_train      = metrics.mean_squared_error(train_y, train_pred)
    mape_ftse      = metrics.mean_absolute_percentage_error(ftse_y, ftse_pred)
    mse_ftse       = metrics.mean_squared_error(ftse_y, ftse_pred)
    mape_sp500test = metrics.mean_absolute_percentage_error(sp500_y, sp500_pred)
    mse_sp500test  = metrics.mean_squared_error(sp500_y, sp500_pred)

    print(f"  MAPE train={mape_train:.4f}")
    print(f"  MAPE SP500 test={mape_sp500test:.4f} | MSE={mse_sp500test:.4f}  [mismo mercado, sin state feat]")
    print(f"  MAPE FTSE  full={mape_ftse:.4f}      | MSE={mse_ftse:.4f}  [cross-market, 135k rows]")
    print(f"  [Ref. 7-4 SP500 group5 full] MAPE=0.2549 (con state features)")
    print(f"  Best params: {search.best_params_}")

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_folder = f"{OUTPUT_PREFIX}/C_1"

    with tempfile.TemporaryDirectory() as tmp:
        summary_line, _ = save_and_upload_results(
            tmp=tmp, gcs_folder=gcs_folder, ts=ts,
            exec_id=exec_id, model_type="xgboost",
            train_raw=sp500_train, train_y=train_y, train_pred=train_pred,
            test_sets=[
                ("FTSE100_full", ftse_full,  ftse_y,  ftse_pred),   # 135k completo
                ("SP500test",    sp500_test, sp500_y, sp500_pred),   # 71k referencia
            ],
            search=search, feat_cols=feat_cols, n_iter=N_ITER,
            train_time=train_time, mape_train=mape_train, mse_train=mse_train,
        )

    return summary_line, mape_ftse, mse_ftse


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Cross-training experiments B and C")
    print("=" * 60)

    print("\nCargando datos desde GCS...")
    sp500_train = pd.read_csv(f"gs://{BUCKET_NAME}/{SP500_TRAIN}")
    sp500_test  = pd.read_csv(f"gs://{BUCKET_NAME}/{SP500_TEST}")
    ftse_train  = pd.read_csv(f"gs://{BUCKET_NAME}/{FTSE_TRAIN}")
    ftse_test   = pd.read_csv(f"gs://{BUCKET_NAME}/{FTSE_TEST}")

    print(f"  SP500 train: {len(sp500_train):,} | SP500 test: {len(sp500_test):,}")
    print(f"  FTSE  train: {len(ftse_train):,}  | FTSE  test: {len(ftse_test):,}")

    # FTSE100 completo (train+test) para usar como test en experimento C
    # En C entrenamos solo en SP500, así que ninguna fila de FTSE100 contamina el train.
    ftse_full = pd.concat([ftse_train, ftse_test], ignore_index=True)
    print(f"  FTSE100 completo (B train + B test): {len(ftse_full):,} filas")

    # Features comunes (determinadas por columnas disponibles en FTSE100)
    feat_cols = get_cross_features(sp500_train, ftse_full)

    global_summary = []

    # Experimento B: FTSE train (108k) → FTSE test (27k)
    if already_done("B-1"):
        print("\nSKIP B-1 — ya existe en GCS")
    else:
        line, mape, mse = run_B(ftse_train, ftse_test, feat_cols)
        global_summary.append(line)
        print(f"\n[B-1 DONE] MAPE_FTSE_test={mape:.4f} | MSE={mse:.4f}")

    # Experimento C: SP500 train (285k) → FTSE full (135k) + SP500 test (71k)
    if already_done("C-1"):
        print("\nSKIP C-1 — ya existe en GCS")
    else:
        line, mape, mse = run_C(sp500_train, sp500_test, ftse_full, feat_cols)
        global_summary.append(line)
        print(f"\n[C-1 DONE] MAPE_FTSE_full={mape:.4f} | MSE={mse:.4f}")

    # Resumen global
    if global_summary:
        upload_string("\n".join(global_summary), f"{OUTPUT_PREFIX}/model_summary.txt")

    print("\n=== EXPERIMENTOS B y C COMPLETADOS ===")
