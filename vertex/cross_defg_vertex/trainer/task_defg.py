"""
trainer/task_defg.py
─────────────────────
Vertex AI Custom Job — Experimentos Cross-training D, E, F/G.

  D-1   XGBoost  FTSE100 FULL (135k) → SP500 FULL (357k)
        ¿Funciona lo aprendido en UK para predecir US?

  E-1   XGBoost  SP500 train (285k) + FTSE100 FULL (135k) → SP500 test (71k)
        ¿Añadir datos UK mejora la predicción en US?

  FG-1  XGBoost  SP500 train (285k) + FTSE100 train (108k) → FTSE100 test (27k) + SP500 test (71k)
        F: ¿Añadir datos US mejora la predicción en UK?
        G: Modelo global — métricas combinadas SP500+FTSE100.
        (F y G comparten el mismo modelo entrenado, evaluado en distintos test sets.)

Principio de datos máximos:
  - D: entrena en FTSE FULL (no se testea FTSE, sin riesgo de leakage).
  - E: entrena en SP500 train + FTSE FULL (no se testea FTSE, sin leakage).
  - FG: entrena en SP500 train + FTSE train (FTSE test queda limpio para evaluar F).
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

# ─── Config ───────────────────────────────────────────────────────────────────
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

PARAM_GRID = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 5, 7, 9],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma":            [0, 0.1, 0.5],
}

META_COLS = {
    "review_id", "summary", "date", "job_title", "overall_rating",
    "overall_rating_cat", "pros", "cons", "author_location", "company",
    "Company", "Ticker", "Sector", "Headquarters Location", "Unnamed: 0",
    "author_location_filledhq", "author_location_filledhq_state",
    "Acronym", "State name", "Headquarters state",
    "wl_balance", "culture_values", "diversity_inclusion",
    "career_opportunities", "compensation_benefits", "senior_management",
}

# ─── GCS helpers ──────────────────────────────────────────────────────────────
_gcs = None

def gcs():
    global _gcs
    if _gcs is None:
        _gcs = storage.Client()
    return _gcs

def upload_file(local, gcs_path):
    gcs().bucket(BUCKET_NAME).blob(gcs_path).upload_from_filename(local)
    print(f"    -> gs://{BUCKET_NAME}/{gcs_path}")

def upload_str(content, gcs_path):
    gcs().bucket(BUCKET_NAME).blob(gcs_path).upload_from_string(content)

def already_done(exec_id):
    prefix = f"{OUTPUT_PREFIX}/{exec_id.replace('-','_')}/Dashboard output"
    return any(True for _ in gcs().bucket(BUCKET_NAME).list_blobs(prefix=prefix))

def read_gcs(path):
    return pd.read_csv(f"gs://{BUCKET_NAME}/{path}", low_memory=False)

# ─── Feature selection ────────────────────────────────────────────────────────
def cross_features(ref_df, ftse_df):
    """Columnas numéricas presentes en ambos datasets, sin metadata."""
    a = set(ref_df.select_dtypes("number").columns) - META_COLS
    b = set(ftse_df.select_dtypes("number").columns) - META_COLS
    cols = sorted(a & b)
    print(f"  Features comunes (group5 sin estado): {len(cols)}")
    return cols

# ─── Core train + eval ────────────────────────────────────────────────────────
def train_xgb(train_x, train_y, label=""):
    model = xgb.XGBRegressor(
        objective="reg:squarederror", tree_method="hist",
        random_state=RANDOM_SEED, n_jobs=N_JOBS, verbosity=0,
    )
    search = RandomizedSearchCV(
        estimator=model, param_distributions=PARAM_GRID,
        n_iter=N_ITER, scoring="neg_root_mean_squared_error",
        cv=N_FOLDS, random_state=RANDOM_SEED, n_jobs=1, verbose=1,
    )
    print(f"  Entrenando XGBoost {label} ({N_ITER} iter × {N_FOLDS} folds, {len(train_x):,} rows)...")
    t0 = datetime.now()
    search.fit(train_x, train_y)
    elapsed = str(datetime.now() - t0)
    print(f"  Completado en {elapsed} | Best params: {search.best_params_}")
    return search, elapsed

def eval_set(search, X, y, label):
    pred = search.best_estimator_.predict(X)
    mape = metrics.mean_absolute_percentage_error(y, pred)
    mse  = metrics.mean_squared_error(y, pred)
    mae  = metrics.mean_absolute_error(y, pred)
    print(f"  [{label}] MAPE={mape:.4f} | MSE={mse:.4f} | MAE={mae:.4f}")
    return pred, {"mape": mape, "mse": mse, "mae": mae}

# ─── Save all artifacts ───────────────────────────────────────────────────────
def save(tmp, gcs_folder, ts, exec_id, model_type, n_iter,
         train_raw, train_y, train_pred, train_time,
         test_sets,   # [(label, raw_df, y, pred, metrics_dict)]
         search, feat_cols):

    def lp(f): return os.path.join(tmp, f)
    def gp(f): return f"{gcs_folder}/{f}"

    mape_train = metrics.mean_absolute_percentage_error(train_y, train_pred)
    mse_train  = metrics.mean_squared_error(train_y, train_pred)

    # Predictions CSVs
    pd.DataFrame({"review_id": train_raw["review_id"].values,
                  "real": train_y.values, "pred": train_pred})\
        .to_csv(lp(f"Train_preds_{ts}.csv"), index=False)
    upload_file(lp(f"Train_preds_{ts}.csv"), gp(f"Train_preds_{ts}.csv"))

    for label, raw_df, y, pred, _ in test_sets:
        pd.DataFrame({"review_id": raw_df["review_id"].values,
                      "real": y.values, "pred": pred})\
            .to_csv(lp(f"Test_preds_{label}_{ts}.csv"), index=False)
        upload_file(lp(f"Test_preds_{label}_{ts}.csv"), gp(f"Test_preds_{label}_{ts}.csv"))

    # Scatter plots
    for y_r, y_p, name in [(train_y, train_pred, "Train")] + \
                           [(y, pred, lbl) for lbl, _, y, pred, _ in test_sets]:
        plt.figure(figsize=(6, 5))
        plt.scatter(y_r, y_p, alpha=0.3, s=1)
        plt.xlabel("Real"); plt.ylabel("Predicted"); plt.title(f"{exec_id} | {name}")
        plt.tight_layout()
        plt.savefig(lp(f"{name}_scatter_{ts}.png"), dpi=100); plt.close()
        upload_file(lp(f"{name}_scatter_{ts}.png"), gp(f"{name}_scatter_{ts}.png"))

    # CV results
    pd.DataFrame(search.cv_results_)\
        .to_csv(lp(f"Building_summary_{ts}.csv"), index=False)
    upload_file(lp(f"Building_summary_{ts}.csv"), gp(f"Building_summary_{ts}.csv"))

    # Feature importance
    fi = pd.DataFrame({"feature": feat_cols,
                        "importance": search.best_estimator_.feature_importances_})\
           .sort_values("importance", ascending=False)
    fi.to_csv(lp(f"feature_importance_{ts}.csv"), index=False)
    upload_file(lp(f"feature_importance_{ts}.csv"), gp(f"feature_importance_{ts}.csv"))

    # Config
    test_m = {lbl: m for lbl, _, _, _, m in test_sets}
    cfg = {
        "execution_id": exec_id, "model_type": model_type,
        "n_iter": n_iter, "n_folds": N_FOLDS,
        "n_features": len(feat_cols),
        "train_rows": len(train_raw),
        "train_time": train_time,
        "best_params": search.best_params_,
        "mape_train": mape_train, "mse_train": mse_train,
        "test_metrics": test_m,
    }
    with open(lp(f"config_{ts}.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    upload_file(lp(f"config_{ts}.json"), gp(f"config_{ts}.json"))

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

    primary_lbl = test_sets[0][0]
    xls = f"Dashboard_output_{ts}.xlsx"
    with pd.ExcelWriter(lp(xls), engine="xlsxwriter") as writer:
        general = {
            "Execution name": [ts], "Execution ID": [exec_id],
            "Model type": [model_type], "PCA": ["No"],
            "n_iter": [n_iter], "n_folds": [N_FOLDS],
            "Train time": [train_time], "Train rows": [len(train_raw)],
            "Metric 1": ["MAPE"], "Metric 1 train": [mape_train],
            "Metric 2": ["MSE"],  "Metric 2 train": [mse_train],
        }
        for lbl, _, _, _, m in test_sets:
            general[f"MAPE_{lbl}"] = [m["mape"]]
            general[f"MSE_{lbl}"]  = [m["mse"]]
        pd.DataFrame(general).to_excel(writer, sheet_name="General", index=False)

        # Train predictions sheet
        tr = pd.DataFrame({"review_id": train_raw["review_id"].values,
                            f"Real_{ts}": train_y.values, f"Pred_{ts}": train_pred})
        tr.to_excel(writer, sheet_name="Train predictions", index=False)

        for lbl, raw_df, y, pred, _ in test_sets:
            sheet = pd.DataFrame({"review_id": raw_df["review_id"].values,
                                   f"Real_{ts}": y.values, f"Pred_{ts}": pred})
            sheet.to_excel(writer, sheet_name=f"Test {lbl}"[:31], index=False)

        deepdive.to_excel(writer, sheet_name="Execution deepdive", index=False)
    upload_file(lp(xls), gp(xls))

    # Pickle
    with open(lp(f"dict_{ts}.pkl"), "wb") as f:
        pickle.dump({"Random search object": search}, f)
    upload_file(lp(f"dict_{ts}.pkl"), gp(f"dict_{ts}.pkl"))

    # Summary line
    parts = [f"{ts}; Execution ID: {exec_id}; Model type: {model_type}; "
             f"Train time: {train_time}; Train rows: {len(train_raw)}; "
             f"MAPE train: {mape_train:.4f}; MSE train: {mse_train:.4f}"]
    for lbl, _, _, _, m in test_sets:
        parts.append(f"MAPE_{lbl}: {m['mape']:.4f}; MSE_{lbl}: {m['mse']:.4f}")
    summary = "; ".join(parts)
    with open(lp("model_summary.txt"), "w") as f:
        f.write(summary + "\n")
    upload_file(lp("model_summary.txt"), gp("model_summary.txt"))
    return summary


# ─── D-1: FTSE FULL → SP500 FULL ─────────────────────────────────────────────
def run_D(ftse_full, sp500_full, feat_cols):
    """Train: FTSE100 completo (135k). Test: SP500 completo (357k)."""
    print(f"\n{'='*60}")
    print("D-1: FTSE FULL (135k) → SP500 FULL (357k)")
    print(f"{'='*60}")

    train_x = ftse_full[feat_cols].fillna(0)
    train_y = ftse_full["overall_rating"]
    test_x  = sp500_full[feat_cols].fillna(0)
    test_y  = sp500_full["overall_rating"]

    search, elapsed = train_xgb(train_x, train_y, label="[D-1 FTSE→SP500]")
    train_pred, m_tr = eval_set(search, train_x, train_y, "FTSE_train (in-sample)")
    test_pred,  m_te = eval_set(search, test_x,  test_y,  "SP500_full (cross-mkt)")

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_folder = f"{OUTPUT_PREFIX}/D_1"
    with tempfile.TemporaryDirectory() as tmp:
        line = save(tmp, gcs_folder, ts, "D-1", "xgboost", N_ITER,
                    ftse_full, train_y, train_pred, elapsed,
                    [("SP500_full", sp500_full, test_y, test_pred, m_te)],
                    search, feat_cols)
    print(f"\n[D-1 DONE] MAPE_SP500_full={m_te['mape']:.4f} | MSE={m_te['mse']:.4f}")
    return line, m_te


# ─── E-1: SP500 train + FTSE FULL → SP500 test ───────────────────────────────
def run_E(sp500_train, sp500_test, ftse_full, feat_cols):
    """Train: SP500 train (285k) + FTSE FULL (135k) = 420k. Test: SP500 test (71k)."""
    print(f"\n{'='*60}")
    print("E-1: SP500 train + FTSE FULL (420k) → SP500 test (71k)")
    print(f"{'='*60}")

    combined  = pd.concat([sp500_train, ftse_full], ignore_index=True)
    train_x   = combined[feat_cols].fillna(0)
    train_y   = combined["overall_rating"]
    test_x    = sp500_test[feat_cols].fillna(0)
    test_y    = sp500_test["overall_rating"]
    print(f"  Train combinado: {len(combined):,} rows (SP500: {len(sp500_train):,} + FTSE: {len(ftse_full):,})")

    search, elapsed = train_xgb(train_x, train_y, label="[E-1 SP500+FTSE→SP500]")
    train_pred, m_tr = eval_set(search, train_x, train_y, "combined train (in-sample)")
    test_pred,  m_te = eval_set(search, test_x,  test_y,  "SP500 test")
    print(f"  [Ref. 7-4 SP500 solo, group5 full] MAPE=0.2549")

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_folder = f"{OUTPUT_PREFIX}/E_1"
    with tempfile.TemporaryDirectory() as tmp:
        line = save(tmp, gcs_folder, ts, "E-1", "xgboost", N_ITER,
                    combined, train_y, train_pred, elapsed,
                    [("SP500test", sp500_test, test_y, test_pred, m_te)],
                    search, feat_cols)
    print(f"\n[E-1 DONE] MAPE_SP500test={m_te['mape']:.4f} | MSE={m_te['mse']:.4f}")
    return line, m_te


# ─── FG-1: SP500 train + FTSE train → FTSE test + SP500 test ─────────────────
def run_FG(sp500_train, sp500_test, ftse_train, ftse_test, feat_cols):
    """
    Train: SP500 train (285k) + FTSE train (108k) = 393k.
    Test F: FTSE test (27k)  — ¿datos SP500 mejoran predicción en UK?
    Test G: SP500 test (71k) + FTSE test (27k) combined — modelo global.
    """
    print(f"\n{'='*60}")
    print("FG-1: SP500 train + FTSE train (393k) → FTSE test + SP500 test")
    print(f"{'='*60}")

    combined  = pd.concat([sp500_train, ftse_train], ignore_index=True)
    train_x   = combined[feat_cols].fillna(0)
    train_y   = combined["overall_rating"]
    ftse_x    = ftse_test[feat_cols].fillna(0)
    ftse_y    = ftse_test["overall_rating"]
    sp500_x   = sp500_test[feat_cols].fillna(0)
    sp500_y   = sp500_test["overall_rating"]
    print(f"  Train combinado: {len(combined):,} rows (SP500: {len(sp500_train):,} + FTSE: {len(ftse_train):,})")

    search, elapsed = train_xgb(train_x, train_y, label="[FG-1 SP500+FTSE→both]")
    train_pred, m_tr  = eval_set(search, train_x, train_y, "combined train (in-sample)")
    ftse_pred,  m_f   = eval_set(search, ftse_x,  ftse_y,  "FTSE test (F)")
    sp500_pred, m_g   = eval_set(search, sp500_x, sp500_y, "SP500 test (G-component)")

    # G: métricas sobre el test combinado SP500+FTSE
    combined_test = pd.concat([sp500_test, ftse_test], ignore_index=True)
    combined_x    = combined_test[feat_cols].fillna(0)
    combined_y    = combined_test["overall_rating"]
    combo_pred, m_combo = eval_set(search, combined_x, combined_y, "SP500+FTSE test combined (G)")

    print(f"  [Ref. B-1 FTSE solo]         MAPE_FTSE será la referencia de F")
    print(f"  [Ref. 7-4 SP500 group5 full] MAPE SP500=0.2549")

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_folder = f"{OUTPUT_PREFIX}/FG_1"
    with tempfile.TemporaryDirectory() as tmp:
        line = save(tmp, gcs_folder, ts, "FG-1", "xgboost", N_ITER,
                    combined, train_y, train_pred, elapsed,
                    [
                        ("FTSE_test",         ftse_test,     ftse_y,    ftse_pred,  m_f),
                        ("SP500_test",         sp500_test,    sp500_y,   sp500_pred, m_g),
                        ("SP500_FTSE_combined",combined_test, combined_y, combo_pred, m_combo),
                    ],
                    search, feat_cols)
    print(f"\n[FG-1 DONE]")
    print(f"  F: MAPE_FTSE_test={m_f['mape']:.4f}      | MSE={m_f['mse']:.4f}")
    print(f"  G: MAPE_SP500_test={m_g['mape']:.4f}     | MSE={m_g['mse']:.4f}")
    print(f"  G: MAPE_combined={m_combo['mape']:.4f}   | MSE={m_combo['mse']:.4f}")
    return line, m_f, m_g, m_combo


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Cross-training experiments D, E, F/G")
    print("=" * 60)

    print("\nCargando datos desde GCS...")
    sp500_train = read_gcs(SP500_TRAIN)
    sp500_test  = read_gcs(SP500_TEST)
    ftse_train  = read_gcs(FTSE_TRAIN)
    ftse_test   = read_gcs(FTSE_TEST)

    print(f"  SP500 train: {len(sp500_train):,} | SP500 test: {len(sp500_test):,}")
    print(f"  FTSE  train: {len(ftse_train):,}  | FTSE  test: {len(ftse_test):,}")

    # Datasets completos (para D y E que pueden usar todo sin leakage)
    ftse_full  = pd.concat([ftse_train,  ftse_test],  ignore_index=True)
    sp500_full = pd.concat([sp500_train, sp500_test], ignore_index=True)
    print(f"  FTSE  FULL: {len(ftse_full):,}  | SP500 FULL: {len(sp500_full):,}")

    # Features comunes (determinadas por FTSE100 = subconjunto de SP500)
    feat_cols = cross_features(sp500_train, ftse_full)

    global_summary = []

    # D-1
    if already_done("D-1"):
        print("\nSKIP D-1 — ya existe en GCS")
    else:
        line, _ = run_D(ftse_full, sp500_full, feat_cols)
        global_summary.append(line)

    # E-1
    if already_done("E-1"):
        print("\nSKIP E-1 — ya existe en GCS")
    else:
        line, _ = run_E(sp500_train, sp500_test, ftse_full, feat_cols)
        global_summary.append(line)

    # FG-1
    if already_done("FG-1"):
        print("\nSKIP FG-1 — ya existe en GCS")
    else:
        line, *_ = run_FG(sp500_train, sp500_test, ftse_train, ftse_test, feat_cols)
        global_summary.append(line)

    if global_summary:
        upload_str("\n".join(global_summary),
                   f"{OUTPUT_PREFIX}/model_summary_defg.txt")

    print("\n=== EXPERIMENTOS D, E, F/G COMPLETADOS ===")
