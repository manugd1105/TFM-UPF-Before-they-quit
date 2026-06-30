"""
trainer/task_grupo1_svr.py
Vertex AI Custom Job — Group 1, exec 1-5 (SVR, string length features).
Reduced iterations to keep runtime manageable.
"""

import os
import json
import pickle
import warnings
import tempfile
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn import metrics
from google.cloud import storage

warnings.filterwarnings("ignore")

BUCKET_NAME   = os.environ.get("GCP_BUCKET", "CHANGE_ME")
EXECUTION_ID  = "20240322_105534"
DATA_PREFIX   = "datos_entrada"
OUTPUT_PREFIX = "grupo_1"

N_ITER      = 10   # 10 combinaciones × 5 folds = 50 fits (~2-7h)
N_FOLDS     = 5
RANDOM_SEED = 8

PARAM_GRID = {
    "C":       [0.01, 0.1, 1.0, 10.0, 100.0],
    "kernel":  ["rbf", "linear", "poly"],
    "gamma":   ["scale", "auto"],
    "epsilon": [0.01, 0.1, 0.5, 1.0],
}

FEATURES_GROUP1 = [
    "summary_charlen", "summary_wordlen",
    "pros_charlen",    "pros_wordlen",
    "cons_charlen",    "cons_wordlen",
]

_gcs_client = None

def gcs():
    global _gcs_client
    if _gcs_client is None:
        _gcs_client = storage.Client()
    return _gcs_client

def upload_file(local_path, gcs_path):
    gcs().bucket(BUCKET_NAME).blob(gcs_path).upload_from_filename(local_path)
    print(f"    -> gs://{BUCKET_NAME}/{gcs_path}")

def upload_string(content, gcs_path):
    gcs().bucket(BUCKET_NAME).blob(gcs_path).upload_from_string(content)


if __name__ == "__main__":
    exec_id    = "1-5"
    gcs_folder = "grupo_1/1_5"

    # Skip if already done
    prefix = f"{gcs_folder}/Dashboard output"
    if any(True for _ in gcs().bucket(BUCKET_NAME).list_blobs(prefix=prefix)):
        print(f"SKIP {exec_id} — already in GCS")
        exit(0)

    print("Loading data from GCS...")
    train_raw = pd.read_csv(f"gs://{BUCKET_NAME}/{DATA_PREFIX}/dataframes_Train_{EXECUTION_ID}.csv")
    test_raw  = pd.read_csv(f"gs://{BUCKET_NAME}/{DATA_PREFIX}/dataframes_Test_{EXECUTION_ID}.csv")
    print(f"Train: {len(train_raw)} rows | Test: {len(test_raw)} rows")

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    avail   = [f for f in FEATURES_GROUP1 if f in train_raw.columns]
    train_x = train_raw[avail].fillna(0)
    train_y = train_raw["overall_rating"]
    test_x  = test_raw[avail].fillna(0)
    test_y  = test_raw["overall_rating"]

    # SVR: n_jobs=1 (una sola matriz kernel en memoria a la vez), max_iter=500
    search = RandomizedSearchCV(
        estimator=SVR(max_iter=2000),
        param_distributions=PARAM_GRID,
        n_iter=N_ITER,
        scoring="neg_root_mean_squared_error",
        cv=N_FOLDS,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbose=1,
    )

    print(f"Training SVR ({N_ITER} iter x {N_FOLDS} folds, {len(avail)} features)...")
    t0         = datetime.now()
    search.fit(train_x, train_y)
    train_time = str(datetime.now() - t0)
    print(f"Done in {train_time}")

    train_pred = search.best_estimator_.predict(train_x)
    test_pred  = search.best_estimator_.predict(test_x)
    mape_train = metrics.mean_absolute_percentage_error(train_y, train_pred)
    mape_test  = metrics.mean_absolute_percentage_error(test_y,  test_pred)
    mse_train  = metrics.mean_squared_error(train_y, train_pred)
    mse_test   = metrics.mean_squared_error(test_y,  test_pred)
    print(f"MAPE test: {mape_test:.4f} | MSE test: {mse_test:.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        def lp(fname): return os.path.join(tmp, fname)
        def gp(fname): return f"{gcs_folder}/{fname}"

        train_real    = pd.concat([train_raw[["review_id"]], train_y.reset_index(drop=True)], axis=1)
        test_real     = pd.concat([test_raw[["review_id"]],  test_y.reset_index(drop=True)],  axis=1)
        train_pred_df = pd.concat([train_raw[["review_id"]], pd.Series(train_pred, name=0)],   axis=1)
        test_pred_df  = pd.concat([test_raw[["review_id"]],  pd.Series(test_pred,  name=0)],   axis=1)

        for df, fname in [
            (train_real,                          f"Train real_{ts}.csv"),
            (test_real,                           f"Test real_{ts}.csv"),
            (train_pred_df,                       f"Train pred_{ts}.csv"),
            (test_pred_df,                        f"Test pred_{ts}.csv"),
            (pd.DataFrame(search.cv_results_),    f"Building summary_{ts}.csv"),
        ]:
            df.to_csv(lp(fname), index=False)
            upload_file(lp(fname), gp(fname))

        for real, pred, name in [(train_y, train_pred, "Training"), (test_y, test_pred, "Testing")]:
            fig_name = f"{name}_comparison_{ts}.png"
            plt.figure()
            plt.scatter(real, pred, alpha=0.3, s=1)
            plt.xlabel("Real"); plt.ylabel("Predicted"); plt.title(f"{name} comparison")
            plt.savefig(lp(fig_name)); plt.close()
            upload_file(lp(fig_name), gp(fig_name))

        cfg_name = f"config_{ts}.json"
        with open(lp(cfg_name), "w") as f:
            json.dump({
                "execution_id": exec_id, "feature_group": "group1",
                "pca_mode": None, "model_type": "svr",
                "n_iter": N_ITER, "n_folds": N_FOLDS,
                "max_iter_svr": 500,
                "best_params": search.best_params_,
            }, f)
        upload_file(lp(cfg_name), gp(cfg_name))

        pkl_name = f"dict_{ts}.pkl"
        with open(lp(pkl_name), "wb") as f:
            pickle.dump({"Random search object": search}, f)
        upload_file(lp(pkl_name), gp(pkl_name))

        lst_params = list(search.cv_results_["params"][0].keys())
        deepdive   = pd.DataFrame({
            col: search.cv_results_[col]
            for col in ["mean_fit_time","std_fit_time","mean_score_time","std_score_time",
                        "mean_test_score","std_test_score","rank_test_score"] +
                       [f"split{s}_test_score" for s in range(N_FOLDS)]
        })
        deepdive["Execution name"] = ts
        hp_df = pd.DataFrame({k: [d[k] for d in search.cv_results_["params"]] for k in lst_params})
        for i, hp in enumerate(lst_params):
            deepdive[f"param_{i}_name"] = hp
            deepdive[f"param_{i}"]      = hp_df[hp]

        xls_name = f"Dashboard output_{ts}.xlsx"
        with pd.ExcelWriter(lp(xls_name), engine="xlsxwriter") as writer:
            pd.DataFrame({
                "Execution name": [ts],     "Execution ID": [exec_id],
                "Model type": ["svr"],      "PCA": ["No"],
                "Number of search iterations": [N_ITER], "Number of folds": [N_FOLDS],
                "Train time": [train_time],
                "Metric 1": ["MAPE"], "Metric 1 train": [mape_train], "Metric 1 test": [mape_test],
                "Metric 2": ["MSE"],  "Metric 2 train": [mse_train],  "Metric 2 test": [mse_test],
            }).to_excel(writer, sheet_name="General", index=False)
            pd.merge(train_real, train_pred_df, on="review_id").rename(
                columns={"overall_rating": f"Real_{ts}", 0: f"Pred_{ts}"}
            ).to_excel(writer, sheet_name="Train predictions", index=False)
            pd.merge(test_real, test_pred_df, on="review_id").rename(
                columns={"overall_rating": f"Real_{ts}", 0: f"Pred_{ts}"}
            ).to_excel(writer, sheet_name="Test predictions", index=False)
            deepdive.to_excel(writer, sheet_name="Execution deepdive", index=False)
        upload_file(lp(xls_name), gp(xls_name))

        summary_line = (
            f"{ts}; Execution ID: {exec_id}; Model type: svr; "
            f"Train time: {train_time}; MAPE train: {mape_train}; MAPE test: {mape_test}; "
            f"MSE train: {mse_train}; MSE test: {mse_test}"
        )
        smry_name = "model_summary.txt"
        with open(lp(smry_name), "w") as f:
            f.write(summary_line + "\n")
        upload_file(lp(smry_name), gp(smry_name))

    print(f"Done. Saved to gs://{BUCKET_NAME}/{gcs_folder}/")
    print(f"MAPE test: {mape_test:.4f} | MSE test: {mse_test:.4f}")
