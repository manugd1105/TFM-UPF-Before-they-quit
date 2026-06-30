"""
trainer/task_grupo5.py
Vertex AI Custom Job — Group 5 executions (all features, no PCA, 5-1 to 5-6).
Reads data from gs://<GCP_BUCKET>, saves results to gs://<GCP_BUCKET>/grupo_5/
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
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics
import xgboost as xgb
from google.cloud import storage

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────
BUCKET_NAME   = os.environ.get("GCP_BUCKET", "CHANGE_ME")
EXECUTION_ID  = "20240322_105534"
DATA_PREFIX   = "datos_entrada"
OUTPUT_PREFIX = "grupo_5"

N_ITER = {
    "elasticnet":    20,
    "knn":           40,
    "decision_tree": 40,
    "random_forest": 40,
    "svr":           10,
    "xgboost":      100,
}
N_FOLDS     = 5
N_JOBS      = 16
RANDOM_SEED = 8

# ─── Hyperparameter grids ─────────────────────────────────────
PARAM_GRIDS = {
    "elasticnet": {
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        "alpha":    [0.001, 0.01, 0.1, 1.0, 10.0],
    },
    "knn": {
        "n_neighbors": [3, 5, 7, 10, 15, 20, 30, 50],
        "weights":     ["uniform", "distance"],
        "metric":      ["euclidean", "manhattan", "minkowski"],
    },
    "decision_tree": {
        "max_depth":         [3, 5, 7, 10, 15, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf":  [1, 2, 5, 10],
        "max_features":      ["sqrt", "log2", None],
    },
    "random_forest": {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 5],
        "max_features":      ["sqrt", "log2"],
    },
    "svr": {
        "C":       [0.01, 0.1, 1.0, 10.0, 100.0],
        "kernel":  ["rbf", "linear", "poly"],
        "gamma":   ["scale", "auto"],
        "epsilon": [0.01, 0.1, 0.5, 1.0],
    },
    "xgboost": {
        "n_estimators":     [100, 200, 300, 500],
        "max_depth":        [3, 5, 7, 9],
        "learning_rate":    [0.01, 0.05, 0.1, 0.2],
        "subsample":        [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma":            [0, 0.1, 0.5],
    },
}

# ─── Feature set (all features) ───────────────────────────────
FEATURES_GROUP5 = [
    "summary_charlen","summary_wordlen","pros_charlen","pros_wordlen","cons_charlen","cons_wordlen",
    "summary_ld_PDT","summary_ld_IN","summary_ld_``","summary_ld_JJS","summary_ld_RBS","summary_ld_NNP",
    "summary_ld_DT","summary_ld_EX","summary_ld_NNS","summary_ld_CC","summary_ld_VBD","summary_ld_.",
    "summary_ld_(","summary_ld_UH","summary_ld_VBN","summary_ld_RP","summary_ld_JJR","summary_ld_WDT",
    "summary_ld_,","summary_ld_VBZ","summary_ld_MD","summary_ld_PRP","summary_ld_PRP$","summary_ld_$",
    "summary_ld_WP","summary_ld_VBG","summary_ld_NN","summary_ld_CD","summary_ld_)","summary_ld_RBR",
    "summary_ld_VBP","summary_ld_WRB","summary_ld_NNPS","summary_ld_TO","summary_ld_RB","summary_ld_VB",
    "summary_ld_JJ","pros_ld_PDT","pros_ld_IN","pros_ld_``","pros_ld_JJS","pros_ld_RBS","pros_ld_NNP",
    "pros_ld_DT","pros_ld_EX","pros_ld_NNS","pros_ld_CC","pros_ld_VBD","pros_ld_POS","pros_ld_.",
    "pros_ld_(","pros_ld_UH","pros_ld_VBN","pros_ld_RP","pros_ld_JJR","pros_ld_WDT","pros_ld_FW",
    "pros_ld_,","pros_ld_VBZ","pros_ld_MD","pros_ld_PRP","pros_ld_PRP$","pros_ld_$","pros_ld_WP",
    "pros_ld_VBG","pros_ld_NN","pros_ld_CD","pros_ld_)","pros_ld_SYM","pros_ld_RBR","pros_ld_VBP",
    "pros_ld_WRB","pros_ld_NNPS","pros_ld_TO","pros_ld_RB","pros_ld_VB","pros_ld_JJ","cons_ld_PDT",
    "cons_ld_IN","cons_ld_``","cons_ld_JJS","cons_ld_RBS","cons_ld_DT","cons_ld_NNP","cons_ld_EX",
    "cons_ld_NNS","cons_ld_CC","cons_ld_VBD","cons_ld_POS","cons_ld_.","cons_ld_(","cons_ld_VBN",
    "cons_ld_UH","cons_ld_RP","cons_ld_JJR","cons_ld_WDT","cons_ld_FW","cons_ld_,","cons_ld_VBZ",
    "cons_ld_MD","cons_ld_PRP","cons_ld_PRP$","cons_ld_$","cons_ld_WP","cons_ld_VBG","cons_ld_NN",
    "cons_ld_CD","cons_ld_)","cons_ld_SYM","cons_ld_RBR","cons_ld_VBP","cons_ld_WRB","cons_ld_NNPS",
    "cons_ld_TO","cons_ld_RB","cons_ld_VB","cons_ld_JJ","pros_emolex_fear","pros_emolex_anger",
    "pros_emolex_anticip","pros_emolex_trust","pros_emolex_surprise","pros_emolex_positive",
    "pros_emolex_negative","pros_emolex_sadness","pros_emolex_disgust","pros_emolex_joy",
    "cons_emolex_fear","cons_emolex_anger","cons_emolex_anticip","cons_emolex_trust",
    "cons_emolex_surprise","cons_emolex_positive","cons_emolex_negative","cons_emolex_sadness",
    "cons_emolex_disgust","cons_emolex_joy","pros_nltk_sia","cons_nltk_sia",
    "ODI_related_vocab_pros","ODI_related_vocab_cons",
    "pros_empath_help","pros_empath_office","pros_empath_money","pros_empath_domestic_work",
    "pros_empath_sleep","pros_empath_occupation","pros_empath_family","pros_empath_vacation",
    "pros_empath_health","pros_empath_swearing_terms","pros_empath_leisure","pros_empath_suffering",
    "pros_empath_wealthy","pros_empath_exercise","pros_empath_home","pros_empath_rage",
    "pros_empath_fun","pros_empath_negative_emotion","pros_empath_payment","pros_empath_achievement",
    "cons_empath_help","cons_empath_office","cons_empath_money","cons_empath_domestic_work",
    "cons_empath_sleep","cons_empath_occupation","cons_empath_family","cons_empath_vacation",
    "cons_empath_health","cons_empath_swearing_terms","cons_empath_leisure","cons_empath_suffering",
    "cons_empath_wealthy","cons_empath_exercise","cons_empath_home","cons_empath_rage",
    "cons_empath_fun","cons_empath_negative_emotion","cons_empath_payment",
    "pros_sim_ODI_Depressed mood","cons_sim_ODI_Depressed mood","pros_sim_ODI_Sleep alterations",
    "cons_sim_ODI_Sleep alterations","pros_sim_ODI_Fatigue","cons_sim_ODI_Fatigue",
    "pros_sim_ODI_Worthlessness","cons_sim_ODI_Worthlessness","pros_sim_ODI_Anhedonia",
    "cons_sim_ODI_Anhedonia","pros_sim_ODI_Appetite alterations","cons_sim_ODI_Appetite alterations",
    "pros_sim_ODI_Cognitive impairment","cons_sim_ODI_Cognitive impairment",
    "pros_sim_ODI_Psycomotor alterations","cons_sim_ODI_Psycomotor alterations",
    "pros_sim_ODI_Suicidal ideation","cons_sim_ODI_Suicidal ideation",
    "pros_sim_JDI_Work itself","cons_sim_JDI_Work itself","pros_sim_JDI_Pay","cons_sim_JDI_Pay",
    "pros_sim_JDI_Promotion","cons_sim_JDI_Promotion","pros_sim_JDI_Supervision",
    "cons_sim_JDI_Supervision","pros_sim_JDI_Coworkers","cons_sim_JDI_Coworkers",
    "GDP per capita","Population","Precipitation","AverageTemperatureF",
    "AverageTemperatureAvgHighF","AverageTemperatureAvgLowF","Republican/lean Rep.",
    "No lean","Democrat/lean Dem.","Crime index","Price per square foot ($)",
    "PercentHighSchoolOrHigher","PercentBachelorsOrHigher","State Tax Rate",
    "Market Capitalization","NumberofEmployees","Market Cap per Employee","Founded",
    "month","year","stock_percchange_month","stock_percchange_year",
    "sp500_percchange_month","sp500_percchange_year",
    "summary_ld_FW","summary_ld_POS","pros_ld_#","cons_ld_WP$","cons_ld_#","cons_ld_LS",
    "summary_ld_#","pros_ld_WP$","pros_ld_LS","summary_ld_SYM","summary_ld_WP$","summary_ld_LS",
    "days from review to t0","days from foundation to t0","days from foundation to review",
    "State name_targenc_median","State name_targenc_mean","Sector_targenc_median","Sector_targenc_mean",
    "State name_Alabama_ohe","State name_Alaska_ohe","State name_Arizona_ohe","State name_Arkansas_ohe",
    "State name_Bermuda_ohe","State name_California_ohe","State name_Canada_ohe","State name_Colorado_ohe",
    "State name_Connecticut_ohe","State name_Delaware_ohe","State name_District of Columbia_ohe",
    "State name_Florida_ohe","State name_Georgia_ohe","State name_Hawaii_ohe","State name_Idaho_ohe",
    "State name_Illinois_ohe","State name_Indiana_ohe","State name_Iowa_ohe","State name_Ireland_ohe",
    "State name_Israel_ohe","State name_Kansas_ohe","State name_Kentucky_ohe","State name_Louisiana_ohe",
    "State name_Maine_ohe","State name_Maryland_ohe","State name_Massachusetts_ohe",
    "State name_Michigan_ohe","State name_Minnesota_ohe","State name_Mississippi_ohe",
    "State name_Missouri_ohe","State name_Montana_ohe","State name_Nebraska_ohe",
    "State name_Netherlands_ohe","State name_Nevada_ohe","State name_New Hampshire_ohe",
    "State name_New Jersey_ohe","State name_New Mexico_ohe","State name_New York_ohe",
    "State name_North Carolina_ohe","State name_North Dakota_ohe","State name_Ohio_ohe",
    "State name_Oklahoma_ohe","State name_Oregon_ohe","State name_Pennsylvania_ohe",
    "State name_Rhode Island_ohe","State name_South Carolina_ohe","State name_South Dakota_ohe",
    "State name_Switzerland_ohe","State name_Tennessee_ohe","State name_Texas_ohe",
    "State name_United Kingdom_ohe","State name_Utah_ohe","State name_Vermont_ohe",
    "State name_Virginia_ohe","State name_Washington_ohe","State name_West Virginia_ohe",
    "State name_Wisconsin_ohe","State name_Wyoming_ohe",
    "Sector_Commercial Services_ohe","Sector_Communications_ohe","Sector_Consumer Durables_ohe",
    "Sector_Consumer Non-Durables_ohe","Sector_Consumer Services_ohe","Sector_Distribution Services_ohe",
    "Sector_Electronic Technology_ohe","Sector_Energy Minerals_ohe","Sector_Finance_ohe",
    "Sector_Health Services_ohe","Sector_Health Technology_ohe","Sector_Industrial Services_ohe",
    "Sector_Non-Energy Minerals_ohe","Sector_Process Industries_ohe","Sector_Producer Manufacturing_ohe",
    "Sector_Retail Trade_ohe","Sector_Technology Services_ohe","Sector_Transportation_ohe",
    "Sector_Utilities_ohe",
]

FEATURE_SETS = {"group5": FEATURES_GROUP5}

# ─── Executions ───────────────────────────────────────────────
EXECUTIONS = [
    ("5-1", "group5", None, "elasticnet"),
    ("5-2", "group5", None, "knn"),
    ("5-3", "group5", None, "decision_tree"),
    ("5-4", "group5", None, "random_forest"),
    ("5-5", "group5", None, "svr"),
    ("5-6", "group5", None, "xgboost"),
]

# ─── GCS helpers ──────────────────────────────────────────────
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

# ─── Model factory ────────────────────────────────────────────
def get_model(model_type):
    if model_type == "elasticnet":    return ElasticNet(random_state=RANDOM_SEED)
    if model_type == "knn":           return KNeighborsRegressor()
    if model_type == "decision_tree": return DecisionTreeRegressor(random_state=RANDOM_SEED)
    if model_type == "random_forest": return RandomForestRegressor(random_state=RANDOM_SEED)
    if model_type == "svr":           return SVR(max_iter=2000)
    if model_type == "xgboost":       return xgb.XGBRegressor(random_state=RANDOM_SEED)
    raise ValueError(f"Unknown model: {model_type}")

# ─── Main execution ───────────────────────────────────────────
global_summary_lines = []

def run_execution(exec_id, feature_group, model_type, train_raw, test_raw):
    print(f"\n{'='*60}")
    print(f"START: {exec_id} | {feature_group} | {model_type}")
    print(f"{'='*60}")

    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_folder = f"{OUTPUT_PREFIX}/{exec_id.replace('-', '_')}"

    available = [f for f in FEATURE_SETS[feature_group] if f in train_raw.columns]
    train_x   = train_raw[available].fillna(0)
    train_y   = train_raw["overall_rating"]
    test_x    = test_raw[available].fillna(0)
    test_y    = test_raw["overall_rating"]

    params    = PARAM_GRIDS[model_type]
    grid_size = 1
    for v in params.values():
        grid_size *= len(v)
    n_iter = min(N_ITER[model_type], grid_size)

    # SVR kernel matrix es O(n²) — con todos los datos y N_JOBS=1 solo hay 1 matriz en memoria
    n_jobs_run = 1 if model_type == "svr" else N_JOBS

    search = RandomizedSearchCV(
        estimator=get_model(model_type),
        param_distributions=params,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=N_FOLDS,
        random_state=RANDOM_SEED,
        n_jobs=n_jobs_run,
        verbose=1,
    )

    print(f"  Training {model_type} ({n_iter} iter x {N_FOLDS} folds, {len(available)} features)...")
    t0         = datetime.now()
    search.fit(train_x, train_y)
    train_time = str(datetime.now() - t0)
    print(f"  Done in {train_time}")

    train_pred = search.best_estimator_.predict(train_x)
    test_pred  = search.best_estimator_.predict(test_x)
    mape_train = metrics.mean_absolute_percentage_error(train_y, train_pred)
    mape_test  = metrics.mean_absolute_percentage_error(test_y,  test_pred)
    mse_train  = metrics.mean_squared_error(train_y, train_pred)
    mse_test   = metrics.mean_squared_error(test_y,  test_pred)
    print(f"  MAPE test: {mape_test:.4f} | MSE test: {mse_test:.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        def lp(fname): return os.path.join(tmp, fname)
        def gp(fname): return f"{gcs_folder}/{fname}"

        train_real    = pd.concat([train_raw[["review_id"]], train_y.reset_index(drop=True)], axis=1)
        test_real     = pd.concat([test_raw[["review_id"]],  test_y.reset_index(drop=True)],  axis=1)
        train_pred_df = pd.concat([train_raw[["review_id"]], pd.Series(train_pred, name=0)],   axis=1)
        test_pred_df  = pd.concat([test_raw[["review_id"]],  pd.Series(test_pred,  name=0)],   axis=1)

        for df, fname in [
            (train_real,                       f"Train real_{ts}.csv"),
            (test_real,                        f"Test real_{ts}.csv"),
            (train_pred_df,                    f"Train pred_{ts}.csv"),
            (test_pred_df,                     f"Test pred_{ts}.csv"),
            (pd.DataFrame(search.cv_results_), f"Building summary_{ts}.csv"),
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
                "execution_id": exec_id, "feature_group": feature_group,
                "pca_mode": None, "model_type": model_type,
                "n_iter": n_iter, "n_folds": N_FOLDS,
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
                "Model type": [model_type], "PCA": ["No"],
                "Number of search iterations": [n_iter], "Number of folds": [N_FOLDS],
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
            f"{ts}; Execution ID: {exec_id}; Model type: {model_type}; "
            f"Train time: {train_time}; MAPE train: {mape_train}; MAPE test: {mape_test}; "
            f"MSE train: {mse_train}; MSE test: {mse_test}"
        )
        smry_name = "model_summary.txt"
        with open(lp(smry_name), "w") as f:
            f.write(summary_line + "\n")
        upload_file(lp(smry_name), gp(smry_name))
        global_summary_lines.append(summary_line)

    print(f"  Saved to gs://{BUCKET_NAME}/{gcs_folder}/")
    return mape_test, mse_test


if __name__ == "__main__":
    print("Loading data from GCS...")
    train_raw = pd.read_csv(f"gs://{BUCKET_NAME}/{DATA_PREFIX}/dataframes_Train_{EXECUTION_ID}.csv")
    test_raw  = pd.read_csv(f"gs://{BUCKET_NAME}/{DATA_PREFIX}/dataframes_Test_{EXECUTION_ID}.csv")
    print(f"Train: {len(train_raw)} rows | Test: {len(test_raw)} rows")

    bucket    = gcs().bucket(BUCKET_NAME)
    completed = set()
    for exec_id, _, _, _ in EXECUTIONS:
        prefix = f"{OUTPUT_PREFIX}/{exec_id.replace('-', '_')}/Dashboard output"
        if any(True for _ in bucket.list_blobs(prefix=prefix)):
            completed.add(exec_id)

    results = []
    for exec_id, fg, pca_mode, model_type in EXECUTIONS:
        if exec_id in completed:
            print(f"SKIP {exec_id} — already completed")
            continue
        try:
            mape, mse = run_execution(exec_id, fg, model_type, train_raw, test_raw)
            results.append({"id": exec_id, "model": model_type, "mape_test": mape, "mse_test": mse})
        except Exception as e:
            print(f"ERROR in {exec_id}: {e}")
            results.append({"id": exec_id, "model": model_type, "error": str(e)})

    upload_string("\n".join(global_summary_lines), f"{OUTPUT_PREFIX}/model_summary.txt")

    print("\n=== FINAL RESULTS ===")
    for r in results:
        if "error" in r:
            print(f"  {r['id']} ({r['model']}): ERROR - {r['error']}")
        else:
            print(f"  {r['id']} ({r['model']}): MAPE={r['mape_test']:.4f} | MSE={r['mse_test']:.4f}")
