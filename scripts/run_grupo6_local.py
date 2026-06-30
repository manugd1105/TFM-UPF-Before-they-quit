"""
run_grupo6_local.py
Group 6 — LightGBM, all feature/PCA combos (6-1 to 6-13).
Reads locally preprocessed data (already StandardScaler'd).
Saves results to 05_outputs_modeling/grupo_6/
"""

import os
import math
import json
import pickle
import warnings
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn import metrics
import lightgbm as lgb

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────
EXECUTION_ID = "20240322_105534"
DATA_DIR     = f"03_outputs_data_preprocessing/output_data_preprocessing_{EXECUTION_ID}"
TRAIN_PATH   = f"{DATA_DIR}/dataframes_Train_{EXECUTION_ID}.csv"
TEST_PATH    = f"{DATA_DIR}/dataframes_Test_{EXECUTION_ID}.csv"
OUTPUT_DIR   = "05_outputs_modeling/grupo_6"

# ─── Config ───────────────────────────────────────────────────
N_ITER_LGBM = 100
N_FOLDS     = 5
N_JOBS      = 4
RANDOM_SEED = 8

# ─── Hyperparameter grid ──────────────────────────────────────
PARAM_GRID = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [3, 5, 7, 9, -1],
    "learning_rate":     [0.01, 0.05, 0.1, 0.2],
    "num_leaves":        [31, 63, 127],
    "subsample":         [0.6, 0.8, 1.0],
    "colsample_bytree":  [0.6, 0.8, 1.0],
    "min_child_samples": [10, 20, 50],
    "reg_alpha":         [0, 0.1, 1.0],
    "reg_lambda":        [0, 0.1, 1.0],
}

# ─── Feature sets (identical to cloud) ───────────────────────
FEATURES_ALL = [
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

_NLP_PREFIXES    = [
    "summary_charlen","summary_wordlen","pros_charlen","pros_wordlen","cons_charlen","cons_wordlen",
    "summary_ld_","pros_ld_","cons_ld_","pros_emolex_","cons_emolex_",
    "pros_nltk_","cons_nltk_","ODI_related_vocab_","pros_empath_","cons_empath_",
]
_SIM_PREFIXES    = ["pros_sim_ODI_","cons_sim_ODI_","pros_sim_JDI_","cons_sim_JDI_"]

FEATURES_GROUP1 = ["summary_charlen","summary_wordlen","pros_charlen","pros_wordlen","cons_charlen","cons_wordlen"]
FEATURES_GROUP3 = [f for f in FEATURES_ALL if any(f.startswith(p) for p in _NLP_PREFIXES)]
FEATURES_GROUP4 = [f for f in FEATURES_ALL if any(f.startswith(p) for p in _NLP_PREFIXES + _SIM_PREFIXES)]

FEATURE_SETS = {
    "group1": FEATURES_GROUP1,
    "group2": FEATURES_ALL,
    "group3": FEATURES_GROUP3,
    "group4": FEATURES_GROUP4,
    "group5": FEATURES_ALL,
}

PCA_FEATURE_GROUPS = {
    "pca0":  ["summary_charlen","summary_wordlen","pros_charlen","pros_wordlen","cons_charlen","cons_wordlen"],
    "pca1":  [f for f in FEATURES_ALL if any(f.startswith(p) for p in ["summary_ld_","pros_ld_","cons_ld_"])],
    "pca2":  [f for f in FEATURES_ALL if "emolex" in f],
    "pca3":  ["pros_nltk_sia","cons_nltk_sia"],
    "pca4":  ["ODI_related_vocab_pros","ODI_related_vocab_cons"],
    "pca5":  [f for f in FEATURES_ALL if "empath" in f],
    "pca6":  [f for f in FEATURES_ALL if "sim_ODI" in f or "sim_JDI" in f],
    "pca7":  ["GDP per capita","Population","Precipitation","AverageTemperatureF",
              "AverageTemperatureAvgHighF","AverageTemperatureAvgLowF","Republican/lean Rep.",
              "No lean","Democrat/lean Dem.","Crime index","Price per square foot ($)",
              "PercentHighSchoolOrHigher","PercentBachelorsOrHigher","State Tax Rate"],
    "pca8":  ["Market Capitalization","NumberofEmployees","Market Cap per Employee","Founded","month","year"],
    "pca9":  ["stock_percchange_month","stock_percchange_year","sp500_percchange_month","sp500_percchange_year"],
    "pca10": ["days from review to t0","days from foundation to t0","days from foundation to review"],
    "pca11": [f for f in FEATURES_ALL if "State name_" in f],
    "pca12": [f for f in FEATURES_ALL if "Sector_" in f],
}

EXECUTIONS = [
    ("6-1",  "group1", None,      "lightgbm"),
    ("6-2",  "group2", "1_dim",   "lightgbm"),
    ("6-3",  "group2", "5_dim",   "lightgbm"),
    ("6-4",  "group2", "1_per_5", "lightgbm"),
    ("6-5",  "group3", None,      "lightgbm"),
    ("6-6",  "group3", "1_dim",   "lightgbm"),
    ("6-7",  "group3", "5_dim",   "lightgbm"),
    ("6-8",  "group3", "1_per_5", "lightgbm"),
    ("6-9",  "group4", None,      "lightgbm"),
    ("6-10", "group4", "1_dim",   "lightgbm"),
    ("6-11", "group4", "5_dim",   "lightgbm"),
    ("6-12", "group4", "1_per_5", "lightgbm"),
    ("6-13", "group5", None,      "lightgbm"),
]

# ─── PCA helpers ──────────────────────────────────────────────
def get_pca_dims(n_features, pca_mode):
    if pca_mode == "1_dim":   return 1
    if pca_mode == "5_dim":   return min(5, n_features)
    if pca_mode == "1_per_5": return max(1, math.ceil(n_features / 5))

def build_pca_config(input_features, pca_mode):
    result = []
    for name, group_feats in PCA_FEATURE_GROUPS.items():
        active = [f for f in group_feats if f in input_features]
        if len(active) < 2:
            continue
        n_dims = min(get_pca_dims(len(active), pca_mode), len(active))
        result.append((name, active, n_dims))
    return result

# ─── Main execution ───────────────────────────────────────────
global_summary_lines = []

def run_execution(exec_id, feature_group, pca_mode, train_raw, test_raw):
    print(f"\n{'='*60}")
    print(f"START: {exec_id} | {feature_group} | PCA={pca_mode} | lightgbm")
    print(f"{'='*60}", flush=True)

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join(OUTPUT_DIR, exec_id.replace("-", "_"))
    os.makedirs(out_dir, exist_ok=True)

    available = [f for f in FEATURE_SETS[feature_group] if f in train_raw.columns]
    train_x   = train_raw[available].fillna(0)
    train_y   = train_raw["overall_rating"]
    test_x    = test_raw[available].fillna(0)
    test_y    = test_raw["overall_rating"]

    dict_out = {}

    if pca_mode is not None:
        for pca_name, feats, n_dims in build_pca_config(list(train_x.columns), pca_mode):
            print(f"  PCA {pca_name}: {len(feats)} features -> {n_dims} dims")
            pca    = PCA(n_components=n_dims)
            tr_pca = pd.DataFrame(pca.fit_transform(train_x[feats]),
                                  columns=[f"{pca_name}_{i}" for i in range(n_dims)],
                                  index=train_x.index)
            te_pca = pd.DataFrame(pca.transform(test_x[feats]),
                                  columns=[f"{pca_name}_{i}" for i in range(n_dims)],
                                  index=test_x.index)
            train_x = pd.concat([train_x.drop(columns=feats), tr_pca], axis=1)
            test_x  = pd.concat([test_x.drop(columns=feats),  te_pca], axis=1)
            dict_out[pca_name] = pca

    grid_size = 1
    for v in PARAM_GRID.values():
        grid_size *= len(v)
    n_iter = min(N_ITER_LGBM, grid_size)

    model  = lgb.LGBMRegressor(random_state=RANDOM_SEED, n_jobs=N_JOBS, verbose=-1)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_GRID,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=N_FOLDS,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        verbose=1,
    )

    print(f"  Training lightgbm ({n_iter} iter x {N_FOLDS} folds, {len(train_x.columns)} features)...", flush=True)
    t0         = datetime.now()
    search.fit(train_x, train_y)
    train_time = str(datetime.now() - t0)
    print(f"  Done in {train_time}", flush=True)

    dict_out["Random search object"] = search

    train_pred = search.best_estimator_.predict(train_x)
    test_pred  = search.best_estimator_.predict(test_x)
    mape_train = metrics.mean_absolute_percentage_error(train_y, train_pred)
    mape_test  = metrics.mean_absolute_percentage_error(test_y,  test_pred)
    mse_train  = metrics.mean_squared_error(train_y, train_pred)
    mse_test   = metrics.mean_squared_error(test_y,  test_pred)
    print(f"  MAPE test: {mape_test:.4f} | MSE test: {mse_test:.4f}", flush=True)

    def fp(fname): return os.path.join(out_dir, fname)

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
        df.to_csv(fp(fname), index=False)

    for real, pred, name in [(train_y, train_pred, "Training"), (test_y, test_pred, "Testing")]:
        fig_name = f"{name}_comparison_{ts}.png"
        plt.figure()
        plt.scatter(real, pred, alpha=0.3, s=1)
        plt.xlabel("Real"); plt.ylabel("Predicted"); plt.title(f"{name} comparison")
        plt.savefig(fp(fig_name)); plt.close()

    with open(fp(f"config_{ts}.json"), "w") as f:
        json.dump({
            "execution_id": exec_id, "feature_group": feature_group,
            "pca_mode": pca_mode, "model_type": "lightgbm",
            "n_iter": n_iter, "n_folds": N_FOLDS,
            "best_params": search.best_params_,
        }, f)

    with open(fp(f"dict_{ts}.pkl"), "wb") as f:
        pickle.dump(dict_out, f)

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
    with pd.ExcelWriter(fp(xls_name), engine="xlsxwriter") as writer:
        pd.DataFrame({
            "Execution name": [ts],       "Execution ID": [exec_id],
            "Model type": ["lightgbm"],   "PCA": ["No" if pca_mode is None else "Yes"],
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

    summary_line = (
        f"{ts}; Execution ID: {exec_id}; Model type: lightgbm; "
        f"Train time: {train_time}; MAPE train: {mape_train}; MAPE test: {mape_test}; "
        f"MSE train: {mse_train}; MSE test: {mse_test}"
    )
    with open(fp("model_summary.txt"), "w") as f:
        f.write(summary_line + "\n")
    global_summary_lines.append(summary_line)

    print(f"  Saved to {out_dir}/", flush=True)
    return mape_test, mse_test


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading local data...", flush=True)
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw  = pd.read_csv(TEST_PATH)
    print(f"Train: {len(train_raw)} rows | Test: {len(test_raw)} rows", flush=True)

    # Skip executions that already have a Dashboard output
    completed = set()
    for exec_id, _, _, _ in EXECUTIONS:
        folder = os.path.join(OUTPUT_DIR, exec_id.replace("-", "_"))
        if any(f.startswith("Dashboard output") for f in os.listdir(folder) if os.path.isdir(folder)):
            completed.add(exec_id)
    if completed:
        print(f"Skipping already completed: {sorted(completed)}", flush=True)

    results = []
    for exec_id, fg, pca_mode, model_type in EXECUTIONS:
        if exec_id in completed:
            print(f"SKIP {exec_id} — already done", flush=True)
            continue
        try:
            mape, mse = run_execution(exec_id, fg, pca_mode, train_raw, test_raw)
            results.append({"id": exec_id, "mape_test": mape, "mse_test": mse})
        except Exception as e:
            print(f"ERROR in {exec_id}: {e}", flush=True)
            results.append({"id": exec_id, "error": str(e)})

    with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w") as f:
        f.write("\n".join(global_summary_lines))

    print("\n=== FINAL RESULTS ===")
    for r in results:
        if "error" in r:
            print(f"  {r['id']}: ERROR - {r['error']}")
        else:
            print(f"  {r['id']}: MAPE={r['mape_test']:.4f} | MSE={r['mse_test']:.4f}")
