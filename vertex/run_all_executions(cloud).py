"""
run_all_executions.py
---------------------
Script to run all pending model executions sequentially.
Results are saved to 05_outputs_modeling/{execution_id}/default_execution/

BEFORE RUNNING: set the N_ITER values below.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
import math
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import metrics
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server use
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — SET THESE BEFORE RUNNING
# ============================================================
N_ITER = {
    "elasticnet":     20,   # TODO: decide
    "knn":            40,   # TODO: decide
    "decision_tree":  40,   # TODO: decide
    "random_forest":  40,   # TODO: decide
    "svr":            20,   # TODO: decide
    "xgboost":       100,   # TODO: decide
}
N_FOLDS        = 5
N_JOBS         = -1
RANDOM_SEED    = 8
EXECUTION_TO_USE = "20240322_105534"
DATA_PATH      = "./03_outputs_data_preprocessing/output_data_preprocessing_" + EXECUTION_TO_USE
OUTPUT_BASE    = "./05_outputs_modeling"
SUMMARY_FILE   = os.path.join(OUTPUT_BASE, "model_summary.txt")

# ============================================================
# HYPERPARAMETER GRIDS
# ============================================================
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
        "n_estimators":    [100, 200, 300, 500],
        "max_depth":       [3, 5, 7, 9],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.6, 0.8, 1.0],
        "colsample_bytree":[0.6, 0.8, 1.0],
        "min_child_weight":[1, 3, 5],
        "gamma":           [0, 0.1, 0.5],
    },
}

# ============================================================
# FEATURE GROUPS
# ============================================================
FEATURES_GROUP1 = [
    "summary_charlen", "summary_wordlen", "pros_charlen",
    "pros_wordlen", "cons_charlen", "cons_wordlen",
]

FEATURES_GROUP2 = [
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

# Group 3: NLP features only (string lengths + linguistic dims + emolex + nltk + odi vocab + empath)
FEATURES_GROUP3 = [f for f in FEATURES_GROUP2 if f in [
    "summary_charlen","summary_wordlen","pros_charlen","pros_wordlen","cons_charlen","cons_wordlen",
    *[f for f in FEATURES_GROUP2 if any(f.startswith(p) for p in [
        "summary_ld_","pros_ld_","cons_ld_","pros_emolex_","cons_emolex_",
        "pros_nltk_","cons_nltk_","ODI_related_vocab_","pros_empath_","cons_empath_",
    ])]
]]

# Group 4: NLP + ODI/JDI similarities
FEATURES_GROUP4 = [f for f in FEATURES_GROUP2 if f in [
    *FEATURES_GROUP3,
    *[f for f in FEATURES_GROUP2 if any(f.startswith(p) for p in ["pros_sim_","cons_sim_"])]
]]

# Group 5: same features as group 2, no PCA
FEATURES_GROUP5 = FEATURES_GROUP2

FEATURE_SETS = {
    "group1": FEATURES_GROUP1,
    "group2": FEATURES_GROUP2,
    "group3": FEATURES_GROUP3,
    "group4": FEATURES_GROUP4,
    "group5": FEATURES_GROUP5,
}

# ============================================================
# PCA GROUP DEFINITIONS (per feature group)
# Each entry: (pca_name, list_of_features)
# Only features present in the input set are used.
# ============================================================
PCA_FEATURE_GROUPS = {
    "pca0":  ["summary_charlen","summary_wordlen","pros_charlen","pros_wordlen","cons_charlen","cons_wordlen"],
    "pca1":  [f for f in FEATURES_GROUP2 if f.endswith(tuple(
                ["_PDT","_IN","_``","_JJS","_RBS","_NNP","_DT","_EX","_NNS","_CC","_VBD","_POS",
                 "_.",  "_(" ,"_UH","_VBN","_RP" ,"_JJR","_WDT","_FW","_,"  ,"_VBZ","_MD" ,
                 "_PRP","_PRP$","_$","_WP","_VBG","_NN" ,"_CD" ,"_)" ,"_SYM","_RBR","_VBP",
                 "_WRB","_NNPS","_TO","_RB" ,"_VB" ,"_JJ" ,"_FW","_POS","_#","_WP$","_LS","_SYM"]))
             and any(f.startswith(p) for p in ["summary_ld_","pros_ld_","cons_ld_"])],
    "pca2":  [f for f in FEATURES_GROUP2 if "emolex" in f],
    "pca3":  ["pros_nltk_sia","cons_nltk_sia"],
    "pca4":  ["ODI_related_vocab_pros","ODI_related_vocab_cons"],
    "pca5":  [f for f in FEATURES_GROUP2 if "empath" in f],
    "pca6":  [f for f in FEATURES_GROUP2 if "sim_ODI" in f or "sim_JDI" in f],
    "pca7":  ["GDP per capita","Population","Precipitation","AverageTemperatureF",
              "AverageTemperatureAvgHighF","AverageTemperatureAvgLowF","Republican/lean Rep.",
              "No lean","Democrat/lean Dem.","Crime index","Price per square foot ($)",
              "PercentHighSchoolOrHigher","PercentBachelorsOrHigher","State Tax Rate"],
    "pca8":  ["Market Capitalization","NumberofEmployees","Market Cap per Employee","Founded","month","year"],
    "pca9":  ["stock_percchange_month","stock_percchange_year","sp500_percchange_month","sp500_percchange_year"],
    "pca10": ["days from review to t0","days from foundation to t0","days from foundation to review"],
    "pca11": [f for f in FEATURES_GROUP2 if "State name_" in f],
    "pca12": [f for f in FEATURES_GROUP2 if "Sector_" in f],
}

def get_pca_dims(n_features, pca_mode):
    """Calculate number of PCA dimensions based on mode."""
    if pca_mode == "1_dim":
        return 1
    elif pca_mode == "5_dim":
        return min(5, n_features)
    elif pca_mode == "1_per_5":
        return max(1, math.ceil(n_features / 5))
    return None

def build_pca_config(input_features, pca_mode):
    """
    Build list of (pca_name, features_in_group, n_dims) for features
    that appear in input_features.
    """
    pca_configs = []
    for pca_name, group_feats in PCA_FEATURE_GROUPS.items():
        # Only use features present in input_features
        active_feats = [f for f in group_feats if f in input_features]
        if len(active_feats) < 2:
            continue  # PCA requires at least 2 features
        n_dims = get_pca_dims(len(active_feats), pca_mode)
        n_dims = min(n_dims, len(active_feats))  # cap at n_features
        pca_configs.append((pca_name, active_feats, n_dims))
    return pca_configs

# ============================================================
# EXECUTION LIST — pending only
# Format: (execution_id, feature_group, pca_mode, model_type)
# pca_mode: None | "1_dim" | "5_dim" | "1_per_5"
# ============================================================
EXECUTIONS = [
    # Group 1 pending
    ("1-5",  "group1", None,      "svr"),

    # Group 2 pending
    ("2-5",  "group2", "5_dim",   "knn"),
    ("2-6",  "group2", "1_per_5", "knn"),
    ("2-7",  "group2", "1_dim",   "decision_tree"),
    ("2-8",  "group2", "5_dim",   "decision_tree"),
    ("2-9",  "group2", "1_per_5", "decision_tree"),
    ("2-13", "group2", "1_dim",   "svr"),
    ("2-14", "group2", "5_dim",   "svr"),
    ("2-15", "group2", "1_per_5", "svr"),

    # Group 3 — Only NLP features
    ("3-1",  "group3", None,      "elasticnet"),
    ("3-2",  "group3", "1_dim",   "elasticnet"),
    ("3-3",  "group3", "5_dim",   "elasticnet"),
    ("3-4",  "group3", "1_per_5", "elasticnet"),
    ("3-5",  "group3", None,      "knn"),
    ("3-6",  "group3", "1_dim",   "knn"),
    ("3-7",  "group3", "5_dim",   "knn"),
    ("3-8",  "group3", "1_per_5", "knn"),
    ("3-9",  "group3", None,      "decision_tree"),
    ("3-10", "group3", "1_dim",   "decision_tree"),
    ("3-11", "group3", "5_dim",   "decision_tree"),
    ("3-12", "group3", "1_per_5", "decision_tree"),
    ("3-13", "group3", None,      "random_forest"),
    ("3-14", "group3", "1_dim",   "random_forest"),
    ("3-15", "group3", "5_dim",   "random_forest"),
    ("3-16", "group3", "1_per_5", "random_forest"),
    ("3-17", "group3", None,      "svr"),
    ("3-18", "group3", "1_dim",   "svr"),
    ("3-19", "group3", "5_dim",   "svr"),
    ("3-20", "group3", "1_per_5", "svr"),
    ("3-21", "group3", None,      "xgboost"),
    ("3-22", "group3", "1_dim",   "xgboost"),
    ("3-23", "group3", "5_dim",   "xgboost"),
    ("3-24", "group3", "1_per_5", "xgboost"),

    # Group 4 — NLP + ODI/JDI similarities
    ("4-1",  "group4", None,      "elasticnet"),
    ("4-2",  "group4", "1_dim",   "elasticnet"),
    ("4-3",  "group4", "5_dim",   "elasticnet"),
    ("4-4",  "group4", "1_per_5", "elasticnet"),
    ("4-5",  "group4", None,      "knn"),
    ("4-6",  "group4", "1_dim",   "knn"),
    ("4-7",  "group4", "5_dim",   "knn"),
    ("4-8",  "group4", "1_per_5", "knn"),
    ("4-9",  "group4", None,      "decision_tree"),
    ("4-10", "group4", "1_dim",   "decision_tree"),
    ("4-11", "group4", "5_dim",   "decision_tree"),
    ("4-12", "group4", "1_per_5", "decision_tree"),
    ("4-13", "group4", None,      "random_forest"),
    ("4-14", "group4", "1_dim",   "random_forest"),
    ("4-15", "group4", "5_dim",   "random_forest"),
    ("4-16", "group4", "1_per_5", "random_forest"),
    ("4-17", "group4", None,      "svr"),
    ("4-18", "group4", "1_dim",   "svr"),
    ("4-19", "group4", "5_dim",   "svr"),
    ("4-20", "group4", "1_per_5", "svr"),
    ("4-21", "group4", None,      "xgboost"),
    ("4-22", "group4", "1_dim",   "xgboost"),
    ("4-23", "group4", "5_dim",   "xgboost"),
    ("4-24", "group4", "1_per_5", "xgboost"),

    # Group 5 — All features, no PCA
    ("5-1",  "group5", None,      "elasticnet"),
    ("5-2",  "group5", None,      "knn"),
    ("5-3",  "group5", None,      "decision_tree"),
    ("5-4",  "group5", None,      "random_forest"),
    ("5-5",  "group5", None,      "svr"),
    ("5-6",  "group5", None,      "xgboost"),
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def get_model(model_type, random_seed):
    if model_type == "elasticnet":
        return ElasticNet(random_state=random_seed)
    elif model_type == "knn":
        return KNeighborsRegressor()
    elif model_type == "decision_tree":
        return DecisionTreeRegressor(random_state=random_seed)
    elif model_type == "random_forest":
        return RandomForestRegressor(random_state=random_seed)
    elif model_type == "svr":
        return SVR()
    elif model_type == "xgboost":
        return xgb.XGBRegressor(random_state=random_seed)
    raise ValueError(f"Unknown model type: {model_type}")

def create_excel_file(file_name, dataframes_dict):
    with pd.ExcelWriter(f"{file_name}.xlsx", engine='xlsxwriter') as writer:
        for sheet_name, df in dataframes_dict.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def transform_list_of_dicts(input_list):
    output_dict = {}
    for d in input_list:
        for key, value in d.items():
            if key not in output_dict:
                output_dict[key] = []
            output_dict[key].append(value)
    return output_dict

# ============================================================
# MAIN
# ============================================================
def run_execution(exec_id, feature_group, pca_mode, model_type):
    print(f"\n{'='*60}")
    print(f"START: {exec_id} | {feature_group} | PCA={pca_mode} | {model_type}")
    print(f"{'='*60}")

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Output directory
    out_dir = os.path.join(OUTPUT_BASE, exec_id, "default_execution")
    os.makedirs(out_dir, exist_ok=True)

    # Load data
    train_raw = pd.read_csv(os.path.join(DATA_PATH, f"dataframes_Train_{EXECUTION_TO_USE}.csv"))
    test_raw  = pd.read_csv(os.path.join(DATA_PATH, f"dataframes_Test_{EXECUTION_TO_USE}.csv"))

    # Filter to selected features
    input_features = FEATURE_SETS[feature_group]
    # Only keep features that exist in the data
    available = [f for f in input_features if f in train_raw.columns]
    all_cols = available + ["overall_rating"]

    train = train_raw[all_cols]
    test  = test_raw[all_cols]

    train_x = train.drop("overall_rating", axis=1).fillna(0)
    train_y = train["overall_rating"]
    test_x  = test.drop("overall_rating", axis=1).fillna(0)
    test_y  = test["overall_rating"]

    dict_out = {}

    # Apply PCA
    if pca_mode is not None:
        pca_configs = build_pca_config(list(train_x.columns), pca_mode)
        for pca_name, feats, n_dims in pca_configs:
            print(f"  PCA {pca_name}: {len(feats)} features → {n_dims} dims")
            pca = PCA(n_components=n_dims)
            tr_pca = pd.DataFrame(
                pca.fit_transform(train_x[feats]),
                columns=[f"{pca_name}_{i}" for i in range(n_dims)],
                index=train_x.index
            )
            te_pca = pd.DataFrame(
                pca.transform(test_x[feats]),
                columns=[f"{pca_name}_{i}" for i in range(n_dims)],
                index=test_x.index
            )
            train_x = pd.concat([train_x.drop(columns=feats), tr_pca], axis=1)
            test_x  = pd.concat([test_x.drop(columns=feats),  te_pca], axis=1)
            dict_out[pca_name] = pca

    # Train model
    n_iter = N_ITER[model_type]
    params = PARAM_GRIDS[model_type]

    # Cap n_iter to grid size to avoid ValueError
    from itertools import product as iproduct
    grid_size = 1
    for v in params.values():
        grid_size *= len(v)
    n_iter = min(n_iter, grid_size)

    model = get_model(model_type, RANDOM_SEED)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=N_FOLDS,
        random_state=RANDOM_SEED,
        n_jobs=N_JOBS,
        verbose=1,
    )

    print(f"  Training {model_type} ({n_iter} iterations × {N_FOLDS} folds)...")
    starttime = datetime.now()
    search.fit(train_x, train_y)
    endtime = datetime.now()
    train_time = str(endtime - starttime)
    print(f"  Done in {train_time}")

    dict_out['Random search object'] = search

    # Predictions & metrics
    train_pred = search.best_estimator_.predict(train_x)
    test_pred  = search.best_estimator_.predict(test_x)

    mape_train = metrics.mean_absolute_percentage_error(train_y, train_pred)
    mape_test  = metrics.mean_absolute_percentage_error(test_y, test_pred)
    mse_train  = metrics.mean_squared_error(train_y, train_pred)
    mse_test   = metrics.mean_squared_error(test_y, test_pred)

    print(f"  MAPE test: {mape_test:.4f} | MSE test: {mse_test:.4f}")

    # Save CSVs
    train_real_df = pd.merge(train_raw['review_id'], train_y, left_index=True, right_index=True)
    test_real_df  = pd.merge(test_raw['review_id'],  test_y,  left_index=True, right_index=True)
    train_pred_df = pd.merge(train_raw['review_id'], pd.DataFrame(train_pred), left_index=True, right_index=True)
    test_pred_df  = pd.merge(test_raw['review_id'],  pd.DataFrame(test_pred),  left_index=True, right_index=True)

    train_real_df.to_csv(os.path.join(out_dir, f"Train real_{current_datetime}.csv"), index=False)
    test_real_df.to_csv( os.path.join(out_dir, f"Test real_{current_datetime}.csv"),  index=False)
    train_pred_df.to_csv(os.path.join(out_dir, f"Train pred_{current_datetime}.csv"), index=False)
    test_pred_df.to_csv( os.path.join(out_dir, f"Test pred_{current_datetime}.csv"),  index=False)

    pd.DataFrame(search.cv_results_).to_csv(
        os.path.join(out_dir, f"Building summary_{current_datetime}.csv"))

    # Plots
    for split, real, pred, name in [
        ("train", train_y, train_pred, "Training"),
        ("test",  test_y,  test_pred,  "Testing"),
    ]:
        plt.figure()
        plt.scatter(real, pred, alpha=0.3, s=1)
        plt.xlabel('Real'); plt.ylabel('Predicted'); plt.title(f'{name} comparison')
        plt.savefig(os.path.join(out_dir, f"{name}_comparisson_{current_datetime}.png"))
        plt.close()

    # Save config snapshot
    config_snap = {
        "execution_id": exec_id, "feature_group": feature_group,
        "pca_mode": pca_mode, "model_type": model_type,
        "n_iter": n_iter, "n_folds": N_FOLDS,
        "best_params": search.best_params_,
    }
    with open(os.path.join(out_dir, f"config_{current_datetime}.json"), 'w') as f:
        json.dump(config_snap, f)

    # Save pickle
    with open(os.path.join(out_dir, f"dict_{current_datetime}.pkl"), 'wb') as f:
        pickle.dump(dict_out, f)

    # Dashboard Excel
    lst_params = list(search.cv_results_['params'][0].keys())
    deepdive = pd.DataFrame({
        col: search.cv_results_[col]
        for col in ['mean_fit_time','std_fit_time','mean_score_time','std_score_time',
                    'mean_test_score','std_test_score','rank_test_score'] +
                   [f'split{s}_test_score' for s in range(N_FOLDS)]
    })
    deepdive['Execution name'] = current_datetime
    hyperparam_df = pd.DataFrame(transform_list_of_dicts(search.cv_results_['params']))
    for i, hp in enumerate(lst_params):
        deepdive[f"param_{i}_name"] = hp
        deepdive[f"param_{i}"]      = hyperparam_df[hp]

    dashboard = {
        'General': pd.DataFrame({
            'Execution name': [current_datetime], 'Execution ID': [exec_id],
            'Model type': [model_type], 'PCA': ["No" if pca_mode is None else "Yes"],
            'Number of search iterations': [n_iter], 'Number of folds': [N_FOLDS],
            'Train time': [train_time],
            'Metric 1': ["MAPE"], 'Metric 1 train': [mape_train], 'Metric 1 test': [mape_test],
            'Metric 2': ["MSE"],  'Metric 2 train': [mse_train],  'Metric 2 test': [mse_test],
        }),
        'Train predictions': pd.merge(train_real_df, train_pred_df, on='review_id').rename(
            columns={'overall_rating': f'Real_{current_datetime}', 0: f'Predicted_{current_datetime}'}),
        'Test predictions': pd.merge(test_real_df, test_pred_df, on='review_id').rename(
            columns={'overall_rating': f'Real_{current_datetime}', 0: f'Predicted_{current_datetime}'}),
        'Execution deepdive': deepdive,
    }
    create_excel_file(os.path.join(out_dir, f"Dashboard output_{current_datetime}.xlsx"), dashboard)

    # model_summary.txt
    summary_line = (f"\n{current_datetime}; Execution ID: {exec_id}; Model type: {model_type}; "
                    f"Train time: {train_time}; MAPE train: {mape_train}; MAPE test: {mape_test}; "
                    f"MSE train: {mse_train}; MSE test: {mse_test}")
    with open(os.path.join(OUTPUT_BASE, exec_id, "model_summary.txt"), "a") as f:
        f.write(summary_line)
    with open(SUMMARY_FILE, "a") as f:
        f.write(summary_line)

    print(f"  Saved to {out_dir}")
    return current_datetime, mape_train, mape_test, mse_train, mse_test


if __name__ == "__main__":
    print(f"Starting {len(EXECUTIONS)} executions...")
    print(f"N_ITER: {N_ITER}")

    results = []
    for exec_id, feature_group, pca_mode, model_type in EXECUTIONS:
        # Skip if already done (folder exists with a Dashboard file)
        out_dir = os.path.join(OUTPUT_BASE, exec_id, "default_execution")
        if os.path.exists(out_dir) and any("Dashboard" in f for f in os.listdir(out_dir)):
            print(f"SKIP {exec_id} — already completed")
            continue

        try:
            ts, mt, mtest, ms, mstest = run_execution(exec_id, feature_group, pca_mode, model_type)
            results.append({"id": exec_id, "model": model_type, "mape_test": mtest, "mse_test": mstest})
        except Exception as e:
            print(f"ERROR in {exec_id}: {e}")
            results.append({"id": exec_id, "model": model_type, "error": str(e)})

    print("\n\n=== FINAL RESULTS ===")
    for r in results:
        if "error" in r:
            print(f"  {r['id']} ({r['model']}): ERROR — {r['error']}")
        else:
            print(f"  {r['id']} ({r['model']}): MAPE test={r['mape_test']:.4f} | MSE test={r['mse_test']:.4f}")
