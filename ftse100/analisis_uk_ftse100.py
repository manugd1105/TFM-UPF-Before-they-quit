"""
Análisis de viabilidad: datos UK / FTSE 100 para el TFM
==========================================================
Lee all_reviews.csv y glassdoor_reviews.csv de la carpeta extradatos,
filtra por mercado UK / FTSE 100, comprueba duplicados y consistencia
de columnas, y genera tablas + visualizaciones.
"""

import os, re, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from collections import Counter

# ─── 0. Configuración ─────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "analisis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#f7f7f7",
    "axes.grid":        True,
    "grid.color":       "white",
    "grid.linewidth":   0.8,
    "font.size":        10,
})

# ─── 1. FTSE 100 — lista completa con variantes de nombre en Glassdoor ────────
# Cada entrada: (ticker_o_id, nombre_oficial, [slugs posibles en glassdoor])
FTSE100 = [
    # Financials
    ("AV",   "Aviva",                       ["Aviva"]),
    ("BARC", "Barclays",                    ["Barclays"]),
    ("BATS", "British-American-Tobacco",    ["British-American-Tobacco", "BAT"]),
    ("BNZL", "Bunzl",                       ["Bunzl"]),
    ("BT.A", "BT-Group",                    ["BT", "BT-Group"]),
    ("CPG",  "Compass-Group",               ["Compass-Group", "Compass"]),
    ("DGE",  "Diageo",                      ["Diageo"]),
    ("EZJ",  "easyJet",                     ["easyJet", "EasyJet"]),
    ("EXPN", "Experian",                    ["Experian"]),
    ("FLTR", "Flutter-Entertainment",       ["Flutter-Entertainment", "Flutter", "Betfair"]),
    ("GLEN", "Glencore",                    ["Glencore"]),
    ("GSK",  "GSK",                         ["GSK", "GlaxoSmithKline", "Glaxo-SmithKline"]),
    ("HLN",  "Haleon",                      ["Haleon"]),
    ("HLMA", "Halma",                       ["Halma"]),
    ("HSBA", "HSBC",                        ["HSBC"]),
    ("IAG",  "International-Airlines-Group",["International-Airlines-Group", "British-Airways", "IAG", "Iberia"]),
    ("IHG",  "InterContinental-Hotels-Group",["InterContinental-Hotels-Group","IHG", "Holiday-Inn"]),
    ("IMB",  "Imperial-Brands",             ["Imperial-Brands", "Imperial-Tobacco"]),
    ("INF",  "Informa",                     ["Informa"]),
    ("ITV",  "ITV",                         ["ITV"]),
    ("JD",   "JD-Sports-Fashion",           ["JD-Sports", "JD-Sports-Fashion"]),
    ("LAND", "Land-Securities",             ["Land-Securities", "Landsec"]),
    ("LGEN", "Legal-and-General",           ["Legal-General", "Legal-and-General"]),
    ("LLOY", "Lloyds-Banking-Group",        ["Lloyds-Banking-Group", "Lloyds-Bank", "Lloyds"]),
    ("LSEG", "London-Stock-Exchange-Group", ["London-Stock-Exchange-Group", "LSEG", "London-Stock-Exchange"]),
    ("MKS",  "Marks-and-Spencer",           ["Marks-Spencer", "Marks-and-Spencer"]),
    ("MNG",  "M-and-G",                     ["M-and-G", "MandG", "Prudential-MandG"]),
    ("MNDI", "Mondi",                       ["Mondi"]),
    ("NG",   "National-Grid",               ["National-Grid"]),
    ("NXT",  "Next",                        ["Next"]),
    ("OCDO", "Ocado",                       ["Ocado", "Ocado-Group"]),
    ("PHX",  "Phoenix-Group",               ["Phoenix-Group"]),
    ("PRU",  "Prudential",                  ["Prudential"]),
    ("PSN",  "Persimmon",                   ["Persimmon"]),
    ("PSON", "Pearson",                     ["Pearson"]),
    ("RB",   "Reckitt",                     ["Reckitt", "Reckitt-Benckiser"]),
    ("RDSA", "Shell",                       ["Shell", "Royal-Dutch-Shell"]),
    ("REL",  "RELX",                        ["RELX", "Reed-Elsevier"]),
    ("RIO",  "Rio-Tinto",                   ["Rio-Tinto"]),
    ("RKT",  "Reckitt",                     ["Reckitt"]),
    ("RMV",  "Rightmove",                   ["Rightmove"]),
    ("RNO",  "Rentokil-Initial",            ["Rentokil-Initial", "Rentokil"]),
    ("RR",   "Rolls-Royce",                 ["Rolls-Royce", "Rolls-Royce-Holdings"]),
    ("RS2K", "RS-Group",                    ["RS-Group", "RS-Components"]),
    ("SBRY", "Sainsburys",                  ["Sainsburys", "Sainsbury"]),
    ("SGRO", "Segro",                       ["Segro", "SEGRO"]),
    ("SJP",  "St-James-Place",              ["St-James-Place", "St-James-s-Place"]),
    ("SKG",  "Smurfit-Kappa",               ["Smurfit-Kappa"]),
    ("SMDS", "DS-Smith",                    ["DS-Smith"]),
    ("SMT",  "Scottish-Mortgage",           ["Scottish-Mortgage-Investment-Trust"]),
    ("SN",   "Smith-and-Nephew",            ["Smith-Nephew", "Smith-and-Nephew"]),
    ("SPX",  "Spirax-Sarco-Engineering",    ["Spirax-Sarco-Engineering", "Spirax-Sarco"]),
    ("SSE",  "SSE",                         ["SSE"]),
    ("STAN", "Standard-Chartered",          ["Standard-Chartered"]),
    ("SVT",  "Severn-Trent",                ["Severn-Trent"]),
    ("TSCO", "Tesco",                       ["Tesco"]),
    ("TW",   "Taylor-Wimpey",               ["Taylor-Wimpey"]),
    ("UL",   "Unilever",                    ["Unilever"]),
    ("UU",   "United-Utilities",            ["United-Utilities"]),
    ("VOD",  "Vodafone",                    ["Vodafone"]),
    ("WPP",  "WPP",                         ["WPP"]),
    ("WTB",  "Whitbread",                   ["Whitbread"]),
    # Mining / Resources
    ("AAL",  "Anglo-American",              ["Anglo-American"]),
    ("ANTO", "Antofagasta",                 ["Antofagasta"]),
    ("BP",   "BP",                          ["BP"]),
    ("BHP",  "BHP",                         ["BHP", "BHP-Group"]),
    ("FRES", "Fresnillo",                   ["Fresnillo"]),
    # Tech / Others
    ("AUTO", "Auto-Trader",                 ["Auto-Trader", "Autotrader"]),
    ("BA",   "BAE-Systems",                 ["BAE-Systems"]),
    ("BAES", "BAE-Systems",                 ["BAE-Systems"]),
    ("CNA",  "Centrica",                    ["Centrica"]),
    ("COB",  "Cobham",                      ["Cobham"]),
    ("CRDA", "Croda-International",         ["Croda-International", "Croda"]),
    ("DCC",  "DCC",                         ["DCC"]),
    ("ENT",  "Entain",                      ["Entain", "GVC-Holdings", "Ladbrokes"]),
    ("FERG", "Ferguson",                    ["Ferguson"]),
    ("ITRK", "Intertek",                    ["Intertek"]),
    ("KGF",  "Kingfisher",                  ["Kingfisher", "B-Q", "Screwfix"]),
    ("MRO",  "Melrose-Industries",          ["Melrose-Industries", "Melrose"]),
    ("NWG",  "NatWest-Group",               ["NatWest-Group", "NatWest", "Royal-Bank-of-Scotland"]),
    ("PSH",  "Pershing-Square",             ["Pershing-Square"]),
    ("PSHP", "Pershing-Square",             ["Pershing-Square"]),
    ("QQ",   "Qinetiq",                     ["Qinetiq", "QinetiQ"]),
    ("SAGE", "Sage-Group",                  ["Sage-Group", "Sage"]),
    ("SDR",  "Schroders",                   ["Schroders"]),
    ("SGLN", "Segro",                       ["Segro"]),
    ("SHI",  "SThree",                      ["SThree"]),
    ("SXS",  "Spirent",                     ["Spirent"]),
    ("WEIR", "Weir-Group",                  ["Weir-Group", "Weir"]),
    ("WMH",  "William-Hill",                ["William-Hill"]),
    ("AZN",  "AstraZeneca",                 ["AstraZeneca"]),
    ("BKG",  "Berkeley-Group",              ["Berkeley-Group", "Berkeley"]),
    ("BLND", "British-Land",                ["British-Land"]),
    ("BOO",  "Boohoo",                      ["Boohoo"]),
    ("BTG",  "BTG",                         ["BTG"]),
    ("DRX",  "Drax-Group",                  ["Drax-Group", "Drax"]),
    ("HWDN", "Howden-Joinery",              ["Howden-Joinery", "Howdens"]),
    ("IMI",  "IMI",                         ["IMI"]),
    ("MNDI", "Mondi",                       ["Mondi"]),
    ("MONI", "Monitise",                    ["Monitise"]),
    ("NMC",  "NMC-Health",                  ["NMC-Health"]),
    ("SMWH", "WH-Smith",                    ["WH-Smith", "WHSmith"]),
    ("STJ",  "St-James-Place",              ["St-James-Place"]),
    ("WDI",  "Wirecard",                    ["Wirecard"]),
    ("3IN",  "3i-Group",                    ["3i-Group", "3i"]),
    ("ADM",  "Admiral-Group",               ["Admiral-Group", "Admiral"]),
    ("BDEV", "Barratt-Developments",        ["Barratt-Developments", "Barratt"]),
    ("BREI", "Brewin-Dolphin",              ["Brewin-Dolphin"]),
    ("FLTRF","Flutter-Entertainment",       ["Flutter-Entertainment"]),
    ("GRG",  "Greggs",                      ["Greggs"]),
    ("HL",   "Hargreaves-Lansdown",         ["Hargreaves-Lansdown"]),
    ("HSX",  "Hiscox",                      ["Hiscox"]),
    ("JUP",  "Jupiter-Fund-Management",     ["Jupiter-Fund-Management", "Jupiter"]),
    ("MNDI2","Mondi",                       ["Mondi"]),
    ("MPI",  "MP-Materials",                ["MP-Materials"]),
    ("RTO",  "Rentokil",                    ["Rentokil"]),
    ("ULVR", "Unilever",                    ["Unilever"]),
    ("DARK", "Darktrace",                   ["Darktrace"]),
    ("OCDO2","Ocado",                       ["Ocado"]),
]

# Deduplicate slugs
all_ftse_slugs = set()
for _, _, slugs in FTSE100:
    all_ftse_slugs.update([s.lower() for s in slugs])

# Palabras clave para filtrado UK en location
UK_KEYWORDS = [
    "england", "scotland", "wales", "northern ireland",
    "united kingdom", " uk,", ", uk", "(uk)", "london",
    "manchester", "birmingham", "edinburgh", "glasgow",
    "leeds", "bristol", "sheffield", "cambridge", "oxford",
    "liverpool", "coventry", "nottingham", "newcastle"
]

# ─── 2. Carga de glassdoor_reviews.csv ────────────────────────────────────────
print("\n" + "="*60)
print("Cargando glassdoor_reviews.csv …")
t0 = time.time()

gd = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "glassdoor_reviews.csv"),
    low_memory=False,
)
print(f"  Filas totales: {len(gd):,}  |  {time.time()-t0:.1f}s")
print(f"  Columnas: {list(gd.columns)}")

# 2a. Filtro UK por location
loc_lower = gd["location"].fillna("").str.lower()
gd_uk_loc  = gd[loc_lower.apply(lambda x: any(k in x for k in UK_KEYWORDS))].copy()
print(f"\n  Reviews con location UK: {len(gd_uk_loc):,}")

# 2b. Filtro FTSE 100 por firm slug
firm_lower = gd["firm"].fillna("").str.lower()
gd_ftse    = gd[firm_lower.apply(lambda x: any(slug in x for slug in all_ftse_slugs))].copy()
print(f"  Reviews de firmas FTSE 100: {len(gd_ftse):,}")

# 2c. Unión (UK por location OR firma FTSE)
gd_uk = pd.concat([gd_uk_loc, gd_ftse]).drop_duplicates().copy()
print(f"  Reviews UK (union): {len(gd_uk):,}")

# Qué firmas FTSE 100 encontramos en glassdoor_reviews
gd_ftse_firms_found = gd_ftse["firm"].str.lower().unique()
ftse_in_gd = sorted([s for s in all_ftse_slugs if any(s in f for f in gd_ftse_firms_found)])
print(f"\n  Slugs FTSE 100 encontrados en glassdoor_reviews: {len(ftse_in_gd)}")

# ─── 3. Carga de all_reviews.csv (chunked — 3.8 GB) ──────────────────────────
print("\n" + "="*60)
print("Cargando all_reviews.csv en chunks …")
t0 = time.time()

CHUNK = 200_000
ar_uk_chunks  = []
ar_all_count  = 0

for chunk in pd.read_csv(
        os.path.join(os.path.dirname(__file__), "all_reviews.csv"),
        chunksize=CHUNK, low_memory=False):

    ar_all_count += len(chunk)

    # Extraer nombre de firma del firm_link
    # Formato: "Reviews/Company-Name-Reviews-E12345.htm" o URL completa
    chunk["firm_slug"] = (
        chunk["firm_link"].fillna("")
        .str.extract(r"Reviews/([^/]+)-Reviews-E\d+\.htm")[0]
        .str.lower()
    )

    # Filtro 1: UK por location/status
    status_lower = chunk["status"].fillna("").str.lower()
    uk_status    = status_lower.apply(lambda x: any(k in x for k in UK_KEYWORDS))

    # Filtro 2: firma FTSE 100
    slug_lower   = chunk["firm_slug"].fillna("")
    ftse_match   = slug_lower.apply(
        lambda x: any(slug in x for slug in all_ftse_slugs) if x else False
    )

    filtered = chunk[uk_status | ftse_match].copy()
    if len(filtered):
        ar_uk_chunks.append(filtered)

    if ar_all_count % 2_000_000 == 0:
        print(f"    … {ar_all_count:,} filas procesadas ({time.time()-t0:.0f}s)")

ar = pd.concat(ar_uk_chunks, ignore_index=True) if ar_uk_chunks else pd.DataFrame()
print(f"\n  Total filas all_reviews: {ar_all_count:,}  |  {time.time()-t0:.1f}s")
print(f"  Reviews UK/FTSE en all_reviews: {len(ar):,}")

# ─── 4. Estadísticas básicas por dataset ─────────────────────────────────────
print("\n" + "="*60)
print("RESUMEN GENERAL")
print("="*60)

def basic_stats(df, name, date_col, rating_col):
    out = {
        "Dataset":        name,
        "Reviews totales":f"{len(df):,}",
        "Columnas":       len(df.columns),
        "Fecha mín":      df[date_col].min() if date_col in df else "—",
        "Fecha máx":      df[date_col].max() if date_col in df else "—",
        "Rating medio":   f"{df[rating_col].dropna().astype(float).mean():.2f}" if rating_col in df else "—",
        "% missing rating": f"{df[rating_col].isna().mean()*100:.1f}%" if rating_col in df else "—",
    }
    return out

stats_gd = basic_stats(gd_uk, "glassdoor_reviews (UK)", "date_review", "overall_rating")
stats_ar = basic_stats(ar,    "all_reviews (UK/FTSE)",  "date",         "rating")

stats_df = pd.DataFrame([stats_gd, stats_ar]).set_index("Dataset")
print(stats_df.to_string())

# ─── 5. Mapeo de columnas entre ambos datasets ────────────────────────────────
# Columnas equivalentes para el TFM (según el formato de 01_outputs_p1)
COL_MAP = {
    "glassdoor_reviews":  ["firm","date_review","job_title","current","location",
                           "overall_rating","work_life_balance","culture_values",
                           "diversity_inclusion","career_opp","comp_benefits",
                           "senior_mgmt","recommend","ceo_approv","outlook",
                           "headline","pros","cons"],
    "all_reviews":        ["firm_link","date","job","status",
                           "rating",
                           "Work/Life Balance","Culture & Values",
                           "Diversity & Inclusion","Career Opportunities",
                           "Compensation and Benefits","Senior Management",
                           "Recommend","CEO Approval","Business Outlook",
                           "title","pros","cons"],
    "TFM_target":         ["company","date","job_title","current","author_location",
                           "overall_rating","wl_balance","culture_values",
                           "diversity_inclusion","career_opportunities",
                           "compensation_benefits","senior_management",
                           "recommend","ceo_approv","outlook",
                           "summary","pros","cons"],
}

alignment_rows = []
for gd_col, ar_col, tfm_col in zip(COL_MAP["glassdoor_reviews"],
                                    COL_MAP["all_reviews"],
                                    COL_MAP["TFM_target"]):
    gd_miss  = f"{gd_uk[gd_col].isna().mean()*100:.1f}%" if gd_col in gd_uk.columns else "AUSENTE"
    ar_miss  = f"{ar[ar_col].isna().mean()*100:.1f}%"    if (len(ar) and ar_col in ar.columns) else "AUSENTE"
    alignment_rows.append({
        "glassdoor_reviews": gd_col,
        "all_reviews":       ar_col,
        "TFM (target)":      tfm_col,
        "GD % missing":      gd_miss,
        "AR % missing":      ar_miss,
    })

align_df = pd.DataFrame(alignment_rows)
print("\n\nMAPA DE COLUMNAS y % MISSING")
print(align_df.to_string(index=False))

# ─── 6. Detección de duplicados inter-dataset ─────────────────────────────────
print("\n\n" + "="*60)
print("DETECCIÓN DE DUPLICADOS")
print("="*60)

if len(ar) > 0:
    # Normalizar texto para comparar
    def norm(s):
        return str(s).lower().strip() if pd.notna(s) else ""

    gd_key = (gd_uk["pros"].fillna("").str.lower().str.strip() + "|" +
               gd_uk["cons"].fillna("").str.lower().str.strip())
    ar_key = (ar["pros"].fillna("").str.lower().str.strip() + "|" +
               ar["cons"].fillna("").str.lower().str.strip())

    gd_set = set(gd_key[gd_key.str.len() > 20])
    ar_set = set(ar_key[ar_key.str.len() > 20])
    overlap = gd_set & ar_set

    print(f"  Keys únicas glassdoor_reviews (UK): {len(gd_set):,}")
    print(f"  Keys únicas all_reviews (UK/FTSE):  {len(ar_set):,}")
    print(f"  Solapamiento exacto (pros+cons):     {len(overlap):,}")
    dup_rate = len(overlap) / max(len(gd_set), 1) * 100
    print(f"  Tasa de duplicados:                  {dup_rate:.2f}%")
else:
    print("  all_reviews sin datos UK — sin comparación.")
    overlap = set()

# ─── 7. FTSE 100 — presencia en ambos datasets ───────────────────────────────
print("\n\n" + "="*60)
print("PRESENCIA DE EMPRESAS FTSE 100")
print("="*60)

ftse_presence = []
for ticker, name, slugs in FTSE100:
    slug_set = [s.lower() for s in slugs]

    gd_rows  = gd[gd["firm"].str.lower().apply(
        lambda x: any(s in x for s in slug_set))].shape[0]
    ar_rows  = (ar[ar["firm_slug"].fillna("").apply(
        lambda x: any(s in x for s in slug_set))].shape[0]
                if len(ar) else 0)

    if gd_rows > 0 or ar_rows > 0:
        ftse_presence.append({
            "Empresa":             name,
            "Slugs buscados":      ", ".join(slugs[:2]),
            "GD reviews":          gd_rows,
            "AR reviews":          ar_rows,
            "Total":               gd_rows + ar_rows,
        })

ftse_df = (pd.DataFrame(ftse_presence)
           .drop_duplicates(subset=["Empresa"])
           .sort_values("Total", ascending=False)
           .reset_index(drop=True))
print(ftse_df.to_string(index=False))
print(f"\n  Empresas FTSE 100 con datos en GD: {(ftse_df['GD reviews']>0).sum()}")
print(f"  Empresas FTSE 100 con datos en AR: {(ftse_df['AR reviews']>0).sum()}")

# ─── 8. Análisis temporal ──────────────────────────────────────────────────────
gd_uk["year"] = pd.to_datetime(gd_uk["date_review"], errors="coerce").dt.year

gd_year = gd_uk["year"].dropna().astype(int)
gd_year_counts = gd_year.value_counts().sort_index()

if len(ar):
    ar["year"] = pd.to_datetime(ar["date"], errors="coerce").dt.year
    ar_year_counts = ar["year"].dropna().astype(int).value_counts().sort_index()
else:
    ar_year_counts = pd.Series(dtype=int)

# ─── 9. VISUALIZACIONES ───────────────────────────────────────────────────────
print("\n\nGenerando visualizaciones …")

# ── Fig 1: Dashboard principal ────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22))
fig.suptitle("Análisis de viabilidad: datos UK / FTSE 100\n(glassdoor_reviews.csv  vs  all_reviews.csv)",
             fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# --- Panel 1: Reviews por año (GD UK) ---
ax1 = fig.add_subplot(gs[0, :2])
if len(gd_year_counts):
    gd_year_counts.plot(kind="bar", ax=ax1, color="#2196F3", alpha=0.85)
ax1.set_title("glassdoor_reviews — Reviews UK por año")
ax1.set_xlabel("Año")
ax1.set_ylabel("Número de reviews")
ax1.tick_params(axis="x", rotation=45)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# --- Panel 2: Reviews por año (AR UK) ---
ax2 = fig.add_subplot(gs[0, 2])
if len(ar_year_counts):
    ar_year_counts.plot(kind="bar", ax=ax2, color="#FF5722", alpha=0.85)
    ax2.set_title("all_reviews — Reviews UK/FTSE por año")
else:
    ax2.text(0.5, 0.5, "Sin datos UK\nen all_reviews", ha="center", va="center",
             fontsize=12, color="gray")
    ax2.set_title("all_reviews — Reviews UK/FTSE por año")
ax2.set_xlabel("Año")
ax2.set_ylabel("Número de reviews")
ax2.tick_params(axis="x", rotation=45)

# --- Panel 3: Distribución rating GD UK ---
ax3 = fig.add_subplot(gs[1, 0])
gd_uk["overall_rating"].dropna().astype(float).hist(
    bins=10, ax=ax3, color="#2196F3", alpha=0.8, edgecolor="white")
ax3.set_title("GD UK — Distribución overall_rating")
ax3.set_xlabel("Rating")
ax3.set_ylabel("Frecuencia")

# --- Panel 4: Distribución rating AR ---
ax4 = fig.add_subplot(gs[1, 1])
if len(ar) and "rating" in ar.columns:
    ar["rating"].dropna().astype(float).hist(
        bins=10, ax=ax4, color="#FF5722", alpha=0.8, edgecolor="white")
ax4.set_title("AR UK/FTSE — Distribución rating")
ax4.set_xlabel("Rating")
ax4.set_ylabel("Frecuencia")

# --- Panel 5: Top 20 empresas FTSE 100 por nº reviews ---
ax5 = fig.add_subplot(gs[1, 2])
if len(ftse_df):
    top20 = ftse_df.head(20)
    ax5.barh(top20["Empresa"][::-1], top20["GD reviews"][::-1],
             color="#2196F3", alpha=0.8, label="GD")
    ax5.barh(top20["Empresa"][::-1], top20["AR reviews"][::-1],
             left=top20["GD reviews"][::-1].values,
             color="#FF5722", alpha=0.8, label="AR")
    ax5.set_title("Top 20 FTSE 100 por reviews")
    ax5.set_xlabel("Reviews")
    ax5.legend(fontsize=8)
    ax5.tick_params(axis="y", labelsize=7)

# --- Panel 6: % missing por columna clave (GD UK) ---
ax6 = fig.add_subplot(gs[2, :2])
key_cols_gd = ["overall_rating","work_life_balance","culture_values",
               "diversity_inclusion","career_opp","comp_benefits",
               "senior_mgmt","recommend","ceo_approv","outlook","pros","cons"]
miss_gd = {c: gd_uk[c].isna().mean()*100 for c in key_cols_gd if c in gd_uk.columns}
miss_s   = pd.Series(miss_gd).sort_values(ascending=True)
miss_s.plot(kind="barh", ax=ax6, color="#2196F3", alpha=0.8)
ax6.set_title("glassdoor_reviews UK — % valores nulos por columna clave")
ax6.set_xlabel("% nulos")
ax6.axvline(20, color="red", linestyle="--", linewidth=0.8, label="20%")
ax6.legend(fontsize=8)

# --- Panel 7: % missing AR ---
ax7 = fig.add_subplot(gs[2, 2])
if len(ar):
    key_cols_ar = ["rating","Work/Life Balance","Culture & Values",
                   "Diversity & Inclusion","Career Opportunities",
                   "Compensation and Benefits","Senior Management",
                   "Recommend","CEO Approval","Business Outlook","pros","cons"]
    miss_ar = {c: ar[c].isna().mean()*100 for c in key_cols_ar if c in ar.columns}
    miss_ar_s = pd.Series(miss_ar).sort_values(ascending=True)
    miss_ar_s.plot(kind="barh", ax=ax7, color="#FF5722", alpha=0.8)
    ax7.set_title("all_reviews UK/FTSE — % nulos clave")
    ax7.set_xlabel("% nulos")
    ax7.axvline(20, color="red", linestyle="--", linewidth=0.8)
else:
    ax7.text(0.5, 0.5, "Sin datos", ha="center", va="center", color="gray")
    ax7.set_title("all_reviews UK/FTSE — % nulos clave")

# --- Panel 8: location breakdown GD UK ---
ax8 = fig.add_subplot(gs[3, 0])
loc_counts = (gd_uk["location"].fillna("Unknown")
              .str.extract(r",\s*([^,]+)$")[0]
              .str.strip()
              .value_counts()
              .head(15))
loc_counts[::-1].plot(kind="barh", ax=ax8, color="#4CAF50", alpha=0.8)
ax8.set_title("GD UK — Top 15 regiones (location)")
ax8.set_xlabel("Reviews")
ax8.tick_params(axis="y", labelsize=7)

# --- Panel 9: Comparativa solapamiento ---
ax9 = fig.add_subplot(gs[3, 1])
sizes  = [len(gd_set) - len(overlap), len(ar_set) - len(overlap), len(overlap)]
labels = ["Solo GD", "Solo AR", f"Solapamiento\n({len(overlap):,})"]
colors = ["#2196F3", "#FF5722", "#9C27B0"]
if sum(sizes) > 0:
    ax9.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 8})
else:
    ax9.text(0.5, 0.5, "Sin solapamiento\ndetectado", ha="center", va="center")
ax9.set_title("Duplicados entre datasets\n(pros+cons exactos)")

# --- Panel 10: Tabla resumen viabilidad ---
ax10 = fig.add_subplot(gs[3, 2])
ax10.axis("off")
viab_data = [
    ["Métrica", "glassdoor_reviews", "all_reviews"],
    ["Reviews UK/FTSE",    f"{len(gd_uk):,}",          f"{len(ar):,}"],
    ["Empresas FTSE encontradas",
     str((ftse_df['GD reviews']>0).sum()),
     str((ftse_df['AR reviews']>0).sum())],
    ["Rating medio UK",
     f"{gd_uk['overall_rating'].dropna().astype(float).mean():.2f}" if len(gd_uk) else "—",
     f"{ar['rating'].dropna().astype(float).mean():.2f}" if len(ar) else "—"],
    ["Solapamiento exacto", f"{len(overlap):,}", "←"],
    ["Columnas compatibles", "18", "19"],
]
tbl = ax10.table(cellText=viab_data[1:], colLabels=viab_data[0],
                  loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.5)
ax10.set_title("Resumen de viabilidad", pad=12)

fig.savefig(os.path.join(OUTPUT_DIR, "fig1_dashboard.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print("  → fig1_dashboard.png")

# ── Fig 2: Mapa de alineación de columnas ────────────────────────────────────
fig2, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")
ax.set_title("Mapa de columnas: glassdoor_reviews ↔ all_reviews ↔ formato TFM",
             fontsize=13, fontweight="bold", pad=15)

col_data   = align_df.values.tolist()
col_labels = align_df.columns.tolist()

tbl2 = ax.table(cellText=col_data, colLabels=col_labels,
                 loc="center", cellLoc="center")
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(8.5)
tbl2.scale(1, 1.6)

# Colorear celdas con alta falta de datos
for (r, c), cell in tbl2.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold")
    elif c in [3, 4]:  # columnas de % missing
        text = cell.get_text().get_text()
        if text == "AUSENTE":
            cell.set_facecolor("#FFCDD2")
        elif text.endswith("%"):
            try:
                val = float(text.replace("%", ""))
                if val > 50:
                    cell.set_facecolor("#FFCDD2")
                elif val > 20:
                    cell.set_facecolor("#FFF9C4")
                else:
                    cell.set_facecolor("#E8F5E9")
            except ValueError:
                pass
    elif r % 2 == 0:
        cell.set_facecolor("#F5F5F5")

fig2.savefig(os.path.join(OUTPUT_DIR, "fig2_column_map.png"), dpi=130, bbox_inches="tight")
plt.close(fig2)
print("  → fig2_column_map.png")

# ── Fig 3: FTSE 100 presencia detallada ──────────────────────────────────────
if len(ftse_df) > 0:
    fig3, axes = plt.subplots(1, 2, figsize=(18, max(6, len(ftse_df)*0.32)))
    fig3.suptitle("Presencia de empresas FTSE 100 en los datasets",
                  fontsize=13, fontweight="bold")

    # Bar chart reviews GD
    ax_l = axes[0]
    top_n = ftse_df[ftse_df["GD reviews"] > 0].sort_values("GD reviews")
    ax_l.barh(top_n["Empresa"], top_n["GD reviews"], color="#2196F3", alpha=0.85)
    ax_l.set_title("glassdoor_reviews — reviews por empresa FTSE")
    ax_l.set_xlabel("Reviews")
    ax_l.tick_params(axis="y", labelsize=8)
    ax_l.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Bar chart reviews AR
    ax_r = axes[1]
    top_n2 = ftse_df[ftse_df["AR reviews"] > 0].sort_values("AR reviews")
    if len(top_n2):
        ax_r.barh(top_n2["Empresa"], top_n2["AR reviews"], color="#FF5722", alpha=0.85)
    else:
        ax_r.text(0.5, 0.5, "Sin empresas FTSE\nen all_reviews", ha="center",
                  va="center", fontsize=12, color="gray", transform=ax_r.transAxes)
    ax_r.set_title("all_reviews — reviews por empresa FTSE")
    ax_r.set_xlabel("Reviews")
    ax_r.tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    fig3.savefig(os.path.join(OUTPUT_DIR, "fig3_ftse_presence.png"), dpi=130, bbox_inches="tight")
    plt.close(fig3)
    print("  → fig3_ftse_presence.png")

# ── Fig 4: Rating por subrating (heatmap correlación) ────────────────────────
sub_cols_gd = ["overall_rating","work_life_balance","culture_values",
               "diversity_inclusion","career_opp","comp_benefits","senior_mgmt"]
sub_cols_gd = [c for c in sub_cols_gd if c in gd_uk.columns]

if len(sub_cols_gd) >= 2:
    corr_data = gd_uk[sub_cols_gd].dropna().astype(float)
    if len(corr_data) > 10:
        fig4, ax4c = plt.subplots(figsize=(8, 6))
        corr = corr_data.corr()
        im = ax4c.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax4c)
        ax4c.set_xticks(range(len(sub_cols_gd)))
        ax4c.set_yticks(range(len(sub_cols_gd)))
        ax4c.set_xticklabels([c.replace("_"," ") for c in sub_cols_gd],
                              rotation=45, ha="right", fontsize=8)
        ax4c.set_yticklabels([c.replace("_"," ") for c in sub_cols_gd], fontsize=8)
        for i in range(len(sub_cols_gd)):
            for j in range(len(sub_cols_gd)):
                ax4c.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center",
                           va="center", fontsize=7,
                           color="white" if abs(corr.iloc[i,j]) > 0.5 else "black")
        ax4c.set_title("GD UK — Correlación entre subratings (glassdoor_reviews)")
        fig4.tight_layout()
        fig4.savefig(os.path.join(OUTPUT_DIR, "fig4_rating_corr.png"),
                     dpi=130, bbox_inches="tight")
        plt.close(fig4)
        print("  → fig4_rating_corr.png")

# ─── 10. Exportar tablas CSV ──────────────────────────────────────────────────
align_df.to_csv(os.path.join(OUTPUT_DIR, "tabla_column_map.csv"), index=False)
ftse_df.to_csv(os.path.join(OUTPUT_DIR, "tabla_ftse_presence.csv"), index=False)
stats_df.to_csv(os.path.join(OUTPUT_DIR, "tabla_resumen.csv"))

# ─── 11. Conclusiones de viabilidad ──────────────────────────────────────────
print("\n" + "="*60)
print("CONCLUSIONES DE VIABILIDAD")
print("="*60)

total_new = len(gd_uk) + len(ar)
print(f"""
1. VOLUMEN DE DATOS
   ─────────────────
   glassdoor_reviews (UK/FTSE): {len(gd_uk):,} reviews
   all_reviews (UK/FTSE):       {len(ar):,} reviews
   Total potencial nuevo:        {total_new:,} reviews

2. COBERTURA FTSE 100
   ────────────────────
   Empresas FTSE 100 presentes en glassdoor_reviews: {(ftse_df['GD reviews']>0).sum() if len(ftse_df) else 0}
   Empresas FTSE 100 presentes en all_reviews:        {(ftse_df['AR reviews']>0).sum() if len(ftse_df) else 0}

3. DUPLICADOS INTER-DATASET
   ──────────────────────────
   Solapamiento exacto (pros+cons): {len(overlap):,} ({dup_rate:.2f}%)
   → {'BAJO riesgo de duplicación' if dup_rate < 5 else 'ATENCIÓN: solapamiento significativo'}

4. COMPATIBILIDAD DE COLUMNAS
   ────────────────────────────
   Las columnas clave (rating, subratings, pros, cons) son equivalentes
   con renombrado simple. Ver fig2_column_map.png para el mapeo completo.
   Columna 'location' en glassdoor_reviews permite filtrar UK directamente.
   En all_reviews el 'status' no incluye location geográfica de forma fiable.

5. RECOMENDACIÓN
   ───────────────
   {'✓ VIABLE: volumen suficiente y columnas compatibles.' if total_new > 5000 else '⚠ LIMITADO: volumen bajo, valorar coste/beneficio.'}
   Priorizar glassdoor_reviews.csv (mejor cobertura UK y columna location).
   Evitar all_reviews.csv si aporta <1000 reviews UK verificadas.
""")

print(f"\nOutputs guardados en: {OUTPUT_DIR}/")
