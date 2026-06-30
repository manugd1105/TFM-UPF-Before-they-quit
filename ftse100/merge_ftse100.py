"""
Unión de glassdoor_reviews.csv + all_reviews.csv
─────────────────────────────────────────────────
• Filtra sólo empresas FTSE 100
• Normaliza columnas a un esquema común
• Deduplica: clave = hash(pros_norm + cons_norm)
  Si una review aparece en ambos datasets se conserva
  la de glassdoor_reviews (tiene location fiable).
• Genera review_id determinista (sha1 hex[:12])
• Guarda en extradatos/ftse100_reviews.csv
"""

import hashlib, os, re, time
import pandas as pd

# ─── FTSE 100 slugs (misma lista que en el análisis) ──────────────────────────
FTSE100_SLUGS = {
    "aviva", "barclays", "british-american-tobacco", "bat", "bunzl",
    "bt", "bt-group", "compass-group", "compass", "diageo", "easyjet",
    "experian", "flutter-entertainment", "flutter", "betfair", "glencore",
    "gsk", "glaxosmithkline", "glaxo-smithkline", "haleon", "halma", "hsbc",
    "international-airlines-group", "british-airways", "iag", "iberia",
    "intercontinental-hotels-group", "ihg", "holiday-inn",
    "imperial-brands", "imperial-tobacco", "informa", "itv",
    "jd-sports", "jd-sports-fashion", "land-securities", "landsec",
    "legal-general", "legal-and-general", "lloyds-banking-group",
    "lloyds-bank", "lloyds", "london-stock-exchange-group", "lseg",
    "london-stock-exchange", "marks-spencer", "marks-and-spencer",
    "m-and-g", "mandg", "mondi", "national-grid", "next",
    "ocado", "ocado-group", "phoenix-group", "prudential", "persimmon",
    "pearson", "reckitt", "reckitt-benckiser", "shell", "royal-dutch-shell",
    "relx", "reed-elsevier", "rio-tinto", "rightmove",
    "rentokil-initial", "rentokil", "rolls-royce", "rolls-royce-holdings",
    "rs-group", "rs-components", "sainsburys", "sainsbury", "segro",
    "st-james-place", "st-james-s-place", "smurfit-kappa", "ds-smith",
    "smith-nephew", "smith-and-nephew", "spirax-sarco-engineering",
    "spirax-sarco", "sse", "standard-chartered", "severn-trent", "tesco",
    "taylor-wimpey", "unilever", "united-utilities", "vodafone", "wpp",
    "whitbread", "anglo-american", "antofagasta", "bp", "bhp", "bhp-group",
    "fresnillo", "auto-trader", "autotrader", "bae-systems", "centrica",
    "croda-international", "croda", "dcc", "entain", "gvc-holdings",
    "ladbrokes", "ferguson", "intertek", "kingfisher", "b-q", "screwfix",
    "melrose-industries", "melrose", "natwest-group", "natwest",
    "royal-bank-of-scotland", "qinetiq", "sage-group", "sage", "schroders",
    "spirent", "weir-group", "weir", "william-hill", "astrazeneca",
    "berkeley-group", "berkeley", "british-land", "drax-group", "drax",
    "howden-joinery", "howdens", "imi", "wh-smith", "whsmith",
    "3i-group", "3i", "admiral-group", "admiral", "barratt-developments",
    "barratt", "brewin-dolphin", "greggs", "hargreaves-lansdown",
    "hiscox", "jupiter-fund-management", "jupiter", "darktrace",
}

def slug_matches_ftse(slug: str) -> bool:
    """Coincidencia exacta o como prefijo/sufijo separado por guión.
    Evita falsos positivos de subcadenas (p.ej. 'sse' en 'credit-suisse',
    'bat' en 'bath-&-body-works').
    """
    s = str(slug).lower().strip()
    for f in FTSE100_SLUGS:
        if s == f or s.startswith(f + "-") or s.endswith("-" + f):
            return True
    return False

def norm_text(s) -> str:
    return str(s).lower().strip() if pd.notna(s) else ""

def review_hash(pros, cons, company, date) -> str:
    raw = norm_text(pros) + "|" + norm_text(cons) + "|" + norm_text(company) + "|" + norm_text(date)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]

def dedup_key(pros, cons) -> str:
    """Clave de solapamiento entre datasets (sólo contenido)."""
    p, c = norm_text(pros), norm_text(cons)
    if len(p) + len(c) < 20:
        return ""          # demasiado corta para ser fiable
    return hashlib.sha1((p + "|" + c).encode()).hexdigest()

BASE = os.path.dirname(__file__)

# ─── 1. glassdoor_reviews — filtro FTSE 100 ───────────────────────────────────
print("Cargando glassdoor_reviews.csv …")
t0 = time.time()

gd_raw = pd.read_csv(os.path.join(BASE, "glassdoor_reviews.csv"), low_memory=False)
print(f"  {len(gd_raw):,} filas totales  ({time.time()-t0:.1f}s)")

gd_ftse = gd_raw[gd_raw["firm"].fillna("").apply(slug_matches_ftse)].copy()
print(f"  {len(gd_ftse):,} filas FTSE 100")

# Normalizar
gd_norm = pd.DataFrame({
    "source":                 "glassdoor_reviews",
    "company":                gd_ftse["firm"].str.lower().str.strip(),
    "date":                   gd_ftse["date_review"],
    "job_title":              gd_ftse["job_title"],
    "current":                gd_ftse["current"],
    "location":               gd_ftse["location"],
    "overall_rating":         pd.to_numeric(gd_ftse["overall_rating"],  errors="coerce"),
    "wl_balance":             pd.to_numeric(gd_ftse["work_life_balance"], errors="coerce"),
    "culture_values":         pd.to_numeric(gd_ftse["culture_values"],   errors="coerce"),
    "diversity_inclusion":    pd.to_numeric(gd_ftse["diversity_inclusion"], errors="coerce"),
    "career_opportunities":   pd.to_numeric(gd_ftse["career_opp"],       errors="coerce"),
    "compensation_benefits":  pd.to_numeric(gd_ftse["comp_benefits"],    errors="coerce"),
    "senior_management":      pd.to_numeric(gd_ftse["senior_mgmt"],      errors="coerce"),
    "recommend":              gd_ftse["recommend"],
    "ceo_approv":             gd_ftse["ceo_approv"],
    "outlook":                gd_ftse["outlook"],
    "summary":                gd_ftse["headline"],
    "pros":                   gd_ftse["pros"],
    "cons":                   gd_ftse["cons"],
    "advice":                 pd.NA,
})

gd_norm["review_id"]  = gd_norm.apply(
    lambda r: review_hash(r.pros, r.cons, r.company, r.date), axis=1)
gd_norm["dedup_key"]  = gd_norm.apply(
    lambda r: dedup_key(r.pros, r.cons), axis=1)

# Registrar claves vistas (para excluir duplicados de all_reviews)
seen_keys = set(gd_norm.loc[gd_norm["dedup_key"] != "", "dedup_key"])
print(f"  → {len(gd_norm):,} reviews normalizadas | {len(seen_keys):,} claves únicas registradas")

# ─── 2. all_reviews — filtro FTSE 100 en chunks (3.8 GB) ─────────────────────
print("\nCargando all_reviews.csv en chunks …")
t1 = time.time()

SLUG_RE = re.compile(r"Reviews/(.+)-Reviews-E\d+\.htm", re.IGNORECASE)

ar_chunks = []
total_read = 0
total_ftse = 0
total_new  = 0

for chunk in pd.read_csv(
        os.path.join(BASE, "all_reviews.csv"),
        chunksize=200_000, low_memory=False):

    total_read += len(chunk)

    # Extraer slug desde firm_link
    chunk["_slug"] = chunk["firm_link"].fillna("").str.extract(SLUG_RE, expand=False).str.lower()
    ftse_mask = chunk["_slug"].apply(lambda x: slug_matches_ftse(x) if x else False)
    sub = chunk[ftse_mask].copy()
    total_ftse += len(sub)

    if len(sub) == 0:
        continue

    # Calcular dedup_key y excluir reviews ya presentes en glassdoor_reviews
    sub["_dkey"] = sub.apply(lambda r: dedup_key(r.pros, r.cons), axis=1)
    sub_new = sub[~sub["_dkey"].isin(seen_keys) | (sub["_dkey"] == "")].copy()
    total_new += len(sub_new)

    # Actualizar seen_keys con las nuevas
    new_keys = set(sub_new.loc[sub_new["_dkey"] != "", "_dkey"])
    seen_keys.update(new_keys)

    if len(sub_new) == 0:
        continue

    # Normalizar
    norm = pd.DataFrame({
        "source":                "all_reviews",
        "company":               sub_new["_slug"],
        "date":                  sub_new["date"],
        "job_title":             sub_new["job"],
        "current":               sub_new["status"],
        "location":              pd.NA,           # no hay location fiable
        "overall_rating":        pd.to_numeric(sub_new["rating"],                    errors="coerce"),
        "wl_balance":            pd.to_numeric(sub_new["Work/Life Balance"],         errors="coerce"),
        "culture_values":        pd.to_numeric(sub_new["Culture & Values"],          errors="coerce"),
        "diversity_inclusion":   pd.to_numeric(sub_new["Diversity & Inclusion"],     errors="coerce"),
        "career_opportunities":  pd.to_numeric(sub_new["Career Opportunities"],      errors="coerce"),
        "compensation_benefits": pd.to_numeric(sub_new["Compensation and Benefits"], errors="coerce"),
        "senior_management":     pd.to_numeric(sub_new["Senior Management"],         errors="coerce"),
        "recommend":             sub_new["Recommend"],
        "ceo_approv":            sub_new["CEO Approval"],
        "outlook":               sub_new["Business Outlook"],
        "summary":               sub_new["title"],
        "pros":                  sub_new["pros"],
        "cons":                  sub_new["cons"],
        "advice":                sub_new["advice"],
    })
    norm["review_id"] = norm.apply(
        lambda r: review_hash(r.pros, r.cons, r.company, r.date), axis=1)
    norm["dedup_key"] = sub_new["_dkey"].values

    ar_chunks.append(norm)

    if total_read % 2_000_000 == 0:
        print(f"  … {total_read:,} leídas | {total_ftse:,} FTSE | {total_new:,} nuevas ({time.time()-t1:.0f}s)")

print(f"  Total leídas: {total_read:,} | FTSE: {total_ftse:,} | Nuevas (no duplicadas): {total_new:,}  ({time.time()-t1:.1f}s)")

ar_norm = pd.concat(ar_chunks, ignore_index=True) if ar_chunks else pd.DataFrame(columns=gd_norm.columns)

# ─── 3. Unión final ────────────────────────────────────────────────────────────
print("\nUniendo datasets …")
combined = pd.concat([gd_norm, ar_norm], ignore_index=True)

# Eliminar columna auxiliar y asegurar review_id único (renombrar dupes edge-case)
combined.drop(columns=["dedup_key"], inplace=True)

# Segunda pasada de dedup por review_id (colisiones hash improbables pero posibles)
before = len(combined)
combined.drop_duplicates(subset=["review_id"], keep="first", inplace=True)
combined.reset_index(drop=True, inplace=True)
print(f"  Filas antes de dedup final: {before:,} → después: {len(combined):,}")

# Reordenar columnas
COLS = ["review_id","source","company","date","job_title","current","location",
        "overall_rating","wl_balance","culture_values","diversity_inclusion",
        "career_opportunities","compensation_benefits","senior_management",
        "recommend","ceo_approv","outlook","summary","pros","cons","advice"]
combined = combined[[c for c in COLS if c in combined.columns]]

# ─── 4. Guardar ───────────────────────────────────────────────────────────────
out_path = os.path.join(BASE, "ftse100_reviews.csv")
combined.to_csv(out_path, index=False)
size_mb = os.path.getsize(out_path) / 1_048_576

print(f"\n{'='*55}")
print(f"  Archivo: ftse100_reviews.csv")
print(f"  Filas totales:        {len(combined):,}")
print(f"  De glassdoor_reviews: {(combined['source']=='glassdoor_reviews').sum():,}")
print(f"  De all_reviews:       {(combined['source']=='all_reviews').sum():,}")
print(f"  Tamaño:               {size_mb:.1f} MB")
print(f"  Duplicados eliminados:{total_ftse - total_new + (before - len(combined)):,}")
print(f"{'='*55}")
print(f"\nResumen por empresa (top 20):")
print(combined.groupby("company")["review_id"].count()
      .sort_values(ascending=False).head(20).to_string())
