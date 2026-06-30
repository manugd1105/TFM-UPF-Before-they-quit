"""
Normalización de ftse100_reviews.csv
──────────────────────────────────────
• Aplica whitelist explícito slug → (ticker, nombre canónico)
• Descarta falsos positivos (university-of-california-berkeley,
  compass-education, vodafone-idea, smithfield's, etc.)
• Agrupa variantes del mismo grupo bajo un único ticker
• Añade columnas  ticker  y  company_canonical
• Sobreescribe ftse100_reviews.csv con la versión limpia
"""

import os
import pandas as pd

BASE = os.path.dirname(__file__)

# ─── Whitelist explícito: slug_en_csv → (ticker_ftse100, nombre_canónico) ─────
# Cada slug debe aparecer tal cual en la columna 'company' del CSV.
# Variantes legítimas (subsidiarias, marcas) se mapean al mismo ticker.
SLUG_TO_FTSE: dict[str, tuple[str, str]] = {
    # ── Financials / Insurance ──────────────────────────────────────────────
    "hsbc":                                  ("HSBA", "HSBC"),
    "hsbc-holdings":                         ("HSBA", "HSBC"),
    "barclays":                              ("BARC", "Barclays"),
    "lloyds-banking-group":                  ("LLOY", "Lloyds Banking Group"),
    "lloyds-bank":                           ("LLOY", "Lloyds Banking Group"),
    "lloyds":                                ("LLOY", "Lloyds Banking Group"),
    "natwest-group":                         ("NWG",  "NatWest Group"),
    "royal-bank-of-scotland-international":  ("NWG",  "NatWest Group"),
    "standard-chartered-bank":               ("STAN", "Standard Chartered"),
    "standard-chartered":                    ("STAN", "Standard Chartered"),
    "aviva":                                 ("AV.",  "Aviva"),
    "prudential":                            ("PRU",  "Prudential"),
    "prudential-plc":                        ("PRU",  "Prudential"),
    "legal-and-general":                     ("LGEN", "Legal & General"),
    "legal-general":                         ("LGEN", "Legal & General"),
    "schroders":                             ("SDR",  "Schroders"),
    "brewin-dolphin":                        ("BREI", "Brewin Dolphin"),
    "rbc-brewin-dolphin":                    ("BREI", "Brewin Dolphin"),
    "st-james-s-place-wealth-management":    ("SJP",  "St. James's Place"),
    "st-james-place":                        ("SJP",  "St. James's Place"),
    "st-james-s-place":                      ("SJP",  "St. James's Place"),
    "admiral-group":                         ("ADM",  "Admiral Group"),
    # ── Consumer Staples / Retail ────────────────────────────────────────────
    "tesco":                                 ("TSCO", "Tesco"),
    "j-sainsbury":                           ("SBRY", "Sainsbury's"),
    "sainsbury-s":                           ("SBRY", "Sainsbury's"),
    "sainsburys":                            ("SBRY", "Sainsbury's"),
    "marks-and-spencer":                     ("MKS",  "Marks & Spencer"),
    "next":                                  ("NXT",  "Next"),
    "jd-sports-fashion":                     ("JD.",  "JD Sports Fashion"),
    "jd-sports":                             ("JD.",  "JD Sports Fashion"),
    "kingfisher":                            ("KGF",  "Kingfisher"),
    "screwfix":                              ("KGF",  "Kingfisher"),     # marca de Kingfisher
    "greggs":                                ("GRG",  "Greggs"),
    "whsmith":                               ("SMWH", "WH Smith"),
    "wh-smith":                              ("SMWH", "WH Smith"),
    "whitbread":                             ("WTB",  "Whitbread"),
    "ocado":                                 ("OCDO", "Ocado Group"),
    "ocado-group":                           ("OCDO", "Ocado Group"),
    "unilever":                              ("ULVR", "Unilever"),
    "reckitt":                               ("RKT",  "Reckitt"),
    "reckitt-benckiser":                     ("RKT",  "Reckitt"),
    "diageo":                                ("DGE",  "Diageo"),
    "imperial-brands":                       ("IMB",  "Imperial Brands"),
    "imperial-tobacco":                      ("IMB",  "Imperial Brands"),
    "bat":                                   ("BATS", "British American Tobacco"),
    "british-american-tobacco":              ("BATS", "British American Tobacco"),
    # ── Healthcare / Pharma ─────────────────────────────────────────────────
    "glaxosmithkline":                       ("GSK",  "GSK"),
    "gsk":                                   ("GSK",  "GSK"),
    "astrazeneca":                           ("AZN",  "AstraZeneca"),
    "smith-and-nephew":                      ("SN.",  "Smith & Nephew"),
    "smith-nephew":                          ("SN.",  "Smith & Nephew"),
    "haleon":                                ("HLN",  "Haleon"),
    # ── Technology / Telecoms ────────────────────────────────────────────────
    "vodafone":                              ("VOD",  "Vodafone"),
    "bt":                                    ("BT.A", "BT Group"),
    "bt-group":                              ("BT.A", "BT Group"),
    "sage":                                  ("SGE",  "Sage Group"),
    "sage-group":                            ("SGE",  "Sage Group"),
    "experian":                              ("EXPN", "Experian"),
    "auto-trader":                           ("AUTO", "Auto Trader"),
    "autotrader":                            ("AUTO", "Auto Trader"),
    "rightmove":                             ("RMV",  "Rightmove"),
    "relx":                                  ("REL",  "RELX"),
    "reed-elsevier-shared-services":         ("REL",  "RELX"),
    "reed-elsevier":                         ("REL",  "RELX"),
    "lseg-london-stock-exchange-group":      ("LSEG", "London Stock Exchange Group"),
    "london-stock-exchange-group":           ("LSEG", "London Stock Exchange Group"),
    "lseg":                                  ("LSEG", "London Stock Exchange Group"),
    "darktrace":                             ("DARK", "Darktrace"),
    "spirent-communications":               ("SPT",  "Spirent Communications"),
    "spirent":                               ("SPT",  "Spirent Communications"),
    # ── Industrials / Engineering ────────────────────────────────────────────
    "rolls-royce":                           ("RR.",  "Rolls-Royce"),
    "rolls-royce-power-systems":             ("RR.",  "Rolls-Royce"),
    "bae-systems":                           ("BA.",  "BAE Systems"),
    "bae-systems-usa":                       ("BA.",  "BAE Systems"),
    "melrose-industries":                    ("MRO",  "Melrose Industries"),
    "melrose":                               ("MRO",  "Melrose Industries"),
    "weir-group":                            ("WEIR", "Weir Group"),
    "weir-esco":                             ("WEIR", "Weir Group"),
    "imi":                                   ("IMI",  "IMI"),
    "imi-critical-engineering":              ("IMI",  "IMI"),
    "imi-norgren":                           ("IMI",  "IMI"),
    "spirax-sarco":                          ("SPX",  "Spirax-Sarco Engineering"),
    "spirax-sarco-engineering":              ("SPX",  "Spirax-Sarco Engineering"),
    "intertek":                              ("ITRK", "Intertek"),
    "halma":                                 ("HLMA", "Halma"),
    "qinetiq-group":                         ("QQ.",  "QinetiQ"),
    "qinetiq":                               ("QQ.",  "QinetiQ"),
    "croda":                                 ("CRDA", "Croda International"),
    "croda-international":                   ("CRDA", "Croda International"),
    "ds-smith":                              ("SMDS", "DS Smith"),
    "bunzl":                                 ("BNZL", "Bunzl"),
    "rs-group":                              ("RS2K", "RS Group"),
    "smurfit-kappa":                         ("SKG",  "Smurfit Kappa"),
    "mondi-group":                           ("MNDI", "Mondi"),
    "mondi":                                 ("MNDI", "Mondi"),
    "howdens-joinery":                       ("HWDN", "Howden Joinery"),
    "howden-joinery":                        ("HWDN", "Howden Joinery"),
    "ferguson-enterprises":                  ("FERG", "Ferguson"),
    "ferguson":                              ("FERG", "Ferguson"),
    # ── Energy / Mining ─────────────────────────────────────────────────────
    "bp":                                    ("BP.",  "BP"),
    "shell":                                 ("SHEL", "Shell"),
    "royal-dutch-shell":                     ("SHEL", "Shell"),
    "centrica":                              ("CNA",  "Centrica"),
    "sse":                                   ("SSE",  "SSE"),
    "sse-airtricity":                        ("SSE",  "SSE"),
    "drax-group":                            ("DRX",  "Drax Group"),
    "drax":                                  ("DRX",  "Drax Group"),
    "national-grid":                         ("NG.",  "National Grid"),
    "severn-trent":                          ("SVT",  "Severn Trent"),
    "united-utilities":                      ("UU.",  "United Utilities"),
    "glencore":                              ("GLEN", "Glencore"),
    "bhp":                                   ("BHP",  "BHP Group"),
    "bhp-group":                             ("BHP",  "BHP Group"),
    "rio-tinto":                             ("RIO",  "Rio Tinto"),
    "anglo-american":                        ("AAL",  "Anglo American"),
    "antofagasta":                           ("ANTO", "Antofagasta"),
    "fresnillo":                             ("FRES", "Fresnillo"),
    # ── Travel / Hospitality / Media ────────────────────────────────────────
    "british-airways":                       ("IAG",  "International Airlines Group"),
    "iberia":                                ("IAG",  "International Airlines Group"),
    "international-airlines-group":          ("IAG",  "International Airlines Group"),
    "easyjet":                               ("EZJ",  "easyJet"),
    "ihg-hotels-and-resorts":               ("IHG",  "IHG Hotels & Resorts"),
    "holiday-inn":                           ("IHG",  "IHG Hotels & Resorts"),
    "holiday-inn-express":                   ("IHG",  "IHG Hotels & Resorts"),
    "intercontinental-hotels-group":         ("IHG",  "IHG Hotels & Resorts"),
    "ihg":                                   ("IHG",  "IHG Hotels & Resorts"),
    "entain":                                ("ENT",  "Entain"),
    "gvc-holdings":                          ("ENT",  "Entain"),
    "ladbrokes":                             ("ENT",  "Entain"),
    "william-hill":                          ("WMH",  "William Hill"),
    "flutter-entertainment":                 ("FLTR", "Flutter Entertainment"),
    "betfair":                               ("FLTR", "Flutter Entertainment"),
    "wpp":                                   ("WPP",  "WPP"),
    "itv":                                   ("ITV",  "ITV"),
    "itv-wales-&-west":                      ("ITV",  "ITV"),
    "informa":                               ("INF",  "Informa"),
    "pearson":                               ("PSON", "Pearson"),
    "compass-group":                         ("CPG",  "Compass Group"),
    # ── Real Estate ─────────────────────────────────────────────────────────
    "segro":                                 ("SGRO", "Segro"),
    "landsec":                               ("LAND", "Land Securities"),
    "land-securities":                       ("LAND", "Land Securities"),
    "british-land-company":                  ("BLND", "British Land"),
    "british-land":                          ("BLND", "British Land"),
    "berkeley-group":                        ("BKG",  "Berkeley Group"),
    "barratt-developments":                  ("BDEV", "Barratt Developments"),
    "barratt":                               ("BDEV", "Barratt Developments"),
    "taylor-wimpey":                         ("TW.",  "Taylor Wimpey"),
    "persimmon":                             ("PSN",  "Persimmon"),
    # ── Logistics / Services ────────────────────────────────────────────────
    "rentokil-initial":                      ("RTO",  "Rentokil Initial"),
    "rentokil":                              ("RTO",  "Rentokil Initial"),
    "dcc":                                   ("DCC",  "DCC"),
    "3i-group":                              ("III",  "3i Group"),
    "3i":                                    ("III",  "3i Group"),
}

# ─── Cargar CSV ────────────────────────────────────────────────────────────────
path_in = os.path.join(BASE, "ftse100_reviews.csv")
print(f"Cargando {path_in} …")
df = pd.read_csv(path_in, low_memory=False)
print(f"  {len(df):,} filas  |  {df['company'].nunique()} slugs únicos")

# ─── Aplicar whitelist ─────────────────────────────────────────────────────────
df["ticker"]            = df["company"].map(lambda s: SLUG_TO_FTSE.get(s, (None, None))[0])
df["company_canonical"] = df["company"].map(lambda s: SLUG_TO_FTSE.get(s, (None, None))[1])

# Separar reconocidos de falsos positivos
known   = df[df["ticker"].notna()].copy()
unknown = df[df["ticker"].isna()].copy()

print(f"\n  Slugs reconocidos (whitelist): {known['company'].nunique()}")
print(f"  Filas conservadas:             {len(known):,}")
print(f"\n  Slugs descartados (falsos positivos): {unknown['company'].nunique()}")
if len(unknown):
    fp_counts = unknown.groupby("company").size().sort_values(ascending=False)
    for slug, n in fp_counts.items():
        print(f"    ✗  {slug:<55} {n:>6} filas")

# ─── Reordenar columnas ────────────────────────────────────────────────────────
COLS = [
    "review_id", "source", "ticker", "company_canonical", "company",
    "date", "job_title", "current", "location",
    "overall_rating", "wl_balance", "culture_values", "diversity_inclusion",
    "career_opportunities", "compensation_benefits", "senior_management",
    "recommend", "ceo_approv", "outlook",
    "summary", "pros", "cons", "advice",
]
known = known[[c for c in COLS if c in known.columns]]

# ─── Resumen final ─────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print(f"  Reviews finales:          {len(known):,}")
print(f"  Empresas FTSE únicas:     {known['ticker'].nunique()}")
print(f"  De glassdoor_reviews:     {(known['source']=='glassdoor_reviews').sum():,}")
print(f"  De all_reviews:           {(known['source']=='all_reviews').sum():,}")
print(f"  Falsos positivos retirados: {len(unknown):,}")
print(f"{'='*58}")

print(f"\nReviews por empresa (ordenadas por volumen):")
summary = (known.groupby(["ticker", "company_canonical"])
           .agg(reviews=("review_id", "count"),
                rating_medio=("overall_rating", "mean"))
           .reset_index()
           .sort_values("reviews", ascending=False))
summary["rating_medio"] = summary["rating_medio"].round(2)
print(summary.to_string(index=False))

# ─── Guardar ───────────────────────────────────────────────────────────────────
known.to_csv(path_in, index=False)
size_mb = os.path.getsize(path_in) / 1_048_576
print(f"\nGuardado: ftse100_reviews.csv  ({size_mb:.1f} MB)")
