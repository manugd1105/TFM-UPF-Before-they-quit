"""
Genera el PDF de resumen del producto TFM — NVIDIA Employee Satisfaction Predictor
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import Image
from reportlab.lib import colors
from datetime import date
from pathlib import Path

OUTPUT = str(Path(__file__).resolve().parent / "TFM_Product_Summary.pdf")

# ── Colores corporativos NVIDIA / TFM ────────────────────────────────────────
GREEN    = HexColor("#76b900")   # NVIDIA green
DARKGRAY = HexColor("#1a1a1a")
MIDGRAY  = HexColor("#4a4a4a")
LIGHTGRAY= HexColor("#f5f5f5")
ACCENT   = HexColor("#005f87")   # azul académico

# ── Estilos ───────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def style(name, **kw):
    return ParagraphStyle(name, **kw)

S = {
    "cover_title": style("ct", fontName="Helvetica-Bold", fontSize=28,
                         textColor=white, alignment=TA_CENTER, spaceAfter=8),
    "cover_sub":   style("cs", fontName="Helvetica", fontSize=14,
                         textColor=HexColor("#dddddd"), alignment=TA_CENTER, spaceAfter=6),
    "cover_meta":  style("cm", fontName="Helvetica", fontSize=11,
                         textColor=HexColor("#bbbbbb"), alignment=TA_CENTER),
    "h1":          style("h1", fontName="Helvetica-Bold", fontSize=16,
                         textColor=ACCENT, spaceBefore=18, spaceAfter=8,
                         borderPadding=(0,0,4,0)),
    "h2":          style("h2", fontName="Helvetica-Bold", fontSize=12,
                         textColor=DARKGRAY, spaceBefore=12, spaceAfter=5),
    "body":        style("bd", fontName="Helvetica", fontSize=10,
                         textColor=MIDGRAY, leading=16, alignment=TA_JUSTIFY,
                         spaceAfter=6),
    "bullet":      style("bl", fontName="Helvetica", fontSize=10,
                         textColor=MIDGRAY, leading=15, leftIndent=16,
                         bulletIndent=4, spaceAfter=3),
    "code":        style("cd", fontName="Courier", fontSize=9,
                         textColor=HexColor("#2d6a4f"), backColor=HexColor("#f0fff4"),
                         leading=14, leftIndent=12, spaceAfter=6),
    "caption":     style("cp", fontName="Helvetica-Oblique", fontSize=9,
                         textColor=HexColor("#888888"), alignment=TA_CENTER, spaceAfter=8),
    "metric_val":  style("mv", fontName="Helvetica-Bold", fontSize=22,
                         textColor=GREEN, alignment=TA_CENTER),
    "metric_lbl":  style("ml", fontName="Helvetica", fontSize=9,
                         textColor=MIDGRAY, alignment=TA_CENTER),
}

def H1(text):
    items = [
        HRFlowable(width="100%", thickness=2, color=GREEN, spaceAfter=4),
        Paragraph(text, S["h1"]),
    ]
    return KeepTogether(items)

def H2(text):
    return Paragraph(text, S["h2"])

def P(text):
    return Paragraph(text, S["body"])

def B(text):
    return Paragraph(f"• {text}", S["bullet"])

def SP(n=6):
    return Spacer(1, n)

# ── Tabla de métricas ─────────────────────────────────────────────────────────
def metric_table(metrics):
    data = [[Paragraph(v, S["metric_val"]) for v, _ in metrics],
            [Paragraph(l, S["metric_lbl"]) for _, l in metrics]]
    t = Table(data, colWidths=[4.2*cm]*len(metrics))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), LIGHTGRAY),
        ("ROUNDEDCORNERS", [6]),
        ("BOX",     (0,0), (-1,-1), 0.5, HexColor("#cccccc")),
        ("GRID",    (0,0), (-1,-1), 0.5, HexColor("#cccccc")),
        ("VALIGN",  (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    return t

def arch_table():
    rows = [
        ["Componente", "Tecnología", "Función"],
        ["Formulario",     "Tally.so",          "Recoge la review del empleado"],
        ["Orquestador",    "n8n (Node.js v20)",  "Recibe webhook y coordina el pipeline"],
        ["API de predicción","FastAPI + Uvicorn", "Feature engineering + inferencia XGBoost"],
        ["Modelo ML",      "XGBoost Regressor",  "Predice overall_rating (escala 1–5)"],
        ["Almacenamiento", "gold.xlsx (openpyxl)","Histórico acumulativo de predicciones"],
        ["Visualización",  "Tableau Desktop",    "Dashboard en tiempo real"],
        ["Túnel público",  "ngrok",              "Expone n8n a internet para Tally"],
    ]
    col_w = [3.8*cm, 4.2*cm, 8.5*cm]
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  ACCENT),
        ("TEXTCOLOR",     (0,0), (-1,0),  white),
        ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
        ("TEXTCOLOR",     (0,1), (-1,-1), MIDGRAY),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [white, LIGHTGRAY]),
        ("GRID",          (0,0), (-1,-1), 0.4, HexColor("#cccccc")),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0), (-1,-1), 6),
        ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
    ]))
    return t

def feature_table():
    rows = [
        ["Grupo de Features", "Nº Features", "Herramienta / Fuente"],
        ["POS tag frequencies",        "~120", "NLTK pos_tag (Penn Treebank)"],
        ["EmoLex affect scores",        "20",  "NRCLex (pros + cons × 10 emociones)"],
        ["Empath scores",               "40",  "Empath (pros + cons × 20 categorías)"],
        ["VADER sentiment",              "2",  "NLTK SentimentIntensityAnalyzer"],
        ["JDI cosine similarities",     "10",  "SentenceTransformer all-roberta-large-v1"],
        ["ODI cosine similarities",     "18",  "SentenceTransformer all-roberta-large-v1"],
        ["ODI vocabulary ratio",         "2",  "Keyword matching (9 dimensiones clínicas)"],
        ["Text lengths",                 "6",  "charlen + wordlen (summary, pros, cons)"],
        ["Company features",             "4",  "Market Cap, Employees, Founded, Cap/Employee"],
        ["Sector OHE + target enc.",    "21",  "19 sectores + Sector_targenc_mean/median"],
        ["Stock % change",               "4",  "yfinance (NVDA + SPY, mes + año)"],
        ["Temporal features",            "5",  "month, year, days desde fundación / t0"],
        ["State features",             "~15",  "OHE + target encoding (Estado del empleado)"],
        ["TOTAL",                      "267",  ""],
    ]
    col_w = [5.5*cm, 2.5*cm, 8.5*cm]
    t = Table(rows, colWidths=col_w)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),  (-1,0),  ACCENT),
        ("TEXTCOLOR",     (0,0),  (-1,0),  white),
        ("FONTNAME",      (0,0),  (-1,0),  "Helvetica-Bold"),
        ("BACKGROUND",    (0,-1), (-1,-1), HexColor("#e8f4fd")),
        ("FONTNAME",      (0,-1), (-1,-1), "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,-1), (-1,-1), ACCENT),
        ("FONTSIZE",      (0,0),  (-1,-1), 9),
        ("FONTNAME",      (0,1),  (-1,-2), "Helvetica"),
        ("TEXTCOLOR",     (0,1),  (-1,-2), MIDGRAY),
        ("ROWBACKGROUNDS",(0,1),  (-1,-2), [white, LIGHTGRAY]),
        ("GRID",          (0,0),  (-1,-1), 0.4, HexColor("#cccccc")),
        ("VALIGN",        (0,0),  (-1,-1), "MIDDLE"),
        ("TOPPADDING",    (0,0),  (-1,-1), 5),
        ("BOTTOMPADDING", (0,0),  (-1,-1), 5),
        ("LEFTPADDING",   (0,0),  (-1,-1), 8),
    ]))
    return t


# ── BUILD ────────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    OUTPUT, pagesize=A4,
    leftMargin=2.2*cm, rightMargin=2.2*cm,
    topMargin=2*cm, bottomMargin=2*cm,
    title="TFM Product Summary — NVIDIA Employee Satisfaction Predictor",
    author="Iker Lleida"
)

story = []

# ── PORTADA ───────────────────────────────────────────────────────────────────
def cover_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(DARKGRAY)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)
    canvas.setFillColor(GREEN)
    canvas.rect(0, A4[1]-1.2*cm, A4[0], 1.2*cm, fill=1, stroke=0)
    canvas.setFillColor(GREEN)
    canvas.rect(0, 0, A4[0], 0.6*cm, fill=1, stroke=0)
    canvas.restoreState()

story.append(SP(120))
story.append(Paragraph("NVIDIA Employee", S["cover_title"]))
story.append(Paragraph("Satisfaction Predictor", S["cover_title"]))
story.append(SP(12))
story.append(Paragraph("Pipeline de Producto — Documento Técnico", S["cover_sub"]))
story.append(SP(30))
story.append(Paragraph("Trabajo de Fin de Máster · Data Analytics", S["cover_meta"]))
story.append(SP(6))
story.append(Paragraph(f"Iker Lleida · {date.today().strftime('%B %Y')}", S["cover_meta"]))
story.append(PageBreak())

# ── 1. RESUMEN EJECUTIVO ──────────────────────────────────────────────────────
story.append(H1("1. Resumen Ejecutivo"))
story.append(P(
    "Este documento describe el diseño, implementación y resultados del sistema de predicción "
    "de satisfacción laboral de empleados de NVIDIA desarrollado como parte del Trabajo de Fin "
    "de Máster en Data Analytics. El sistema combina técnicas avanzadas de Procesamiento del "
    "Lenguaje Natural (NLP) con modelos de Machine Learning para estimar, en tiempo real, "
    "el nivel de satisfacción de un empleado a partir de una review textual."
))
story.append(P(
    "El pipeline de producto completo abarca desde la recogida de datos mediante un formulario "
    "web hasta la visualización automática en un dashboard de Tableau Desktop, pasando por un "
    "motor de predicción basado en XGBoost entrenado sobre 356.594 reviews de empleados de "
    "empresas del S&P 500 extraídas de Glassdoor."
))
story.append(SP(10))
story.append(metric_table([
    ("356.594", "Reviews de entrenamiento"),
    ("267",     "Features del modelo"),
    ("0.655",   "R² en datos P1"),
    ("0.731",   "RMSE en datos P1"),
    ("500",     "Empresas S&P 500"),
    ("<2s",     "Latencia predicción"),
]))
story.append(SP(10))

# ── 2. PROBLEMA Y MOTIVACIÓN ─────────────────────────────────────────────────
story.append(H1("2. Problema y Motivación"))
story.append(P(
    "La satisfacción laboral es un predictor crítico de la retención de talento, la productividad "
    "y el bienestar organizacional. Sin embargo, las encuestas tradicionales tienen baja tasa de "
    "respuesta y capturan únicamente métricas cuantitativas, perdiendo el rico contenido semántico "
    "presente en el texto libre de las reviews de empleados."
))
story.append(H2("Objetivo principal"))
story.append(P(
    "Desarrollar un modelo de predicción que, dado el texto libre de una review de empleado "
    "(resumen, pros, contras), estime automáticamente el <b>overall_rating</b> en escala 1–5 "
    "y lo clasifique como <b>Alto</b> (≥4) o <b>Bajo</b> (&lt;4), integrando además variables "
    "contextuales de la empresa, sector y mercado."
))
story.append(H2("Caso de uso: NVIDIA"))
story.append(P(
    "NVIDIA (ticker: NVDA) se utiliza como empresa piloto para la demo en tiempo real. Con "
    "945 reviews históricas disponibles y una distribución de satisfacción predominantemente "
    "positiva (68% Alto), sirve como banco de pruebas ideal para validar el pipeline end-to-end."
))

# ── 3. DATOS ──────────────────────────────────────────────────────────────────
story.append(H1("3. Datos"))
story.append(H2("3.1 Fuente de datos"))
story.append(P(
    "Las reviews fueron extraídas de <b>Glassdoor</b> mediante web scraping para las 500 empresas "
    "del índice S&P 500. Cada review incluye texto libre (summary, pros, cons) y valoraciones "
    "numéricas en seis dimensiones: work-life balance, culture &amp; values, diversity &amp; "
    "inclusion, career opportunities, compensation &amp; benefits, y senior management."
))
story.append(H2("3.2 Enriquecimiento"))
story.append(P("A cada review se le añadieron variables contextuales externas:"))
for item in [
    "Datos de empresa: capitalización bursátil, número de empleados, año de fundación, sector (GICS).",
    "Datos de mercado: variación porcentual del precio de la acción a 1 mes y 1 año (yfinance).",
    "Datos macroeconómicos del estado del empleado: PIB per cápita, temperatura media, índice de criminalidad, etc.",
    "Variables temporales: mes, año y días desde la fundación de la empresa hasta la review.",
]:
    story.append(B(item))
story.append(H2("3.3 Preprocesamiento"))
story.append(P(
    "Se aplicó <b>StandardScaler</b> a todas las features numéricas, <b>One-Hot Encoding</b> "
    "para sector y estado, y <b>Target Encoding</b> (media y mediana del target por categoría) "
    "para capturar el efecto de sector y estado sobre la satisfacción. El split train/test "
    "fue 80/20 estratificado, resultando en 285.275 reviews de entrenamiento y 71.319 de test."
))

# ── 4. FEATURE ENGINEERING ────────────────────────────────────────────────────
story.append(H1("4. Feature Engineering (267 features)"))
story.append(P(
    "El feature engineering es el núcleo del sistema. Se extraen 267 features distribuidas en "
    "13 grupos que capturan distintas dimensiones del contenido textual y el contexto empresarial:"
))
story.append(SP(6))
story.append(feature_table())
story.append(SP(8))
story.append(H2("4.1 Similitudes semánticas JDI / ODI"))
story.append(P(
    "Se calculan similitudes coseno entre el embedding de pros/cons del empleado y vectores "
    "de referencia de dos marcos teóricos: el <b>Job Descriptive Index (JDI)</b> con 5 dimensiones "
    "(trabajo en sí, salario, promoción, supervisión, compañeros) y el <b>Organizational "
    "Depression Index (ODI)</b> con 9 dimensiones clínicas (ánimo depresivo, alteraciones del "
    "sueño, fatiga, etc.). Los embeddings se generan con el modelo "
    "<b>all-roberta-large-v1</b> (1024 dimensiones)."
))

# ── 5. MODELO ─────────────────────────────────────────────────────────────────
story.append(H1("5. Modelo de Machine Learning"))
story.append(H2("5.1 Selección de modelo"))
story.append(P(
    "Se evaluaron múltiples algoritmos (ElasticNet, K-Nearest Neighbors, Decision Tree, "
    "Random Forest, XGBoost) mediante validación cruzada con 5 folds, optimizando el "
    "<b>neg_root_mean_squared_error</b>. XGBoost Regressor fue seleccionado como modelo final "
    "por su rendimiento superior y capacidad para capturar interacciones no lineales entre features."
))
story.append(H2("5.2 Hiperparámetros del modelo final"))

hp_data = [
    ["Parámetro", "Valor"],
    ["n_estimators", "500"],
    ["max_depth", "9"],
    ["learning_rate", "0.05"],
    ["objective", "reg:squarederror"],
    ["tree_method", "hist"],
    ["random_state", "8"],
]
hp_t = Table(hp_data, colWidths=[6*cm, 5*cm])
hp_t.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,0),  ACCENT),
    ("TEXTCOLOR",     (0,0), (-1,0),  white),
    ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0,0), (-1,-1), 9),
    ("FONTNAME",      (0,1), (-1,-1), "Courier"),
    ("TEXTCOLOR",     (0,1), (-1,-1), MIDGRAY),
    ("ROWBACKGROUNDS",(0,1), (-1,-1), [white, LIGHTGRAY]),
    ("GRID",          (0,0), (-1,-1), 0.4, HexColor("#cccccc")),
    ("TOPPADDING",    (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING",   (0,0), (-1,-1), 8),
]))
story.append(hp_t)
story.append(SP(10))

story.append(H2("5.3 Features más importantes (Top 5)"))
for rank, feat, imp in [
    ("1º", "cons_sim_ODI_Worthlessness",   "0.0847"),
    ("2º", "cons_charlen",                 "0.0508"),
    ("3º", "pros_nltk_sia",                "0.0318"),
    ("4º", "cons_emolex_disgust",          "0.0207"),
    ("5º", "pros_charlen",                 "0.0194"),
]:
    story.append(B(f"<b>{rank}</b> {feat} — importancia: {imp}"))

story.append(SP(8))
story.append(H2("5.4 Nota sobre serialización"))
story.append(P(
    "El modelo fue entrenado con sklearn 1.7.2 y XGBoost 2.x en el entorno de entrenamiento. "
    "Al cargarlo con sklearn 1.4.2 (entorno local), el base_score de XGBoost no se restaura "
    "correctamente desde el pickle, produciendo un desplazamiento sistemático en las predicciones. "
    "Esto se diagnosticó y documentó durante el desarrollo del pipeline de producción."
))

# ── 6. PIPELINE DE PRODUCTO ───────────────────────────────────────────────────
story.append(H1("6. Pipeline de Producto en Tiempo Real"))
story.append(P(
    "El sistema de demo en tiempo real implementa un pipeline end-to-end completamente automatizado "
    "que conecta la recogida de feedback del empleado con la visualización de la predicción en "
    "un dashboard analítico:"
))
story.append(SP(6))
story.append(Paragraph(
    "Tally.so  →  ngrok  →  n8n  →  FastAPI /predict  →  gold.xlsx  →  Tableau Desktop",
    style("flow", fontName="Courier-Bold", fontSize=11, textColor=ACCENT,
          alignment=TA_CENTER, spaceAfter=12, spaceBefore=6,
          backColor=HexColor("#f0f4ff"), borderPadding=10)
))
story.append(SP(6))
story.append(arch_table())
story.append(SP(10))

story.append(H2("6.1 Flujo de datos"))
for step, desc in [
    ("1. Empleado", "Rellena el formulario Tally.so con: summary, pros, cons, y 6 ratings numéricos (1–5)."),
    ("2. Tally → n8n", "Tally envía un webhook POST al endpoint público de ngrok que tuneliza a n8n (localhost:5678)."),
    ("3. n8n Parse", "El nodo 'Parse Tally Fields' extrae los campos del JSON de Tally y los normaliza."),
    ("4. n8n → FastAPI", "HTTP POST a localhost:8000/predict con los campos de texto y ratings."),
    ("5. Feature Engineering", "La FastAPI computa 267 features en tiempo real: POS tags, EmoLex, Empath, VADER, similitudes JDI/ODI, datos de empresa y mercado."),
    ("6. Predicción XGBoost", "El modelo infiere el overall_rating (1–5) y binariza: ≥4 → Alto, <4 → Bajo."),
    ("7. Append Excel", "POST a /append-gold: openpyxl añade la nueva fila al gold.xlsx."),
    ("8. Tableau", "Cmd+Shift+F5 refresca la conexión Live al Excel y la nueva predicción aparece en el dashboard."),
]:
    story.append(B(f"<b>{step}:</b> {desc}"))

story.append(SP(10))
story.append(H2("6.2 Campos del formulario Tally"))
for field, tipo, desc in [
    ("summary",              "Texto corto",  "Resumen de la experiencia del empleado"),
    ("pros",                 "Texto largo",  "Lo mejor de trabajar en la empresa"),
    ("cons",                 "Texto largo",  "Aspectos a mejorar"),
    ("wl_balance",           "Escala 1–5",   "Work-life balance"),
    ("culture_values",       "Escala 1–5",   "Cultura y valores"),
    ("diversity_inclusion",  "Escala 1–5",   "Diversidad e inclusión"),
    ("career_opportunities", "Escala 1–5",   "Oportunidades de carrera"),
    ("compensation_benefits","Escala 1–5",   "Compensación y beneficios"),
    ("senior_management",    "Escala 1–5",   "Alta dirección"),
    ("department",           "Desplegable",  "Departamento del empleado"),
    ("seniority",            "Desplegable",  "Nivel de seniority"),
    ("gender",               "Desplegable",  "Género (opcional)"),
]:
    story.append(B(f"<b>{field}</b> ({tipo}): {desc}"))

# ── 7. ESTRUCTURA DEL GOLD.XLSX ───────────────────────────────────────────────
story.append(H1("7. Estructura del Gold Dataset"))
story.append(P(
    "El fichero <b>gold.xlsx</b> actúa como repositorio central de predicciones, combinando "
    "945 reviews históricas de NVIDIA (batch predict) con las nuevas submissions en tiempo real. "
    "Sirve como fuente de datos para Tableau Desktop."
))
cols_data = [
    ["Columna", "Tipo", "Descripción"],
    ["timestamp",              "Datetime", "Fecha y hora de la predicción (ISO 8601)"],
    ["submission_date",        "Date",     "Fecha de la review"],
    ["summary",                "String",   "Texto del resumen"],
    ["pros",                   "String",   "Texto de pros"],
    ["cons",                   "String",   "Texto de cons"],
    ["department",             "String",   "Departamento del empleado"],
    ["seniority",              "String",   "Nivel de seniority / job title"],
    ["gender",                 "String",   "Género"],
    ["wl_balance",             "Float",    "Work-life balance (1–5)"],
    ["culture_values",         "Float",    "Cultura y valores (1–5)"],
    ["diversity_inclusion",    "Float",    "Diversidad e inclusión (1–5)"],
    ["career_opportunities",   "Float",    "Oportunidades de carrera (1–5)"],
    ["compensation_benefits",  "Float",    "Compensación (1–5)"],
    ["senior_management",      "Float",    "Alta dirección (1–5)"],
    ["predicted_rating",       "Float",    "Predicción del modelo (1.0–5.0)"],
    ["satisfaction",           "String",   "Binarización: Alto (≥4) / Bajo (<4)"],
    ["confidence",             "Float",    "Distancia al umbral normalizada (0–1)"],
]
ct = Table(cols_data, colWidths=[4.5*cm, 2*cm, 10*cm])
ct.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,0),  ACCENT),
    ("TEXTCOLOR",     (0,0), (-1,0),  white),
    ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0,0), (-1,-1), 9),
    ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
    ("TEXTCOLOR",     (0,1), (-1,-1), MIDGRAY),
    ("ROWBACKGROUNDS",(0,1), (-1,-1), [white, LIGHTGRAY]),
    ("GRID",          (0,0), (-1,-1), 0.4, HexColor("#cccccc")),
    ("TOPPADDING",    (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("LEFTPADDING",   (0,0), (-1,-1), 7),
]))
story.append(ct)

# ── 8. STACK TECNOLÓGICO ──────────────────────────────────────────────────────
story.append(H1("8. Stack Tecnológico"))
tech_data = [
    ["Capa", "Tecnología", "Versión"],
    ["ML / Modelo",        "XGBoost Regressor",          "2.0.3"],
    ["Feature Engineering","NLTK, NRCLex, Empath, VADER", "—"],
    ["Embeddings",         "sentence-transformers",       "all-roberta-large-v1"],
    ["API",                "FastAPI + Uvicorn",           "0.115 / 0.32"],
    ["Orquestación",       "n8n",                        "2.27.4"],
    ["Datos",              "pandas, openpyxl",            "2.x / 3.1.4"],
    ["Formulario",         "Tally.so",                   "—"],
    ["Túnel",              "ngrok",                      "3.39.8"],
    ["Visualización",      "Tableau Desktop",            "—"],
    ["Runtime",            "Python 3.11 / Node.js 20",   "—"],
    ["Entorno",            "Anaconda (macOS)",            "—"],
]
tt = Table(tech_data, colWidths=[4.5*cm, 7*cm, 5*cm])
tt.setStyle(TableStyle([
    ("BACKGROUND",    (0,0), (-1,0),  ACCENT),
    ("TEXTCOLOR",     (0,0), (-1,0),  white),
    ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0,0), (-1,-1), 9),
    ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
    ("TEXTCOLOR",     (0,1), (-1,-1), MIDGRAY),
    ("ROWBACKGROUNDS",(0,1), (-1,-1), [white, LIGHTGRAY]),
    ("GRID",          (0,0), (-1,-1), 0.4, HexColor("#cccccc")),
    ("TOPPADDING",    (0,0), (-1,-1), 5),
    ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ("LEFTPADDING",   (0,0), (-1,-1), 8),
]))
story.append(tt)

# ── 9. LIMITACIONES ───────────────────────────────────────────────────────────
story.append(H1("9. Limitaciones y Trabajo Futuro"))
story.append(H2("9.1 Limitaciones actuales"))
for lim in [
    "El modelo fue entrenado exclusivamente con datos de empresas del S&P 500 estadounidenses; "
    "su generalización a otros mercados o sectores requiere validación adicional.",
    "Las features de empresa (Market Cap, sector) están hardcodeadas para NVIDIA en el pipeline "
    "de tiempo real; para otras empresas sería necesario parametrizar el endpoint.",
    "La incompatibilidad entre sklearn 1.7.2 (entrenamiento) y 1.4.2 (producción) genera un "
    "desplazamiento en el base_score de XGBoost que afecta a la escala absoluta de las predicciones.",
    "El dashboard requiere refresh manual en Tableau Desktop; una solución de producción utilizaría "
    "Tableau Server o Tableau Cloud con scheduled extracts.",
    "ngrok en plan gratuito genera una URL distinta en cada reinicio, lo que requiere actualizar "
    "el webhook de Tally.so.",
]:
    story.append(B(lim))

story.append(H2("9.2 Trabajo futuro"))
for fut in [
    "Despliegue del modelo en Google Cloud Vertex AI para mayor escalabilidad y latencia controlada.",
    "Entrenamiento de modelos específicos por empresa o sector para mejorar la precisión.",
    "Implementación de fine-tuning de modelos de lenguaje (BERT, RoBERTa) como alternativa "
    "al feature engineering manual.",
    "Integración de análisis de sentimiento a nivel de aspecto (ABSA) para descomponer la "
    "satisfacción por dimensión JDI/ODI.",
    "Dashboard en Tableau Cloud con refresh automático y alertas en tiempo real.",
]:
    story.append(B(fut))

# ── 10. CONCLUSIONES ──────────────────────────────────────────────────────────
story.append(H1("10. Conclusiones"))
story.append(P(
    "Este trabajo demuestra la viabilidad de construir un sistema de predicción de satisfacción "
    "laboral en tiempo real combinando NLP clásico, embeddings contextuales y modelos de gradient "
    "boosting. El pipeline implementado va más allá del prototipo académico: es un producto "
    "funcional con integración end-to-end entre formulario web, API de predicción y dashboard "
    "analítico, ejecutándose en tiempo real con una latencia inferior a 2 segundos."
))
story.append(P(
    "La aplicación a NVIDIA demuestra que el modelo captura patrones semánticos significativos "
    "en las reviews de empleados: la importancia de features como <i>cons_sim_ODI_Worthlessness</i> "
    "o <i>cons_emolex_disgust</i> revela que el modelo ha aprendido a identificar señales de "
    "malestar laboral directamente del texto, sin necesidad de labels explícitos sobre "
    "dimensiones psicológicas."
))
story.append(P(
    "El pipeline de producto valida la arquitectura propuesta como base para un sistema de "
    "monitorización continua del clima laboral corporativo, con potencial de escalado a cualquier "
    "empresa con suficiente volumen de feedback textual de empleados."
))

story.append(SP(20))
story.append(HRFlowable(width="100%", thickness=1, color=HexColor("#cccccc")))
story.append(SP(6))
story.append(Paragraph(
    f"Documento generado automáticamente · {date.today().strftime('%d %B %Y')} · TFM Data Analytics",
    style("foot", fontName="Helvetica-Oblique", fontSize=8,
          textColor=HexColor("#aaaaaa"), alignment=TA_CENTER)
))

# ── GENERAR ───────────────────────────────────────────────────────────────────
doc.build(story)
print(f"PDF generado: {OUTPUT}")
