"""
generate_crosstraining_report.py
─────────────────────────────────
Genera un PDF con los resultados de los experimentos de cross-training
SP500 × FTSE100.
"""

import io
import json
import os
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from google.cloud import storage

# ── Datos ──────────────────────────────────────────────────────────────────────
PROJECT_ID = os.environ.get("GCP_PROJECT", "CHANGE_ME")
BUCKET_NAME = os.environ.get("GCP_BUCKET", "CHANGE_ME")
client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

TIMESTAMPS = {
    "B_1":  "20260614_045436",
    "C_1":  "20260614_072301",
    "D_1":  "20260614_085354",
    "E_1":  "20260614_115115",
    "FG_1": "20260614_144022",
}

def load_config(exp):
    ts = TIMESTAMPS[exp]
    blob = bucket.blob(f"cross_training/{exp}/config_{ts}.json")
    return json.loads(blob.download_as_text()), ts

def load_fi(exp):
    ts = TIMESTAMPS[exp]
    blob = bucket.blob(f"cross_training/{exp}/feature_importance_{ts}.csv")
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))

configs = {exp: load_config(exp)[0] for exp in TIMESTAMPS}
fi_dfs  = {exp: load_fi(exp) for exp in TIMESTAMPS}

# ── Colores ────────────────────────────────────────────────────────────────────
BLUE   = "#2196F3"
GREEN  = "#4CAF50"
ORANGE = "#FF9800"
RED    = "#F44336"
PURPLE = "#9C27B0"
GREY   = "#757575"
DARK   = "#212121"

C_MAP = {
    "A":    "#546E7A",
    "B-1":  BLUE,
    "C-1":  GREEN,
    "D-1":  ORANGE,
    "E-1":  PURPLE,
    "F":    "#00BCD4",
    "G":    "#795548",
}

OUTPUT = str(Path(__file__).resolve().parent.parent / "results/Cross_Training_Results.pdf")

# ── Tabla maestra ──────────────────────────────────────────────────────────────
RESULTS = [
    # (label, train_desc, n_train, test_desc, mape, mse, gap_train_test, color)
    ("A  (baseline)",    "SP500",              "285k", "SP500 (71k)",       0.2549, 0.6802, 0.055,  C_MAP["A"]),
    ("B-1",              "FTSE train",         "108k", "FTSE test (27k)",   0.2479, 0.6810, 0.052,  C_MAP["B-1"]),
    ("C-1  SP500→FTSE",  "SP500",              "285k", "FTSE full (135k)",  0.2438, 0.7558, 0.047,  C_MAP["C-1"]),
    ("C-1  SP500→SP500", "SP500",              "285k", "SP500 (71k)",       0.2557, 0.6824, 0.059,  C_MAP["C-1"]),
    ("D-1  FTSE→SP500",  "FTSE full",          "135k", "SP500 full (357k)", 0.2675, 0.7956, 0.104,  C_MAP["D-1"]),
    ("E-1",              "SP500 + FTSE full",  "421k", "SP500 (71k)",       0.2546, 0.6784, 0.051,  C_MAP["E-1"]),
    ("F  (FG-1)",        "SP500 + FTSE train", "394k", "FTSE test (27k)",   0.2453, 0.6704, 0.042,  C_MAP["F"]),
    ("G  (FG-1)",        "SP500 + FTSE train", "394k", "SP500 (71k)",       0.2547, 0.6799, 0.052,  C_MAP["G"]),
    ("G  global",        "SP500 + FTSE train", "394k", "SP500+FTSE (98k)",  0.2521, 0.6773, 0.049,  C_MAP["G"]),
]

FINDINGS = [
    ("1. FTSE funciona bien en su propio mercado  (B-1: 0.2479)",
     "Con solo 108k reviews y sin features de estado geográfico, el modelo UK supera\n"
     "al baseline SP500 (0.2549). El NLP captura bien los patrones de satisfacción\n"
     "incluso con menos datos."),

    ("2. SP500 → FTSE generaliza sorprendentemente bien  (C-1: 0.2438)",
     "El modelo entrenado en empresas americanas predice satisfacción en UK mejor\n"
     "que el modelo nativo UK. El mayor volumen de entrenamiento (285k vs 108k) supera\n"
     "la diferencia de mercado. Los patrones lingüísticos son universales."),

    ("3. Transferencia asimétrica: US→UK mejor que UK→US  (D-1: 0.2675)",
     "FTSE→SP500 pierde 0.013 MAPE respecto al baseline, mientras SP500→FTSE gana\n"
     "0.004. La asimetría se explica por el diferencial de datos de entrenamiento\n"
     "(135k vs 285k), no por diferencias estructurales entre mercados."),

    ("4. Añadir FTSE no mejora SP500  (E-1: 0.2546 ≈ A: 0.2549)",
     "Con 421k filas combinadas, la mejora en SP500 es inapreciable (+0.0003).\n"
     "El modelo SP500 ya estaba bien optimizado; los datos UK no añaden señal nueva\n"
     "para el mercado US."),

    ("5. Añadir SP500 mejora levemente FTSE  (F: 0.2453 vs B-1: 0.2479)",
     "El modelo combinado (394k) gana 0.003 MAPE sobre el modelo nativo FTSE.\n"
     "Los datos SP500 actúan como regularización: el modelo aprende patrones\n"
     "NLP más robustos y generalizables."),

    ("6. Un modelo global funciona bien en ambos mercados  (G: 0.2521)",
     "Un único XGBoost entrenado en SP500+FTSE obtiene MAPE=0.2521 sobre el test\n"
     "combinado. Los mecanismos de satisfacción laboral en texto inglés son\n"
     "suficientemente universales para un modelo unificado."),
]


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#F8F9FA")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.tick_params(colors=DARK, labelsize=9)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold", color=DARK, pad=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9, color=GREY)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=GREY)
    ax.grid(axis="y", color="#EEEEEE", linewidth=0.8)


with PdfPages(OUTPUT) as pdf:

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINA 1 — Portada + tabla
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.patch.set_facecolor("white")

    # Header strip
    ax_hdr = fig.add_axes([0, 0.88, 1, 0.12])
    ax_hdr.set_facecolor(DARK)
    ax_hdr.axis("off")
    ax_hdr.text(0.5, 0.65, "Cross-Training SP500 × FTSE100 — Resultados completos",
                ha="center", va="center", fontsize=16, fontweight="bold",
                color="white", transform=ax_hdr.transAxes)
    ax_hdr.text(0.5, 0.20, "XGBoost  ·  267 features (group5 sin estado)  ·  n_iter=200  ·  CV=5  ·  Vertex AI n2-highmem-16",
                ha="center", va="center", fontsize=9, color="#BBBBBB",
                transform=ax_hdr.transAxes)

    # Table
    ax_tbl = fig.add_axes([0.02, 0.04, 0.96, 0.82])
    ax_tbl.axis("off")

    col_labels = ["Experimento", "Train data", "N train", "Test data", "MAPE test", "MSE test", "Gap train/test"]
    cell_data = [
        [r[0], r[1], r[2], r[3], f"{r[4]:.4f}", f"{r[5]:.4f}", f"{r[6]:.3f}"]
        for r in RESULTS
    ]

    tbl = ax_tbl.table(
        cellText=cell_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 2.1)

    # Style header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor(DARK)
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Style rows
    baseline_mape = 0.2549
    for i, row in enumerate(RESULTS):
        mape = row[4]
        color = row[7]
        for j in range(len(col_labels)):
            cell = tbl[(i + 1, j)]
            cell.set_facecolor("#F0F4FF" if i % 2 == 0 else "white")
            if j == 4:  # MAPE column — highlight good/bad
                if mape < baseline_mape - 0.003:
                    cell.set_facecolor("#C8E6C9")  # green
                    cell.set_text_props(fontweight="bold", color="#1B5E20")
                elif mape > baseline_mape + 0.010:
                    cell.set_facecolor("#FFCCBC")  # red-ish
                    cell.set_text_props(color="#BF360C")
                else:
                    cell.set_text_props(fontweight="bold")

    # Baseline annotation
    ax_tbl.text(0.01, 0.01, "* Baseline A: MAPE=0.2549 (SP500 group5 completo con state features)  "
                "| Verde = mejor que baseline  |  Naranja = peor",
                fontsize=7.5, color=GREY, transform=ax_tbl.transAxes)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINA 2 — Gráficos comparativos
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.suptitle("Comparativa de MAPE por experimento y mercado", fontsize=13,
                 fontweight="bold", color=DARK, y=0.97)

    # — Gráfico 1: MAPE sobre FTSE100 test —
    ax = axes[0]
    ftse_exps  = ["B-1\nFTSE→FTSE", "C-1\nSP500→FTSE", "F\nSP500+FTSE→FTSE"]
    ftse_mapes = [0.2479, 0.2438, 0.2453]
    ftse_colors = [C_MAP["B-1"], C_MAP["C-1"], C_MAP["F"]]
    bars = ax.bar(ftse_exps, ftse_mapes, color=ftse_colors, width=0.5, edgecolor="white", linewidth=1.5)
    ax.axhline(0.2549, color=GREY, linestyle="--", linewidth=1, label="Baseline A (SP500)")
    for bar, val in zip(bars, ftse_mapes):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.002, f"{val:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(0.22, 0.29)
    ax.legend(fontsize=8)
    style_ax(ax, title="Test en FTSE100", ylabel="MAPE")

    # — Gráfico 2: MAPE sobre SP500 test —
    ax = axes[1]
    sp_exps  = ["A\nSP500→SP500", "C-1 ref\nSP500→SP500\n(sin estado)", "E-1\nSP500+FTSE\n→SP500", "G\nSP500+FTSE\n→SP500", "D-1\nFTSE→SP500"]
    sp_mapes = [0.2549, 0.2557, 0.2546, 0.2547, 0.2675]
    sp_colors = [C_MAP["A"], C_MAP["C-1"], C_MAP["E-1"], C_MAP["G"], C_MAP["D-1"]]
    bars = ax.bar(sp_exps, sp_mapes, color=sp_colors, width=0.5, edgecolor="white", linewidth=1.5)
    ax.axhline(0.2549, color=GREY, linestyle="--", linewidth=1)
    for bar, val in zip(bars, sp_mapes):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.001, f"{val:.4f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_ylim(0.22, 0.29)
    ax.tick_params(axis="x", labelsize=7)
    style_ax(ax, title="Test en SP500", ylabel="MAPE")

    # — Gráfico 3: Gap train/test (overfitting) —
    ax = axes[2]
    gap_labels = ["A", "B-1", "C-1\n(FTSE)", "C-1\n(SP500)", "D-1", "E-1", "F", "G"]
    gap_trains = [0.2549-0.055, 0.1959, 0.1967, 0.1967, 0.1631, 0.2034, 0.2030, 0.2030]
    gap_tests  = [0.2549, 0.2479, 0.2438, 0.2557, 0.2675, 0.2546, 0.2453, 0.2547]
    gap_colors_list = [C_MAP["A"], C_MAP["B-1"], C_MAP["C-1"], C_MAP["C-1"],
                       C_MAP["D-1"], C_MAP["E-1"], C_MAP["F"], C_MAP["G"]]
    x = np.arange(len(gap_labels))
    w = 0.35
    bars1 = ax.bar(x - w/2, gap_trains, w, label="Train", color=[c + "AA" for c in gap_colors_list],
                   edgecolor="white")
    bars2 = ax.bar(x + w/2, gap_tests,  w, label="Test",  color=gap_colors_list, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(gap_labels, fontsize=8)
    ax.set_ylim(0.10, 0.31)
    ax.legend(fontsize=8)
    style_ax(ax, title="MAPE train vs test (overfitting)", ylabel="MAPE")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINA 3 — Feature importances
    # ══════════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    fig.suptitle("Top 10 Feature Importances por experimento (XGBoost gain)",
                 fontsize=13, fontweight="bold", color=DARK, y=0.98)

    exp_labels = {
        "B_1": "B-1: FTSE → FTSE",
        "C_1": "C-1: SP500 → FTSE",
        "D_1": "D-1: FTSE → SP500",
        "E_1": "E-1: SP500+FTSE → SP500",
        "FG_1": "FG-1: SP500+FTSE → FTSE+SP500",
    }
    exp_colors = {
        "B_1": C_MAP["B-1"],
        "C_1": C_MAP["C-1"],
        "D_1": C_MAP["D-1"],
        "E_1": C_MAP["E-1"],
        "FG_1": C_MAP["F"],
    }

    exp_order = ["B_1", "C_1", "D_1", "E_1", "FG_1"]
    axs_flat = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]]
    axes[1, 2].axis("off")

    def shorten(name):
        replacements = {
            "cons_sim_ODI_": "cons·ODI·",
            "pros_sim_ODI_": "pros·ODI·",
            "cons_sim_JDI_": "cons·JDI·",
            "pros_sim_JDI_": "pros·JDI·",
            "cons_emolex_": "cons·EL·",
            "pros_emolex_": "pros·EL·",
            "cons_empath_": "cons·EP·",
            "pros_empath_": "pros·EP·",
            "cons_nltk_sia": "cons·NLTK",
            "pros_nltk_sia": "pros·NLTK",
            "cons_charlen": "cons·charlen",
            "cons_wordlen": "cons·wordlen",
            "pros_charlen": "pros·charlen",
            "pros_wordlen": "pros·wordlen",
            "Sector_targenc_mean": "Sector·targenc·mean",
        }
        for k, v in replacements.items():
            name = name.replace(k, v)
        return name[:35]

    for ax, exp in zip(axs_flat, exp_order):
        fi = fi_dfs[exp].head(10)
        feats = [shorten(f) for f in fi["feature"]]
        imps  = fi["importance"].values
        color = exp_colors[exp]
        bars  = ax.barh(range(len(feats)), imps, color=color, alpha=0.85, edgecolor="white")
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlim(0, imps[0] * 1.25)
        for bar, val in zip(bars, imps):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va="center", fontsize=7, color=DARK)
        style_ax(ax, title=exp_labels[exp], xlabel="Importance (gain)")

    # Legend box in empty subplot
    ax_leg = axes[1, 2]
    ax_leg.axis("off")
    legend_text = (
        "Abreviaturas:\n\n"
        "cons·ODI·X  →  similitud semántica JDI/ODI (contras)\n"
        "pros·ODI·X  →  similitud semántica JDI/ODI (pros)\n"
        "cons·EL·X   →  EmoLex NRC (contras)\n"
        "pros·EL·X   →  EmoLex NRC (pros)\n"
        "cons·EP·X   →  Empath category (contras)\n"
        "pros·EP·X   →  Empath category (pros)\n"
        "cons·NLTK   →  NLTK VADER sentiment (contras)\n"
        "pros·NLTK   →  NLTK VADER sentiment (pros)\n\n"
        "La feature más importante en todos\n"
        "los experimentos es:\n"
        "cons·ODI·Worthlessness\n"
        "(similitud semántica entre los\n"
        "contras y la dimensión ODI\n"
        "'sentimiento de inutilidad')"
    )
    ax_leg.text(0.05, 0.95, legend_text, transform=ax_leg.transAxes,
                fontsize=8.5, va="top", ha="left", color=DARK,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F4FF",
                          edgecolor="#BBBBBB", linewidth=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINA 4 — Hallazgos y conclusiones
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    ax_hdr = fig.add_axes([0, 0.91, 1, 0.09])
    ax_hdr.set_facecolor(DARK)
    ax_hdr.axis("off")
    ax_hdr.text(0.5, 0.5, "Hallazgos principales y conclusiones",
                ha="center", va="center", fontsize=15, fontweight="bold",
                color="white", transform=ax_hdr.transAxes)

    ax_body = fig.add_axes([0.04, 0.02, 0.92, 0.87])
    ax_body.axis("off")

    y = 0.97
    colors_find = [C_MAP["B-1"], C_MAP["C-1"], C_MAP["D-1"],
                   C_MAP["E-1"], C_MAP["F"], C_MAP["G"]]

    for (title, body), color in zip(FINDINGS, colors_find):
        # Colored bullet
        ax_body.add_patch(mpatches.FancyBboxPatch(
            (0, y - 0.005), 0.008, 0.038,
            boxstyle="round,pad=0.002",
            facecolor=color, edgecolor="none",
            transform=ax_body.transAxes))
        ax_body.text(0.015, y + 0.018, title,
                     transform=ax_body.transAxes,
                     fontsize=9.5, fontweight="bold", color=DARK, va="top")
        ax_body.text(0.015, y - 0.002, body,
                     transform=ax_body.transAxes,
                     fontsize=8.5, color="#444444", va="top",
                     linespacing=1.4)
        y -= 0.162

    # Summary box
    summary = (
        "Conclusión global:  Los patrones de satisfacción laboral capturados por NLP en reseñas Glassdoor en inglés son\n"
        "suficientemente universales para transferirse entre los mercados S&P 500 (US) y FTSE 100 (UK) con degradación mínima.\n"
        "La feature más predictiva en todos los escenarios — cons_sim_ODI_Worthlessness — apunta a que la intensidad con la\n"
        "que un empleado expresa sentimientos de inutilidad en los contras es el predictor más robusto y universal de satisfacción."
    )
    ax_body.text(0.5, 0.01, summary, transform=ax_body.transAxes,
                 ha="center", va="bottom", fontsize=8.5, color=DARK,
                 style="italic", linespacing=1.5,
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F5E9",
                           edgecolor="#4CAF50", linewidth=1.2))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ══════════════════════════════════════════════════════════════════════════
    # PÁGINA 5 — Resumen ejecutivo (tabla de preguntas/respuestas)
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")

    ax_hdr = fig.add_axes([0, 0.91, 1, 0.09])
    ax_hdr.set_facecolor("#1565C0")
    ax_hdr.axis("off")
    ax_hdr.text(0.5, 0.5, "Resumen ejecutivo — Preguntas de investigación",
                ha="center", va="center", fontsize=15, fontweight="bold",
                color="white", transform=ax_hdr.transAxes)

    ax_tbl = fig.add_axes([0.04, 0.05, 0.92, 0.84])
    ax_tbl.axis("off")

    qa_data = [
        ["¿Funciona FTSE100 en su propio mercado?",
         "Sí", "MAPE=0.2479 (B-1) — mejor que baseline SP500 (0.2549)"],
        ["¿Generaliza SP500 → UK?",
         "Sí, muy bien", "MAPE=0.2438 (C-1) — ¡mejor que el modelo nativo UK!"],
        ["¿Generaliza UK → US?",
         "Parcialmente", "MAPE=0.2675 (D-1) — degradación de +0.013 vs baseline"],
        ["¿El mix SP500+FTSE mejora SP500?",
         "No", "MAPE=0.2546 (E-1) — diferencia inapreciable (+0.0003)"],
        ["¿El mix SP500+FTSE mejora FTSE?",
         "Levemente", "MAPE=0.2453 (F) vs 0.2479 (B-1) — mejora de 0.003"],
        ["¿Un modelo global SP500+FTSE funciona?",
         "Sí", "MAPE=0.2521 (G) sobre test combinado SP500+FTSE"],
        ["¿Las features son universales?",
         "Sí", "cons_sim_ODI_Worthlessness es la #1 en todos los experimentos"],
        ["¿Hay overfitting problemático?",
         "No más de lo habitual", "Gap train/test ~0.05 en todos salvo D-1 (0.10 = domain shift)"],
        ["¿Las state features importan?",
         "Apenas", "C-1 SP500→SP500 sin state: 0.2557 vs A con state: 0.2549 (+0.0008)"],
    ]

    tbl2 = ax_tbl.table(
        cellText=qa_data,
        colLabels=["Pregunta de investigación", "Respuesta", "Métrica / Evidencia"],
        loc="center",
        cellLoc="left",
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(9)
    tbl2.scale(1, 2.5)
    tbl2.auto_set_column_width([0, 1, 2])

    for j in range(3):
        tbl2[(0, j)].set_facecolor("#1565C0")
        tbl2[(0, j)].set_text_props(color="white", fontweight="bold")

    resp_colors = {
        "Sí": "#C8E6C9", "Sí, muy bien": "#A5D6A7", "Levemente": "#DCEDC8",
        "Parcialmente": "#FFE0B2", "No": "#FFECB3",
        "No más de lo habitual": "#E8EAF6", "Apenas": "#E8EAF6",
    }
    for i, row in enumerate(qa_data):
        for j in range(3):
            cell = tbl2[(i+1, j)]
            bg = "#F8F9FA" if i % 2 == 0 else "white"
            cell.set_facecolor(bg)
        resp_cell = tbl2[(i+1, 1)]
        resp = row[1]
        resp_cell.set_facecolor(resp_colors.get(resp, "#F8F9FA"))
        resp_cell.set_text_props(fontweight="bold")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"PDF generado: {OUTPUT}")
