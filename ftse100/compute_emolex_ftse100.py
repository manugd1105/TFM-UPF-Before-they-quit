"""
fix_emolex_ftse100.py
─────────────────────
Computa features EmoLex (NRC Emotion Lexicon) para los datasets FTSE100
y sobreescribe las 20 columnas pros/cons_emolex_* que estaban todas a 0.

Usa nrclex v4 con load_raw_text (tokenización + lemmatización via TextBlob).
La proporción de cada emoción = count_emoción / total_etiquetas_emocionales.
"""

import time
from pathlib import Path
import pandas as pd
from nrclex import NRCLex

# ── Configuración ──────────────────────────────────────────────────────────────
# Los CSVs deben estar en ftse100/data/ (ver ftse100/data/README.md)
_DATA_DIR = Path(__file__).resolve().parent / "data"
FILES = {
    "train": str(_DATA_DIR / "dataframes_Train_scrapeadas.csv"),
    "test":  str(_DATA_DIR / "dataframes_Test_scrapeadas.csv"),
}

EMOTIONS = [
    "fear", "anger", "anticip", "trust", "surprise",
    "positive", "negative", "sadness", "disgust", "joy",
]
NRC_KEYS = [
    "fear", "anger", "anticipation", "trust", "surprise",
    "positive", "negative", "sadness", "disgust", "joy",
]
EMOLEX_COLS = (
    [f"pros_emolex_{e}" for e in EMOTIONS] +
    [f"cons_emolex_{e}" for e in EMOTIONS]
)

# ── Función de scoring ─────────────────────────────────────────────────────────
_lex = NRCLex()          # instancia reutilizable (evita recargar lexicón cada vez)

def emolex_row(pros: str, cons: str) -> list:
    """Devuelve 20 valores EmoLex: primero pros (10) luego cons (10)."""
    result = []
    for text in (pros, cons):
        if not isinstance(text, str) or not text.strip():
            result.extend([0.0] * 10)
            continue
        _lex.load_raw_text(text)
        freqs = _lex.affect_frequencies
        result.extend(freqs.get(k, 0.0) for k in NRC_KEYS)
    return result


# ── Procesamiento ──────────────────────────────────────────────────────────────
for split, path in FILES.items():
    print(f"\n{'='*60}")
    print(f"  Procesando {split}: {path}")
    t0 = time.time()

    df = pd.read_csv(path, low_memory=False)
    n  = len(df)
    print(f"  Filas: {n:,}")

    # Sanity-check: confirmar que estaban a 0
    zero_before = (df[EMOLEX_COLS] == 0).all().all()
    print(f"  EmoLex todos a 0 antes del fix: {zero_before}")

    # Computar en batches con progreso
    BATCH = 5_000
    all_scores = []

    for start in range(0, n, BATCH):
        batch = df.iloc[start : start + BATCH]
        scores = [
            emolex_row(row["pros"], row["cons"])
            for _, row in batch[["pros", "cons"]].iterrows()
        ]
        all_scores.extend(scores)
        elapsed = time.time() - t0
        done = start + len(batch)
        pct  = done / n * 100
        rate = done / elapsed if elapsed > 0 else 1
        eta  = (n - done) / rate if rate > 0 else 0
        print(f"  {pct:5.1f}%  |  {done:>7,}/{n:,}  |  "
              f"{elapsed:5.1f}s  |  ETA {eta:.0f}s", end="\r", flush=True)

    print()

    # Asignar columnas
    scores_df = pd.DataFrame(all_scores, columns=EMOLEX_COLS)
    df[EMOLEX_COLS] = scores_df.values

    # Verificar
    nonzero = (df[EMOLEX_COLS] != 0).any(axis=1).sum()
    print(f"  Filas con algun EmoLex != 0: {nonzero:,} / {n:,} ({nonzero/n*100:.1f}%)")
    print(f"  pros_emolex_trust  media: {df['pros_emolex_trust'].mean():.5f}")
    print(f"  cons_emolex_negative media: {df['cons_emolex_negative'].mean():.5f}")

    # Guardar
    df.to_csv(path, index=False)
    print(f"  Guardado. Tiempo total: {time.time()-t0:.1f}s")

print("\nFix EmoLex FTSE100 completado.")
