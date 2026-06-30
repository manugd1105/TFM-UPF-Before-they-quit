"""
upload_ftse100_data.py
─────────────────────
Sube los CSVs de FTSE100 (con EmoLex ya arreglado) a GCS.
  gs://<GCP_BUCKET>/datos_entrada/dataframes_Train_ftse100.csv
  gs://<GCP_BUCKET>/datos_entrada/dataframes_Test_ftse100.csv
"""

import os
from pathlib import Path
from google.cloud import storage

PROJECT_ID  = os.environ.get("GCP_PROJECT", "CHANGE_ME")
BUCKET_NAME = os.environ.get("GCP_BUCKET",  "CHANGE_ME")

# CSVs deben estar en ftse100/data/ (ver ftse100/data/README.md)
_DATA_DIR = Path(__file__).resolve().parent.parent / "ftse100" / "data"
FILES = {
    str(_DATA_DIR / "dataframes_Train_scrapeadas.csv"):
        "datos_entrada/dataframes_Train_ftse100.csv",
    str(_DATA_DIR / "dataframes_Test_scrapeadas.csv"):
        "datos_entrada/dataframes_Test_ftse100.csv",
}

client = storage.Client(project=PROJECT_ID)
bucket = client.bucket(BUCKET_NAME)

for local, gcs_path in FILES.items():
    size_mb = os.path.getsize(local) / 1_048_576
    print(f"Subiendo {os.path.basename(local)} ({size_mb:.0f} MB) → gs://{BUCKET_NAME}/{gcs_path}")
    bucket.blob(gcs_path).upload_from_filename(local)
    print(f"  OK")

print("\nFTSE100 data subida a GCS.")
