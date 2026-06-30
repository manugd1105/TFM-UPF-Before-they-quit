"""
upload_data.py
Sube los CSVs de entrenamiento/test preprocesados al nuevo bucket de GCS.
Usa el CSV local directamente (sin pasar por BigQuery) para evitar la sanitización
de nombres de columna que causaba el problema anterior.

USO:
  export GCP_PROJECT="tu-nuevo-proyecto"
  export GCP_BUCKET="tu-nuevo-bucket"
  python upload_data.py
"""

import os
import sys
from google.cloud import storage

PROJECT_ID  = os.environ.get("GCP_PROJECT", "CHANGE_ME")
BUCKET_NAME = os.environ.get("GCP_BUCKET",  "CHANGE_ME")

EXECUTION_ID = "20240322_105534"
LOCAL_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "03_outputs_data_preprocessing",
    f"output_data_preprocessing_{EXECUTION_ID}",
)

FILES = [
    f"dataframes_Train_{EXECUTION_ID}.csv",
    f"dataframes_Test_{EXECUTION_ID}.csv",
]

GCS_PREFIX = "datos_entrada"


def main():
    if "CHANGE_ME" in (PROJECT_ID, BUCKET_NAME):
        print("ERROR: define GCP_PROJECT y GCP_BUCKET como variables de entorno antes de ejecutar.")
        sys.exit(1)

    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)

    for fname in FILES:
        local_path = os.path.join(LOCAL_DATA_DIR, fname)
        if not os.path.exists(local_path):
            print(f"ERROR: no encontrado {local_path}")
            sys.exit(1)

        gcs_path = f"{GCS_PREFIX}/{fname}"
        blob = bucket.blob(gcs_path)

        if blob.exists():
            print(f"  YA EXISTE: gs://{BUCKET_NAME}/{gcs_path} — saltando")
            continue

        size_mb = os.path.getsize(local_path) / 1024 / 1024
        print(f"  Subiendo {fname} ({size_mb:.0f} MB) → gs://{BUCKET_NAME}/{gcs_path} ...")
        blob.upload_from_filename(local_path)
        print(f"  OK")

    print(f"\nDatos disponibles en gs://{BUCKET_NAME}/{GCS_PREFIX}/")


if __name__ == "__main__":
    main()
