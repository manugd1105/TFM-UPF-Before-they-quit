"""
submit_all.py
Punto de entrada único: sube los datos y lanza todos los jobs de Vertex AI
para los grupos 2-6 (los que tienen features afectadas por BigQuery).

USO:
  export GCP_PROJECT="tu-nuevo-proyecto"
  export GCP_BUCKET="tu-nuevo-bucket"
  gcloud auth application-default login
  gcloud auth application-default set-quota-project $GCP_PROJECT
  python submit_all.py

El script:
  1. Sube los CSVs locales al bucket (si no existen ya)
  2. Compila y sube el paquete de cada grupo
  3. Lanza el Custom Job en Vertex AI
  4. Espera 30s entre jobs para no saturar las APIs

Grupo 1 se omite (solo 6 features de longitud de string, no afectadas por BQ).
"""

import os
import sys
import time
import importlib.util

PROJECT_ID  = os.environ.get("GCP_PROJECT", "CHANGE_ME")
BUCKET_NAME = os.environ.get("GCP_BUCKET",  "CHANGE_ME")

VERTEX_DIR = os.path.dirname(os.path.abspath(__file__))


def load_submit(script_name):
    path = os.path.join(VERTEX_DIR, script_name)
    spec = importlib.util.spec_from_file_location("submit_mod", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    if "CHANGE_ME" in (PROJECT_ID, BUCKET_NAME):
        print("ERROR: define GCP_PROJECT y GCP_BUCKET como variables de entorno antes de ejecutar.")
        print("  export GCP_PROJECT='tu-nuevo-proyecto'")
        print("  export GCP_BUCKET='tu-nuevo-bucket'")
        sys.exit(1)

    print(f"PROJECT  : {PROJECT_ID}")
    print(f"BUCKET   : {BUCKET_NAME}")
    print()

    # ── Paso 1: subir datos ──────────────────────────────────────
    print("=" * 60)
    print("PASO 1: Subiendo CSVs locales a GCS...")
    print("=" * 60)
    import upload_data
    upload_data.main()
    print()

    # ── Paso 2: lanzar jobs ──────────────────────────────────────
    jobs = [
        ("GRUPO 2 (334 features + PCA, 19 ejecuciones)",   "submit_grupo2.py"),
        ("GRUPO 3 (NLP features + PCA, 24 ejecuciones)",   "submit_group3.py"),
        ("GRUPO 4 (NLP+JDI/ODI + PCA, 24 ejecuciones)",   "submit_grupo4.py"),
        ("GRUPO 5 (334 features sin PCA, 6 ejecuciones)",  "submit_grupo5.py"),
        ("GRUPO 6 (LightGBM, 13 ejecuciones)",             "submit_grupo6.py"),
    ]

    for label, script in jobs:
        print("=" * 60)
        print(f"LANZANDO: {label}")
        print("=" * 60)
        mod = load_submit(script)
        mod.build_package()
        mod.upload_package()
        mod.submit_job()
        print(f"✓ Job lanzado: {label}")
        print()
        time.sleep(30)

    print("=" * 60)
    print("TODOS LOS JOBS LANZADOS")
    print(f"Monitoriza en:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print("=" * 60)


if __name__ == "__main__":
    main()
