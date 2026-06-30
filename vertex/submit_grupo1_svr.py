"""
submit_grupo1_svr.py
Lanza solo 1-5 (SVR) con iteraciones reducidas (5 iter × 5 folds = 25 fits).
"""

import os
import subprocess
import sys

from google.cloud import aiplatform, storage

PROJECT_ID    = os.environ.get("GCP_PROJECT", "CHANGE_ME")
REGION        = "europe-southwest1"
BUCKET_NAME   = os.environ.get("GCP_BUCKET",  "CHANGE_ME")
PACKAGE_LOCAL = os.path.join(os.path.dirname(__file__), "grupo1_vertex", "dist", "trainer-0.1.tar.gz")
PACKAGE_DIR   = os.path.join(os.path.dirname(__file__), "grupo1_vertex")


def build_package():
    print("Empaquetando trainer grupo 1 SVR...")
    subprocess.run([sys.executable, "setup.py", "sdist"], cwd=PACKAGE_DIR, check=True)
    print(f"Paquete creado: {PACKAGE_LOCAL}")


def upload_package():
    print("Subiendo paquete a GCS...")
    client = storage.Client(project=PROJECT_ID)
    client.bucket(BUCKET_NAME).blob("packages/trainer_grupo1-0.1.tar.gz").upload_from_filename(PACKAGE_LOCAL)
    print("Paquete subido.")


def submit_job():
    print("Enviando Custom Job (Grupo 1 SVR) a Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")

    bootstrap = (
        "pip install --quiet xgboost gcsfs xlsxwriter google-cloud-storage "
        "scikit-learn pandas matplotlib numpy && "
        "python -c \""
        "from google.cloud import storage; "
        f"storage.Client().bucket('{BUCKET_NAME}').blob('packages/trainer_grupo1-0.1.tar.gz')"
        ".download_to_filename('/tmp/trainer_grupo1.tar.gz')"
        "\" && "
        "pip install --quiet /tmp/trainer_grupo1.tar.gz && "
        "python -m trainer.task_grupo1_svr"
    )

    job = aiplatform.CustomJob(
        display_name="tfm_grupo1_svr",
        worker_pool_specs=[{
            "machine_spec": {"machine_type": "n2-highmem-16"},
            "replica_count": 1,
            "container_spec": {
                "image_uri": "python:3.10-slim",
                "command":   ["bash", "-c"],
                "args":      [bootstrap],
                "env": [
                    {"name": "GCP_BUCKET",  "value": BUCKET_NAME},
                    {"name": "GCP_PROJECT", "value": PROJECT_ID},
                ],
            },
        }],
    )
    job.submit()
    print(f"\nJob lanzado: tfm_grupo1_svr")
    print(f"Resultados en: gs://{BUCKET_NAME}/grupo_1/1_5/")


if __name__ == "__main__":
    build_package()
    upload_package()
    submit_job()
