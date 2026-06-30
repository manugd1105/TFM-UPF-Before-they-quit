"""
submit_cross_BC.py
───────────────────
Empaqueta y lanza el Custom Job en Vertex AI para los experimentos B y C.

  B-1  XGBoost  FTSE100 train (108k) → FTSE100 test (27k)
  C-1  XGBoost  SP500 train (285k)   → FTSE100 full (135k) + SP500 test (71k)

Outputs en:
  gs://<GCP_BUCKET>/cross_training/B_1/
  gs://<GCP_BUCKET>/cross_training/C_1/
"""

import os
import subprocess
import sys

from google.cloud import aiplatform, storage

PROJECT_ID    = os.environ.get("GCP_PROJECT", "CHANGE_ME")
REGION        = "europe-southwest1"
BUCKET_NAME   = os.environ.get("GCP_BUCKET",  "CHANGE_ME")
PACKAGE_GCS   = f"gs://{BUCKET_NAME}/packages/trainer_cross_BC-0.1.tar.gz"
PACKAGE_LOCAL = os.path.join(os.path.dirname(__file__), "cross_vertex", "dist", "trainer-0.1.tar.gz")
PACKAGE_DIR   = os.path.join(os.path.dirname(__file__), "cross_vertex")


def build_package():
    print("Empaquetando trainer cross B&C...")
    subprocess.run(
        [sys.executable, "setup.py", "sdist"],
        cwd=PACKAGE_DIR,
        check=True,
    )
    print(f"Paquete creado: {PACKAGE_LOCAL}")


def upload_package():
    print(f"Subiendo paquete a {PACKAGE_GCS}...")
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    bucket.blob("packages/trainer_cross_BC-0.1.tar.gz").upload_from_filename(PACKAGE_LOCAL)
    print("Paquete subido.")


def submit_job():
    print("Enviando Custom Job (Cross B&C) a Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")

    bootstrap = (
        "apt-get update -qq && "
        "pip install --quiet xgboost gcsfs xlsxwriter google-cloud-storage "
        "scikit-learn pandas matplotlib numpy && "
        "python -c \""
        "from google.cloud import storage; "
        f"storage.Client().bucket('{BUCKET_NAME}').blob('packages/trainer_cross_BC-0.1.tar.gz')"
        ".download_to_filename('/tmp/trainer_cross_BC.tar.gz')"
        "\" && "
        "pip install --quiet /tmp/trainer_cross_BC.tar.gz && "
        "python -m trainer.task_cross"
    )

    job = aiplatform.CustomJob(
        display_name="tfm_cross_BC",
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "n2-highmem-16",
            },
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

    print(f"\nJob lanzado!")
    print(f"\nMonitoriza en:")
    print(f"  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={PROJECT_ID}")
    print(f"\nResultados en:")
    print(f"  gs://{BUCKET_NAME}/cross_training/B_1/")
    print(f"  gs://{BUCKET_NAME}/cross_training/C_1/")


if __name__ == "__main__":
    build_package()
    upload_package()
    submit_job()
