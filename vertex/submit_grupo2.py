"""
submit_grupo2.py
Empaqueta el trainer y lanza el Custom Job en Vertex AI para el Grupo 2.
"""

import os
import subprocess
import sys

from google.cloud import aiplatform, storage

PROJECT_ID    = os.environ.get("GCP_PROJECT", "CHANGE_ME")
REGION        = "europe-southwest1"
BUCKET_NAME   = os.environ.get("GCP_BUCKET",  "CHANGE_ME")
PACKAGE_GCS   = f"gs://{BUCKET_NAME}/packages/trainer_grupo2-0.1.tar.gz"
PACKAGE_LOCAL = os.path.join(os.path.dirname(__file__), "grupo2_vertex", "dist", "trainer-0.1.tar.gz")
PACKAGE_DIR   = os.path.join(os.path.dirname(__file__), "grupo2_vertex")


def build_package():
    print("Empaquetando trainer grupo 2...")
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
    bucket.blob("packages/trainer_grupo2-0.1.tar.gz").upload_from_filename(PACKAGE_LOCAL)
    print("Paquete subido.")


def submit_job():
    print("Enviando Custom Job (Grupo 2) a Vertex AI...")
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=f"gs://{BUCKET_NAME}")

    bootstrap = (
        "pip install --quiet xgboost gcsfs xlsxwriter google-cloud-storage "
        "scikit-learn pandas matplotlib numpy && "
        "python -c \""
        "from google.cloud import storage; "
        f"storage.Client().bucket('{BUCKET_NAME}').blob('packages/trainer_grupo2-0.1.tar.gz')"
        ".download_to_filename('/tmp/trainer_grupo2.tar.gz')"
        "\" && "
        "pip install --quiet /tmp/trainer_grupo2.tar.gz && "
        "python -m trainer.task_grupo2"
    )

    job = aiplatform.CustomJob(
        display_name="tfm_grupo2",
        worker_pool_specs=[{
            "machine_spec": {
                "machine_type": "n2-highmem-16",  # 16 vCPUs, 128 GB RAM
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
    print(f"  gs://{BUCKET_NAME}/grupo_2/2_1/ ... gs://{BUCKET_NAME}/grupo_2/2_19/")


if __name__ == "__main__":
    build_package()
    upload_package()
    submit_job()
