from setuptools import setup, find_packages

setup(
    name="trainer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "xgboost",
        "lightgbm",
        "gcsfs",
        "xlsxwriter",
        "google-cloud-storage",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "numpy",
    ],
)
