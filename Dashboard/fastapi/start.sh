#!/bin/bash
# Arrancar el FastAPI de predicción de satisfacción laboral NVIDIA
# Ejecutar desde la carpeta donde está main.py

# Evita crash SIGSEGV por dos runtimes OpenMP (XGBoost + sklearn/numpy en macOS)
export KMP_DUPLICATE_LIB_OK=TRUE

echo "🚀 Arrancando FastAPI..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
