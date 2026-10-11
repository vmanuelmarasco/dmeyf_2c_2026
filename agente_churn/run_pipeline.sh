#!/usr/bin/env bash
# Corre el pipeline completo con config/experimento.yaml.
# Para corridas largas, lanzarlo con tmux o nohup (no desde una pestaña del navegador):
#   nohup caffeinate -i bash run_pipeline.sh > data/pipeline.log 2>&1 &
# (-u en python: los mensajes se escriben al log al instante, no al final)
set -euo pipefail
C=${1:-config/experimento.yaml}
python3 -u src/01_data_ingestion.py --config "$C"
python3 -u src/02_eda_financial_diagnosis.py
python3 -u src/03_feature_engineering_drift.py --config "$C"
python3 -u src/04_optuna_tuning.py --config "$C"
python3 -u src/05_semillerio_corte.py --config "$C"
python3 -u src/06_business_shap_mlops.py --config "$C"
