#!/usr/bin/env bash
# Test de humo: corre el pipeline completo sobre datos sintéticos con parámetros mínimos.
# Uso (desde la raíz del repo):  bash tests/smoke_test.sh
set -euo pipefail
T=tests/tmp
C=tests/smoke_config.yaml
rm -rf "$T" && mkdir -p "$T"

python3 tests/make_synthetic_data.py --output_path $T/competencia_crudo.csv
python3 src/01_data_ingestion.py --config $C
python3 src/02_eda_financial_diagnosis.py --input_path $T/dataset_con_clase.parquet --output_summary $T/eda.json
python3 src/03_feature_engineering_drift.py --config $C
python3 src/04_optuna_tuning.py --config $C --db_path $T/optuna.db
python3 src/05_semillerio_corte.py --config $C
python3 src/06_business_shap_mlops.py --config $C --mlflow_uri sqlite:///$T/mlflow.db
echo "✅ SMOKE TEST OK: el pipeline corrió de punta a punta"
