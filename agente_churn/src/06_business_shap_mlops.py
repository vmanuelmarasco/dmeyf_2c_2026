#!/usr/bin/env python3
"""
src/06_business_shap_mlops.py — Agente 6: Explicabilidad TreeSHAP, resumen de negocio y MLflow

## Propósito
1. Cargar un modelo del semillerío final (data/exp/exp<N>/modelos/) y explicar sus predicciones
   sobre una muestra del mes a entregar con TreeSHAP (beeswarm, barras y un waterfall).
2. Resumir las variables más influyentes con la DIRECCIÓN de su efecto (si valores altos
   suben o bajan la probabilidad de baja). Son hipótesis para negocio, no causas.
3. Registrar el experimento en MLflow (config, ganancia en test, k*) con backend SQLite.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")  # sin pantalla
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap

from config import NO_FEATURES, cargar_config


def generate_shap_plots(model: lgb.Booster, X_test: pd.DataFrame, output_dir: Path) -> dict:
    """Genera gráficos de importancia global (Beeswarm) y local (Waterfall) con TreeSHAP."""
    print("\n" + "=" * 65)
    print("🌳 GENERANDO EXPLICABILIDAD CON TREESHAP")
    print("=" * 65)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Gráfico Beeswarm (Importancia Global)
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.title("Importancia Global de Atributos Bancarios (TreeSHAP Beeswarm)", fontsize=12)
    plt.tight_layout()
    beeswarm_path = output_dir / "shap_beeswarm.png"
    plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  • Gráfico Beeswarm guardado en: {beeswarm_path}")

    # 2. Gráfico Bar Plot (Importancia Promedio Absoluta)
    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Top 15 Variables por Impacto Promedio |SHAP|", fontsize=12)
    plt.tight_layout()
    bar_path = output_dir / "shap_bar.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  • Gráfico Bar Plot guardado en: {bar_path}")

    # 3. Gráfico Waterfall para el Cliente #0 (Explicación Local)
    plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.title("Explicación Local de Churn para Cliente Individual", fontsize=12)
    plt.tight_layout()
    waterfall_path = output_dir / "shap_waterfall_sample.png"
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  • Gráfico Waterfall local guardado en: {waterfall_path}")

    # Calcular ranking de variables principales
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:10]
    top_features = X_test.columns[top_indices].tolist()

    return {
        "beeswarm_path": str(beeswarm_path),
        "bar_path": str(bar_path),
        "waterfall_path": str(waterfall_path),
        "top_features": top_features,
    }


def resumen_variables(shap_values, X: pd.DataFrame, top: int = 10) -> list[dict]:
    """Top variables por |SHAP| medio, con el signo de la relación valor ↔ efecto."""
    vals = shap_values.values
    imp = np.abs(vals).mean(axis=0)
    filas = []
    for i in np.argsort(imp)[::-1][:top]:
        x = X.iloc[:, i].astype(float).values
        ok = ~np.isnan(x)
        corr = np.corrcoef(x[ok], vals[ok, i])[0, 1] if ok.sum() > 2 and np.std(x[ok]) > 0 else np.nan
        if np.isnan(corr):
            direccion = "sin dirección clara"
        elif corr > 0:
            direccion = "valores ALTOS suben la probabilidad de baja"
        else:
            direccion = "valores BAJOS suben la probabilidad de baja"
        filas.append({"variable": X.columns[i], "shap_medio_abs": float(imp[i]),
                      "correlacion_valor_shap": None if np.isnan(corr) else float(corr), "direccion": direccion})
    return filas


def main():
    parser = argparse.ArgumentParser(description="Agente 6: SHAP, resumen y MLflow")
    parser.add_argument("--config", default="config/experimento.yaml")
    parser.add_argument("--mlflow_uri", default="sqlite:///db/mlflow.db",
                        help="MLflow 3 ya no admite el backend de archivos ./mlruns")
    parser.add_argument("--muestra", type=int, default=5000)
    args = parser.parse_args()
    cfg = cargar_config(args.config)
    exp_dir = cfg["exp_dir"]
    output_dir = exp_dir / "reports"

    modelos = sorted((exp_dir / "modelos").glob("lgbm_semilla_*.txt"))
    if not modelos:
        raise FileNotFoundError("No hay modelos finales: corré antes 05_semillerio_corte.py")
    modelo = lgb.Booster(model_file=str(modelos[0]))
    print(f"📥 Modelo: {modelos[0].name} (1 de {len(modelos)} del semillerío)")

    df = pd.read_parquet(cfg["fe_path"], filters=[("foto_mes", "==", cfg["meses"]["future"])])
    X = df[modelo.feature_name()]
    X = X.sample(min(args.muestra, len(X)), random_state=cfg["semilla_primigenia"])

    shap_info = generate_shap_plots(modelo, X, output_dir)
    shap_values = shap.TreeExplainer(modelo)(X)
    variables = resumen_variables(shap_values, X)

    resumen_05 = json.loads((exp_dir / "resumen_05.json").read_text(encoding="utf-8"))
    informe = {"experimento": cfg["experimento"], "envios_k_opt": resumen_05.get("k_opt"),
               "ganancia_test": resumen_05.get("test", {}), "variables_influyentes": variables}
    informe_path = output_dir / "resumen_negocio.json"
    informe_path.write_text(json.dumps(informe, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n💼 Variables más influyentes (hipótesis para negocio, no causas):")
    for v in variables:
        print(f"   • {v['variable']:<35} {v['direccion']}")

    Path("db").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment("Churn_DMEyF2026")
    with mlflow.start_run(run_name=f"exp{cfg['experimento']}"):
        mlflow.log_params({
            "meses_train": str(cfg["meses"]["train"]), "mes_test": cfg["meses"]["test"],
            "meses_train_final": str(cfg["meses"]["train_final"]), "mes_future": cfg["meses"]["future"],
            "target_positivos": str(cfg["target"]["positivos"]),
            "undersampling": cfg["entrenamiento"]["undersampling"],
            "n_semillas": resumen_05.get("n_semillas"), "k_opt": resumen_05.get("k_opt"),
        })
        for k, v in resumen_05.get("test", {}).items():
            if k != "envios":
                mlflow.log_metric(f"test_{k}", v)
        optuna_json = exp_dir / "optuna_mejores.json"
        for p in [shap_info["beeswarm_path"], shap_info["bar_path"], shap_info["waterfall_path"],
                  informe_path, optuna_json, exp_dir / "ganancia_test.csv", exp_dir / "drift_psi.csv",
                  Path(args.config)]:
            if Path(p).exists():
                mlflow.log_artifact(str(p))
    print(f"\n📈 Registrado en MLflow ({args.mlflow_uri}). Para verlo: mlflow ui --backend-store-uri {args.mlflow_uri}")


if __name__ == "__main__":
    main()
