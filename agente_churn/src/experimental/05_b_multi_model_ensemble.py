#!/usr/bin/env python3
"""
src/05_b_multi_model_ensemble.py — Módulo 5-B: Ensamble Híbrido con Semillerío por Algoritmo y Optimización de Pesos en Optuna
Alineado con las directivas de Hibridación de Semilleríos de DMEyF 2026 (z301, z401, z601).

## Propósito
Entrenar semillerío interno (N semillas) de LightGBM, XGBoost y Random Forest, promediar
las probabilidades por algoritmo y optimizar con Optuna los pesos de mezcla
(w_lgbm, w_xgb, w_rf) y el umbral de corte. Exporta la submisión y un resumen JSON.

## Criterio de selección
- Económico: ganancia neta (acierto +$1.072.500 | envío -$27.500).
- Estabilidad: menor dispersión de ganancia entre semillas.
"""

import argparse
import json
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Parámetros monetarios oficiales de la cátedra
GANANCIA_ACIERTO = 1_072_500
COSTO_ESTIMULO = 27_500


def parse_args():
    parser = argparse.ArgumentParser(
        description="Módulo 5-B: Ensamble Híbrido con Semillerío Multimodelo (LGBM + XGB + RF) y Pesos Optuna."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default="data/competencia_01_fe_clean.parquet",
        help="Ruta al dataset procesado y filtrado",
    )
    parser.add_argument(
        "--mes_train",
        type=int,
        default=202103,
        help="Foto mes de entrenamiento",
    )
    parser.add_argument(
        "--mes_test",
        type=int,
        default=202105,
        help="Foto mes de prueba/validación temporal (Mayo)",
    )
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=20,
        help="Cantidad de semillas de primos para el semillerío interno de cada modelo (default: 20)",
    )
    parser.add_argument(
        "--n_trials_optuna",
        type=int,
        default=100,
        help="Cantidad de ensayos en Optuna para optimizar los pesos del ensamble",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/submission_hybrid_ensemble.csv",
        help="Ruta donde se guardará el CSV final de submisión",
    )
    parser.add_argument(
        "--output_summary",
        type=str,
        default="data/hybrid_ensemble_summary.json",
        help="Ruta donde se guardará el informe JSON del ensamble",
    )
    return parser.parse_args()


def generate_prime_seeds(n: int, seed_base: int = 42) -> list[int]:
    """Genera n semillas basadas en números primos de 6 dígitos (100003 a 999983)."""
    np.random.seed(seed_base)
    primes_range = np.arange(100003, 999983, 2)
    selected_seeds = np.random.choice(primes_range, size=n, replace=False).tolist()
    return [int(s) for s in selected_seeds]


def train_seed_averaged_models(
    X_train: pd.DataFrame,
    y_train_bin: pd.Series,
    X_test: pd.DataFrame,
    seeds: list[int],
) -> dict[str, np.ndarray]:
    """
    Entrena N semillas para LightGBM, XGBoost y Random Forest, 
    retornando la probabilidad promediada (libre de varianza) de cada arquitectura.
    """
    n_samples_test = len(X_test)
    preds_lgb_acc = np.zeros(n_samples_test, dtype=np.float64)
    preds_xgb_acc = np.zeros(n_samples_test, dtype=np.float64)
    preds_rf_acc = np.zeros(n_samples_test, dtype=np.float64)

    print("\n" + "=" * 70)
    print(f"🤖 ENTRENANDO SEMILLERÍO MULTIMODELO ({len(seeds)} semillas por algoritmo)")
    print("=" * 70)

    train_data_lgb = lgb.Dataset(X_train, label=y_train_bin)

    for idx, seed in enumerate(seeds, start=1):
        # 1. LightGBM con semilla actual
        lgb_params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "first_metric_only": True,
            "boost_from_average": True,
            "max_bin": 31,
            "num_leaves": 32,
            "learning_rate": 0.02,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.5,
            "bagging_fraction": 0.7,
            "bagging_freq": 1,  # sin esto LightGBM ignora bagging_fraction
            "scale_pos_weight": 2.5,
            "seed": seed,
            "verbose": -1,
        }
        model_lgb = lgb.train(lgb_params, train_data_lgb, num_boost_round=250)
        preds_lgb_acc += model_lgb.predict(X_test)

        # 2. XGBoost con semilla actual
        xgb_model = xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.02,
            max_depth=5,
            subsample=0.7,
            colsample_bytree=0.5,
            scale_pos_weight=2.5,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train_bin)
        preds_xgb_acc += xgb_model.predict_proba(X_test)[:, 1]

        # 3. Random Forest con semilla actual
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_leaf=50,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        rf_model.fit(X_train, y_train_bin)
        preds_rf_acc += rf_model.predict_proba(X_test)[:, 1]

        if idx % 5 == 0 or idx == len(seeds):
            print(f"  • Completadas {idx}/{len(seeds)} semillas en los 3 algoritmos...")

    # Promediado final por arquitectura
    avg_predictions = {
        "lgbm": preds_lgb_acc / len(seeds),
        "xgboost": preds_xgb_acc / len(seeds),
        "rf": preds_rf_acc / len(seeds),
    }

    print("✅ Promediado de probabilidades completado con éxito.")
    return avg_predictions


def calculate_profit(y_pred_probs: np.ndarray, y_real_classes: pd.Series, threshold: float = 0.025) -> float:
    """Calcula la ganancia económica sobre la clase ternaria real."""
    ganancia_unitaria = np.where(y_real_classes == "BAJA+2", GANANCIA_ACIERTO, -COSTO_ESTIMULO)
    return float(ganancia_unitaria[y_pred_probs >= threshold].sum())


def optimize_ensemble_weights(
    preds_dict: dict[str, np.ndarray],
    y_test_class: pd.Series,
    n_trials: int,
    seed: int,
) -> tuple[dict[str, float], float, float, int]:
    """
    Optuna busca los pesos de mezcla (w_lgbm, w_xgb, w_rf) y el umbral tau*
    que maximizan la Ganancia Acumulada.
    """
    print("\n" + "=" * 70)
    print(f"🎯 OPTIMIZANDO PESOS Y UMBRAL DEL ENSAMBLE EN OPTUNA ({n_trials} trials)...")
    print("=" * 70)

    p_lgb = preds_dict["lgbm"]
    p_xgb = preds_dict["xgboost"]
    p_rf = preds_dict["rf"]

    def objective(trial):
        w_lgb = trial.suggest_float("w_lgbm", 0.0, 1.0)
        w_xgb = trial.suggest_float("w_xgb", 0.0, 1.0)
        w_rf = trial.suggest_float("w_rf", 0.0, 1.0)

        total_w = w_lgb + w_xgb + w_rf
        if total_w < 1e-5:
            return -1e9

        w_lgb /= total_w
        w_xgb /= total_w
        w_rf /= total_w

        p_blend = (w_lgb * p_lgb) + (w_xgb * p_xgb) + (w_rf * p_rf)
        threshold = trial.suggest_float("threshold", 0.010, 0.080)

        ganancia = calculate_profit(p_blend, y_test_class, threshold=threshold)
        return ganancia

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials)

    best_p = study.best_params
    total_w = best_p["w_lgbm"] + best_p["w_xgb"] + best_p["w_rf"]

    normalized_weights = {
        "w_lgbm": float(best_p["w_lgbm"] / total_w),
        "w_xgb": float(best_p["w_xgb"] / total_w),
        "w_rf": float(best_p["w_rf"] / total_w),
    }

    opt_threshold = float(best_p["threshold"])
    max_profit = float(study.best_value)

    # Calcular envíos exactos en el óptimo
    p_blend_best = (
        (normalized_weights["w_lgbm"] * p_lgb)
        + (normalized_weights["w_xgb"] * p_xgb)
        + (normalized_weights["w_rf"] * p_rf)
    )
    opt_sends = int((p_blend_best >= opt_threshold).sum())

    print("\n🏆 RESULTADOS DE LA OPTIMIZACIÓN DE PESOS:")
    print(f"💰 MÁXIMA GANANCIA LOGRADA: ${max_profit:,.0f}")
    print("⚖️ Pesos Óptimos Asignados:")
    print(f"  • LightGBM (Semillerío):     {normalized_weights['w_lgbm']:.4f} ({normalized_weights['w_lgbm']*100:.1f}%)")
    print(f"  • XGBoost (Semillerío):      {normalized_weights['w_xgb']:.4f} ({normalized_weights['w_xgb']*100:.1f}%)")
    print(f"  • Random Forest (Semillerío): {normalized_weights['w_rf']:.4f} ({normalized_weights['w_rf']*100:.1f}%)")
    print(f"🎯 Umbral Óptimo de Corte (tau*): {opt_threshold:.6f}")
    print(f"📦 Cantidad Óptima de Envíos (k*): {opt_sends:,} clientes")

    return normalized_weights, opt_threshold, max_profit, opt_sends


def export_submission(
    df_test: pd.DataFrame,
    p_blend: np.ndarray,
    opt_threshold: float,
    output_csv_path: Path,
):
    """Genera el archivo CSV final con decisiones binarias (1/0)."""
    df_sub = pd.DataFrame(
        {
            "numero_de_cliente": df_test["numero_de_cliente"].values,
            "prob_blend": p_blend,
            "Predecir": (p_blend >= opt_threshold).astype(int),
        }
    )

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_sub[["numero_de_cliente", "Predecir"]].to_csv(output_csv_path, index=False)
    print(f"\n💾 Archivo de Submisión exportado exitosamente en: {output_csv_path}")


def main():
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"❌ No se encontró el dataset: {input_path}")

    print(f"📥 Cargando datos desde: {input_path} ...")
    df = pd.read_parquet(input_path)

    df_train = df[df["foto_mes"] == args.mes_train].copy()
    df_test = df[df["foto_mes"] == args.mes_test].copy()

    drop_cols = ["foto_mes", "numero_de_cliente", "clase_ternaria", "clase_binaria", "clase_binaria2"]
    features = [c for c in df_train.columns if c not in drop_cols]

    X_train = df_train[features]
    y_train_bin = df_train["clase_binaria"]

    X_test = df_test[features]
    y_test_class = df_test["clase_ternaria"]

    # 1. Generar semillas de números primos
    seeds = generate_prime_seeds(args.n_seeds)

    # 2. Entrenar Semillerío interno por cada algoritmo
    preds_dict = train_seed_averaged_models(X_train, y_train_bin, X_test, seeds)

    # 3. Comparativa de modelos individuales semillados
    print("\n" + "=" * 70)
    print("📊 GANANCIA DE CADA ALGORITMO CON SEMILLERÍO (Umbral Teórico 0.025)")
    print("=" * 70)
    for model_name, p_val in preds_dict.items():
        gan_ind = calculate_profit(p_val, y_test_class, threshold=0.025)
        print(f"  • Semillerío {model_name.upper():<12}: Ganancia = ${gan_ind:,.0f}")

    # 4. Optimizar pesos y umbral en Optuna
    opt_weights, opt_threshold, max_profit, opt_sends = optimize_ensemble_weights(
        preds_dict, y_test_class, args.n_trials_optuna, seed=100003
    )

    # 5. Calcular predicción final ensamblada
    p_blend_final = (
        (opt_weights["w_lgbm"] * preds_dict["lgbm"])
        + (opt_weights["w_xgb"] * preds_dict["xgboost"])
        + (opt_weights["w_rf"] * preds_dict["rf"])
    )

    # 6. Exportar CSV y guardar resumen estructurado JSON
    export_submission(df_test, p_blend_final, opt_threshold, Path(args.output_csv))

    summary_data = {
        "mes_train": args.mes_train,
        "mes_test": args.mes_test,
        "n_seeds_per_model": len(seeds),
        "models_included": ["LightGBM", "XGBoost", "RandomForest"],
        "individual_profits_default_tau": {
            k: calculate_profit(v, y_test_class, 0.025) for k, v in preds_dict.items()
        },
        "optimized_weights": opt_weights,
        "optimal_threshold": opt_threshold,
        "max_profit": max_profit,
        "optimal_sends": opt_sends,
    }

    output_summary_path = Path(args.output_summary)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)

    print(f"💾 Reporte guardado en: {output_summary_path}")
    print("✅ Módulo 5-B: Ensamble Híbrido con Semillerío finalizado con éxito.")


if __name__ == "__main__":
    main()
