#!/usr/bin/env python3
"""
src/04_optuna_tuning.py — Agente 4: Optimización bayesiana de LightGBM (Optuna)

## Propósito (flujo de z494_TareaHogar_04, en Python)
1. Tomar SOLO los meses de `meses.train`. El mes de test no se mira acá.
2. Target clase01: 1 si la clase está en `target.positivos` (por defecto BAJA+1 y BAJA+2).
3. Undersampling: quedarse con una fracción de los CONTINUA (`entrenamiento.undersampling`),
   con la semilla primigenia. Los positivos y los BAJA+1 entran siempre.
4. Optuna (TPE) busca learning_rate, feature_fraction, num_leaves y min_data_in_leaf
   maximizando el AUC de una validación cruzada estratificada de 5 folds, con early stopping.
   (z494 usa AUC en la BO; con clase01 ya no se puede cortar por prob > 1/40.)
5. Guarda los mejores hiperparámetros y la cantidad de árboles en
   data/exp/exp<N>/optuna_mejores.json. El estudio queda en db/optuna.db y se puede retomar.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from config import NO_FEATURES, cargar_config


def preparar_train(cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(cfg["fe_path"], filters=[("foto_mes", "in", cfg["meses"]["train"])])
    df["clase01"] = df["clase_ternaria"].isin(cfg["target"]["positivos"]).astype(int)

    rng = np.random.default_rng(cfg["semilla_primigenia"])
    azar = rng.random(len(df))
    us = cfg["entrenamiento"]["undersampling"]
    entra = (azar <= us) | df["clase_ternaria"].isin(["BAJA+1", "BAJA+2"])
    df = df[entra]

    features = [c for c in df.columns if c not in NO_FEATURES]
    print(f"📦 Train {cfg['meses']['train']} con undersampling {us}: {len(df):,} filas, {len(features)} variables")
    print(df["clase_ternaria"].value_counts().to_string())
    return df[features], df["clase01"]


def main():
    parser = argparse.ArgumentParser(description="Agente 4: Optuna sobre LightGBM")
    parser.add_argument("--config", default="config/experimento.yaml")
    parser.add_argument("--db_path", default="db/optuna.db")
    args = parser.parse_args()
    cfg = cargar_config(args.config)
    opt = cfg["optuna"]
    semilla = cfg["semilla_primigenia"]

    X, y = preparar_train(cfg)
    dtrain = lgb.Dataset(X, label=y, free_raw_data=False)
    rangos = opt["rangos"]

    def objective(trial: optuna.Trial) -> float:
        params = dict(opt["fijos"])
        params.update({
            "learning_rate": trial.suggest_float("learning_rate", *rangos["learning_rate"], log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", *rangos["feature_fraction"]),
            "num_leaves": trial.suggest_int("num_leaves", *rangos["num_leaves"], log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", *rangos["min_data_in_leaf"], log=True),
            "seed": semilla,
        })
        cv = lgb.cv(
            params, dtrain,
            num_boost_round=5000,
            nfold=opt["xval_folds"], stratified=True, seed=semilla,
            callbacks=[lgb.early_stopping(int(50 + 5 / params["learning_rate"]), verbose=False)],
        )
        auc = cv["valid auc-mean"]
        trial.set_user_attr("best_iter", len(auc))  # con early stopping, el largo es la mejor iteración
        return float(auc[-1])

    Path(args.db_path).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="maximize",
        study_name=f"exp{cfg['experimento']}",
        storage=f"sqlite:///{Path(args.db_path).absolute()}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=semilla),
    )
    pendientes = opt["n_trials"] - len([t for t in study.trials if t.state.is_finished()])
    if pendientes > 0:
        print(f"\n🚀 Optuna: {pendientes} trials (estudio exp{cfg['experimento']}, se retoma si se corta)")
        study.optimize(objective, n_trials=pendientes)

    mejor = study.best_trial
    resultado = {
        "auc_cv": mejor.value,
        "best_iter": mejor.user_attrs["best_iter"],
        "params": {**opt["fijos"], **mejor.params},
        "undersampling": cfg["entrenamiento"]["undersampling"],
        "trial": mejor.number,
    }
    salida = cfg["exp_dir"] / "optuna_mejores.json"
    salida.write_text(json.dumps(resultado, indent=2), encoding="utf-8")

    print(f"\n🏆 Mejor trial #{mejor.number}: AUC CV {mejor.value:.5f}, {resultado['best_iter']} árboles")
    for k, v in mejor.params.items():
        print(f"   • {k:<18} {v}")
    print(f"💾 {salida}")


if __name__ == "__main__":
    main()
