"""Experimento 01 - Harness de medicion.

Entrena el modelo de referencia (trial 25, 238 features) con N semillas sobre
los dos folds con gap de 2 meses, y PERSISTE LA PROBABILIDAD POR CLIENTE.

Ese ultimo punto es el objetivo del script. Hasta ahora los experimentos
guardaban solamente la ganancia escalar, asi que cualquier pregunta nueva
("y si cortaramos en otro lugar?") obligaba a reentrenar. Con las
probabilidades en disco, toda pregunta sobre la REGLA DE CORTE se responde
sin volver a entrenar nada.

Sin early stopping: el numero de iteraciones es fijo e identico en todos los
brazos. De esa forma el mes de validacion no participa de ninguna seleccion y
la ganancia que reporta no esta inflada por haber elegido la iteracion ahi.

Uso:
    nohup python3 -u experimentos/exp01_harness.py > experimentos/salidas/harness.log 2>&1 &
"""

import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as cfg
import preprocesamiento as prep
import feature_engineering as fe

CHECKPOINTS = (100, 150, 200, 300)

# Smoke test: EXP01_MAX_SEMILLAS=1 corre una sola semilla para verificar que el
# script no se cae en la linea 300 antes de lanzar la corrida completa.
MAX_SEMILLAS = int(os.environ.get("EXP01_MAX_SEMILLAS", "0")) or None
RUTA_METRICAS = cfg.SALIDAS / "exp01_metricas.csv"


def construir_base() -> tuple:
    """Pipeline de preprocesamiento + feature engineering. Cachea en parquet."""
    if cfg.CACHE_BASE.is_file():
        print(f"[base] leyendo cache {cfg.CACHE_BASE.name}", flush=True)
        base = pd.read_pickle(cfg.CACHE_BASE)
        features = prep.cargar_lista_features(
            cfg.PROYECTO / "features_candidato_fe.csv"
        )
        return base, features

    t0 = time.time()
    ruta = prep.resolver_ruta_fuente()
    print(f"[base] fuente {ruta}", flush=True)
    original = prep.cargar_competencia_01(ruta)
    print(f"[base] crudo {original.shape}", flush=True)

    preprocesada, correcciones = prep.preprocesar_competencia_01(original)
    del original
    gc.collect()
    print(f"[base] preprocesada {preprocesada.shape}", flush=True)
    print(correcciones.to_string(index=False), flush=True)

    features = prep.cargar_lista_features(
        cfg.PROYECTO / "features_candidato_fe.csv"
    )
    entrada, _ = prep.seleccionar_entrada_feature_engineering(
        preprocesada, features
    )
    del preprocesada
    gc.collect()

    base, f_temporales, f_carga = fe.construir_features_candidato(entrada)
    del entrada
    gc.collect()
    print(
        f"[base] con features {base.shape} "
        f"(temporales={len(f_temporales)}, carga={len(f_carga)})",
        flush=True,
    )

    base, infinitos = prep.convertir_infinitos_en_nulos(base, features)
    print(f"[base] infinitos convertidos a nulo: {infinitos}", flush=True)
    columnas = ["numero_de_cliente", "foto_mes", "clase_ternaria",
                "target_baja2"] + features
    base = base[columnas]
    base.to_pickle(cfg.CACHE_BASE)
    print(f"[base] cache escrito en {time.time()-t0:.0f}s", flush=True)
    return base, features


def ganancia_por_fila(clase_ternaria) -> np.ndarray:
    """SIEMPRE desde la ternaria. BAJA+1 y CONTINUA cuestan lo mismo pero son
    clientes distintos; construirla desde la binaria de entrenamiento es el
    error clasico."""
    return np.where(
        np.asarray(clase_ternaria) == "BAJA+2",
        cfg.GANANCIA_BAJA2,
        -cfg.COSTO_ESTIMULO,
    ).astype(np.float64)


def main() -> None:
    base, features = construir_base()
    print(f"[base] {len(features)} features", flush=True)

    for fold in cfg.FOLDS:
        cfg.verificar_gap(fold)
    print("[base] gap temporal >= 2 verificado en los dos folds", flush=True)

    filas = []
    if RUTA_METRICAS.is_file():
        filas = pd.read_csv(RUTA_METRICAS).to_dict("records")
        hechos = {(r["fold"], r["semilla"], r["iteraciones"]) for r in filas}
        print(f"[reanudar] {len(hechos)} mediciones ya en disco", flush=True)
    else:
        hechos = set()

    for fold in cfg.FOLDS:
        nombre = fold["nombre"]
        m_train = base["foto_mes"].isin(fold["train"])
        m_valid = base["foto_mes"].eq(fold["validacion"])

        X_train = base.loc[m_train, features].to_numpy(dtype=np.float32)
        y_train = base.loc[m_train, "target_baja2"].to_numpy(dtype=np.int8)
        X_valid = base.loc[m_valid, features].to_numpy(dtype=np.float32)
        clase_valid = base.loc[m_valid, "clase_ternaria"].to_numpy()
        gan_fila = ganancia_por_fila(clase_valid)

        # El bloque de validacion se guarda una sola vez por fold: es la
        # verdad contra la que se mide cualquier regla de corte.
        ruta_verdad = cfg.PREDICCIONES / f"verdad_{nombre}.csv"
        if not ruta_verdad.is_file():
            pd.DataFrame({
                "numero_de_cliente": base.loc[m_valid, "numero_de_cliente"].values,
                "foto_mes": base.loc[m_valid, "foto_mes"].values,
                "clase_ternaria": clase_valid,
                "target_baja2": base.loc[m_valid, "target_baja2"].values,
            }).to_csv(ruta_verdad, index=False)

        print(
            f"\n[{nombre}] train {fold['train']} {X_train.shape} "
            f"pos={int(y_train.sum())} | valid {fold['validacion']} "
            f"{X_valid.shape} pos={int((clase_valid=='BAJA+2').sum())}",
            flush=True,
        )

        for semilla in cfg.SEMILLAS_MEDICION[:MAX_SEMILLAS]:
            if all((nombre, semilla, c) in hechos for c in CHECKPOINTS):
                continue
            t0 = time.time()
            params = {
                **cfg.PARAMETROS_TRIAL25,
                "seed": semilla,
                "feature_fraction_seed": semilla,
                "bagging_seed": semilla,
                "data_random_seed": semilla,
            }
            ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            modelo = lgb.train(
                params=params,
                train_set=ds,
                num_boost_round=max(CHECKPOINTS),
                callbacks=[lgb.log_evaluation(period=0)],
            )

            for corte in CHECKPOINTS:
                p = modelo.predict(X_valid, num_iteration=corte)
                np.save(
                    cfg.PREDICCIONES / f"p_{nombre}_{semilla}_{corte}.npy",
                    p.astype(np.float32),
                )
                estimular = p >= cfg.UMBRAL_TEORICO
                filas.append({
                    "fold": nombre,
                    "mes_validacion": fold["validacion"],
                    "semilla": semilla,
                    "iteraciones": corte,
                    "ganancia_umbral_teorico": float(gan_fila[estimular].sum()),
                    "estimulados": int(estimular.sum()),
                    "suma_probabilidades": float(p.sum()),
                })

            pd.DataFrame(filas).to_csv(RUTA_METRICAS, index=False)
            ultimo = [f for f in filas if f["semilla"] == semilla
                      and f["fold"] == nombre and f["iteraciones"] == 150][0]
            print(
                f"  semilla {semilla}: {time.time()-t0:5.1f}s  "
                f"ganancia@150iter={ultimo['ganancia_umbral_teorico']/1e6:7.2f}M  "
                f"k={ultimo['estimulados']}",
                flush=True,
            )
            del modelo, ds
            gc.collect()

        del X_train, y_train, X_valid
        gc.collect()

    print(f"\n[fin] metricas en {RUTA_METRICAS}", flush=True)
    print(f"[fin] predicciones en {cfg.PREDICCIONES}", flush=True)


if __name__ == "__main__":
    main()
