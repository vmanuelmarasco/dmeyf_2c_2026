#!/usr/bin/env python3
"""
src/05_semillerio_corte.py — Agente 5: Semillerío de LightGBM, corte por envíos y entrega

## Propósito (z494 "Producción" + z601 "cortes" + z301 "público/privado")
FASE TEST (mide, no decide hiperparámetros):
  1. Entrena N semillas con los mejores hiperparámetros de Optuna sobre `meses.train`,
     SIN undersampling y con min_data_in_leaf normalizado (min_data_in_leaf / undersampling), como z494.
  2. Predice `meses.test`, promedia las semillas (semillerío) y arma la curva de ganancia por
     CANTIDAD DE ENVÍOS (con clase01 ya no se puede cortar por probabilidad 1/40).
  3. Parte el mes de test en 30% / 70% (semilla_kaggle) para ver si la ganancia se sostiene
     en dos mitades distintas (el público/privado de z494).
  4. Elige k* = cantidad de envíos que maximiza la ganancia del semillerío (curva suavizada).
FASE FINAL (la entrega):
  5. Reentrena las N semillas sobre `meses.train_final`, predice `meses.future`,
     promedia y entrega los ids de los k* clientes con mayor probabilidad
     (una columna, sin encabezado).

Salidas en data/exp/exp<N>/: ganancia_test.csv, resumen_05.json, modelos/, prediccion_future.csv,
entrega_exp<N>_k<k*>.csv
"""

import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import NO_FEATURES, cargar_config, ganancia_por_cliente, semillas_primas


def cargar_meses(cfg: dict, meses: list[int]) -> pd.DataFrame:
    df = pd.read_parquet(cfg["fe_path"], filters=[("foto_mes", "in", meses)])
    df["clase01"] = df["clase_ternaria"].isin(cfg["target"]["positivos"]).astype(int)
    return df


def params_finales(cfg: dict, mejores: dict) -> dict:
    p = dict(mejores["params"])
    # z494: el modelo final se entrena sin undersampling, entonces la hoja mínima se escala
    p["min_data_in_leaf"] = int(round(p["min_data_in_leaf"] / mejores["undersampling"]))
    p.pop("metric", None)
    return p


def entrenar_y_predecir(df_train, df_pred, features, params, n_iter, semillas, guardar_en=None):
    dtrain = lgb.Dataset(df_train[features], label=df_train["clase01"])
    preds = []
    for i, s in enumerate(semillas, 1):
        modelo = lgb.train({**params, "seed": s}, dtrain, num_boost_round=n_iter)
        preds.append(modelo.predict(df_pred[features]))
        if guardar_en is not None:
            modelo.save_model(str(guardar_en / f"lgbm_semilla_{s}.txt"))
        if i % 5 == 0 or i == len(semillas):
            print(f"   • {i}/{len(semillas)} semillas")
    return np.vstack(preds)  # (n_semillas, n_clientes)


def ganancia_en_cortes(prob, clase, cortes, mascara=None) -> np.ndarray:
    """Ganancia de enviar a los `k` clientes con mayor prob, para cada k de `cortes`.
    Con `mascara`, solo suma la ganancia de los clientes de ese subconjunto (público o privado),
    igual que Kaggle: el corte es sobre la lista completa y cada leaderboard mira su parte."""
    orden = np.argsort(-prob, kind="stable")
    g = ganancia_por_cliente(np.asarray(clase)[orden])
    if mascara is not None:
        g = np.where(np.asarray(mascara)[orden], g, 0)
    acumulada = np.cumsum(g)
    cortes = np.minimum(cortes, len(acumulada))
    return acumulada[cortes - 1]


def fase_test(cfg, mejores, features, params, semillas, cortes):
    m = cfg["meses"]
    print(f"\n🧪 FASE TEST: entreno en {m['train']} y mido en {m['test']}")
    df_train = cargar_meses(cfg, m["train"])
    df_test = cargar_meses(cfg, [m["test"]])
    if df_test["clase_ternaria"].isna().any():
        raise ValueError(f"El mes de test {m['test']} tiene clientes sin clase: no sirve para medir.")

    P = entrenar_y_predecir(df_train, df_test, features, params, mejores["best_iter"], semillas)
    prob = P.mean(axis=0)
    clase = df_test["clase_ternaria"].values

    # Público / privado simulados (estratificado por clase, como realidad_inicializar de z494)
    rng = np.random.default_rng(cfg["semilla_kaggle"])
    publico = np.zeros(len(df_test), dtype=bool)
    for c in np.unique(clase):
        idx = np.flatnonzero(clase == c)
        publico[rng.choice(idx, size=int(round(0.3 * len(idx))), replace=False)] = True

    por_semilla = np.vstack([ganancia_en_cortes(p, clase, cortes) for p in P])
    tabla = pd.DataFrame({
        "envios": cortes,
        "ganancia_semillerio": ganancia_en_cortes(prob, clase, cortes),
        "ganancia_publico": ganancia_en_cortes(prob, clase, cortes, publico) / 0.3,
        "ganancia_privado": ganancia_en_cortes(prob, clase, cortes, ~publico) / 0.7,
        "media_semillas_sueltas": por_semilla.mean(axis=0),
        "desvio_semillas_sueltas": por_semilla.std(axis=0),
    })
    tabla = tabla.astype({c: float for c in tabla.columns if c != "envios"})
    tabla["suavizada"] = tabla["ganancia_semillerio"].rolling(3, center=True, min_periods=1).mean()
    k_opt = int(tabla.loc[tabla["suavizada"].idxmax(), "envios"])
    tabla.to_csv(cfg["exp_dir"] / "ganancia_test.csv", index=False)

    print("\n💰 Ganancia en el mes de test por cantidad de envíos:")
    print(tabla.to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    fila = tabla[tabla["envios"] == k_opt].iloc[0]
    print(f"\n🎯 k* = {k_opt:,} envíos  |  ganancia semillerío {fila['ganancia_semillerio']:,.0f}"
          f"  |  semilla suelta {fila['media_semillas_sueltas']:,.0f} ± {fila['desvio_semillas_sueltas']:,.0f}")
    print("   (k* se elige mirando test: esa ganancia es algo optimista; la honesta es la de las vecinas de k*)")
    return k_opt, fila.to_dict()


def fase_final(cfg, mejores, features, params, semillas, k_opt):
    m = cfg["meses"]
    print(f"\n🚀 FASE FINAL: entreno en {m['train_final']} y predigo {m['future']}")
    df_train = cargar_meses(cfg, m["train_final"])
    df_future = cargar_meses(cfg, [m["future"]])
    if df_train["clase_ternaria"].isna().any():
        raise ValueError(f"train_final {m['train_final']} incluye clientes sin clase. Revisá la config.")

    carpeta_modelos = cfg["exp_dir"] / "modelos"
    carpeta_modelos.mkdir(exist_ok=True)
    P = entrenar_y_predecir(df_train, df_future, features, params, mejores["best_iter"], semillas, carpeta_modelos)

    pred = df_future[["numero_de_cliente"]].astype("int64")  # ids enteros, nunca "123.0"
    pred["prob"] = P.mean(axis=0)
    pred = pred.sort_values("prob", ascending=False).reset_index(drop=True)
    pred["Predicted"] = (pred.index < k_opt).astype(int)
    pred.to_csv(cfg["exp_dir"] / "prediccion_future.csv", index=False)

    # Entrega: una sola columna con los ids de los k* clientes elegidos, sin encabezado
    salida = cfg["exp_dir"] / f"entrega_exp{cfg['experimento']}_k{k_opt}.csv"
    pred.loc[pred["Predicted"] == 1, ["numero_de_cliente"]].to_csv(salida, index=False, header=False)
    print(f"📤 Entrega: {salida}  ({k_opt:,} ids de {len(pred):,} clientes)")
    return salida


def main():
    parser = argparse.ArgumentParser(description="Agente 5: semillerío, corte por envíos y entrega")
    parser.add_argument("--config", default="config/experimento.yaml")
    parser.add_argument("--fase", choices=["test", "final", "ambas"], default="ambas")
    args = parser.parse_args()
    cfg = cargar_config(args.config)

    mejores = json.loads((cfg["exp_dir"] / "optuna_mejores.json").read_text(encoding="utf-8"))
    params = params_finales(cfg, mejores)
    semillas = semillas_primas(cfg["semilla_primigenia"], cfg["semillerio"]["n_semillas"])
    c = cfg["semillerio"]["cortes"]
    cortes = np.arange(c["desde"], c["hasta"] + 1, c["paso"])

    columnas = pd.read_parquet(cfg["fe_path"], filters=[("foto_mes", "==", cfg["meses"]["test"])]).columns
    features = [col for col in columnas if col not in NO_FEATURES]
    print(f"⚙️  {len(semillas)} semillas primas, {mejores['best_iter']} árboles, "
          f"min_data_in_leaf normalizado = {params['min_data_in_leaf']}")

    resumen_path = cfg["exp_dir"] / "resumen_05.json"
    resumen = json.loads(resumen_path.read_text(encoding="utf-8")) if resumen_path.exists() else {}

    if args.fase in ("test", "ambas"):
        k_opt, fila = fase_test(cfg, mejores, features, params, semillas, cortes)
        resumen.update({"k_opt": k_opt, "test": {k: float(v) for k, v in fila.items()},
                        "n_semillas": len(semillas), "meses": cfg["meses"], "experimento": cfg["experimento"]})
    if args.fase in ("final", "ambas"):
        if "k_opt" not in resumen:
            raise ValueError("Falta k*: corré primero --fase test")
        resumen["entrega"] = str(fase_final(cfg, mejores, features, params, semillas, resumen["k_opt"]))

    resumen_path.write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(f"💾 {resumen_path}")


if __name__ == "__main__":
    main()
