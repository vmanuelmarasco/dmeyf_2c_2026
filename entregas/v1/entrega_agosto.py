"""Entrenamiento final y prediccion de agosto 2021.

Corre despues de competencia_02.ipynb, que deja la base y las features en
competencia_01.duckdb.

DECISIONES, y por que cada una:

- Features: las 940 validadas en el Experimento 02 (144 crudas + 796 temporales).
  Le ganan al crudo en los dos folds: mayo +6,30 M (p=0,0008), junio +49,18 M
  (p=0,0000019).
- Meses de train: 202103-202106, los cuatro con clase_ternaria completa. El
  ultimo mes etiquetado esta a 2 meses de agosto, igual que el gap de los folds.
- Iteraciones: 150 fijas, sin early stopping. No hay mes de validacion donde
  elegirlas sin contaminar la estimacion (fuga F6).
- Hiperparametros: trial 25. Fueron optimizados sobre el set viejo de 238
  features, asi que son conservadores para 940. Optuna sobre el set nuevo es
  la mejora pendiente.
- Semillas: las 5 declaradas en el CheckIn. El Experimento 01 mostro que
  ensamblar satura en 5 semillas, asi que no hace falta mas.
- Corte: 0,025. El Experimento 01 probo que ninguna regla alternativa
  transfiere entre meses; 0,025 gana por no tener parametro que transferir.
- ORDER BY explicito en todas las lecturas: DuckDB paraleliza y "select *" sin
  orden no es determinista. Sin esto las predicciones no se pueden re-alinear
  con los clientes, que es un bug silencioso y fatal.
"""
import numpy as np, pandas as pd, duckdb, lightgbm as lgb, time
from pathlib import Path

GAN, COST = 1_072_500, 27_500
UMBRAL = COST / (GAN + COST)
SEMILLAS = [100129, 100183, 100151, 500041, 900007]
MESES_TRAIN = [202103, 202104, 202105, 202106]
MES_SCORING = 202108
ITERACIONES = 150
PARAMS = {
    "objective": "binary", "metric": "None", "boosting_type": "gbdt",
    "max_depth": 12, "num_leaves": 57, "learning_rate": 0.0144493962,
    "min_data_in_leaf": 1614, "feature_fraction": 0.2613620757,
    "bagging_fraction": 0.5276823218, "bagging_freq": 1,
    "lambda_l1": 6.957248086e-05, "lambda_l2": 4.968141077e-06,
    "min_gain_to_split": 1.5074952598, "max_bin": 31,
    "feature_pre_filter": False, "verbosity": -1,
}
FOLDS = [{"nombre": "mayo", "train": [202103], "valid": 202105},
         {"nombre": "junio", "train": [202103, 202104], "valid": 202106}]

con = duckdb.connect("competencia_01.duckdb")

# ORDER BY determinista. Es la correccion del bug de alineacion.
print("[1] reconstruyendo 'modelado' con ORDER BY explicito...", flush=True)
con.execute(f"""
    create or replace table modelado as
    select c.*, f.* exclude (numero_de_cliente, foto_mes)
    from competencia_01 c
    join features_temporales f using (numero_de_cliente, foto_mes)
    where c.foto_mes in ({",".join(map(str, MESES_TRAIN))})
       or c.foto_mes = {MES_SCORING}
    order by foto_mes, numero_de_cliente
""")

base = con.execute(
    "select * from modelado order by foto_mes, numero_de_cliente").df()
crudas = [r[0] for r in con.execute("describe competencia_01").fetchall()]
prohibidas = {"numero_de_cliente", "foto_mes", "clase_ternaria"}
fechas = {"Master_Fvencimiento", "Master_Finiciomora", "Master_fultimo_cierre",
          "Master_fechaalta", "Visa_Fvencimiento", "Visa_Finiciomora",
          "Visa_fultimo_cierre", "Visa_fechaalta"}
features = [c for c in base.columns if c not in prohibidas | fechas]
base[features] = base[features].replace([np.inf, -np.inf], np.nan)
base["y"] = (base["clase_ternaria"] == "BAJA+2").astype("int8")
print(f"    filas {len(base):,} | features {len(features)}", flush=True)
assert not (prohibidas & set(features))


def entrenar(meses_train, features, semilla, X_pred):
    m = base["foto_mes"].isin(meses_train).to_numpy()
    modelo = lgb.train(
        {**PARAMS, "seed": semilla, "bagging_seed": semilla,
         "feature_fraction_seed": semilla, "data_random_seed": semilla},
        lgb.Dataset(base.loc[m, features].to_numpy(dtype=np.float32),
                    label=base.loc[m, "y"].to_numpy()),
        num_boost_round=ITERACIONES)
    return modelo.predict(X_pred)


# ---- Promedio vs mediana: se decide en los DOS folds, no en uno ----
print("\n[2] eligiendo el metodo de ensamble en los dos folds...", flush=True)
gana_mediana = []
for f in FOLDS:
    mv = (base["foto_mes"] == f["valid"]).to_numpy()
    X_va = base.loc[mv, features].to_numpy(dtype=np.float32)
    gan = np.where(base.loc[mv, "clase_ternaria"].to_numpy() == "BAJA+2", GAN, -COST)
    P = np.vstack([entrenar(f["train"], features, s, X_va) for s in SEMILLAS])
    g_prom = gan[P.mean(axis=0) >= UMBRAL].sum() / 1e6
    g_med = gan[np.median(P, axis=0) >= UMBRAL].sum() / 1e6
    gana_mediana.append(g_med > g_prom)
    print(f"    {f['nombre']:6s}: promedio {g_prom:7.2f}M | mediana {g_med:7.2f}M "
          f"| delta {g_med-g_prom:+6.2f}M", flush=True)

METODO = "mediana" if all(gana_mediana) else "promedio"
print(f"    -> metodo elegido: {METODO.upper()} "
      f"(la mediana tenia que ganar en los dos folds)", flush=True)

# ---- Modelo final y prediccion de agosto ----
print(f"\n[3] entrenando final con {MESES_TRAIN} -> prediciendo {MES_SCORING}",
      flush=True)
ms = (base["foto_mes"] == MES_SCORING).to_numpy()
X_ago = base.loc[ms, features].to_numpy(dtype=np.float32)
clientes = base.loc[ms, "numero_de_cliente"].to_numpy()
assert len(clientes) == len(set(clientes)), "clientes repetidos en agosto"

P = {}
for s in SEMILLAS:
    t0 = time.time()
    P[s] = entrenar(MESES_TRAIN, features, s, X_ago)
    print(f"    semilla {s}: {time.time()-t0:5.1f}s", flush=True)

M = np.vstack([P[s] for s in SEMILLAS])
prob = M.mean(axis=0) if METODO == "promedio" else np.median(M, axis=0)
estimular = (prob >= UMBRAL).astype("int8")

# ---- Entrega ----
# astype(int64): si la clave queda en float, pandas escribe "12159854.0" y
# Kaggle puede no matchearla contra su ground truth. Bug silencioso.
entrega = pd.DataFrame({"numero_de_cliente": clientes.astype("int64"),
                        "Predicted": estimular})
entrega.to_csv("entrega_agosto.csv", index=False)

detalle = pd.DataFrame({
    "numero_de_cliente": clientes.astype("int64"),
    "probabilidad": prob,
    "Predicted": estimular,
    "desvio_entre_semillas": M.std(axis=0, ddof=1),
    **{f"probabilidad_{s}": P[s] for s in SEMILLAS},
}).sort_values("probabilidad", ascending=False)
detalle.to_csv("entrega_agosto_detalle.csv", index=False)

print(f"\n[4] ENTREGA LISTA", flush=True)
print(f"    entrega_agosto.csv          {len(entrega):,} filas (numero_de_cliente, Predicted)")
print(f"    entrega_agosto_detalle.csv  con probabilidades por semilla")
print(f"    clientes a estimular: {int(estimular.sum()):,} "
      f"({100*estimular.mean():.2f}% de la cartera)")
print(f"    metodo de ensamble: {METODO} | corte: {UMBRAL}")
print(f"    ganancia esperada si agosto se parece a junio: "
      f"~{int(estimular.sum())/8834*400.51:.0f}M (muy aproximado)")
con.close()
