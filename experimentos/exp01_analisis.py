"""Experimento 01 - Analisis: la REGLA DE CORTE.

Hipotesis experimental
----------------------
H1: el umbral teorico 0,025 no es el corte que maximiza la ganancia, porque
    las probabilidades de LightGBM no estan calibradas en la cola superior.
    Existe una regla de corte, con su parametro elegido FUERA del mes que se
    mide, que gana significativamente mas que 0,025.
H0: 0,025 es insuperable; cualquier ganancia extra viene de haber elegido el
    corte en el mismo mes donde se lo mide.

Por que importa: el break-even 0,025 = 27.500 / (1.072.500 + 27.500) es el
corte optimo SOLO si P(BAJA+2) esta bien calibrada. Si en la zona del corte el
modelo subestima el riesgo, hay clientes con probabilidad predicha < 0,025 cuya
tasa real de baja supera 0,025: son rentables y no se los estimula.

Diseno
------
Ceteris paribus estricto: un unico modelo (trial 25, 238 features, 150
iteraciones fijas), las mismas semillas, los mismos meses. Lo UNICO que cambia
entre brazos es la regla con la que se decide a quien estimular.

Seleccion cruzada entre meses: el parametro de cada regla se elige en un mes y
se evalua en el otro. Es lo que evita el sesgo de elegir el corte donde se mide
(ese sesgo vale ~+20M y es la trampa principal de este experimento).

Regla de decision, fijada ANTES de ver los resultados
-----------------------------------------------------
Una regla reemplaza a 0,025 solo si cumple las tres:
  1. gana en los dos sentidos del cruce (mayo->junio y junio->mayo);
  2. Wilcoxon de rangos signados apareado por semilla con p < 0,05;
  3. mejora >= +1% de la ganancia de 0,025 (umbral de relevancia practica).
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as cfg

ITERACIONES = 150


def cargar(fold: str):
    verdad = pd.read_csv(cfg.PREDICCIONES / f"verdad_{fold}.csv")
    clase = verdad["clase_ternaria"].to_numpy()
    gan = np.where(clase == "BAJA+2", cfg.GANANCIA_BAJA2, -cfg.COSTO_ESTIMULO)
    y = (clase == "BAJA+2").astype(np.int8)
    probas = {}
    for s in cfg.SEMILLAS_MEDICION:
        r = cfg.PREDICCIONES / f"p_{fold}_{s}_{ITERACIONES}.npy"
        if r.is_file():
            probas[s] = np.load(r).astype(np.float64)
    return gan, y, probas


def g_topk(gan, p, k):
    if k <= 0:
        return 0.0
    orden = np.argsort(-p, kind="mergesort")
    return float(gan[orden[:k]].sum())


def curva(gan, p):
    orden = np.argsort(-p, kind="mergesort")
    return np.concatenate(([0.0], np.cumsum(gan[orden])))


def k_optimo(gan, p):
    ac = curva(gan, p)
    return int(np.argmax(ac)), float(ac.max())


# ---------------------------------------------------------------- reglas
# Cada regla: ajusta(gan,y,p) -> parametro ; aplica(p,parametro) -> k
REGLAS = {}

REGLAS["R0_umbral_teorico"] = (
    lambda gan, y, p: cfg.UMBRAL_TEORICO,
    lambda p, u: int((p >= u).sum()),
)
REGLAS["R1_k_absoluto"] = (
    lambda gan, y, p: k_optimo(gan, p)[0],
    lambda p, k: int(min(k, len(p))),
)
REGLAS["R2_fraccion_poblacion"] = (
    lambda gan, y, p: k_optimo(gan, p)[0] / len(p),
    lambda p, f: int(round(f * len(p))),
)
REGLAS["R3_multiplo_masa_riesgo"] = (
    lambda gan, y, p: k_optimo(gan, p)[0] / p.sum(),
    lambda p, c: int(round(c * p.sum())),
)
REGLAS["R4_umbral_empirico"] = (
    lambda gan, y, p: float(np.sort(p)[::-1][max(k_optimo(gan, p)[0] - 1, 0)]),
    lambda p, u: int((p >= u).sum()),
)


def _isotonica(gan, y, p):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


REGLAS["R5_recalibrado_isotonico"] = (
    _isotonica,
    lambda p, iso: int((iso.predict(p) >= cfg.UMBRAL_TEORICO).sum()),
)


def main() -> None:
    datos = {f["nombre"]: cargar(f["nombre"]) for f in cfg.FOLDS}
    semillas = sorted(set(datos["mayo"][2]) & set(datos["junio"][2]))
    print(f"semillas disponibles en los dos folds: {len(semillas)}\n")

    for nombre, (gan, y, probas) in datos.items():
        n = len(gan)
        pos = int(y.sum())
        print(f"[{nombre}] {n:,} clientes, {pos:,} BAJA+2 ({100*pos/n:.3f}%), "
              f"techo {pos*cfg.GANANCIA_BAJA2/1e6:,.0f}M")
    print()

    # ---------------- resultado principal: cruce entre meses ----------------
    filas = []
    for ajuste, medicion in [("mayo", "junio"), ("junio", "mayo")]:
        gan_a, y_a, p_a = datos[ajuste]
        gan_m, y_m, p_m = datos[medicion]
        for regla, (fit, apply_) in REGLAS.items():
            for s in semillas:
                par = fit(gan_a, y_a, p_a[s])
                k = apply_(p_m[s], par)
                filas.append({
                    "regla": regla,
                    "ajuste": ajuste,
                    "medicion": medicion,
                    "semilla": s,
                    "k": k,
                    "ganancia": g_topk(gan_m, p_m[s], k),
                })
    res = pd.DataFrame(filas)
    res.to_csv(cfg.SALIDAS / "exp01_reglas_corte.csv", index=False)

    print("=== Ganancia por regla de corte (parametro elegido en el OTRO mes) ===")
    for medicion in ["junio", "mayo"]:
        sub = res[res.medicion == medicion]
        base = sub[sub.regla == "R0_umbral_teorico"].set_index("semilla").ganancia
        print(f"\n  medido en {medicion.upper()} (parametro ajustado en "
              f"{'mayo' if medicion=='junio' else 'junio'}):")
        print(f"    {'regla':26s} {'media':>9s} {'delta':>9s} {'desvio_dif':>11s} "
              f"{'p_wilcoxon':>11s} {'gana':>7s} {'k_medio':>8s}")
        for regla in REGLAS:
            d = sub[sub.regla == regla].set_index("semilla")
            g = d.ganancia
            dif = (g - base).values / 1e6
            if regla == "R0_umbral_teorico":
                print(f"    {regla:26s} {g.mean()/1e6:8.2f}M {'--':>9s} "
                      f"{'--':>11s} {'--':>11s} {'--':>7s} {d.k.mean():8.0f}")
                continue
            try:
                pw = stats.wilcoxon(dif).pvalue
                pw_s = f"{pw:.4f}"
            except Exception:
                pw_s = "n/a"
            print(f"    {regla:26s} {g.mean()/1e6:8.2f}M {dif.mean():+8.2f}M "
                  f"{dif.std(ddof=1):10.2f}M {pw_s:>11s} "
                  f"{int((dif>0).sum())}/{len(dif):<4d} {d.k.mean():8.0f}")

    # ---------------- cuanto vale el sesgo de elegir el corte in-sample -----
    print("\n=== Sesgo de elegir el corte en el MISMO mes (no es una opcion valida) ===")
    for nombre, (gan, y, probas) in datos.items():
        d0 = np.mean([g_topk(gan, probas[s], int((probas[s] >= cfg.UMBRAL_TEORICO).sum()))
                      for s in semillas]) / 1e6
        dop = np.mean([k_optimo(gan, probas[s])[1] for s in semillas]) / 1e6
        kop = np.mean([k_optimo(gan, probas[s])[0] for s in semillas])
        print(f"  {nombre}: 0,025 -> {d0:7.2f}M | optimo in-sample -> {dop:7.2f}M "
              f"(k={kop:.0f})  sesgo optimista {dop-d0:+.2f}M")

    # ---------------- calibracion: la causa mecanica ------------------------
    print("\n=== Calibracion en la cola superior (por que 0,025 corta temprano) ===")
    for nombre, (gan, y, probas) in datos.items():
        p = np.mean([probas[s] for s in semillas], axis=0)
        orden = np.argsort(-p)
        print(f"  {nombre}:  {'tramo':>14s} {'p_predicha':>11s} {'tasa_real':>10s} {'ratio':>7s}")
        for lo, hi in [(0, 2500), (2500, 5000), (5000, 10000), (10000, 20000)]:
            idx = orden[lo:hi]
            pr, re = p[idx].mean(), y[idx].mean()
            print(f"  {'':9s}{f'{lo}-{hi}':>14s} {pr:11.4f} {re:10.4f} "
                  f"{pr/max(re,1e-9):7.2f}")

    # ---------------- incertidumbre del leaderboard -------------------------
    print("\n=== Incertidumbre del leaderboard (Montecarlo 30%, metodo z301) ===")
    rng = np.random.default_rng(cfg.SEMILLAS_ENTREGA[0])
    for nombre, (gan, y, probas) in datos.items():
        p = np.mean([probas[s] for s in semillas], axis=0)
        ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
        est = p >= cfg.UMBRAL_TEORICO
        out = []
        for _ in range(400):
            m = np.concatenate([rng.choice(ip, int(.3*len(ip)), False),
                                rng.choice(ineg, int(.3*len(ineg)), False)])
            out.append(gan[m][est[m]].sum() / 0.3)
        out = np.array(out) / 1e6
        print(f"  {nombre}: media {out.mean():7.2f}M  desvio {out.std(ddof=1):6.2f}M  "
              f"IC95 [{np.percentile(out,2.5):.1f}, {np.percentile(out,97.5):.1f}]  "
              f"ancho {np.percentile(out,97.5)-np.percentile(out,2.5):.1f}M")

    print(f"\n[fin] detalle en {cfg.SALIDAS/'exp01_reglas_corte.csv'}")


if __name__ == "__main__":
    main()
