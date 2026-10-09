"""Etapa 11 - Datos sinteticos con la misma forma que la Primera Competencia.

PARA QUE SIRVE
--------------
1. Probar p10_adaptativo.py de punta a punta en segundos, antes de lanzar la
   corrida real de una hora (regla 11: correr primero en chico para no fallar
   en la linea 300).
2. Tener una VERDAD conocida. En los datos reales nunca se sabe si una
   decision del pipeline fue buena; aca si, porque el mecanismo que genera
   las bajas lo escribimos nosotros, y agosto tiene etiqueta oculta.

QUE IMITA DEL PROBLEMA REAL
---------------------------
- Una fila por (cliente, mes), meses 202103..202108, y la clase ternaria se
  DEDUCE de la presencia en M+1 y M+2, igual que en p01_base.py.
- Prevalencia de BAJA+2 baja (~1 %, el real es ~0,6 %: se sube un poco para
  que la demo tenga suficientes positivos con 30.000 clientes).
- Un riesgo latente que evoluciona por cliente (AR(1)). Las variables son
  mediciones ruidosas de ese riesgo, asi que la historia (lags, medias
  moviles) tiene senal real que el mes actual solo no tiene.
- Variables de monto con inflacion mensual: deriva honesta, PSI alto.
- Una variable constante, una variable "rota" (todo cero en 202105, como
  ccajas_depositos en el dataset real) y variables de ruido puro.
- Clientes que desaparecen un mes y vuelven, para ejercitar los lags por
  calendario y no por fila.
"""

import numpy as np
import pandas as pd

MESES = (202103, 202104, 202105, 202106, 202107, 202108)

# Variables con senal temporal: son las que el catalogo de la etapa 2 usa
# para construir lags, medias moviles y tendencias en modo sintetico.
VARIABLES_TEMPORALES = (
    "ctrx_quarter", "mpayroll", "mcuentas_saldo", "cmobile_app_trx",
    "mtarjeta_visa_consumo", "cproductos",
)


def _sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def generar(n_clientes: int = 30_000, semilla: int = 100129):
    """Devuelve (base, verdad_scoring).

    base : DataFrame con numero_de_cliente, foto_mes, features y
        clase_ternaria (NULL en 202107 y 202108, como en el real).
    verdad_scoring : DataFrame con la clase verdadera de 202108, que el
        pipeline NO ve. Solo sirve para auditar la entrega al final.
    """
    rng = np.random.default_rng(semilla)
    n_meses_simulados = len(MESES) + 2  # +2 para poder etiquetar 202108

    # Riesgo latente: persistente (AR 0,9) mas una TENDENCIA propia de cada
    # cliente. La tendencia es lo que el mes actual solo no revela y la
    # historia (lags, deltas, pendientes) si: es la senal que justifica la
    # etapa 2. Una parte de los clientes se esta deteriorando de verdad.
    tendencia = np.where(rng.random(n_clientes) < 0.15,
                         rng.normal(0.35, 0.15, n_clientes), rng.normal(0, 0.05, n_clientes))
    riesgo = np.zeros((n_clientes, n_meses_simulados))
    riesgo[:, 0] = rng.normal(0, 1, n_clientes)
    for t in range(1, n_meses_simulados):
        riesgo[:, t] = 0.9 * riesgo[:, t - 1] + tendencia + rng.normal(0, 0.3, n_clientes)

    # Presencia: el cliente que se va no vuelve. La probabilidad de irse
    # despues del mes t crece con el riesgo de ese mes.
    presente = np.ones((n_clientes, n_meses_simulados), dtype=bool)
    for t in range(1, n_meses_simulados):
        p_irse = _sigmoide(-6.3 + 1.6 * riesgo[:, t - 1])
        se_va = rng.random(n_clientes) < p_irse
        presente[:, t] = presente[:, t - 1] & ~se_va

    # Huecos: 60 clientes faltan un mes en el medio y vuelven.
    con_hueco = rng.choice(n_clientes, 60, replace=False)
    mes_hueco = rng.integers(1, len(MESES) - 1, 60)
    presente[con_hueco, mes_hueco] = False

    filas = []
    ids = np.arange(n_clientes) + 10_000_000
    tiene_visa = rng.random(n_clientes) < 0.7
    sueldo_base = rng.lognormal(12.5, 0.6, n_clientes)
    for t in range(n_meses_simulados):
        r = riesgo[:, t]
        inflacion = 1.035 ** t
        ruido = lambda escala: rng.normal(0, escala, n_clientes)
        df = pd.DataFrame({
            "numero_de_cliente": ids,
            "_t": t,
            "ctrx_quarter": rng.poisson(np.exp(3.2 - 0.45 * r)),
            "mpayroll": np.where(r + ruido(0.8) < 1.6,
                                 sueldo_base * inflacion * np.exp(ruido(0.1)), 0.0),
            "mcuentas_saldo": sueldo_base * inflacion * np.exp(0.3 - 0.35 * r + ruido(0.7)),
            "cmobile_app_trx": rng.poisson(np.exp(2.0 - 0.3 * r)),
            "mtarjeta_visa_consumo": np.where(
                tiene_visa, sueldo_base * 0.3 * inflacion * np.exp(-0.3 * r + ruido(0.6)), np.nan),
            "cproductos": np.clip(np.round(5 - 0.6 * r + ruido(1.0)), 1, 12),
            "cliente_edad": 0,  # se completa abajo
            "cliente_constante": 1.0,
            "ccajas_rota": rng.poisson(2.0, n_clientes).astype(float),
        })
        for j in range(8):
            df[f"ruido_{j}"] = rng.normal(0, 1, n_clientes)
        df = df[presente[:, t]]
        filas.append(df)

    base = pd.concat(filas, ignore_index=True)
    edad = pd.Series(rng.integers(20, 80, n_clientes), index=ids)
    base["cliente_edad"] = base["numero_de_cliente"].map(edad).to_numpy() + base["_t"] // 12

    # Calendario: los meses simulados mas alla de 202108 solo existen para
    # etiquetar; se descartan despues de construir la clase.
    calendario = list(MESES) + [202109, 202110]
    base["foto_mes"] = base["_t"].map(dict(enumerate(calendario)))

    # Variable rota: todo cero en 202105, como ccajas_depositos en el real.
    base.loc[base["foto_mes"] == 202105, "ccajas_rota"] = 0.0

    base = construir_clase_ternaria(base.drop(columns="_t"))
    verdad = base.loc[base["foto_mes"] == 202108,
                      ["numero_de_cliente", "clase_ternaria"]].copy()

    base = base[base["foto_mes"].isin(MESES)].copy()
    # Igual que en el real: sin M+2 observado, la clase no se puede saber.
    base.loc[base["foto_mes"] >= 202107, "clase_ternaria"] = None
    columnas = ["numero_de_cliente", "foto_mes"] + [
        c for c in base.columns if c not in ("numero_de_cliente", "foto_mes")]
    return base[columnas].reset_index(drop=True), verdad.reset_index(drop=True)


def construir_clase_ternaria(base: pd.DataFrame) -> pd.DataFrame:
    """Version pandas de la construccion de p01_base.py.

    Mismo criterio: la presencia se mira por CALENDARIO (indice de mes), no
    por la fila siguiente, asi un cliente con hueco en abril es BAJA+1 en
    marzo y no CONTINUA.
    """
    base = base.copy()
    indice = (base["foto_mes"] // 100) * 12 + base["foto_mes"] % 100
    presentes = set(zip(base["numero_de_cliente"], indice))
    ultimo = indice.max()
    m1 = np.array([(c, i + 1) in presentes for c, i in zip(base["numero_de_cliente"], indice)])
    m2 = np.array([(c, i + 2) in presentes for c, i in zip(base["numero_de_cliente"], indice)])
    clase = np.where(~m1, "BAJA+1", np.where(~m2, "BAJA+2", "CONTINUA")).astype(object)
    # Borde de la ventana: no se puede observar M+1 / M+2.
    clase[(indice + 1 > ultimo).to_numpy()] = None
    clase[((indice + 2 > ultimo) & m1).to_numpy()] = None
    base["clase_ternaria"] = clase
    return base


if __name__ == "__main__":
    b, v = generar()
    print(b.shape)
    print(b.groupby("foto_mes")["clase_ternaria"].value_counts(dropna=False).unstack())
    print(v["clase_ternaria"].value_counts())
