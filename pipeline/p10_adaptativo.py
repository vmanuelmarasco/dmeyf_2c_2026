"""Etapa 10 - Pipeline secuencial y adaptativo.

QUE ES
------
Un pipeline que no ejecuta pasos a ciegas: cada etapa mide, y lo que mide
decide que hace la siguiente. Todas las decisiones quedan escritas, con su
evidencia numerica, en un objeto de contexto (ContextoPipeline) que se
persiste a disco despues de cada decision.

    Etapa 1  Ingesta y diagnostico     -> gates, data quality, desbalance
    Etapa 2  Ingenieria iterativa      -> cada bloque entra solo si gana
    Etapa 3  Optimizacion condicional  -> espacio segun dimensionalidad, poda
    Etapa 4  Corte y entrega           -> regla de corte, modelo final, CSV

LO QUE SE HEREDA DE competencia.ipynb Y DEL EXPERIMENTO 01
----------------------------------------------------------
Cada uno de estos puntos es una decision que se tomo por un error o un
resultado previo. No son preferencias de estilo.

1. NO hay k-fold aleatorio. La validacion es temporal con gap de 2 meses
   (p00_config.FOLDS), porque k-fold mezcla meses y clientes. El primer
   baseline de competencia.ipynb tenia gap 1 y daba 311 M inflados.
2. La unidad de comparacion es la SEMILLA, apareada. El ruido que domina es
   el del arbol (desvio entre semillas: 13,18 M en junio), no el de la
   muestra. Cinco semillas solo resuelven efectos de ~20 M: por eso el banco
   usa 20 y Wilcoxon apareado.
3. Un cambio gana solo si gana en LOS DOS folds. Mayo y junio se contradicen
   en iteraciones y en corte (Exp01, ambos p < 0,01).
4. Iteraciones FIJAS (150) e identicas entre brazos. Sin early stopping: el
   mes de validacion no participa de ninguna seleccion.
5. La ganancia se calcula SIEMPRE desde clase_ternaria, nunca desde el
   target binario.
6. El corte 0,025 se mantiene salvo que otra regla, elegida en un mes y
   medida en el otro, gane en los dos sentidos (Exp01: ninguna lo logro).
7. Nada que cambie el modelo entra sin medirse. Las correcciones de data
   quality que detecta la etapa 1 no se aplican: se encolan como candidatos
   de la etapa 2 y compiten como cualquier feature.

DONDE ME APARTO DE LA PLANTILLA, Y POR QUE
------------------------------------------
- "Validacion cruzada Out-of-Fold": reemplazada por los folds temporales de
  arriba. OOF aleatorio es fuga en este problema.
- "Reinyectar la metrica para corregir en el siguiente intento": solo se
  reintenta ante fallas TECNICAS (error, columna faltante, fuga de futuro).
  Reintentar variantes de un bloque porque "no mejoro" es elegir y medir en
  el mismo lugar (fuga F6): con suficientes intentos, alguno gana por azar.
- "Ajustar umbrales segun costo/beneficio": el umbral de costo/beneficio YA
  es 0,025 (= 27.500 / 1.100.000). La etapa 4 solo lo cambia si una regla
  alternativa supera la prueba cruzada entre meses.

USO
---
    # demo en segundos (datos sinteticos)
    python3 -u pipeline/p10_adaptativo.py --modo sintetico

    # smoke test real (2 semillas, 6 trials) antes de la corrida larga
    python3 -u pipeline/p10_adaptativo.py --modo real --max-semillas 2 --optuna-trials 6

    # corrida real completa (~1 h): desacoplada de la sesion, regla 11
    nohup python3 -u pipeline/p10_adaptativo.py --modo real \\
        > pipeline/salidas/p10_real.log 2>&1 &

Codigos de salida: 0 ok, 2 fallo un gate (datos o fuga), 1 otro error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import p00_config as cfg

# Se sube cuando cambia algo que invalida las mediciones cacheadas
# (por ejemplo, la forma de entrenar o de calcular la ganancia).
VERSION_BANCO = 1

COLUMNAS_CLAVE = ("numero_de_cliente", "foto_mes", "clase_ternaria")
COLUMNAS_PROHIBIDAS = frozenset({
    "numero_de_cliente", "foto_mes", "clase_ternaria", "target_baja2",
})


# ==========================================================================
# DECISIONES DE MODELADO
# ==========================================================================
# Todo lo que esta aca es una decision de modelado, no de infraestructura.
# Cada valor hay que poder defenderlo en el examen del 10 de diciembre.
# ==========================================================================

DECISIONES_DE_MODELADO = {
    # Banco de medicion (Exp01: 150 fijas, 20 semillas, los dos folds).
    "iteraciones": 150,
    "semillas_medicion": cfg.SEMILLAS_MEDICION,
    "semillas_entrega": cfg.SEMILLAS_ENTREGA,

    # Criterio para que un cambio entre (fijado ANTES de ver resultados).
    "alpha": 0.05,                    # Wilcoxon apareado, unilateral
    "mejora_minima_relativa": 0.01,   # +1 % de la ganancia del incumbente

    # Poda secuencial: cada cuantas semillas se revisa si seguir midiendo.
    "semillas_antes_de_podar": 5,
    "paso_semillas": 5,
    "alpha_poda": 0.10,

    # Etapa 1 - diagnostico.
    "umbral_desbalance": 0.05,        # prevalencia del positivo por debajo
    "tasas_undersampling": (0.5, 0.25, 0.10),
    "margen_no_inferioridad": 0.01,   # se acepta perder hasta 1 %...
    "aceleracion_minima": 1.5,        # ...solo si entrena 1,5x mas rapido
    "psi_alto": 0.25,
    "fraccion_rota": 0.99,            # ceros o nulos en un mes...
    "fraccion_normal_max": 0.90,      # ...cuando en el resto de los meses es menor

    # Etapa 2 - ingenieria.
    "max_intentos_por_candidato": 2,
    "mes_corte_gate_f5": 202105,

    # Etapa 3 - optimizacion. Se elige en mayo y se confirma en junio.
    "optuna_trials": 40,
    "optuna_timeout_min": 45,
    "optuna_checkpoints": (50, 100, 150),
    "optuna_fold_seleccion": "mayo",
    "optuna_folds_confirmacion": ("junio",),
    "optuna_top_confirmar": 3,
    "piso_relativo_primer_checkpoint": 0.5,
}

# El perfil sintetico achica el presupuesto: la idea es que corra en minutos.
PERFIL_SINTETICO = {
    "semillas_medicion": cfg.SEMILLAS_MEDICION[:10],
    "optuna_trials": 15,
    "optuna_timeout_min": 5,
    "semillas_antes_de_podar": 5,
    "paso_semillas": 5,
}

PARAMETROS_BASE_SINTETICO = {
    "objective": "binary", "metric": "None", "verbosity": -1,
    "feature_pre_filter": False, "boost_from_average": True,
    "learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 300,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "max_bin": 63,
    # Con 30.000 filas, 14 threads pasan mas tiempo coordinandose que
    # entrenando (la primera corrida gasto 770 s de CPU de sistema).
    "num_threads": 4, "force_col_wise": True,
}


# ==========================================================================
# Excepciones: separan "hay que abortar" de "se descarta y se sigue"
# ==========================================================================

class GateFallido(Exception):
    """Invariante de datos roto. El pipeline NO puede seguir: cualquier
    resultado posterior seria sobre datos malos."""


class FugaDeFuturo(Exception):
    """Un candidato lee meses posteriores. Se descarta el candidato, no el
    pipeline."""

    def __init__(self, columnas):
        self.columnas = list(columnas)
        super().__init__(f"F5: {len(self.columnas)} columnas leen el futuro: "
                         f"{self.columnas[:8]}")


class SinHistoria(Exception):
    """El bloque solo produce columnas que no existen en el train de algun
    fold: la ventana temporal no alcanza para medirlo."""

    def __init__(self, columnas):
        self.columnas = list(columnas)
        super().__init__(f"{len(self.columnas)} columnas sin historia en el train de "
                         f"algun fold (la ventana no alcanza): {self.columnas[:6]}")


class ColumnasFaltantes(Exception):
    def __init__(self, columnas):
        self.columnas = list(columnas)
        super().__init__(f"faltan columnas: {self.columnas}")


# ==========================================================================
# Contexto: el estado del pipeline
# ==========================================================================

@dataclass
class Decision:
    etapa: str
    punto: str
    resultado: str
    motivo: str
    evidencia: dict = field(default_factory=dict)
    momento: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


@dataclass
class Brazo:
    """Una configuracion completa a medir: features + parametros + muestreo."""
    nombre: str
    features: tuple
    params: dict
    tasa_continua: float = 1.0   # 1.0 = sin undersampling
    iteraciones: int = 150

    def con(self, **cambios) -> "Brazo":
        datos = asdict(self)
        datos.update(cambios)
        datos["features"] = tuple(datos["features"])
        return Brazo(**datos)


class ContextoPipeline:
    """Guarda metricas, transformaciones aprobadas y el log de decisiones.

    Se persiste completo (estado.json) despues de cada decision, y cada
    decision ademas se agrega a decisiones.jsonl apenas se toma. Si el
    proceso muere, lo decidido hasta ahi queda en disco.
    """

    def __init__(self, modo: str, carpeta: Path, decisiones_modelado: dict):
        self.modo = modo
        self.carpeta = carpeta
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.dm = decisiones_modelado
        self.inicio = datetime.now().isoformat(timespec="seconds")
        self.decisiones: list[Decision] = []
        self.errores: list[dict] = []
        self.diagnostico: dict = {}
        self.metricas: dict = {}
        self.incumbente: Optional[Brazo] = None
        self.historial_incumbentes: list[str] = []
        self.bloques_aprobados: list[dict] = []
        self.bloques_rechazados: list[dict] = []
        self.candidatos_pendientes_etapa2: list = []
        self.regla_corte: dict = {"nombre": "R0_umbral_teorico",
                                  "umbral": cfg.UMBRAL_TEORICO}
        self.entrega: dict = {}
        self.etapas_completas: list[str] = []
        self._ruta_log = carpeta / "log.txt"

    # ---- registro ---------------------------------------------------------

    def log(self, mensaje: str) -> None:
        linea = f"{datetime.now():%H:%M:%S} {mensaje}"
        print(linea, flush=True)
        with open(self._ruta_log, "a") as f:
            f.write(linea + "\n")

    def decidir(self, etapa, punto, resultado, motivo, **evidencia) -> Decision:
        d = Decision(etapa, punto, resultado, motivo, _a_json(evidencia))
        self.decisiones.append(d)
        with open(self.carpeta / "decisiones.jsonl", "a") as f:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
        self.log(f"[{etapa}] DECISION {punto}: {resultado} -- {motivo}")
        self.guardar()
        return d

    def registrar_error(self, etapa, donde, error: BaseException) -> dict:
        info = {"etapa": etapa, "donde": donde, "tipo": type(error).__name__,
                "mensaje": str(error)[:2000],
                "traceback": traceback.format_exc()[-3000:]}
        self.errores.append(info)
        self.log(f"[{etapa}] ERROR en {donde}: {info['tipo']}: {info['mensaje'][:300]}")
        self.guardar()
        return info

    def nuevo_incumbente(self, brazo: Brazo, motivo: str) -> None:
        self.incumbente = brazo
        self.historial_incumbentes.append(f"{brazo.nombre}: {motivo}")
        self.guardar()

    def guardar(self) -> None:
        estado = {
            "modo": self.modo, "inicio": self.inicio,
            "etapas_completas": self.etapas_completas,
            "incumbente": None if self.incumbente is None else {
                "nombre": self.incumbente.nombre,
                "n_features": len(self.incumbente.features),
                "features": list(self.incumbente.features),
                "params": self.incumbente.params,
                "tasa_continua": self.incumbente.tasa_continua,
                "iteraciones": self.incumbente.iteraciones},
            "historial_incumbentes": self.historial_incumbentes,
            "metricas": self.metricas,
            "bloques_aprobados": self.bloques_aprobados,
            "bloques_rechazados": self.bloques_rechazados,
            "regla_corte": self.regla_corte,
            "entrega": self.entrega,
            "diagnostico": self.diagnostico,
            "decisiones": [asdict(d) for d in self.decisiones],
            "errores": self.errores,
            "decisiones_de_modelado": self.dm,
        }
        tmp = self.carpeta / "estado.json.tmp"
        tmp.write_text(json.dumps(_a_json(estado), indent=2, ensure_ascii=False))
        tmp.replace(self.carpeta / "estado.json")  # escritura atomica

    def tabla_decisiones(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(d) for d in self.decisiones])


def _a_json(x):
    """Convierte numpy/tuplas/Path a tipos que json entiende."""
    if isinstance(x, dict):
        return {str(k): _a_json(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_a_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


# ==========================================================================
# Utilidades del problema
# ==========================================================================

def ganancia_por_fila(clase_ternaria) -> np.ndarray:
    """SIEMPRE desde la ternaria: BAJA+1 y CONTINUA cuestan lo mismo pero son
    clientes distintos, y la binaria de entrenamiento los confunde."""
    return np.where(np.asarray(clase_ternaria) == "BAJA+2",
                    cfg.GANANCIA_BAJA2, -cfg.COSTO_ESTIMULO).astype(np.float64)


def corregir_probabilidad(p: np.ndarray, tasa_continua: float) -> np.ndarray:
    """Deshace el sesgo del undersampling de CONTINUA.

    Si se conserva una fraccion t de los negativos, el modelo aprende odds
    inflados por 1/t. Multiplicando los odds por t se vuelve a la escala
    original, y recien ahi tiene sentido cortar en 0,025. Sin esta correccion,
    undersamplear y cortar en 0,025 estimula de mas.
    (Es aproximada porque BAJA+1 no se submuestrea, pero BAJA+1 es ~0,6 % de
    los negativos.)
    """
    if tasa_continua >= 1.0:
        return p
    return tasa_continua * p / (tasa_continua * p + 1.0 - p)


def huella_columna(serie: pd.Series) -> str:
    """Firma barata del contenido de una columna, para el cache del banco:
    si el codigo de una feature cambia, su firma cambia y no se reusan
    mediciones viejas por error."""
    v = pd.to_numeric(serie, errors="coerce").to_numpy(dtype=np.float64)
    finitos = v[np.isfinite(v)]
    return f"{len(v)}:{np.isnan(v).sum()}:{finitos.sum():.6e}:{(finitos ** 2).sum():.6e}"


def p_wilcoxon(diferencias, alternativa: str) -> float:
    d = np.asarray(diferencias, dtype=float)
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, alternative=alternativa).pvalue)
    except ValueError:
        return 1.0


def lag_por_calendario(df: pd.DataFrame, variables, k: int) -> pd.DataFrame:
    """Valor de cada variable k meses ANTES, buscado por calendario.

    No se usa groupby().shift(k): si el cliente falto un mes, shift leeria
    una fila que no es la de hace k meses (el mismo problema de los 23
    clientes con hueco que obliga al CROSS JOIN en p01_base.py).
    """
    indice = cfg.indice_mes(df["foto_mes"])
    pasado = df[["numero_de_cliente"] + list(variables)].copy()
    pasado["_im"] = indice + k  # la fila del mes M-k se "mueve" al mes M
    actual = pd.DataFrame({"numero_de_cliente": df["numero_de_cliente"].to_numpy(),
                           "_im": indice.to_numpy()})
    unido = actual.merge(pasado, on=["numero_de_cliente", "_im"], how="left")
    unido.index = df.index
    return unido[list(variables)]


# ==========================================================================
# Catalogo de candidatos de la etapa 2
# ==========================================================================

@dataclass
class Candidato:
    """Un bloque de transformaciones que compite por entrar al modelo.

    construir(df, params) -> (nuevas_columnas: DataFrame, columnas_a_quitar)
    Recibe la tabla como parametro (exigencia del gate F5: se tiene que poder
    construir sobre una tabla truncada).
    """
    nombre: str
    construir: Callable
    params: dict
    origen: str = "catalogo"
    intento: int = 1


def _verificar_columnas(df, variables):
    faltan = [v for v in variables if v not in df.columns]
    if faltan:
        raise ColumnasFaltantes(faltan)


def fe_lag_delta(df, params):
    k, variables = params["k"], params["variables"]
    _verificar_columnas(df, variables)
    lag = lag_por_calendario(df, variables, k)
    nuevas = {}
    for v in variables:
        nuevas[f"{v}__lag{k}"] = lag[v]
        nuevas[f"{v}__delta{k}"] = df[v] - lag[v]
    return pd.DataFrame(nuevas, index=df.index), []


def fe_ventana3(df, params):
    """Media movil de 3 meses, ratio contra la media y tendencia.

    Con 3 puntos equiespaciados, la pendiente de minimos cuadrados (lo que
    calcula regr_slope en SQL) es exactamente (x_t - x_{t-2}) / 2.
    """
    variables = params["variables"]
    _verificar_columnas(df, variables)
    l1 = lag_por_calendario(df, variables, 1)
    l2 = lag_por_calendario(df, variables, 2)
    nuevas = {}
    for v in variables:
        tres = pd.concat([df[v], l1[v], l2[v]], axis=1)
        media = tres.mean(axis=1, skipna=False)
        nuevas[f"{v}__media3"] = media
        nuevas[f"{v}__ratio_media3"] = df[v] / media.where(media.abs() > 0)
        nuevas[f"{v}__tendencia3"] = (df[v] - l2[v]) / 2.0
    return pd.DataFrame(nuevas, index=df.index), []


def fe_rank(df, params):
    """Rank percentil de montos. Con particion='mes' neutraliza la inflacion
    sin mirar el futuro. Con particion='global' rankea contra TODA la tabla,
    incluido el mes de scoring: es la fuga F5 de manual, y el gate la ataja."""
    variables = params["variables"]
    _verificar_columnas(df, variables)
    nuevas = {}
    for v in variables:
        if params.get("particion", "mes") == "global":
            nuevas[f"{v}__rank"] = df[v].rank(pct=True)
        else:
            nuevas[f"{v}__rank"] = df.groupby("foto_mes")[v].rank(pct=True)
    return pd.DataFrame(nuevas, index=df.index), []


def fe_reparar_rotas(df, params):
    """Pone NULL en los meses donde la variable esta rota, en vez de dejar
    ceros que el modelo lee como 'el cliente no tuvo actividad'."""
    nuevas, quitar = {}, []
    for v, meses in params["rotas"].items():
        if v not in df.columns:
            continue
        nuevas[f"{v}__reparada"] = df[v].where(~df["foto_mes"].isin(meses))
        quitar.append(v)
    return pd.DataFrame(nuevas, index=df.index), quitar


def fe_quitar(df, params):
    """Ablacion: sacar columnas (por ejemplo, las de deriva alta)."""
    return pd.DataFrame(index=df.index), list(params["columnas"])


def fe_trampa_lead(df, params):
    """SOLO DEMO SINTETICA: usa el valor del mes SIGUIENTE. Existe para
    mostrar que el gate F5 descarta una feature que lee el futuro."""
    variables = params["variables"]
    _verificar_columnas(df, variables)
    futuro = lag_por_calendario(df, variables, -1)
    return pd.DataFrame({f"{v}__lead1": futuro[v] for v in variables}, index=df.index), []


def catalogo_candidatos(modo: str, variables_temporales, montos) -> list:
    """Orden en que se prueban los bloques. El orden importa: la seleccion es
    golosa, y cada bloque se compara contra el incumbente de ese momento."""
    vt = list(variables_temporales)
    cands = [
        Candidato("lag2_delta2", fe_lag_delta, {"k": 2, "variables": vt}),
        Candidato("ventana3", fe_ventana3, {"variables": vt}),
        Candidato("rank_montos_mes", fe_rank, {"variables": list(montos), "particion": "mes"}),
    ]
    if modo == "sintetico":
        # En el real, lag1/delta1 ya esta dentro de las 238 features.
        cands.insert(0, Candidato("lag1_delta1", fe_lag_delta, {"k": 1, "variables": vt}))
        # Tres candidatos que existen para ejercitar los caminos de error:
        cands += [
            # fuga: el gate F5 lo descarta (y no hay correccion posible)
            Candidato("trampa_lead1", fe_trampa_lead, {"variables": vt[:2]}, origen="demo"),
            # fuga corregible: el corrector cambia particion global -> mes
            Candidato("rank_global", fe_rank,
                      {"variables": ["mcuentas_saldo"], "particion": "global"}, origen="demo"),
            # error tecnico corregible: una variable que no existe
            Candidato("lag3_con_typo", fe_lag_delta,
                      {"k": 3, "variables": vt[:3] + ["variable_que_no_existe"]}, origen="demo"),
        ]
    return cands


def corregir_candidato(cand: Candidato, diagnostico: dict) -> Optional[Candidato]:
    """Retroalimentacion: a partir del diagnostico de la falla, propone UNA
    variante corregida, o None si no hay correccion legitima.

    Solo se corrigen fallas tecnicas. Una degradacion de la ganancia NO se
    corrige: tunear el bloque mirando la validacion es fuga de seleccion.
    """
    p = dict(cand.params)
    tipo = diagnostico["tipo"]
    if tipo == "columnas_faltantes":
        quedan = [v for v in p.get("variables", []) if v not in diagnostico["columnas"]]
        if not quedan:
            return None
        p["variables"] = quedan
        motivo = f"sin {diagnostico['columnas']}"
    elif tipo == "fuga" and p.get("particion") == "global":
        p["particion"] = "mes"
        motivo = "particion global -> por mes"
    elif tipo == "fuga" and "variables" in p and cand.construir is not fe_trampa_lead:
        culpables = {c.split("__")[0] for c in diagnostico["columnas"]}
        quedan = [v for v in p["variables"] if v not in culpables]
        if not quedan:
            return None
        p["variables"] = quedan
        motivo = f"sin las variables que fugan: {sorted(culpables)}"
    else:
        return None
    nuevo = Candidato(f"{cand.nombre}~v{cand.intento + 1}", cand.construir, p,
                      origen=f"correccion de {cand.nombre} ({motivo})",
                      intento=cand.intento + 1)
    return nuevo


def gate_sin_fuga_pandas(construir, params, df, mes_corte) -> None:
    """F5 exacto (version pandas de p02_fuga.gate_sin_fuga_de_futuro).

    Construye el bloque sobre la tabla completa y sobre la tabla truncada en
    mes_corte, y compara las filas <= mes_corte. Si una feature solo mira el
    pasado, truncar el futuro no puede cambiarla. Cualquier diferencia es
    prueba, no indicio.
    """
    completa, _ = construir(df, params)
    m = df["foto_mes"] <= mes_corte
    truncada, _ = construir(df.loc[m], params)
    culpables = []
    for c in completa.columns:
        a = completa.loc[m, c].to_numpy(dtype=np.float64)
        b = truncada[c].reindex(completa.index[m]).to_numpy(dtype=np.float64)
        iguales = (a == b) | (np.isnan(a) & np.isnan(b))
        if not iguales.all():
            culpables.append(c)
    if culpables:
        raise FugaDeFuturo(culpables)


# ==========================================================================
# Banco de medicion: entrena, mide y cachea
# ==========================================================================

class BancoMedicion:
    """Mide un Brazo en los folds temporales, una semilla por vez.

    Cada medicion se agrega a disco apenas termina (regla 11): si el proceso
    se cae, al relanzar se reusan las mediciones y solo se entrena lo que
    falta. La clave del cache es la firma del brazo, que incluye la huella del
    CONTENIDO de cada columna, no solo su nombre.
    """

    def __init__(self, ctx: ContextoPipeline, base: pd.DataFrame, carpeta_cache: Path):
        self.ctx = ctx
        self.base = base
        self.carpeta = carpeta_cache
        (self.carpeta / "pred").mkdir(parents=True, exist_ok=True)
        self.ruta_cache = self.carpeta / "mediciones.csv"
        self.cache = {}
        if self.ruta_cache.is_file():
            for r in pd.read_csv(self.ruta_cache).to_dict("records"):
                self.cache[(r["firma"], r["fold"], int(r["semilla"]))] = r
        self.huellas: dict = {}
        self.folds = {f["nombre"]: f for f in cfg.FOLDS}
        self._matrices = {}
        clase = base["clase_ternaria"].to_numpy()
        self._mascaras = {}
        for nombre, f in self.folds.items():
            mt = base["foto_mes"].isin(f["train"]).to_numpy()
            mv = (base["foto_mes"] == f["validacion"]).to_numpy()
            self._mascaras[nombre] = (mt, mv)
        self.clase = clase
        self.y = (clase == "BAJA+2").astype(np.int8)

    def registrar_huellas(self, columnas) -> None:
        for c in columnas:
            if c not in self.huellas:
                self.huellas[c] = huella_columna(self.base[c])

    def olvidar(self, columnas) -> None:
        """Se llama al descartar columnas: si despues se vuelve a crear una
        columna con el mismo nombre (una variante corregida), su huella y su
        matriz tienen que recalcularse, no reusarse."""
        columnas = set(columnas)
        for c in columnas:
            self.huellas.pop(c, None)
        for clave in [k for k in self._matrices if columnas.intersection(k[0])]:
            self._matrices.pop(clave)

    def firma(self, b: Brazo) -> str:
        self.registrar_huellas(b.features)
        contenido = json.dumps({
            "v": VERSION_BANCO, "modo": self.ctx.modo,
            "features": [(f, self.huellas[f]) for f in b.features],
            "params": b.params, "tasa": b.tasa_continua, "iter": b.iteraciones,
        }, sort_keys=True, default=str)
        return hashlib.sha1(contenido.encode()).hexdigest()

    def _matriz(self, b: Brazo, fold: str):
        clave = (tuple(b.features), fold)
        if clave not in self._matrices:
            if len(self._matrices) >= 4:  # memoria: el real pesa ~0,3 GB por matriz
                self._matrices.pop(next(iter(self._matrices)))
            mt, mv = self._mascaras[fold]
            X = self.base[list(b.features)]
            self._matrices[clave] = (X.loc[mt].to_numpy(dtype=np.float32),
                                     X.loc[mv].to_numpy(dtype=np.float32))
        return self._matrices[clave]

    def entrenar_y_predecir(self, b: Brazo, fold: str, semilla: int):
        mt, mv = self._mascaras[fold]
        X_tr, X_va = self._matriz(b, fold)
        y_tr = self.y[mt]
        clase_tr = self.clase[mt]
        if b.tasa_continua < 1.0:
            rng = np.random.default_rng(semilla)
            keep = (clase_tr != "CONTINUA") | (rng.random(len(y_tr)) < b.tasa_continua)
            X_tr, y_tr = X_tr[keep], y_tr[keep]
        params = {**b.params, "seed": semilla, "bagging_seed": semilla,
                  "feature_fraction_seed": semilla, "data_random_seed": semilla,
                  "verbosity": -1}
        t0 = time.time()
        modelo = lgb.train(params, lgb.Dataset(X_tr, label=y_tr),
                           num_boost_round=b.iteraciones)
        p = corregir_probabilidad(modelo.predict(X_va), b.tasa_continua)
        return p, time.time() - t0

    def medir(self, b: Brazo, semillas, folds=None, guardar_pred=True) -> pd.DataFrame:
        firma = self.firma(b)
        folds = list(folds or self.folds)
        filas = []
        for fold in folds:
            _, mv = self._mascaras[fold]
            gan = ganancia_por_fila(self.clase[mv])
            for s in semillas:
                ruta_pred = self.carpeta / "pred" / f"{firma[:16]}_{fold}_{s}.npy"
                clave = (firma, fold, int(s))
                if clave in self.cache and (not guardar_pred or ruta_pred.is_file()):
                    filas.append(self.cache[clave])
                    continue
                p, seg = self.entrenar_y_predecir(b, fold, s)
                estim = p >= cfg.UMBRAL_TEORICO
                r = {"firma": firma, "brazo": b.nombre, "fold": fold, "semilla": int(s),
                     "ganancia": float(gan[estim].sum()), "estimulados": int(estim.sum()),
                     "suma_p": float(p.sum()), "segundos": round(seg, 2)}
                if guardar_pred:
                    np.save(ruta_pred, p.astype(np.float32))
                self.cache[clave] = r
                pd.DataFrame([r]).to_csv(self.ruta_cache, mode="a", index=False,
                                         header=not self.ruta_cache.is_file())
                filas.append(r)
        return pd.DataFrame(filas)

    def predicciones(self, b: Brazo, fold: str, semilla: int) -> np.ndarray:
        firma = self.firma(b)
        ruta = self.carpeta / "pred" / f"{firma[:16]}_{fold}_{semilla}.npy"
        if not ruta.is_file():
            self.medir(b, [semilla], [fold], guardar_pred=True)
        return np.load(ruta).astype(np.float64)

    # ---- comparacion apareada con poda secuencial -----------------------

    def comparar(self, inc: Brazo, cand: Brazo, folds=None, modo="superioridad",
                 alpha=None) -> dict:
        """Compara apareado por semilla. Mide de a tandas y PODA en cuanto la
        evidencia alcanza para saber que el candidato no va a ganar.

        modo='superioridad'   : gana si en TODOS los folds mejora con
                                p < alpha y mejora >= mejora_minima.
        modo='no_inferioridad': pasa si en ningun fold es peor con
                                p < alpha y pierde menos del margen.
        """
        dm = self.ctx.dm
        alpha = alpha or dm["alpha"]
        folds = list(folds or self.folds)
        semillas = list(dm["semillas_medicion"])
        n = min(dm["semillas_antes_de_podar"], len(semillas))
        while True:
            m_inc = self.medir(inc, semillas[:n], folds)
            m_can = self.medir(cand, semillas[:n], folds)
            por_fold = {}
            for f in folds:
                a = m_inc[m_inc.fold == f].set_index("semilla")["ganancia"]
                b = m_can[m_can.fold == f].set_index("semilla")["ganancia"]
                d = (b - a.reindex(b.index)).to_numpy()
                media_inc = float(a.mean())
                t = stats.t.ppf(0.95, max(len(d) - 1, 1))
                se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.inf
                por_fold[f] = {
                    "n": len(d), "media_incumbente": media_inc,
                    "media_candidato": float(b.mean()),
                    "delta": float(d.mean()),
                    # abs(): con un incumbente en perdida, el signo no se invierte.
                    "delta_rel": float(d.mean() / max(abs(media_inc), 1.0)),
                    "cota_sup_90": float(d.mean() + t * se),
                    "p_mejor": p_wilcoxon(d, "greater"),
                    "p_peor": p_wilcoxon(d, "less"),
                    "semillas_a_favor": int((d > 0).sum()),
                    "seg_incumbente": float(m_inc[m_inc.fold == f]["segundos"].mean()),
                    "seg_candidato": float(m_can[m_can.fold == f]["segundos"].mean()),
                }
            quedan_semillas = n < len(semillas)

            if modo == "superioridad":
                minimo = {f: dm["mejora_minima_relativa"] * abs(por_fold[f]["media_incumbente"])
                          for f in folds}
                peor = [f for f in folds if por_fold[f]["p_peor"] < dm["alpha_poda"]]
                inalcanzable = [f for f in folds if por_fold[f]["cota_sup_90"] < minimo[f]]
                if quedan_semillas and (peor or inalcanzable):
                    motivo = (f"podado con {n} semillas: peor con evidencia en {peor}"
                              if peor else
                              f"podado con {n} semillas: la cota superior no alcanza la "
                              f"mejora minima en {inalcanzable}")
                    return {"aceptado": False, "podado": True, "motivo": motivo,
                            "por_fold": por_fold}
                if quedan_semillas:
                    n = min(n + dm["paso_semillas"], len(semillas))
                    continue
                ok = all(por_fold[f]["p_mejor"] < alpha and
                         por_fold[f]["delta"] >= minimo[f] for f in folds)
                motivo = ("gana en todos los folds" if ok else
                          "no gana en todos los folds con significancia y mejora minima")
            else:
                peor = [f for f in folds if por_fold[f]["p_peor"] < alpha]
                if quedan_semillas and peor:
                    return {"aceptado": False, "podado": True,
                            "motivo": f"podado con {n} semillas: peor con evidencia en {peor}",
                            "por_fold": por_fold}
                if quedan_semillas:
                    n = min(n + dm["paso_semillas"], len(semillas))
                    continue
                ok = (not peor) and all(por_fold[f]["delta_rel"] >= -dm["margen_no_inferioridad"]
                                        for f in folds)
                motivo = ("no inferior en todos los folds" if ok else
                          "inferior en algun fold")
            return {"aceptado": ok, "podado": False, "motivo": motivo, "por_fold": por_fold}


def resumen_fold(veredicto: dict) -> str:
    partes = []
    for f, r in veredicto["por_fold"].items():
        partes.append(f"{f}: {r['delta'] / 1e6:+.2f}M ({r['delta_rel']:+.2%}) "
                      f"p={r['p_mejor']:.3f} {r['semillas_a_favor']}/{r['n']}")
    return " | ".join(partes)


# ==========================================================================
# El pipeline
# ==========================================================================

class PipelineAdaptativo:

    def __init__(self, modo: str, carpeta: Optional[Path] = None,
                 max_semillas: Optional[int] = None, **cambios_modelado):
        dm = dict(DECISIONES_DE_MODELADO)
        if modo == "sintetico":
            dm.update(PERFIL_SINTETICO)
        dm.update(cambios_modelado)
        if max_semillas:
            dm["semillas_medicion"] = tuple(dm["semillas_medicion"])[:max_semillas]
            dm["semillas_antes_de_podar"] = min(dm["semillas_antes_de_podar"], max_semillas)
        sello = datetime.now().strftime("%Y%m%d_%H%M%S")
        carpeta = carpeta or (cfg.SALIDAS / f"p10_{modo}_{sello}")
        self.ctx = ContextoPipeline(modo, carpeta, dm)
        self.carpeta_cache = cfg.SALIDAS / f"p10_cache_{modo}"
        self.base: Optional[pd.DataFrame] = None
        self.banco: Optional[BancoMedicion] = None
        self.verdad_scoring = None
        self.variables_temporales: list = []
        self.montos: list = []
        self.parametro_regla = None

    # ---- carga ------------------------------------------------------------

    def _cargar(self):
        ctx = self.ctx
        if ctx.modo == "sintetico":
            import p11_sintetico as sint
            base, self.verdad_scoring = sint.generar()
            features = [c for c in base.columns if c not in COLUMNAS_PROHIBIDAS]
            self.variables_temporales = list(sint.VARIABLES_TEMPORALES)
            self.montos = ["mpayroll", "mcuentas_saldo", "mtarjeta_visa_consumo"]
            params = dict(PARAMETROS_BASE_SINTETICO)
            nombre = "baseline_sintetico"
        else:
            sys.path.insert(0, str(cfg.PROYECTO / "experimentos"))
            import config as exp_cfg
            import feature_engineering as fe
            import preprocesamiento as prep
            if not exp_cfg.CACHE_BASE.is_file():
                raise GateFallido(
                    f"No existe {exp_cfg.CACHE_BASE}. Se construye con "
                    "python3 experimentos/exp01_harness.py (~4 min).")
            base = pd.read_pickle(exp_cfg.CACHE_BASE)
            base["clase_ternaria"] = base["clase_ternaria"].astype(object).where(
                base["clase_ternaria"].notna(), None)
            features = prep.cargar_lista_features(cfg.PROYECTO / "features_candidato_fe.csv")
            self.variables_temporales = list(fe.VARIABLES_LAG1)
            self.montos = [v for v in fe.VARIABLES_LAG1 if v.startswith("m")]
            params = dict(exp_cfg.PARAMETROS_TRIAL25)
            nombre = "trial25_238"
        base = base.sort_values(["foto_mes", "numero_de_cliente"]).reset_index(drop=True)
        return base, features, params, nombre

    # ---- etapa 1 ----------------------------------------------------------

    def etapa1_ingesta_y_diagnostico(self) -> None:
        ctx, dm = self.ctx, self.ctx.dm
        ctx.log("[E1] ===== Ingesta y diagnostico inicial =====")
        base, features, params, nombre = self._cargar()
        ctx.log(f"[E1] base {base.shape}, {len(features)} features iniciales")

        # --- Gates: si alguno falla, no hay nada que decidir ---------------
        faltan = [c for c in COLUMNAS_CLAVE if c not in base.columns]
        if faltan:
            raise GateFallido(f"faltan columnas clave: {faltan}")
        faltan = [c for c in features if c not in base.columns]
        if faltan:
            raise GateFallido(f"features declaradas que no estan en la base: {faltan[:10]}")
        dup = int(base.duplicated(["numero_de_cliente", "foto_mes"]).sum())
        if dup:
            raise GateFallido(f"granularidad: {dup} pares (cliente, mes) repetidos")
        prohibidas = sorted(COLUMNAS_PROHIBIDAS.intersection(features))
        if prohibidas:
            raise GateFallido(f"F4: columnas prohibidas como predictores: {prohibidas}")
        for f in cfg.FOLDS:
            cfg.verificar_gap(f)
        con_clase = base.loc[base.foto_mes == cfg.MES_SCORING, "clase_ternaria"].notna().sum()
        if con_clase:
            raise GateFallido(f"F2: {con_clase} filas de {cfg.MES_SCORING} tienen clase")
        for mes in cfg.MESES_TARGET_MADURO:
            nulos = base.loc[base.foto_mes == mes, "clase_ternaria"].isna().sum()
            if nulos:
                raise GateFallido(f"madurez: {mes} tiene {nulos} filas sin clase")
        ctx.log("[E1] gates OK: claves, granularidad, F1, F2, F4, madurez")

        # --- Diagnostico de calidad ------------------------------------------
        meses = sorted(base.foto_mes.unique())
        nulos = base.groupby("foto_mes")[features].agg(lambda s: s.isna().mean()).T
        ceros = base.groupby("foto_mes")[features].agg(lambda s: (s == 0).mean()).T
        psi = pd.DataFrame({m: {c: _psi(base.loc[base.foto_mes == meses[0], c],
                                        base.loc[base.foto_mes == m, c])
                                for c in features} for m in meses[1:]})
        nulos.to_csv(ctx.carpeta / "e1_nulos_por_mes.csv")
        ceros.to_csv(ctx.carpeta / "e1_ceros_por_mes.csv")
        psi.to_csv(ctx.carpeta / "e1_psi_vs_primer_mes.csv")

        prevalencia = (base[base.foto_mes.isin(cfg.MESES_TARGET_MADURO)]
                       .groupby("foto_mes")["clase_ternaria"]
                       .apply(lambda s: (s == "BAJA+2").mean()))
        ctx.diagnostico = {
            "filas": len(base), "features": len(features), "meses": meses,
            "prevalencia_baja2_por_mes": prevalencia.round(5).to_dict(),
            "nulos_medio": float(nulos.values.mean()),
        }

        # D1.1 constantes: se eliminan sin medir. Un arbol nunca puede partir
        # por una columna constante, asi que no hay nada que comparar.
        constantes = [c for c in features if base[c].nunique(dropna=False) <= 1]
        if constantes:
            features = [c for c in features if c not in constantes]
        ctx.decidir("E1", "D1.1 constantes",
                    f"elimino {len(constantes)}" if constantes else "ninguna",
                    "una columna constante no puede generar ningun split",
                    columnas=constantes)

        # D1.2 variables rotas: ceros o nulos casi totales en un mes y no en
        # el resto (como ccajas_depositos en 202105). NO se corrigen aca: se
        # encola la reparacion como candidato y compite en la etapa 2.
        # Solo cuenta como rotura si la variable venia NORMAL en algun mes
        # anterior: un lag1 100 % nulo en 202103 no esta roto, es que no hay
        # febrero. En cambio una rotura en el mes de scoring si se detecta.
        rotas = {}
        for c in features:
            for i, m in enumerate(meses[1:], start=1):
                otros = [x for x in meses if x != m]
                for tabla in (ceros, nulos):
                    if (tabla.loc[c, m] >= dm["fraccion_rota"]
                            and tabla.loc[c, otros].median() < dm["fraccion_normal_max"]
                            and (tabla.loc[c, meses[:i]] < dm["fraccion_normal_max"]).any()):
                        rotas.setdefault(c, []).append(int(m))
        if rotas:
            ctx.candidatos_pendientes_etapa2.append(
                Candidato("reparar_rotas", fe_reparar_rotas, {"rotas": rotas},
                          origen="E1 D1.2"))
        ctx.decidir("E1", "D1.2 variables rotas",
                    f"{len(rotas)} detectadas -> candidato 'reparar_rotas' a la etapa 2"
                    if rotas else "ninguna",
                    "corregir sin medir es una decision de modelado tomada a ciegas",
                    rotas=rotas)

        # D1.3 deriva: PSI alto contra el primer mes. Tampoco se borra: se
        # encola una ablacion. Parte de la deriva es honesta (inflacion) y el
        # modelo puede necesitar la variable igual.
        psi_max = psi.max(axis=1)
        derivantes = sorted(psi_max[psi_max > dm["psi_alto"]].index)
        if derivantes:
            ctx.candidatos_pendientes_etapa2.append(
                Candidato("quitar_deriva_alta", fe_quitar, {"columnas": derivantes},
                          origen="E1 D1.3"))
        ctx.decidir("E1", "D1.3 deriva (PSI)",
                    f"{len(derivantes)} con PSI > {dm['psi_alto']} -> ablacion a la etapa 2"
                    if derivantes else "sin deriva alta",
                    "la deriva puede ser honesta (inflacion); se decide midiendo",
                    columnas=derivantes,
                    top=psi_max.sort_values(ascending=False).head(10).round(3).to_dict())

        # --- Banco e incumbente inicial --------------------------------------
        self.base = base
        self.banco = BancoMedicion(ctx, base, self.carpeta_cache)
        inc = Brazo(nombre, tuple(features), params, iteraciones=dm["iteraciones"])
        ctx.nuevo_incumbente(inc, "punto de partida")
        m = self.banco.medir(inc, dm["semillas_medicion"])
        ctx.metricas["baseline"] = _resumen_medicion(m)
        ctx.log(f"[E1] baseline: {_texto_resumen(ctx.metricas['baseline'])}")

        # D1.4 desbalance.
        prev_media = float(prevalencia.mean())
        if prev_media >= dm["umbral_desbalance"]:
            ctx.decidir("E1", "D1.4 desbalance", "ruta estandar",
                        f"prevalencia {prev_media:.3%} >= {dm['umbral_desbalance']:.0%}",
                        prevalencia=prev_media)
        else:
            ctx.decidir("E1", "D1.4 desbalance", "activo modulo de muestreo",
                        f"prevalencia {prev_media:.3%} < {dm['umbral_desbalance']:.0%}",
                        prevalencia=prev_media)
            self._modulo_muestreo()
        ctx.etapas_completas.append("E1")
        ctx.guardar()

    def _modulo_muestreo(self) -> None:
        """Undersampling de CONTINUA como PALANCA DE COMPUTO (Exp01, punto 5).

        No se busca "arreglar" el desbalance: el corte 0,025 ya pondera el
        costo asimetrico. Se busca entrenar mas rapido sin perder ganancia,
        para poder medir con mas semillas. Por eso la prueba es de NO
        INFERIORIDAD (no perder), no de superioridad (ganar), y ademas se
        exige una aceleracion minima.

        Se prueban tasas de mayor a menor y se corta en la primera que falla:
        si 0,25 ya pierde, 0,10 no va a ser mejor.
        """
        ctx, dm, banco = self.ctx, self.ctx.dm, self.banco
        original, elegido = ctx.incumbente, ctx.incumbente
        for tasa in dm["tasas_undersampling"]:
            cand = original.con(nombre=f"{original.nombre}_us{tasa}", tasa_continua=tasa)
            try:
                # Siempre contra el original: comparar contra la tasa anterior
                # dejaria acumular perdidas chicas que juntas superan el margen.
                v = banco.comparar(original, cand, modo="no_inferioridad")
            except Exception as e:  # noqa: BLE001 -- se registra y se sigue
                ctx.registrar_error("E1", f"undersampling {tasa}", e)
                break
            seg_inc = np.mean([r["seg_incumbente"] for r in v["por_fold"].values()])
            seg_can = np.mean([r["seg_candidato"] for r in v["por_fold"].values()])
            aceleracion = seg_inc / max(seg_can, 1e-9)
            pasa = v["aceptado"] and aceleracion >= dm["aceleracion_minima"]
            ctx.decidir("E1", f"D1.4 undersampling {tasa}",
                        "acepto" if pasa else "rechazo",
                        f"{v['motivo']}; aceleracion {aceleracion:.1f}x; {resumen_fold(v)}",
                        veredicto=v, aceleracion=aceleracion)
            if not pasa:
                break
            elegido = cand
        if elegido is not original:
            ctx.nuevo_incumbente(elegido, f"undersampling CONTINUA a {elegido.tasa_continua}")

    # ---- etapa 2 ----------------------------------------------------------

    def etapa2_ingenieria_iterativa(self) -> None:
        ctx, dm, banco, base = self.ctx, self.ctx.dm, self.banco, self.base
        ctx.log("[E2] ===== Ingenieria iterativa =====")
        cola = list(ctx.candidatos_pendientes_etapa2) + catalogo_candidatos(
            ctx.modo, self.variables_temporales, self.montos)

        while cola:
            cand = cola.pop(0)
            ctx.log(f"[E2] candidato {cand.nombre} (intento {cand.intento}, {cand.origen})")
            nuevas_cols = []
            try:
                # 1. Gate F5 sobre la definicion del bloque.
                gate_sin_fuga_pandas(cand.construir, cand.params, base,
                                     dm["mes_corte_gate_f5"])
                # 2. Construccion sobre la base completa.
                nuevas, quitar = cand.construir(base, cand.params)
                nuevas = nuevas.replace([np.inf, -np.inf], np.nan)
                vacias = [c for c in nuevas.columns if nuevas[c].isna().all()]
                nuevas = nuevas.drop(columns=vacias)
                # Cobertura en train. Una columna 100 % nula en el train de un
                # fold no se puede validar en ese fold: el modelo nunca la usa
                # y lo que se mide es ruido de feature_fraction. Con 6 meses:
                #   - el fold mayo entrena solo con 202103 -> ningun lag existe
                #   - el fold junio entrena con 202103-04 -> solo existe lag1
                # Decision adaptativa: el bloque se mide SOLO en los folds
                # donde todas sus columnas tienen historia, y queda escrito que
                # la evidencia es de menos folds. Las columnas que no tienen
                # historia en ningun fold se descartan antes de medir.
                cobertura = {f: [c for c in nuevas.columns
                                 if nuevas.loc[banco._mascaras[f][0], c].notna().any()]
                             for f in banco.folds}
                nunca = [c for c in nuevas.columns
                         if not any(c in cs for cs in cobertura.values())]
                if nunca:
                    nuevas = nuevas.drop(columns=nunca)
                if nuevas.empty and not quitar:
                    raise SinHistoria(nunca)
                folds_medibles = [f for f in banco.folds
                                  if set(nuevas.columns) <= set(cobertura[f])]
                if not folds_medibles:
                    raise SinHistoria(list(nuevas.columns))
                if nuevas.empty and not quitar:
                    raise ValueError("el bloque no produce ninguna columna con informacion")
                choque = [c for c in nuevas.columns if c in base.columns]
                if choque:
                    raise ValueError(f"columnas que ya existen: {choque[:5]}")
                for c in nuevas.columns:
                    base[c] = nuevas[c].astype(np.float32)
                nuevas_cols = list(nuevas.columns)
                feats = [f for f in ctx.incumbente.features if f not in quitar] + nuevas_cols
                prohibidas = COLUMNAS_PROHIBIDAS.intersection(feats)
                if prohibidas:
                    raise GateFallido(f"F4 en {cand.nombre}: {sorted(prohibidas)}")
                brazo = ctx.incumbente.con(nombre=f"{ctx.incumbente.nombre}+{cand.nombre}",
                                           features=tuple(feats))
                # 3. Medicion apareada con poda.
                v = banco.comparar(ctx.incumbente, brazo, folds=folds_medibles,
                                   modo="superioridad")
                v["folds_medidos"] = folds_medibles
                if nunca:
                    v["columnas_sin_historia"] = nunca
            except GateFallido:
                raise
            except Exception as e:  # noqa: BLE001 -- un candidato roto no tira el pipeline
                base.drop(columns=nuevas_cols, inplace=True, errors="ignore")
                banco.olvidar(nuevas_cols)
                diag = self._diagnosticar(e)
                ctx.registrar_error("E2", cand.nombre, e)
                corregido = (corregir_candidato(cand, diag)
                             if cand.intento < dm["max_intentos_por_candidato"] else None)
                ctx.bloques_rechazados.append({"bloque": cand.nombre, "motivo": diag["tipo"],
                                               "detalle": str(e)[:300]})
                ctx.decidir("E2", f"bloque {cand.nombre}", "descarto por falla",
                            f"{diag['tipo']}: {str(e)[:200]}"
                            + (f" -> reintento como {corregido.nombre}" if corregido else
                               " -> sin correccion posible"),
                            diagnostico=diag)
                if corregido:
                    cola.insert(0, corregido)
                continue

            aviso = ("" if len(folds_medibles) == len(banco.folds) else
                     f" [EVIDENCIA PARCIAL: solo {folds_medibles}, el resto no tiene "
                     f"historia en train]")
            registro = {"bloque": cand.nombre, "origen": cand.origen,
                        "n_nuevas": len(nuevas_cols), "n_quitadas": len(quitar),
                        "veredicto": v}
            if v["aceptado"]:
                ctx.nuevo_incumbente(brazo, f"entra {cand.nombre}")
                ctx.bloques_aprobados.append(registro)
                ctx.decidir("E2", f"bloque {cand.nombre}", "ACEPTO",
                            f"{v['motivo']}; {resumen_fold(v)}{aviso}", veredicto=v)
            else:
                base.drop(columns=nuevas_cols, inplace=True)
                banco.olvidar(nuevas_cols)
                ctx.bloques_rechazados.append(registro)
                # Sin reintento: ver corregir_candidato() sobre la fuga F6.
                ctx.decidir("E2", f"bloque {cand.nombre}", "rechazo",
                            f"{v['motivo']}; {resumen_fold(v)}{aviso}", veredicto=v)

        m = banco.medir(ctx.incumbente, dm["semillas_medicion"])
        ctx.metricas["post_E2"] = _resumen_medicion(m)
        ctx.log(f"[E2] incumbente: {ctx.incumbente.nombre} "
                f"({len(ctx.incumbente.features)} features) "
                f"{_texto_resumen(ctx.metricas['post_E2'])}")
        ctx.etapas_completas.append("E2")
        ctx.guardar()

    @staticmethod
    def _diagnosticar(e: Exception) -> dict:
        """Traduce una excepcion a un diagnostico que el corrector entiende."""
        if isinstance(e, FugaDeFuturo):
            return {"tipo": "fuga", "columnas": e.columnas}
        if isinstance(e, SinHistoria):
            return {"tipo": "sin_historia", "columnas": e.columnas}
        if isinstance(e, ColumnasFaltantes):
            return {"tipo": "columnas_faltantes", "columnas": e.columnas}
        if isinstance(e, MemoryError):
            return {"tipo": "memoria"}
        return {"tipo": "error", "mensaje": f"{type(e).__name__}: {e}"[:500]}

    # ---- etapa 3 ----------------------------------------------------------

    def _espacio_busqueda(self, p: int, n: int) -> dict:
        """El espacio depende de la dimensionalidad resultante de la etapa 2.

        - feature_fraction: con muchas columnas y alta redundancia (lags y
          deltas de la misma variable), fracciones bajas decorrelacionan los
          arboles. Con pocas columnas, una fraccion baja deja arboles sin
          las variables buenas.
        - min_data_in_leaf: se escala con las filas de train, porque lo que
          importa es cuantos BAJA+2 puede haber en una hoja.
        - num_leaves: se acota con p bajo, donde no hay interacciones que
          justifiquen arboles muy anchos.
        """
        if p > 200:
            ff = (0.10, 0.50)
        elif p > 50:
            ff = (0.20, 0.80)
        else:
            ff = (0.50, 1.00)
        hojas = (8, 256) if p > 50 else (8, 64)
        min_hoja = (max(20, int(n * 0.0005)), max(100, int(n * 0.02)))
        return {"feature_fraction": ff, "num_leaves": hojas, "min_data_in_leaf": min_hoja,
                "learning_rate": (0.005, 0.1), "bagging_fraction": (0.4, 1.0),
                "lambda_l1": (1e-8, 10.0), "lambda_l2": (1e-8, 10.0),
                "min_gain_to_split": (0.0, 5.0)}

    def etapa3_optimizacion_condicional(self) -> None:
        import optuna
        ctx, dm, banco = self.ctx, self.ctx.dm, self.banco
        ctx.log("[E3] ===== Seleccion de modelo y optimizacion condicional =====")
        inc = ctx.incumbente
        fold = dm["optuna_fold_seleccion"]
        mt, mv = banco._mascaras[fold]
        n_train = int(mt.sum() * (inc.tasa_continua if inc.tasa_continua < 1 else 1))
        espacio = self._espacio_busqueda(len(inc.features), n_train)
        ctx.decidir("E3", "espacio de busqueda", f"p={len(inc.features)}, n~{n_train:,}",
                    "rangos ajustados a la dimensionalidad (ver _espacio_busqueda)",
                    espacio=espacio)

        X_tr, X_va = banco._matriz(inc, fold)
        y_tr, clase_tr = banco.y[mt], banco.clase[mt]
        gan = ganancia_por_fila(banco.clase[mv])
        semilla = int(dm["semillas_medicion"][0])
        if inc.tasa_continua < 1.0:
            keep = ((clase_tr != "CONTINUA")
                    | (np.random.default_rng(semilla).random(len(y_tr)) < inc.tasa_continua))
            X_tr, y_tr = X_tr[keep], y_tr[keep]
        ds = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
        checkpoints = tuple(dm["optuna_checkpoints"])
        referencia = {}  # ganancia del incumbente en cada checkpoint (trial 0)

        def objetivo(trial):
            params = dict(inc.params)
            for k, (lo, hi) in espacio.items():
                if k in ("num_leaves", "min_data_in_leaf"):
                    params[k] = trial.suggest_int(k, lo, hi, log=True)
                elif k in ("learning_rate", "lambda_l1", "lambda_l2"):
                    params[k] = trial.suggest_float(k, lo, hi, log=True)
                else:
                    params[k] = trial.suggest_float(k, lo, hi)
            params.update({"seed": semilla, "bagging_seed": semilla,
                           "feature_fraction_seed": semilla, "verbosity": -1})
            ultimo = {}

            def poda(env):
                it = env.iteration + 1
                if it not in checkpoints:
                    return
                p = corregir_probabilidad(env.model.predict(X_va), inc.tasa_continua)
                g = float(gan[p >= cfg.UMBRAL_TEORICO].sum())
                ultimo["g"] = g
                trial.report(g, it)
                if trial.number == 0:
                    referencia[it] = g
                # Piso duro: en el primer checkpoint, si no llega a la mitad de
                # lo que hacia el incumbente, no vale la pena seguir.
                elif (it == checkpoints[0] and it in referencia
                      and g < dm["piso_relativo_primer_checkpoint"] * referencia[it]):
                    raise optuna.TrialPruned(f"piso: {g / 1e6:.1f}M en {it} iter")
                if trial.should_prune():
                    raise optuna.TrialPruned(f"mediana: {g / 1e6:.1f}M en {it} iter")

            lgb.train(params, ds, num_boost_round=max(checkpoints), callbacks=[poda])
            return ultimo["g"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        estudio = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=semilla),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
            storage=f"sqlite:///{ctx.carpeta / 'e3_optuna.db'}",
            study_name="e3", load_if_exists=True)
        # Trial 0 = los parametros del incumbente, recortados al espacio: da la
        # referencia del piso y garantiza que el estudio "ve" al incumbente.
        estudio.enqueue_trial({k: float(np.clip(inc.params.get(k, (lo + hi) / 2), lo, hi))
                               if not isinstance(lo, int) else
                               int(np.clip(inc.params.get(k, lo), lo, hi))
                               for k, (lo, hi) in espacio.items()})
        try:
            estudio.optimize(objetivo, n_trials=dm["optuna_trials"],
                             timeout=dm["optuna_timeout_min"] * 60, catch=(Exception,))
        except Exception as e:  # noqa: BLE001
            ctx.registrar_error("E3", "optuna", e)

        trials = estudio.trials_dataframe()
        trials.to_csv(ctx.carpeta / "e3_trials.csv", index=False)
        estados = trials["state"].value_counts().to_dict() if len(trials) else {}
        ctx.decidir("E3", "busqueda", f"{len(trials)} trials", "estados por trial",
                    estados=estados)

        completos = [t for t in estudio.trials
                     if t.state == optuna.trial.TrialState.COMPLETE and t.number != 0]
        completos.sort(key=lambda t: t.value, reverse=True)
        top = completos[:dm["optuna_top_confirmar"]]
        if not top:
            ctx.decidir("E3", "confirmacion", "mantengo parametros", "ningun trial completo")
            ctx.etapas_completas.append("E3")
            return

        # Confirmacion en el fold que la busqueda NO vio, con Bonferroni
        # porque se confirman varios a la vez.
        alpha = dm["alpha"] / len(top)
        ganadores = []
        for t in top:
            params = {**inc.params, **t.params}
            cand = inc.con(nombre=f"{inc.nombre}+optuna{t.number}", params=params)
            try:
                v = banco.comparar(inc, cand, folds=dm["optuna_folds_confirmacion"],
                                   alpha=alpha)
            except Exception as e:  # noqa: BLE001
                ctx.registrar_error("E3", f"confirmacion trial {t.number}", e)
                continue
            ctx.decidir("E3", f"confirmar trial {t.number}",
                        "gana" if v["aceptado"] else "no gana",
                        f"{v['motivo']} (alpha Bonferroni {alpha:.3f}); {resumen_fold(v)}",
                        valor_seleccion=t.value, veredicto=v)
            if v["aceptado"]:
                d = np.mean([r["delta"] for r in v["por_fold"].values()])
                ganadores.append((d, cand))
        if ganadores:
            ganadores.sort(key=lambda x: x[0], reverse=True)
            ctx.nuevo_incumbente(ganadores[0][1], "hiperparametros confirmados fuera de la busqueda")
        m = banco.medir(ctx.incumbente, dm["semillas_medicion"])
        ctx.metricas["post_E3"] = _resumen_medicion(m)
        ctx.etapas_completas.append("E3")
        ctx.guardar()

    # ---- etapa 4 ----------------------------------------------------------

    def etapa4_corte_y_entrega(self) -> None:
        ctx, dm, banco = self.ctx, self.ctx.dm, self.banco
        ctx.log("[E4] ===== Calibracion del corte y entrega =====")
        inc = ctx.incumbente
        semillas = list(dm["semillas_medicion"])
        try:
            self._elegir_regla_de_corte(inc, semillas)
        except Exception as e:  # noqa: BLE001 -- sin regla nueva, queda 0,025
            ctx.registrar_error("E4", "regla de corte", e)
            ctx.decidir("E4", "regla de corte", "R0_umbral_teorico (fallback)",
                        "fallo la evaluacion de reglas alternativas")
        self._entrenar_y_entregar(inc)
        ctx.etapas_completas.append("E4")
        ctx.guardar()

    def _elegir_regla_de_corte(self, inc: Brazo, semillas) -> None:
        """Replica el Experimento 01 de forma automatica: cada regla elige su
        parametro en un mes y se mide en el otro, en los dos sentidos."""
        from sklearn.isotonic import IsotonicRegression
        ctx, dm, banco = self.ctx, self.ctx.dm, self.banco
        folds = list(banco.folds)
        gan = {f: ganancia_por_fila(banco.clase[banco._mascaras[f][1]]) for f in folds}
        pos = {f: banco.y[banco._mascaras[f][1]] for f in folds}

        def mejor_k(p, g):
            orden = np.argsort(-p)
            acum = np.cumsum(g[orden])
            k = int(np.argmax(acum)) + 1
            return k, float(p[orden][k - 1])

        def aplicar(regla, param, p, g):
            if regla == "R0_umbral_teorico":
                return float(g[p >= cfg.UMBRAL_TEORICO].sum())
            if regla == "R1_k_absoluto":
                return float(g[np.argsort(-p)[:param]].sum())
            if regla == "R2_fraccion":
                k = int(round(param * len(p)))
                return float(g[np.argsort(-p)[:k]].sum())
            if regla == "R3_umbral_empirico":
                return float(g[p >= param].sum())
            if regla == "R4_isotonica":
                return float(g[param.predict(p) >= cfg.UMBRAL_TEORICO].sum())
            raise ValueError(regla)

        def ajustar(regla, p, g, y):
            k, umbral = mejor_k(p, g)
            if regla == "R1_k_absoluto":
                return k
            if regla == "R2_fraccion":
                return k / len(p)
            if regla == "R3_umbral_empirico":
                return umbral
            if regla == "R4_isotonica":
                return IsotonicRegression(out_of_bounds="clip").fit(p, y)
            return None

        reglas = ["R1_k_absoluto", "R2_fraccion", "R3_umbral_empirico", "R4_isotonica"]
        filas = []
        for s in semillas:
            preds = {f: banco.predicciones(inc, f, s) for f in folds}
            for origen in folds:
                for destino in folds:
                    if origen == destino:
                        continue
                    g0 = aplicar("R0_umbral_teorico", None, preds[destino], gan[destino])
                    for r in reglas:
                        param = ajustar(r, preds[origen], gan[origen], pos[origen])
                        g = aplicar(r, param, preds[destino], gan[destino])
                        filas.append({"semilla": s, "sentido": f"{origen}->{destino}",
                                      "regla": r, "ganancia_r0": g0, "ganancia": g,
                                      "delta": g - g0})
        tabla = pd.DataFrame(filas)
        tabla.to_csv(ctx.carpeta / "e4_reglas_corte.csv", index=False)

        evaluacion = {}
        for r in reglas:
            por_sentido = {}
            for sentido, t in tabla[tabla.regla == r].groupby("sentido"):
                por_sentido[sentido] = {
                    "delta_medio": float(t.delta.mean()),
                    "delta_rel": float(t.delta.mean() / max(abs(t.ganancia_r0.mean()), 1.0)),
                    "p_mejor": p_wilcoxon(t.delta, "greater")}
            gana = all(v["p_mejor"] < dm["alpha"] and
                       v["delta_rel"] >= dm["mejora_minima_relativa"]
                       for v in por_sentido.values())
            evaluacion[r] = {"gana": gana, "por_sentido": por_sentido}
        ganadoras = [r for r, v in evaluacion.items() if v["gana"]]
        if ganadoras:
            mejor = max(ganadoras, key=lambda r: min(
                v["delta_rel"] for v in evaluacion[r]["por_sentido"].values()))
            ctx.regla_corte = {"nombre": mejor}
            motivo = "gana en los dos sentidos del cruce entre meses"
        else:
            mejor = "R0_umbral_teorico"
            motivo = ("ninguna regla gana en los dos sentidos; 0,025 no tiene "
                      "parametro que transferir")
        ctx.decidir("E4", "regla de corte", mejor, motivo, evaluacion=evaluacion)

        # Diagnostico de calibracion: suma de probabilidades vs positivos reales.
        calib = {}
        for f in folds:
            p = np.mean([banco.predicciones(inc, f, s) for s in semillas], axis=0)
            calib[f] = {"suma_p": float(p.sum()), "positivos": int(pos[f].sum())}
        ctx.diagnostico["calibracion"] = calib

        # Parametro final de la regla: se ajusta en el fold mas reciente, el
        # unico que ve dos meses de train como el modelo final ve cuatro.
        if mejor != "R0_umbral_teorico":
            f = folds[-1]
            p = np.mean([banco.predicciones(inc, f, s) for s in semillas], axis=0)
            self.parametro_regla = ajustar(mejor, p, gan[f], pos[f])
            ctx.regla_corte["parametro"] = (self.parametro_regla if mejor != "R4_isotonica"
                                            else "IsotonicRegression ajustada en " + f)

    def _entrenar_y_entregar(self, inc: Brazo) -> None:
        ctx, dm, base = self.ctx, self.ctx.dm, self.base
        meses_train = tuple(cfg.MESES_TARGET_MADURO)
        cfg.verificar_gap({"nombre": "final", "train": meses_train,
                           "validacion": cfg.MES_SCORING})
        mt = base["foto_mes"].isin(meses_train).to_numpy()
        ms = (base["foto_mes"] == cfg.MES_SCORING).to_numpy()
        X = base[list(inc.features)]
        X_tr = X.loc[mt].to_numpy(dtype=np.float32)
        X_sc = X.loc[ms].to_numpy(dtype=np.float32)
        clase_tr = base.loc[mt, "clase_ternaria"].to_numpy()
        y_tr = (clase_tr == "BAJA+2").astype(np.int8)
        preds = []
        for s in dm["semillas_entrega"]:
            Xs, ys = X_tr, y_tr
            if inc.tasa_continua < 1.0:
                keep = ((clase_tr != "CONTINUA")
                        | (np.random.default_rng(s).random(len(y_tr)) < inc.tasa_continua))
                Xs, ys = X_tr[keep], y_tr[keep]
            params = {**inc.params, "seed": s, "bagging_seed": s,
                      "feature_fraction_seed": s, "data_random_seed": s, "verbosity": -1}
            modelo = lgb.train(params, lgb.Dataset(Xs, label=ys),
                               num_boost_round=inc.iteraciones)
            preds.append(corregir_probabilidad(modelo.predict(X_sc), inc.tasa_continua))
            ctx.log(f"[E4] modelo final semilla {s} listo")
        p = np.mean(preds, axis=0)

        regla = ctx.regla_corte
        if regla["nombre"] == "R0_umbral_teorico":
            estimular = p >= cfg.UMBRAL_TEORICO
        elif regla["nombre"] == "R1_k_absoluto":
            estimular = np.zeros(len(p), bool)
            estimular[np.argsort(-p)[:self.parametro_regla]] = True
        elif regla["nombre"] == "R2_fraccion":
            estimular = np.zeros(len(p), bool)
            estimular[np.argsort(-p)[:int(round(self.parametro_regla * len(p)))]] = True
        elif regla["nombre"] == "R3_umbral_empirico":
            estimular = p >= self.parametro_regla
        else:
            estimular = self.parametro_regla.predict(p) >= cfg.UMBRAL_TEORICO

        sello = datetime.now().strftime("%Y%m%d_%H%M%S")
        ruta = ctx.carpeta / f"entrega_{ctx.modo}_{sello}.csv"
        entrega = pd.DataFrame({"numero_de_cliente": base.loc[ms, "numero_de_cliente"].to_numpy(),
                                "Predicted": estimular.astype(int)})
        entrega.to_csv(ruta, index=False)
        pd.DataFrame({"numero_de_cliente": entrega.numero_de_cliente, "prob": p}).to_csv(
            ctx.carpeta / f"probabilidades_{ctx.modo}_{sello}.csv", index=False)
        ctx.entrega = {"archivo": ruta.name, "estimulados": int(estimular.sum()),
                       "fraccion": float(estimular.mean()), "regla": regla["nombre"],
                       "semillas": list(dm["semillas_entrega"]),
                       "meses_train": list(meses_train)}

        # Solo en sintetico existe la verdad de agosto: auditoria de la entrega.
        if self.verdad_scoring is not None:
            v = self.verdad_scoring.set_index("numero_de_cliente")["clase_ternaria"]
            clase = entrega["numero_de_cliente"].map(v).to_numpy()
            g = ganancia_por_fila(clase)
            ctx.entrega["ganancia_real_sintetica"] = float(g[estimular].sum())
            ctx.entrega["ganancia_optima_sintetica"] = float(g[g > 0].sum())
        ctx.decidir("E4", "entrega", ruta.name,
                    f"{int(estimular.sum()):,} estimulados con {regla['nombre']}",
                    **{k: v for k, v in ctx.entrega.items() if k != "archivo"})

    # ---- orquestacion -----------------------------------------------------

    def correr(self) -> ContextoPipeline:
        """Secuencial: cada etapa usa lo que dejo la anterior. Si una etapa
        opcional falla, el pipeline sigue con el ultimo incumbente valido; si
        falla la etapa 1 o un gate, aborta."""
        ctx = self.ctx
        ctx.log(f"[p10] modo={ctx.modo} carpeta={ctx.carpeta}")
        self.etapa1_ingesta_y_diagnostico()
        for etapa in (self.etapa2_ingenieria_iterativa, self.etapa3_optimizacion_condicional):
            try:
                etapa()
            except GateFallido:
                raise
            except Exception as e:  # noqa: BLE001 -- degradacion controlada
                ctx.registrar_error(etapa.__name__, "etapa completa", e)
                ctx.decidir(etapa.__name__, "falla de etapa", "sigo con el incumbente",
                            f"{type(e).__name__}: {str(e)[:200]}")
        self.etapa4_corte_y_entrega()
        ctx.log("[p10] fin")
        return ctx


# ==========================================================================
# Helpers de reporte
# ==========================================================================

def _psi(ref: pd.Series, act: pd.Series, bins: int = 10) -> float:
    """PSI con cortes por cuantiles del mes de referencia. Los nulos son una
    categoria aparte: que aparezcan nulos nuevos tambien es deriva."""
    ref = pd.to_numeric(ref, errors="coerce")
    act = pd.to_numeric(act, errors="coerce")
    cortes = np.unique(np.nanquantile(ref.dropna(), np.linspace(0, 1, bins + 1))) \
        if ref.notna().any() else np.array([])
    if len(cortes) < 2:
        return 0.0
    cortes[0], cortes[-1] = -np.inf, np.inf

    def dist(s):
        c = pd.cut(s.dropna(), cortes).value_counts(sort=False).to_numpy()
        d = np.append(c, s.isna().sum()).astype(float)
        return np.clip(d / d.sum(), 1e-4, None)

    a, b = dist(ref), dist(act)
    return float(np.sum((b - a) * np.log(b / a)))


def _resumen_medicion(m: pd.DataFrame) -> dict:
    return {f: {"media": float(t.ganancia.mean()), "desvio": float(t.ganancia.std(ddof=1)),
                "n": int(len(t)), "estimulados": float(t.estimulados.mean())}
            for f, t in m.groupby("fold")}


def _texto_resumen(r: dict) -> str:
    return " | ".join(f"{f}: {v['media'] / 1e6:.2f}M +/- {v['desvio'] / 1e6:.2f}M (n={v['n']})"
                      for f, v in r.items())


def correr(modo: str = "sintetico", **kwargs) -> ContextoPipeline:
    return PipelineAdaptativo(modo, **kwargs).correr()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--modo", choices=("sintetico", "real"), default="sintetico")
    ap.add_argument("--max-semillas", type=int, default=None,
                    help="smoke test: limita las semillas de medicion")
    ap.add_argument("--optuna-trials", type=int, default=None)
    args = ap.parse_args()
    cambios = {} if args.optuna_trials is None else {"optuna_trials": args.optuna_trials}
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
    pipe = PipelineAdaptativo(args.modo, max_semillas=args.max_semillas, **cambios)
    try:
        pipe.correr()
        return 0
    except GateFallido as e:
        pipe.ctx.registrar_error("gate", "abortado", e)
        print(f"\n[p10] ABORTADO por gate: {e}", flush=True)
        return 2
    except Exception as e:  # noqa: BLE001 -- ultima red: deja el estado escrito
        pipe.ctx.registrar_error("p10", "abortado", e)
        print(f"\n[p10] ABORTADO: {type(e).__name__}: {e}", flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
