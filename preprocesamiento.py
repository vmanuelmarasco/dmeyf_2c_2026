"""Preprocesamiento reproducible del candidato de 238 features.

El modulo parte de la tabla original ``competencia_01`` (en este proyecto,
persistida como ``competencia_01_crudo.csv``), aplica solamente las
correcciones confirmadas y reconstruye las 13 features de anclaje aceptadas.
No imputa nulos, no recorta outliers y no depende de vistas de DuckDB ni del
estado de ejecucion de ``competencia.ipynb``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


MESES_ESPERADOS: Tuple[int, ...] = (
    202103,
    202104,
    202105,
    202106,
    202107,
    202108,
)
MESES_TARGET_MADURO: Tuple[int, ...] = (202103, 202104, 202105, 202106)
FILAS_ESPERADAS = 983_061

CONTEOS_ESPERADOS: Mapping[int, Mapping[str, int]] = {
    202103: {"filas": 162_900, "targets": 162_900, "baja2": 960},
    202104: {"filas": 163_284, "targets": 163_284, "baja2": 1_139},
    202105: {"filas": 163_768, "targets": 163_768, "baja2": 870},
    202106: {"filas": 164_114, "targets": 164_114, "baja2": 1_098},
    202107: {"filas": 164_348, "targets": 0, "baja2": 0},
    202108: {"filas": 164_647, "targets": 0, "baja2": 0},
}

COLUMNAS_CLAVE: Tuple[str, str] = ("numero_de_cliente", "foto_mes")
COLUMNAS_CONSUMO_DUPLICADAS: Tuple[str, str] = (
    "Master_mconsumototal",
    "Visa_mconsumototal",
)
COLUMNAS_INDICADORES_CORRECCION: Tuple[str, str] = (
    "Master_limite_financiacion_fue_corregido",
    "Visa_limite_financiacion_fue_corregido",
)
COLUMNAS_ANCLAJE: Tuple[str, ...] = (
    "tiene_payroll",
    "coperaciones_recurrentes_total",
    "cantidad_marcas_tarjeta",
    "cantidad_marcas_tarjeta_con_uso",
    "visa_sin_uso",
    "master_sin_uso",
    "cantidad_canales_usados",
    "tiene_operaciones_recurrentes",
    "tiene_tarjeta_sin_uso",
    "cliente_anclado",
    "usa_homebanking_transaccional",
    "usa_mobile_app_transaccional",
    "actividad_digital_transaccional",
)

FEATURES_FINALES_ESPERADAS = 238
FEATURES_ENTRADA_ESPERADAS = 156
FEATURES_LAG_ESPERADAS = 47
FEATURES_CARGA_ESPERADAS = 35

RUTA_HISTORICA_FUENTE = Path(
    "/Users/manuelmarasco/Library/CloudStorage/"
    "GoogleDrive-hola@manuelmarasco.com/Mi unidad/DMEyF/2026/"
    "notebooks/data/competencia_01_crudo.csv"
)


class ErrorAuditoria(RuntimeError):
    """Indica que un invariante obligatorio del pipeline no se cumple."""


def _fallar_si(errores: Sequence[str], contexto: str) -> None:
    if errores:
        detalle = "\n - ".join(errores)
        raise ErrorAuditoria(f"{contexto}:\n - {detalle}")


def resolver_ruta_fuente(
    ruta: Optional[Union[str, Path]] = None,
    directorio_proyecto: Optional[Union[str, Path]] = None,
) -> Path:
    """Resuelve la fuente sin depender del directorio de trabajo del notebook.

    Prioridad: argumento explicito, variable ``COMPETENCIA_01_PATH``, archivos
    locales habituales y, por ultimo, la ubicacion historica encontrada en la
    bitacora. Si ninguna existe, falla e informa todas las alternativas.
    """

    candidatos: List[Path] = []
    if ruta is not None:
        candidatos.append(Path(ruta).expanduser())

    ruta_entorno = os.environ.get("COMPETENCIA_01_PATH")
    if ruta_entorno:
        candidatos.append(Path(ruta_entorno).expanduser())

    raiz = (
        Path(directorio_proyecto).expanduser()
        if directorio_proyecto is not None
        else Path(__file__).resolve().parent
    )
    candidatos.extend(
        [
            raiz / "competencia_01_crudo.csv",
            raiz / "data" / "competencia_01_crudo.csv",
            raiz / "competencia_01.parquet",
            raiz / "data" / "competencia_01.parquet",
            raiz / "competencia_01.csv",
            raiz / "data" / "competencia_01.csv",
            RUTA_HISTORICA_FUENTE,
        ]
    )

    vistos = set()
    candidatos_unicos = []
    for candidato in candidatos:
        normalizado = candidato.resolve(strict=False)
        if normalizado not in vistos:
            vistos.add(normalizado)
            candidatos_unicos.append(normalizado)

    for candidato in candidatos_unicos:
        if candidato.is_file():
            return candidato

    listado = "\n - ".join(str(candidato) for candidato in candidatos_unicos)
    raise FileNotFoundError(
        "No se encontro la tabla original competencia_01. "
        "Pase su ruta a resolver_ruta_fuente() o defina "
        "COMPETENCIA_01_PATH. Rutas revisadas:\n - "
        + listado
    )


def cargar_competencia_01(
    ruta: Union[str, Path],
    columnas: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Carga la tabla original desde CSV o Parquet, sin transformarla."""

    ruta = Path(ruta).expanduser().resolve(strict=False)
    if not ruta.is_file():
        raise FileNotFoundError(f"No existe la fuente: {ruta}")

    sufijos = "".join(ruta.suffixes).lower()
    if sufijos.endswith(".parquet"):
        base = pd.read_parquet(ruta, columns=list(columnas) if columnas else None)
    elif sufijos.endswith(".csv") or sufijos.endswith(".csv.gz"):
        base = pd.read_csv(
            ruta,
            usecols=list(columnas) if columnas else None,
            low_memory=False,
            memory_map=not sufijos.endswith(".gz"),
        )
    else:
        raise ValueError(
            f"Formato no soportado para {ruta}. Use CSV, CSV.GZ o Parquet."
        )

    faltantes = [columna for columna in COLUMNAS_CLAVE if columna not in base]
    if faltantes:
        raise ValueError(
            "La fuente no parece ser competencia_01; faltan columnas clave: "
            + ", ".join(faltantes)
        )
    columnas_derivadas = sorted(
        {"mes_0", "clase_ternaria", "target_baja2"}.intersection(base.columns)
    )
    if columnas_derivadas:
        raise ValueError(
            "La fuente ya contiene columnas derivadas ("
            + ", ".join(columnas_derivadas)
            + "). Use la tabla cruda competencia_01, no una exportacion "
            "con etiquetas parciales."
        )
    if columnas is None and base.shape[1] != 154:
        raise ValueError(
            f"La fuente tiene {base.shape[1]} columnas; la tabla original "
            "competencia_01 esperada tiene 154."
        )
    for columna in COLUMNAS_CLAVE:
        valores = pd.to_numeric(base[columna], errors="raise")
        if (
            valores.isna().any()
            or not np.isfinite(valores.to_numpy(dtype="float64")).all()
            or not np.equal(valores, np.floor(valores)).all()
        ):
            raise ValueError(
                f"La clave {columna} contiene nulos o valores no enteros."
            )
        # La fuente contiene un ID escrito en notacion cientifica; la
        # conversion explicita evita que el motor lo deje como float.
        base[columna] = valores.astype("int64")
    return base


def _indice_mes(valores: pd.Series) -> pd.Series:
    numericos = pd.to_numeric(valores, errors="raise").astype("int64")
    return (numericos // 100) * 12 + (numericos % 100)


def _suma_componentes(base: pd.DataFrame, columnas: Sequence[str]) -> pd.Series:
    """Suma componentes ausentes como cero, pero conserva all-null como nulo."""

    return base.loc[:, list(columnas)].sum(axis=1, skipna=True, min_count=1)


def aplicar_correcciones_calidad(
    base: pd.DataFrame,
    copiar: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica solamente las dos correcciones de limites confirmadas.

    Tambien elimina las dos copias de consumo total y crea la suma de cajas
    aceptada en el candidato actual. Devuelve un reporte con los casos tocados.
    """

    requeridas = {
        "Master_mfinanciacion_limite",
        "Master_mlimitecompra",
        "Visa_mfinanciacion_limite",
        "Visa_mlimitecompra",
        "ccajas_depositos",
        "ccajas_otras",
        *COLUMNAS_CONSUMO_DUPLICADAS,
    }
    faltantes = sorted(requeridas.difference(base.columns))
    if faltantes:
        raise ValueError(
            "Faltan columnas para las correcciones confirmadas: "
            + ", ".join(faltantes)
        )

    salida = base.copy() if copiar else base
    filas_reporte = []

    for marca in ("Master", "Visa"):
        limite_financiacion = f"{marca}_mfinanciacion_limite"
        limite_compra = f"{marca}_mlimitecompra"
        indicador = f"{marca}_limite_financiacion_fue_corregido"

        condicion = (
            salida[limite_compra].gt(0)
            & salida[limite_financiacion]
            .div(salida[limite_compra].where(salida[limite_compra].ne(0)))
            .gt(10)
        ).fillna(False)

        salida[indicador] = condicion.astype("int8")
        salida.loc[condicion, limite_financiacion] = (
            0.9 * salida.loc[condicion, limite_compra]
        )
        filas_reporte.append(
            {
                "correccion": indicador,
                "filas_corregidas": int(condicion.sum()),
            }
        )

    salida = salida.drop(columns=list(COLUMNAS_CONSUMO_DUPLICADAS))
    salida["ccajas_depositos_mas_otras"] = _suma_componentes(
        salida, ("ccajas_depositos", "ccajas_otras")
    )
    return salida, pd.DataFrame(filas_reporte)


def crear_features_anclaje(
    base: pd.DataFrame,
    copiar: bool = True,
) -> pd.DataFrame:
    """Reproduce las 13 features finales de las dos vistas de anclaje."""

    columnas_recurrentes = (
        "ccuenta_debitos_automaticos",
        "ctarjeta_visa_debitos_automaticos",
        "ctarjeta_master_debitos_automaticos",
        "cpagomiscuentas",
        "cpagodeservicios",
    )
    requeridas = {
        "cpayroll_trx",
        "mpayroll",
        "ctarjeta_visa",
        "ctarjeta_master",
        "ctarjeta_visa_transacciones",
        "ctarjeta_master_transacciones",
        "chomebanking_transacciones",
        "cmobile_app_trx",
        "catm_trx",
        "catm_trx_other",
        "ccajas_transacciones",
        "ccallcenter_transacciones",
        *columnas_recurrentes,
    }
    faltantes = sorted(requeridas.difference(base.columns))
    if faltantes:
        raise ValueError(
            "Faltan columnas para las features de anclaje: "
            + ", ".join(faltantes)
        )

    salida = base.copy() if copiar else base
    salida["tiene_payroll"] = (
        salida["cpayroll_trx"].fillna(0).gt(0)
        | salida["mpayroll"].fillna(0).gt(0)
    ).astype("int8")
    salida["coperaciones_recurrentes_total"] = _suma_componentes(
        salida, columnas_recurrentes
    )

    visa_tiene = salida["ctarjeta_visa"].fillna(0).gt(0)
    master_tiene = salida["ctarjeta_master"].fillna(0).gt(0)
    visa_usa = salida["ctarjeta_visa_transacciones"].fillna(0).gt(0)
    master_usa = salida["ctarjeta_master_transacciones"].fillna(0).gt(0)

    salida["cantidad_marcas_tarjeta"] = (
        visa_tiene.astype("int8") + master_tiene.astype("int8")
    )
    salida["cantidad_marcas_tarjeta_con_uso"] = (
        visa_usa.astype("int8") + master_usa.astype("int8")
    )
    salida["visa_sin_uso"] = (visa_tiene & ~visa_usa).astype("int8")
    salida["master_sin_uso"] = (master_tiene & ~master_usa).astype("int8")

    uso_atm = (
        salida["catm_trx"].fillna(0) + salida["catm_trx_other"].fillna(0)
    ).gt(0)
    canales = (
        salida["chomebanking_transacciones"].fillna(0).gt(0).astype("int8")
        + salida["cmobile_app_trx"].fillna(0).gt(0).astype("int8")
        + uso_atm.astype("int8")
        + salida["ccajas_transacciones"].fillna(0).gt(0).astype("int8")
        + salida["ccallcenter_transacciones"].fillna(0).gt(0).astype("int8")
    )
    salida["cantidad_canales_usados"] = canales.astype("int8")

    salida["tiene_operaciones_recurrentes"] = (
        salida["coperaciones_recurrentes_total"].gt(0).fillna(False).astype("int8")
    )
    salida["tiene_tarjeta_sin_uso"] = (
        salida["cantidad_marcas_tarjeta"]
        .gt(salida["cantidad_marcas_tarjeta_con_uso"])
        .astype("int8")
    )
    salida["cliente_anclado"] = (
        salida["tiene_payroll"].eq(1)
        & salida["coperaciones_recurrentes_total"].gt(0).fillna(False)
        & salida["cantidad_marcas_tarjeta"].gt(0)
    ).astype("int8")

    usa_homebanking = salida["chomebanking_transacciones"].fillna(0).gt(0)
    usa_mobile = salida["cmobile_app_trx"].fillna(0).gt(0)
    salida["usa_homebanking_transaccional"] = usa_homebanking.astype("int8")
    salida["usa_mobile_app_transaccional"] = usa_mobile.astype("int8")
    salida["actividad_digital_transaccional"] = (
        usa_homebanking | usa_mobile
    ).astype("int8")
    return salida


def construir_target_temporal(
    base: pd.DataFrame,
    copiar: bool = True,
) -> pd.DataFrame:
    """Construye clase y target por presencia futura del mismo cliente.

    La continuidad se evalua contra el mes calendario exacto. No se propaga
    ninguna etiqueta entre filas. Julio y agosto quedan censurados desde el
    inicio, aunque sea posible observar una etiqueta parcial en julio.
    """

    faltantes = sorted(set(COLUMNAS_CLAVE).difference(base.columns))
    if faltantes:
        raise ValueError("Faltan columnas para construir el target: " + ", ".join(faltantes))

    salida = base.copy() if copiar else base
    duplicados = int(salida.duplicated(list(COLUMNAS_CLAVE)).sum())
    if duplicados:
        raise ErrorAuditoria(
            "No se puede construir el target: hay "
            f"{duplicados:,} claves cliente-mes duplicadas."
        )

    salida = salida.sort_values(list(COLUMNAS_CLAVE)).reset_index(drop=True)
    indice_actual = _indice_mes(salida["foto_mes"])
    indice_siguiente = indice_actual.groupby(
        salida["numero_de_cliente"], sort=False
    ).shift(-1)
    indice_dos_meses = indice_actual.groupby(
        salida["numero_de_cliente"], sort=False
    ).shift(-2)

    presente_mes_1 = indice_siguiente.eq(indice_actual + 1)
    presente_mes_2 = indice_dos_meses.eq(indice_actual + 2)
    target_maduro = salida["foto_mes"].isin(MESES_TARGET_MADURO)

    clase = np.full(len(salida), None, dtype=object)
    clase[target_maduro & ~presente_mes_1] = "BAJA+1"
    clase[target_maduro & presente_mes_1 & ~presente_mes_2] = "BAJA+2"
    clase[target_maduro & presente_mes_1 & presente_mes_2] = "CONTINUA"
    salida["clase_ternaria"] = pd.Categorical(
        clase,
        categories=["CONTINUA", "BAJA+1", "BAJA+2"],
    )

    target = pd.Series(pd.NA, index=salida.index, dtype="Int8")
    target.loc[target_maduro] = (
        salida.loc[target_maduro, "clase_ternaria"]
        .eq("BAJA+2")
        .astype("int8")
    )
    salida["target_baja2"] = target
    return salida


def preprocesar_competencia_01(
    base_original: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta el orden aceptado: calidad, anclaje y target temporal."""

    base, reporte = aplicar_correcciones_calidad(base_original, copiar=True)
    base = crear_features_anclaje(base, copiar=False)
    base = construir_target_temporal(base, copiar=False)
    return base, reporte


def cargar_lista_features(ruta: Union[str, Path]) -> List[str]:
    """Lee y valida la lista congelada de 238 predictores."""

    tabla = pd.read_csv(ruta)
    if list(tabla.columns) != ["feature"]:
        raise ErrorAuditoria(
            "features_candidato_fe.csv debe contener una unica columna 'feature'."
        )
    features = tabla["feature"].astype(str).tolist()
    errores = []
    if len(features) != FEATURES_FINALES_ESPERADAS:
        errores.append(
            f"se esperaban {FEATURES_FINALES_ESPERADAS} features y hay {len(features)}"
        )
    if len(set(features)) != len(features):
        errores.append("la lista contiene nombres duplicados")
    _fallar_si(errores, "Lista de features invalida")
    return features


def seleccionar_entrada_feature_engineering(
    base: pd.DataFrame,
    features_finales: Sequence[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Conserva las 156 entradas congeladas y las columnas de control."""

    if not base.columns.is_unique:
        duplicadas = base.columns[base.columns.duplicated()].unique().tolist()
        raise ErrorAuditoria(
            "La entrada contiene columnas duplicadas: " + ", ".join(duplicadas)
        )
    features_entrada = list(features_finales[:FEATURES_ENTRADA_ESPERADAS])
    faltantes = [feature for feature in features_entrada if feature not in base]
    errores = []
    if len(features_entrada) != FEATURES_ENTRADA_ESPERADAS:
        errores.append(
            f"se esperaban {FEATURES_ENTRADA_ESPERADAS} entradas y hay "
            f"{len(features_entrada)}"
        )
    if faltantes:
        errores.append("faltan entradas: " + ", ".join(faltantes))
    _fallar_si(errores, "No se pudo preparar la entrada de feature engineering")

    columnas_control = [
        "numero_de_cliente",
        "foto_mes",
        "clase_ternaria",
        "target_baja2",
    ]
    return base.loc[:, columnas_control + features_entrada].copy(), features_entrada


def convertir_infinitos_en_nulos(
    base: pd.DataFrame,
    columnas: Sequence[str],
    copiar: bool = True,
) -> Tuple[pd.DataFrame, int]:
    """Convierte solamente ``+/-inf`` a nulo, sin imputar valores faltantes."""

    faltantes = [columna for columna in columnas if columna not in base]
    if faltantes:
        raise ValueError(
            "No se pueden sanear infinitos; faltan columnas: "
            + ", ".join(faltantes)
        )
    salida = base.copy() if copiar else base
    reemplazados = 0
    for columna in columnas:
        valores = pd.to_numeric(salida[columna], errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )
        mascara = np.isinf(valores)
        cantidad = int(mascara.sum())
        if cantidad:
            salida.loc[mascara, columna] = np.nan
            reemplazados += cantidad
    return salida, reemplazados


def auditar_preprocesamiento(base: pd.DataFrame) -> pd.DataFrame:
    """Ejecuta las auditorias obligatorias previas al feature engineering."""

    errores: List[str] = []
    faltantes = sorted(
        {
            *COLUMNAS_CLAVE,
            "clase_ternaria",
            "target_baja2",
            *COLUMNAS_ANCLAJE,
            *COLUMNAS_INDICADORES_CORRECCION,
        }.difference(base.columns)
    )
    if faltantes:
        errores.append("faltan columnas obligatorias: " + ", ".join(faltantes))
        _fallar_si(errores, "Fallo la auditoria de preprocesamiento")

    duplicados = int(base.duplicated(list(COLUMNAS_CLAVE)).sum())
    if duplicados:
        errores.append(f"hay {duplicados:,} filas cliente-mes duplicadas")
    claves_nulas = int(base.loc[:, list(COLUMNAS_CLAVE)].isna().sum().sum())
    if claves_nulas:
        errores.append(f"hay {claves_nulas:,} valores nulos en las claves")

    meses = tuple(sorted(pd.to_numeric(base["foto_mes"]).unique().tolist()))
    if meses != MESES_ESPERADOS:
        errores.append(f"meses observados {meses}; se esperaban {MESES_ESPERADOS}")
    if len(base) != FILAS_ESPERADAS:
        errores.append(f"hay {len(base):,} filas; se esperaban {FILAS_ESPERADAS:,}")

    duplicadas_presentes = [
        columna for columna in COLUMNAS_CONSUMO_DUPLICADAS if columna in base
    ]
    if duplicadas_presentes:
        errores.append(
            "siguen presentes consumos duplicados: " + ", ".join(duplicadas_presentes)
        )

    for marca in ("Master", "Visa"):
        ratio = base[f"{marca}_mfinanciacion_limite"].div(
            base[f"{marca}_mlimitecompra"].where(
                base[f"{marca}_mlimitecompra"].gt(0)
            )
        )
        restantes = int(ratio.gt(10).fillna(False).sum())
        if restantes:
            errores.append(
                f"quedan {restantes:,} limites anomalos sin corregir para {marca}"
            )

    filas_resumen = []
    for mes in MESES_ESPERADOS:
        bloque = base.loc[base["foto_mes"].eq(mes)]
        fila = {
            "foto_mes": mes,
            "filas": int(len(bloque)),
            "targets_disponibles": int(bloque["target_baja2"].notna().sum()),
            "casos_baja2": int(bloque["target_baja2"].eq(1).sum()),
        }
        filas_resumen.append(fila)
        esperado = CONTEOS_ESPERADOS[mes]
        for observado, clave_esperada in (
            (fila["filas"], "filas"),
            (fila["targets_disponibles"], "targets"),
            (fila["casos_baja2"], "baja2"),
        ):
            if observado != esperado[clave_esperada]:
                errores.append(
                    f"{mes}: {clave_esperada}={observado:,}; "
                    f"se esperaba {esperado[clave_esperada]:,}"
                )

    censurados_con_target = int(
        base.loc[base["foto_mes"].isin((202107, 202108)), "target_baja2"]
        .notna()
        .sum()
    )
    if censurados_con_target:
        errores.append(
            f"julio/agosto tienen {censurados_con_target:,} targets no nulos"
        )

    _fallar_si(errores, "Fallo la auditoria de preprocesamiento")
    return pd.DataFrame(filas_resumen)


def _familia_feature(
    feature: str,
    features_lag: set,
    features_carga: set,
) -> str:
    if feature in features_lag:
        return "lag_delta_eventos"
    if feature in features_carga:
        return "carga_financiera"
    if feature in COLUMNAS_ANCLAJE:
        return "anclaje"
    if feature.startswith("Master_"):
        return "tarjeta_master"
    if feature.startswith("Visa_"):
        return "tarjeta_visa"
    if any(token in feature for token in ("prestamo", "inversion", "plazo_fijo")):
        return "prestamos_inversiones"
    if feature.startswith("cseguro") or feature == "ccaja_seguridad":
        return "seguros"
    if any(
        token in feature
        for token in ("payroll", "debito_automatico", "pagodeservicios", "pagomiscuentas")
    ):
        return "payroll_debitos_pagos"
    if any(token in feature for token in ("transferencia", "cheque", "forex")):
        return "transferencias_cheques_forex"
    if any(
        token in feature
        for token in ("homebanking", "mobile", "atm", "cajas", "trx", "autoservicio")
    ):
        return "canales_actividad"
    if any(token in feature for token in ("cuenta", "caja_ahorro", "saldo")):
        return "cuentas_saldos"
    if any(token in feature for token in ("cliente", "rentabilidad", "margen", "producto")):
        return "cliente_rentabilidad"
    return "otras_base"


def resumir_nulos_por_familia(
    base: pd.DataFrame,
    features_finales: Sequence[str],
    features_lag: Sequence[str],
    features_carga: Sequence[str],
) -> pd.DataFrame:
    """Informa nulos por familia sin alterar ningun valor."""

    lag = set(features_lag)
    carga = set(features_carga)
    filas = []
    for feature in features_finales:
        filas.append(
            {
                "feature": feature,
                "familia": _familia_feature(feature, lag, carga),
                "nulos": int(base[feature].isna().sum()),
                "filas": int(len(base)),
            }
        )
    detalle = pd.DataFrame(filas)
    resumen = (
        detalle.groupby("familia", as_index=False)
        .agg(
            features=("feature", "size"),
            nulos=("nulos", "sum"),
            celdas=("filas", "sum"),
        )
        .sort_values("familia")
        .reset_index(drop=True)
    )
    resumen["pct_nulos"] = 100.0 * resumen["nulos"] / resumen["celdas"]
    return resumen


def _cantidad_infinitos(base: pd.DataFrame, columnas: Iterable[str]) -> int:
    total = 0
    for columna in columnas:
        valores = pd.to_numeric(base[columna], errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )
        total += int(np.isinf(valores).sum())
    return total


def auditar_features(
    base: pd.DataFrame,
    features_finales: Sequence[str],
    features_lag: Sequence[str],
    features_carga: Sequence[str],
    variables_lag: Sequence[str],
) -> Dict[str, object]:
    """Valida contrato 47 + 35 = 82, lista final, infinitos y lags."""

    errores: List[str] = []
    if not base.columns.is_unique:
        errores.append("la salida contiene nombres de columna duplicados")
    if len(base) != FILAS_ESPERADAS:
        errores.append(
            f"la salida tiene {len(base):,} filas; se esperaban {FILAS_ESPERADAS:,}"
        )
    meses = tuple(sorted(pd.to_numeric(base["foto_mes"]).unique().tolist()))
    if meses != MESES_ESPERADOS:
        errores.append(f"meses observados {meses}; se esperaban {MESES_ESPERADOS}")
    duplicados_clave = int(base.duplicated(list(COLUMNAS_CLAVE)).sum())
    if duplicados_clave:
        errores.append(
            f"la salida tiene {duplicados_clave:,} claves cliente-mes duplicadas"
        )
    claves_nulas = int(base.loc[:, list(COLUMNAS_CLAVE)].isna().sum().sum())
    if claves_nulas:
        errores.append(f"la salida tiene {claves_nulas:,} valores nulos en claves")
    faltantes = [feature for feature in features_finales if feature not in base]
    if len(features_finales) != FEATURES_FINALES_ESPERADAS:
        errores.append(
            f"hay {len(features_finales)} features finales; "
            f"se esperaban {FEATURES_FINALES_ESPERADAS}"
        )
    if len(features_lag) != FEATURES_LAG_ESPERADAS:
        errores.append(
            f"se generaron {len(features_lag)} features lag/delta/evento; "
            f"se esperaban {FEATURES_LAG_ESPERADAS}"
        )
    if len(features_carga) != FEATURES_CARGA_ESPERADAS:
        errores.append(
            f"se generaron {len(features_carga)} features de carga; "
            f"se esperaban {FEATURES_CARGA_ESPERADAS}"
        )
    lag_esperadas = list(
        features_finales[
            FEATURES_ENTRADA_ESPERADAS:
            FEATURES_ENTRADA_ESPERADAS + FEATURES_LAG_ESPERADAS
        ]
    )
    carga_esperadas = list(
        features_finales[
            FEATURES_ENTRADA_ESPERADAS + FEATURES_LAG_ESPERADAS:
        ]
    )
    if list(features_lag) != lag_esperadas:
        errores.append(
            "los nombres/orden de las 47 features lag no coinciden con el CSV"
        )
    if list(features_carga) != carga_esperadas:
        errores.append(
            "los nombres/orden de las 35 features de carga no coinciden con el CSV"
        )
    if faltantes:
        errores.append(f"faltan {len(faltantes)} features: " + ", ".join(faltantes))
    _fallar_si(errores, "Fallo la auditoria de features")

    no_numericas = [
        feature
        for feature in features_finales
        if not pd.api.types.is_numeric_dtype(base[feature])
    ]
    if no_numericas:
        errores.append("features no numericas: " + ", ".join(no_numericas))

    infinitos = _cantidad_infinitos(base, features_finales)
    if infinitos:
        errores.append(f"hay {infinitos:,} valores infinitos")

    ordenada = base.sort_values(list(COLUMNAS_CLAVE)).reset_index(drop=True)
    indice = _indice_mes(ordenada["foto_mes"])
    indice_anterior = indice.groupby(
        ordenada["numero_de_cliente"], sort=False
    ).shift(1)
    historia_esperada = indice.sub(indice_anterior).eq(1)
    historia_observada = ordenada["_tiene_historia_lag1"].eq(1)
    discrepancias_historia = int(historia_esperada.ne(historia_observada).sum())
    if discrepancias_historia:
        errores.append(
            f"_tiene_historia_lag1 difiere en {discrepancias_historia:,} filas"
        )

    carga_dependiente_lag = [
        feature
        for feature in features_carga
        if feature == "payroll_cambio_relativo_lag1"
        or feature.endswith("_sobre_payroll_lag1")
        or feature.endswith("_sobre_payroll_delta1")
    ]
    columnas_dependientes_lag = list(features_lag) + carga_dependiente_lag
    fuera_de_continuidad_no_nulos = int(
        ordenada.loc[~historia_esperada, columnas_dependientes_lag]
        .notna()
        .sum()
        .sum()
    )
    if fuera_de_continuidad_no_nulos:
        errores.append(
            "hay "
            f"{fuera_de_continuidad_no_nulos:,} valores dependientes de lag "
            "sin mes anterior consecutivo"
        )

    discrepancias_lag = []
    for variable in variables_lag:
        nombre_lag = f"{variable}_lag1"
        esperado = (
            ordenada.groupby("numero_de_cliente", sort=False)[variable]
            .shift(1)
            .where(historia_esperada)
        )
        actual = ordenada[nombre_lag]
        esperado_np = pd.to_numeric(esperado, errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )
        actual_np = pd.to_numeric(actual, errors="coerce").to_numpy(
            dtype="float64", na_value=np.nan
        )
        if not np.allclose(
            actual_np,
            esperado_np,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            discrepancias_lag.append(nombre_lag)
    if discrepancias_lag:
        errores.append("lags con valores incorrectos: " + ", ".join(discrepancias_lag))

    _fallar_si(errores, "Fallo la auditoria de features")
    return {
        "filas": int(len(base)),
        "features_finales": int(len(features_finales)),
        "features_faltantes": int(len(faltantes)),
        "features_lag_delta_eventos": int(len(features_lag)),
        "features_carga_financiera": int(len(features_carga)),
        "features_generadas": int(len(features_lag) + len(features_carga)),
        "valores_infinitos": int(infinitos),
        "discrepancias_continuidad": int(discrepancias_historia),
        "lags_con_discrepancias": int(len(discrepancias_lag)),
    }
