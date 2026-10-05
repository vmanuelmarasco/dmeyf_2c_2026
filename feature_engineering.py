"""Construccion reproducible del candidato de feature engineering.

El punto de entrada espera la salida de ``competencia_01_fe_anclaje_final``
y agrega solamente los bloques aceptados durante el desarrollo:

1. lag 1, delta 1 y eventos de actividad;
2. carga financiera relativa al payroll.

No incluye los bloques descartados de tarjetas, fechas ni flujo.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd


VARIABLES_LAG1 = [
    "ctrx_quarter",
    "coperaciones_recurrentes_total",
    "cantidad_canales_usados",
    "cproductos",
    "mcaja_ahorro",
    "mcuentas_saldo",
    "mpasivos_margen",
    "mrentabilidad",
    "ctarjeta_debito_transacciones",
    "mautoservicio",
    "ctarjeta_visa_transacciones",
    "mtarjeta_visa_consumo",
    "chomebanking_transacciones",
    "cmobile_app_trx",
    "catm_trx",
    "catm_trx_other",
    "cpayroll_trx",
    "mpayroll",
    "mtransferencias_recibidas",
    "mtransferencias_emitidas",
]


def _suma_sin_null(series: List[pd.Series]) -> pd.Series:
    tabla = pd.concat(series, axis=1)
    resultado = tabla.fillna(0).sum(axis=1)
    return resultado.mask(tabla.isna().all(axis=1))


def _ratio_seguro(
    numerador: pd.Series,
    denominador: pd.Series,
) -> pd.Series:
    return numerador / denominador.where(denominador > 0)


def construir_features_candidato(
    base_entrada: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Agrega los 47 features temporales y los 35 de carga financiera.

    Parameters
    ----------
    base_entrada:
        DataFrame con una fila por cliente y foto_mes, proveniente de la
        vista ``competencia_01_fe_anclaje_final``.

    Returns
    -------
    base:
        Copia ordenada de la base con los features agregados.
    features_lag1:
        Nombres de los 47 features lag/delta/evento.
    features_carga:
        Nombres de los 35 features de carga financiera.
    """

    columnas_clave = {"numero_de_cliente", "foto_mes"}
    faltantes_clave = columnas_clave.difference(base_entrada.columns)
    if faltantes_clave:
        raise ValueError(
            "Faltan columnas clave: "
            + ", ".join(sorted(faltantes_clave))
        )

    faltantes_lag = set(VARIABLES_LAG1).difference(base_entrada.columns)
    if faltantes_lag:
        raise ValueError(
            "Faltan columnas para construir lag1: "
            + ", ".join(sorted(faltantes_lag))
        )

    base = (
        base_entrada.copy()
        .sort_values(["numero_de_cliente", "foto_mes"])
        .reset_index(drop=True)
    )

    anio = base["foto_mes"] // 100
    mes = base["foto_mes"] % 100
    base["_indice_mes"] = anio * 12 + mes
    base["_indice_mes_anterior"] = (
        base.groupby("numero_de_cliente", sort=False)["_indice_mes"]
        .shift(1)
    )

    historia_valida = (
        base["_indice_mes"] - base["_indice_mes_anterior"] == 1
    )
    base["_tiene_historia_lag1"] = historia_valida.astype("int8")

    nuevas_lag = {}
    lags = {}
    deltas = {}

    for variable in VARIABLES_LAG1:
        lag_crudo = (
            base.groupby("numero_de_cliente", sort=False)[variable]
            .shift(1)
        )
        lag_valido = lag_crudo.where(historia_valida)
        delta_valido = base[variable] - lag_valido

        nuevas_lag[f"{variable}_lag1"] = lag_valido
        nuevas_lag[f"{variable}_delta1"] = delta_valido
        lags[variable] = lag_valido
        deltas[variable] = delta_valido

    def crear_evento(condicion: pd.Series) -> pd.Series:
        condicion_limpia = condicion.fillna(False).astype("int8")
        return pd.Series(
            np.where(historia_valida, condicion_limpia, np.nan),
            index=base.index,
        )

    nuevas_lag["apago_payroll_lag1"] = crear_evento(
        (lags["cpayroll_trx"] > 0) & (base["cpayroll_trx"] == 0)
    )
    nuevas_lag["apago_recurrentes_lag1"] = crear_evento(
        (lags["coperaciones_recurrentes_total"] > 0)
        & (base["coperaciones_recurrentes_total"] == 0)
    )
    nuevas_lag["apago_visa_lag1"] = crear_evento(
        (lags["ctarjeta_visa_transacciones"] > 0)
        & (base["ctarjeta_visa_transacciones"] == 0)
    )
    nuevas_lag["apago_homebanking_lag1"] = crear_evento(
        (lags["chomebanking_transacciones"] > 0)
        & (base["chomebanking_transacciones"] == 0)
    )
    nuevas_lag["migracion_atm_ajeno_lag1"] = crear_evento(
        (deltas["catm_trx"] < 0) & (deltas["catm_trx_other"] > 0)
    )
    nuevas_lag["caida_ctrx_20pct_lag1"] = crear_evento(
        (lags["ctrx_quarter"] > 0)
        & (base["ctrx_quarter"] < 0.80 * lags["ctrx_quarter"])
    )
    nuevas_lag["caida_canales_lag1"] = crear_evento(
        base["cantidad_canales_usados"]
        < lags["cantidad_canales_usados"]
    )

    base = pd.concat(
        [base, pd.DataFrame(nuevas_lag, index=base.index)],
        axis=1,
    )

    historia_valida_carga = base["_tiene_historia_lag1"] == 1

    def lag1_serie(serie: pd.Series) -> pd.Series:
        return (
            serie.groupby(base["numero_de_cliente"], sort=False)
            .shift(1)
            .where(historia_valida_carga)
        )

    payroll_actual = base["mpayroll"]
    payroll_lag1 = lag1_serie(payroll_actual)
    payroll_positivo_actual = payroll_actual.where(payroll_actual > 0)
    payroll_positivo_lag1 = payroll_lag1.where(payroll_lag1 > 0)

    tabla_payroll_referencia = pd.concat(
        [payroll_positivo_actual, payroll_positivo_lag1],
        axis=1,
    )
    payroll_referencia_2m = (
        tabla_payroll_referencia.mean(axis=1, skipna=True)
        .mask(tabla_payroll_referencia.isna().all(axis=1))
    )

    saldos_tarjeta_positivos = base[
        ["Visa_msaldopesos", "Master_msaldopesos"]
    ].clip(lower=0)

    deuda_tarjetas_pesos = _suma_sin_null(
        [
            saldos_tarjeta_positivos["Visa_msaldopesos"],
            saldos_tarjeta_positivos["Master_msaldopesos"],
        ]
    )
    pago_minimo_tarjetas = _suma_sin_null(
        [base["Visa_mpagominimo"], base["Master_mpagominimo"]]
    )
    prestamos_totales = _suma_sin_null(
        [
            base["mprestamos_personales"],
            base["mprestamos_prendarios"],
            base["mprestamos_hipotecarios"],
        ]
    )
    descubierto_utilizado = (-base["mcuenta_corriente"]).clip(lower=0)
    deuda_total_pesos = _suma_sin_null(
        [
            deuda_tarjetas_pesos,
            prestamos_totales,
            descubierto_utilizado,
        ]
    )

    numeradores_carga = {
        "comisiones": base["mcomisiones"],
        "comisiones_mantenimiento": base["mcomisiones_mantenimiento"],
        "deuda_tarjetas_pesos": deuda_tarjetas_pesos,
        "pago_minimo_tarjetas": pago_minimo_tarjetas,
        "prestamos_totales": prestamos_totales,
        "descubierto_utilizado": descubierto_utilizado,
        "deuda_total_pesos": deuda_total_pesos,
        "saldo_cuentas": base["mcuentas_saldo"],
    }

    nuevas_carga = {
        "payroll_referencia_2m": payroll_referencia_2m,
        "sin_payroll_monto_actual": (
            payroll_actual.fillna(0).le(0).astype("int8")
        ),
        "payroll_cambio_relativo_lag1": (
            (payroll_actual - payroll_lag1)
            / payroll_lag1.where(payroll_lag1 > 0)
        ),
    }

    for nombre, numerador_actual in numeradores_carga.items():
        numerador_lag1 = lag1_serie(numerador_actual)
        ratio_actual = _ratio_seguro(numerador_actual, payroll_actual)
        ratio_referencia = _ratio_seguro(
            numerador_actual,
            payroll_referencia_2m,
        )
        ratio_lag1 = _ratio_seguro(numerador_lag1, payroll_lag1)

        nuevas_carga[f"{nombre}_sobre_payroll_actual"] = ratio_actual
        nuevas_carga[
            f"{nombre}_sobre_payroll_referencia_2m"
        ] = ratio_referencia
        nuevas_carga[f"{nombre}_sobre_payroll_lag1"] = ratio_lag1
        nuevas_carga[f"{nombre}_sobre_payroll_delta1"] = (
            ratio_actual - ratio_lag1
        )

    base = pd.concat(
        [base, pd.DataFrame(nuevas_carga, index=base.index)],
        axis=1,
    )

    features_lag1 = list(nuevas_lag.keys())
    features_carga = list(nuevas_carga.keys())
    features_generadas = features_lag1 + features_carga
    base[features_generadas] = base[features_generadas].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return base, features_lag1, features_carga
