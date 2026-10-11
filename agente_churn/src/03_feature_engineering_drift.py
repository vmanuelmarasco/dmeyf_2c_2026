#!/usr/bin/env python3
"""
src/03_feature_engineering_drift.py — Agente 3: Feature Engineering en SQL y reporte de drift

## Propósito
0. Construir variables de dominio bancario (VARIABLES_BANCARIAS): tarjetas consolidadas,
   uso del límite, liquidez, deuda, sueldo, canales, vínculo y rentabilidad.
1. Construir variables históricas por cliente en DuckDB (z402), también sobre las bancarias:
   lags, deltas, media móvil y pendiente de 3 meses.
2. Opcional: reemplazar variables por su percent_rank dentro de cada foto_mes (z402),
   para neutralizar inflación y drift.
3. Medir el drift (PSI) entre los meses del entrenamiento final y el mes a predecir (z701)
   y dejar un REPORTE. No se borra nada en automático: la cátedra analiza caso por caso.
   Las variables a sacar se ponen a mano en `fe.excluir` de la config.

## Reglas anti-fuga
- Solo LAG (nunca LEAD) y ventanas hacia atrás (`ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`).
- Los lags se calculan sobre TODOS los meses (julio incluido, aunque no tenga clase),
  y recién después se filtran los meses que usa el pipeline.

## Nota sobre z501 (cluster con Random Forest)
z501 es una herramienta para PERFILAR a los BAJA+2, no para crear una variable.
Se corre aparte con el script de la cátedra; por eso aquí no hay `rf_cluster`.

Salida: data/exp/exp<N>/fe.parquet y data/exp/exp<N>/drift_psi.csv
"""

import argparse

import duckdb
import numpy as np
import pandas as pd

from config import NO_FEATURES, cargar_config, meses_necesarios


def columnas_numericas(con, tabla: str) -> list[str]:
    tipos_num = ("INTEGER", "BIGINT", "DOUBLE", "FLOAT", "DECIMAL", "HUGEINT", "SMALLINT", "TINYINT")
    return [
        nombre for nombre, tipo, *_ in con.execute(f"DESCRIBE {tabla}").fetchall()
        if tipo.upper().startswith(tipos_num) and nombre not in NO_FEATURES
    ]


def _suma(*cols: str) -> str:
    return " + ".join(f"COALESCE({c}, 0)" for c in cols)


def _ratio(num: str, den: str) -> str:
    return f"({num}) / NULLIF({den}, 0)"


# Variables de dominio bancario: (nombre, expresión SQL, columnas que necesita, qué representa).
# Solo se crean si existen todas sus columnas. Después reciben lags y deltas como cualquier otra,
# así que también aparecen sus cambios mes a mes (por ejemplo, la caída de la liquidez total).
VARIABLES_BANCARIAS = [
    # --- Tarjetas: Visa + Master consolidadas (la caída de consumos es señal de baja en el EDA)
    ("tc_saldo_total", _suma("Visa_msaldototal", "Master_msaldototal"),
     ["Visa_msaldototal", "Master_msaldototal"], "Deuda total en tarjetas"),
    ("tc_consumo_total", _suma("Visa_mconsumototal", "Master_mconsumototal"),
     ["Visa_mconsumototal", "Master_mconsumototal"], "Consumo total con tarjetas"),
    ("tc_limite_total", _suma("Visa_mlimitecompra", "Master_mlimitecompra"),
     ["Visa_mlimitecompra", "Master_mlimitecompra"], "Límite de compra total"),
    ("tc_pagado_total", _suma("Visa_mpagado", "Master_mpagado"),
     ["Visa_mpagado", "Master_mpagado"], "Pagos de tarjeta del mes"),
    ("tc_uso_limite", _ratio(_suma("Visa_msaldototal", "Master_msaldototal"),
                             _suma("Visa_mlimitecompra", "Master_mlimitecompra")),
     ["Visa_msaldototal", "Master_msaldototal", "Visa_mlimitecompra", "Master_mlimitecompra"],
     "Qué parte del límite está usando"),
    ("tc_pago_sobre_saldo", _ratio(_suma("Visa_mpagado", "Master_mpagado"),
                                   _suma("Visa_msaldototal", "Master_msaldototal")),
     ["Visa_mpagado", "Master_mpagado", "Visa_msaldototal", "Master_msaldototal"],
     "Cuánto de la deuda paga (cancelar todo antes de irse)"),
    ("tc_pagominimo_sobre_saldo", _ratio(_suma("Visa_mpagominimo", "Master_mpagominimo"),
                                         _suma("Visa_msaldototal", "Master_msaldototal")),
     ["Visa_mpagominimo", "Master_mpagominimo", "Visa_msaldototal", "Master_msaldototal"],
     "Peso del pago mínimo sobre la deuda"),
    ("tc_peor_status", "GREATEST(COALESCE(Visa_status, -1), COALESCE(Master_status, -1))",
     ["Visa_status", "Master_status"], "Peor estado entre las dos tarjetas (-1 = sin tarjetas)"),
    ("tc_delinquency", "GREATEST(COALESCE(Visa_delinquency, 0), COALESCE(Master_delinquency, 0))",
     ["Visa_delinquency", "Master_delinquency"], "Mora en alguna de las dos tarjetas"),
    ("tc_activas", "CAST(COALESCE(Visa_status, -1) = 0 AS INTEGER) + CAST(COALESCE(Master_status, -1) = 0 AS INTEGER)",
     ["Visa_status", "Master_status"], "Cantidad de tarjetas en estado normal (0, 1 o 2)"),
    ("sin_visa", "CAST(Visa_status IS NULL AS INTEGER)",
     ["Visa_status"], "No tiene Visa (en el EDA: 3% de BAJA+2, 5 veces la tasa base)"),
    # --- Liquidez, inversión y deuda: el "vaciamiento" de cuentas
    ("inversiones_pesos", _suma("mplazo_fijo_pesos", "minversion1_pesos", "minversion2"),
     ["mplazo_fijo_pesos", "minversion1_pesos", "minversion2"], "Plazos fijos e inversiones en pesos"),
    ("prestamos_total", _suma("mprestamos_personales", "mprestamos_prendarios", "mprestamos_hipotecarios"),
     ["mprestamos_personales", "mprestamos_prendarios", "mprestamos_hipotecarios"], "Deuda total por préstamos"),
    ("ratio_tc_saldos", _ratio(_suma("Visa_msaldototal", "Master_msaldototal"), _suma("mcaja_ahorro", "mcuenta_corriente")),
     ["Visa_msaldototal", "Master_msaldototal", "mcaja_ahorro", "mcuenta_corriente"],
     "Deuda de tarjetas sobre plata en cuentas"),
    ("ratio_pasivos_activos", _ratio("mpasivos_margen", "mactivos_margen"),
     ["mpasivos_margen", "mactivos_margen"], "Margen que deja como ahorrista vs como deudor"),
    # --- Sueldo: el cliente que deja de cobrar en el banco
    ("payroll_total", _suma("mpayroll", "mpayroll2"), ["mpayroll", "mpayroll2"], "Sueldo total acreditado"),
    ("cobra_sueldo", f"CAST({_suma('cpayroll_trx', 'cpayroll2_trx')} > 0 AS INTEGER)",
     ["cpayroll_trx", "cpayroll2_trx"], "Cobra el sueldo en el banco"),
    ("saldo_sobre_sueldo", _ratio("mcuentas_saldo", _suma("mpayroll", "mpayroll2")),
     ["mcuentas_saldo", "mpayroll", "mpayroll2"], "Cuántos sueldos tiene guardados"),
    ("transferencias_neto", "COALESCE(mtransferencias_recibidas, 0) - COALESCE(mtransferencias_emitidas, 0)",
     ["mtransferencias_recibidas", "mtransferencias_emitidas"], "Plata que entra menos plata que sale por transferencias"),
    # --- Uso del banco y canales
    ("trx_digitales", _suma("chomebanking_transacciones", "cmobile_app_trx"),
     ["chomebanking_transacciones", "cmobile_app_trx"], "Transacciones por homebanking y app"),
    ("trx_atencion", _suma("ccajas_transacciones", "ccallcenter_transacciones"),
     ["ccajas_transacciones", "ccallcenter_transacciones"], "Transacciones por caja y call center"),
    ("ratio_digital", _ratio(_suma("chomebanking_transacciones", "cmobile_app_trx"),
                             _suma("chomebanking_transacciones", "cmobile_app_trx", "ccajas_transacciones", "ccallcenter_transacciones")),
     ["chomebanking_transacciones", "cmobile_app_trx", "ccajas_transacciones", "ccallcenter_transacciones"],
     "Qué parte de su operatoria es digital"),
    ("debitos_automaticos_total", _suma("ccuenta_debitos_automaticos", "ctarjeta_visa_debitos_automaticos", "ctarjeta_master_debitos_automaticos"),
     ["ccuenta_debitos_automaticos", "ctarjeta_visa_debitos_automaticos", "ctarjeta_master_debitos_automaticos"],
     "Débitos automáticos adheridos (vínculo: hay que darlos de baja para irse)"),
    ("pagos_servicios_total", _suma("cpagodeservicios", "cpagomiscuentas"),
     ["cpagodeservicios", "cpagomiscuentas"], "Pagos de servicios del mes"),
    ("ratio_ctrx_antiguedad", _ratio("ctrx_quarter", "cliente_antiguedad"),
     ["ctrx_quarter", "cliente_antiguedad"], "Actividad relativa a la antigüedad"),
    # --- Vínculo y rentabilidad
    ("seguros_total", _suma("cseguro_vida", "cseguro_auto", "cseguro_vivienda", "cseguro_accidentes_personales"),
     ["cseguro_vida", "cseguro_auto", "cseguro_vivienda", "cseguro_accidentes_personales"], "Seguros contratados"),
    ("rentabilidad_por_producto", _ratio("mrentabilidad", "cproductos"),
     ["mrentabilidad", "cproductos"], "Rentabilidad mensual por producto"),
    ("comisiones_por_producto", _ratio("mcomisiones", "cproductos"),
     ["mcomisiones", "cproductos"], "Comisiones por producto (¿le cobran mucho?)"),
    ("edad_al_alta", "cliente_edad - cliente_antiguedad / 12.0",
     ["cliente_edad", "cliente_antiguedad"], "Edad que tenía al hacerse cliente"),
]


# Montos en pesos creados en VARIABLES_BANCARIAS (los ratios no van: ya no dependen de la inflación)
MONTOS_BANCARIOS = {"tc_saldo_total", "tc_consumo_total", "tc_limite_total", "tc_pagado_total",
                    "inversiones_pesos", "prestamos_total", "payroll_total", "transferencias_neto"}


def es_monetaria(v: str) -> bool:
    """Convención del diccionario de datos: las variables que empiezan con "m" son montos en pesos."""
    return (v[:1] == "m" and v[1:2].isalpha()) or v.startswith(("Visa_m", "Master_m")) or v in MONTOS_BANCARIOS


def construir_features(con, cfg: dict):
    fe = cfg["fe"]
    origen = cfg["datos"]["con_clase"]
    excluir = fe.get("excluir") or []
    # Una columna original excluida se saca ANTES del FE (no genera lags ni deltas ni bancarias);
    # una variable derivada (ej. mpayroll_delta2) se saca DESPUÉS, del resultado final.
    originales = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{origen}')").fetchall()}
    excluir_orig = [c for c in excluir if c in originales]
    exclude_sql = f" EXCLUDE ({', '.join(excluir_orig)})" if excluir_orig else ""
    con.execute(f"CREATE OR REPLACE VIEW raw AS SELECT *{exclude_sql} FROM read_parquet('{origen}')")
    if excluir:
        print(f"🚫 Excluidas por config (revisión de drift): {excluir}")

    # 1) Variables de dominio bancario (antes de los lags, así ellas también tienen historia)
    if fe.get("bancarias", True):
        disponibles = set(columnas_numericas(con, "raw"))
        creadas = [(n, e) for n, e, req, _ in VARIABLES_BANCARIAS if set(req) <= disponibles and n not in excluir]
        faltantes = [n for n, e, req, _ in VARIABLES_BANCARIAS if not set(req) <= disponibles]
        campos_banc = ", ".join(f"{e} AS {n}" for n, e in creadas)
        con.execute(f"CREATE OR REPLACE VIEW raw2 AS SELECT *{', ' + campos_banc if campos_banc else ''} FROM raw")
        print(f"🏦 {len(creadas)} variables bancarias creadas"
              + (f" ({len(faltantes)} omitidas por faltar columnas: {', '.join(faltantes)})" if faltantes else ""))
    else:
        con.execute("CREATE OR REPLACE VIEW raw2 AS SELECT * FROM raw")

    variables = columnas_numericas(con, "raw2")
    print(f"🛠️  Feature engineering sobre {len(variables)} variables numéricas...")

    # 2) percent_rank por mes (opcional). "monetarias" = todas las variables en pesos.
    pedido = fe.get("rank_por_mes") or []
    if isinstance(pedido, str):
        pedido = [pedido]
    rank_vars = [v for v in variables if v in pedido or ("monetarias" in pedido and es_monetaria(v))]
    if rank_vars:
        reemplazos = ", ".join(
            f"CASE WHEN {v} IS NULL THEN NULL ELSE PERCENT_RANK() OVER "
            f"(PARTITION BY foto_mes, {v} IS NULL ORDER BY {v}) END AS {v}"
            for v in rank_vars
        )
        con.execute(f"CREATE OR REPLACE VIEW base AS SELECT * REPLACE ({reemplazos}) FROM raw2")
        print(f"   • percent_rank por mes en {len(rank_vars)} variables"
              + (f": {rank_vars}" if len(rank_vars) <= 10 else " (monetarias: posición dentro del mes, no pesos)"))
    else:
        con.execute("CREATE OR REPLACE VIEW base AS SELECT * FROM raw2")

    # 3) lags, deltas y ventanas
    campos = []
    for v in variables:
        for k in fe.get("lags", []):
            campos.append(f"LAG({v}, {k}) OVER w AS {v}_lag{k}")
        for k in fe.get("deltas", []):
            campos.append(f"{v} - LAG({v}, {k}) OVER w AS {v}_delta{k}")
    for v in [v for v in fe.get("ventana_variables", []) if v in variables]:
        campos.append(f"AVG({v}) OVER w3 AS {v}_avg3")
        campos.append(f"REGR_SLOPE({v}, periodo0) OVER w3 AS {v}_slope3")

    meses = ", ".join(str(m) for m in meses_necesarios(cfg))
    select_campos = ",\n        ".join(campos) or "NULL AS sin_features"
    query = f"""
    CREATE OR REPLACE TABLE fe AS
    SELECT * EXCLUDE (periodo0) FROM (
        SELECT *,
        {select_campos}
        FROM (SELECT *, CAST(foto_mes / 100 AS INTEGER) * 12 + foto_mes % 100 AS periodo0 FROM base)
        WINDOW
            w  AS (PARTITION BY numero_de_cliente ORDER BY foto_mes),
            w3 AS (PARTITION BY numero_de_cliente ORDER BY foto_mes ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)
    )
    WHERE foto_mes IN ({meses})
    """
    con.execute(query)
    columnas_fe = {r[0] for r in con.execute("DESCRIBE fe").fetchall()}
    excluir_deriv = [c for c in excluir if c not in originales and c in columnas_fe]
    if excluir_deriv:
        con.execute(f"CREATE OR REPLACE TABLE fe AS SELECT * EXCLUDE ({', '.join(excluir_deriv)}) FROM fe")
    no_encontradas = [c for c in excluir if c not in originales and c not in columnas_fe]
    if no_encontradas:
        print(f"⚠️  En fe.excluir pero no existen (¿error de tipeo?): {no_encontradas}")
    n = con.execute("SELECT COUNT(*) FROM fe").fetchone()[0]
    ncol = len(con.execute("DESCRIBE fe").fetchall())
    print(f"✅ {n:,} filas y {ncol} columnas (solo meses {meses}).")


def psi(referencia: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index con cortes por cuantiles de la referencia."""
    ref = referencia[~np.isnan(referencia)]
    act = actual[~np.isnan(actual)]
    if len(ref) == 0 or len(act) == 0:
        return 0.0
    bins = np.unique(np.percentile(ref, np.linspace(0, 100, buckets + 1)))
    if len(bins) < 2:
        return 0.0
    bins[0], bins[-1] = -np.inf, np.inf
    p_ref = np.histogram(ref, bins)[0] / len(ref)
    p_act = np.histogram(act, bins)[0] / len(act)
    p_ref = np.where(p_ref == 0, 1e-6, p_ref)
    p_act = np.where(p_act == 0, 1e-6, p_act)
    return float(np.sum((p_act - p_ref) * np.log(p_act / p_ref)))


def reporte_drift(con, cfg: dict, muestra: int = 50_000):
    m = cfg["meses"]
    # Referencia: los meses del ENTRENAMIENTO FINAL, porque es el modelo que predice el futuro
    # (y tiene historia completa para los lags; los meses de train pueden no tenerla).
    train = ", ".join(str(x) for x in m["train_final"])
    ref = con.execute(f"SELECT * FROM fe WHERE foto_mes IN ({train}) USING SAMPLE reservoir({muestra} ROWS) REPEATABLE (1)").df()
    act = con.execute(f"SELECT * FROM fe WHERE foto_mes = {m['future']} USING SAMPLE reservoir({muestra} ROWS) REPEATABLE (1)").df()
    cols = [c for c in ref.columns if c not in NO_FEATURES and pd.api.types.is_numeric_dtype(ref[c])]

    filas = []
    for c in cols:
        r, a = ref[c].astype(float).values, act[c].astype(float).values
        filas.append({
            "variable": c,
            "psi": psi(r, a),
            "pct_nulos_train": float(np.isnan(r).mean() * 100),
            "pct_nulos_future": float(np.isnan(a).mean() * 100),
            "pct_ceros_train": float((r == 0).mean() * 100),
            "pct_ceros_future": float((a == 0).mean() * 100),
        })
    tabla = pd.DataFrame(filas).sort_values("psi", ascending=False)
    tabla["revisar"] = (tabla["psi"] > 0.25) | ((tabla["pct_nulos_future"] - tabla["pct_nulos_train"]).abs() > 10)
    salida = cfg["exp_dir"] / "drift_psi.csv"
    tabla.to_csv(salida, index=False)

    print(f"\n🛡️  Drift: train_final {m['train_final']} vs future {m['future']} (z701). Top 15 por PSI:")
    print(tabla.head(15).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    marcadas = tabla[tabla["revisar"]]
    print(f"\n   {len(marcadas)} variables marcadas para revisar (PSI > 0,25 o salto de nulos > 10 pp).")
    if len(marcadas):
        print("   Las primeras 10 (un lag con muchos nulos en train suele ser falta de historia, no drift):")
        print(marcadas.head(10)[["variable", "psi", "pct_nulos_train", "pct_nulos_future"]]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"   Reporte completo: {salida}")
    print("   Nada se eliminó. Si decidís sacar alguna, agregala a fe.excluir en la config y volvé a correr este paso.")


def main():
    parser = argparse.ArgumentParser(description="Agente 3: FE en SQL y reporte de drift")
    parser.add_argument("--config", default="config/experimento.yaml")
    args = parser.parse_args()
    cfg = cargar_config(args.config)

    con = duckdb.connect()
    construir_features(con, cfg)
    reporte_drift(con, cfg)
    con.execute(f"COPY fe TO '{cfg['fe_path']}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"\n💾 Guardado en {cfg['fe_path']}")


if __name__ == "__main__":
    main()
