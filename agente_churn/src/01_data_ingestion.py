#!/usr/bin/env python3
"""
src/01_data_ingestion.py — Agente 1: Ingestión y construcción de clase_ternaria

## Propósito
Leer el dataset crudo de la competencia y calcular `clase_ternaria` como en z101 / z494:
- BAJA+1: el cliente está en t y no está en t+1.
- BAJA+2: está en t y en t+1, pero no en t+2.
- CONTINUA: el resto.
El último mes queda sin clase y el anteúltimo solo puede tener BAJA+1 (no se ve t+2).
Si el archivo ya trae `clase_ternaria`, se respeta.

Todo se hace en DuckDB, sin cargar el dataset entero en memoria.
Salida: `datos.con_clase` (parquet) + reporte por mes en consola.
"""

import argparse
from pathlib import Path

import duckdb

from config import cargar_config

SQL_CLASE = """
CREATE OR REPLACE TABLE con_clase AS
WITH base AS (
    SELECT *, CAST(foto_mes / 100 AS INTEGER) * 12 + foto_mes % 100 AS periodo0
    FROM crudo
),
leads AS (
    SELECT numero_de_cliente, periodo0,
           LEAD(periodo0, 1) OVER w AS periodo1,
           LEAD(periodo0, 2) OVER w AS periodo2
    FROM base
    WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY periodo0)
),
topes AS (SELECT MAX(periodo0) AS ultimo FROM base)
SELECT b.* EXCLUDE (periodo0),
    CASE
        WHEN l.periodo0 < t.ultimo - 1 AND l.periodo1 = l.periodo0 + 1
             AND (l.periodo2 IS NULL OR l.periodo0 + 2 < l.periodo2) THEN 'BAJA+2'
        WHEN l.periodo0 < t.ultimo
             AND (l.periodo1 IS NULL OR l.periodo0 + 1 < l.periodo1) THEN 'BAJA+1'
        WHEN l.periodo0 < t.ultimo - 1 THEN 'CONTINUA'
        ELSE NULL
    END AS clase_ternaria
FROM base b
JOIN leads l USING (numero_de_cliente, periodo0)
CROSS JOIN topes t
"""


def main():
    parser = argparse.ArgumentParser(description="Agente 1: ingestión y clase_ternaria")
    parser.add_argument("--config", default="config/experimento.yaml")
    args = parser.parse_args()
    cfg = cargar_config(args.config)

    crudo = cfg["datos"]["crudo"]
    salida = Path(cfg["datos"]["con_clase"])
    if not Path(crudo).exists():
        raise FileNotFoundError(f"No se encontró el dataset crudo: {crudo}")

    con = duckdb.connect()
    print(f"📥 Leyendo {crudo} ...")
    if crudo.endswith(".parquet"):
        con.execute(f"CREATE TABLE crudo AS SELECT * FROM read_parquet('{crudo}')")
    else:
        con.execute(f"CREATE TABLE crudo AS SELECT * FROM read_csv_auto('{crudo}', sample_size=-1)")

    columnas = [r[0] for r in con.execute("DESCRIBE crudo").fetchall()]
    for col in ["numero_de_cliente", "foto_mes"]:
        if col not in columnas:
            raise KeyError(f"Falta la columna obligatoria {col}")

    # En el CSV hay ids escritos en notación científica (ej. "4.8e+07"), y eso hace que DuckDB
    # lea toda la columna como decimal. Se fuerzan a enteros para que la entrega no salga con ".0".
    no_enteros = con.execute(
        "SELECT COUNT(*) FROM crudo WHERE numero_de_cliente <> ROUND(numero_de_cliente)").fetchone()[0]
    if no_enteros:
        raise ValueError(f"{no_enteros} numero_de_cliente no son enteros: revisar el archivo")
    con.execute("""CREATE OR REPLACE TABLE crudo AS SELECT * REPLACE (
        CAST(numero_de_cliente AS BIGINT) AS numero_de_cliente,
        CAST(foto_mes AS INTEGER) AS foto_mes) FROM crudo""")

    if "clase_ternaria" in columnas:
        print("ℹ️  El archivo ya trae clase_ternaria: se usa tal cual.")
        con.execute("CREATE TABLE con_clase AS SELECT * FROM crudo")
    else:
        print("🛠️  Calculando clase_ternaria (lógica de z101 / z494)...")
        con.execute(SQL_CLASE)

    n, ncol = con.execute("SELECT COUNT(*) FROM con_clase").fetchone()[0], len(columnas)
    print(f"✅ {n:,} filas y {ncol} columnas.\n")

    print("📊 Clase por foto_mes (NULL = sin clase completa):")
    print(con.execute("""
        PIVOT (SELECT foto_mes, COALESCE(clase_ternaria, 'SIN_CLASE') AS clase FROM con_clase)
        ON clase USING COUNT(*) GROUP BY foto_mes ORDER BY foto_mes
    """).df().to_string(index=False))

    m = cfg["meses"]
    # Un mes tiene clase completa solo si aparecen CONTINUA (en el anteúltimo solo hay BAJA+1)
    sin_clase = {r[0] for r in con.execute(
        "SELECT foto_mes FROM con_clase GROUP BY foto_mes "
        "HAVING SUM(CASE WHEN clase_ternaria = 'CONTINUA' THEN 1 ELSE 0 END) = 0"
    ).fetchall()}
    for etiqueta, meses in [("train", m["train"]), ("test", [m["test"]]), ("train_final", m["train_final"])]:
        malos = [x for x in meses if x in sin_clase]
        if malos:
            print(f"⚠️  ATENCIÓN: {etiqueta} incluye meses sin clase completa: {malos}. Revisá config/experimento.yaml")

    salida.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY con_clase TO '{salida}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    print(f"\n💾 Guardado en {salida}")


if __name__ == "__main__":
    main()
