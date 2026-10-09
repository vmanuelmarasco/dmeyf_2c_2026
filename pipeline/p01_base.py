"""Etapa 01 - Carga del crudo y construccion del target ternario.

QUE HACE
--------
1. Carga el CSV crudo en DuckDB.
2. Verifica la granularidad: una fila por cliente por mes, sin repetidos.
3. Construye clase_ternaria desde la PRESENCIA del cliente en meses futuros.
4. Corre los gates: conteos por mes y casos de BAJA+2 contra lo ya validado.

POR QUE EL TARGET SE CONSTRUYE Y NO VIENE EN EL DATASET
-------------------------------------------------------
El crudo no trae la clase. La baja de un cliente no es un campo: es el hecho
de que el cliente DEJA DE APARECER. Entonces la etiqueta de un mes M se
deduce mirando si el cliente esta en M+1 y en M+2:

    esta en M+1 = NO  ->  BAJA+1   (ya se fue; llegamos tarde)
    esta en M+1 = SI, esta en M+2 = NO  ->  BAJA+2   (esto es lo que se predice)
    esta en M+1 = SI, esta en M+2 = SI  ->  CONTINUA

Uso:
    python3 pipeline/p01_base.py
"""

import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p00_config as cfg


def main() -> None:
    t0 = time.time()
    ruta = cfg.resolver_ruta_fuente()
    print(f"[01] fuente: {ruta}", flush=True)

    # read_only=False porque vamos a escribir tablas. El archivo se crea si no
    # existe, y persiste entre etapas.
    con = duckdb.connect(str(cfg.BASE_DUCKDB))

    # ----------------------------------------------------------------------
    # 1. Carga del crudo
    # ----------------------------------------------------------------------
    # sample_size = -1 fuerza a DuckDB a leer TODO el archivo para inferir los
    # tipos, en vez de adivinarlos con las primeras 20.000 filas. Sin esto, una
    # columna que es entera arriba y decimal recien en la fila 500.000 se
    # infiere como entera y DuckDB falla o trunca. Con 154 columnas de datos
    # bancarios, es casi seguro que pasa.
    con.execute(f"""
        CREATE OR REPLACE TABLE crudo AS
        SELECT *
        FROM read_csv_auto('{ruta}', sample_size = -1);
    """)
    filas, columnas = con.execute(
        "SELECT count(*), count(*) FROM crudo LIMIT 1"
    ).fetchone()[0], len(con.execute("DESCRIBE crudo").fetchall())
    print(f"[01] crudo cargado: {filas:,} filas x {columnas} columnas "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ----------------------------------------------------------------------
    # 2. GATE de granularidad
    # ----------------------------------------------------------------------
    # El par (numero_de_cliente, foto_mes) es la clave primaria compuesta. Si
    # se repitiera, todo lo que viene despues se rompe EN SILENCIO: lag() y
    # lead() asumen una fila por cliente por mes, asi que con duplicados
    # lead(presencia, 1) no devuelve "el mes siguiente" sino "la otra fila del
    # mismo mes", y el target sale mal sin tirar ningun error.
    duplicados = con.execute("""
        SELECT count(*) FROM (
            SELECT numero_de_cliente, foto_mes
            FROM crudo
            GROUP BY numero_de_cliente, foto_mes
            HAVING count(*) > 1
        );
    """).fetchone()[0]
    if duplicados:
        raise AssertionError(
            f"GATE granularidad: {duplicados} pares (cliente, mes) repetidos."
        )
    print("[01] GATE granularidad OK: la clave no se repite", flush=True)

    # ----------------------------------------------------------------------
    # 3. GATE de conteos por mes
    # ----------------------------------------------------------------------
    conteos = con.execute("""
        SELECT foto_mes, count(*) AS filas
        FROM crudo GROUP BY foto_mes ORDER BY foto_mes;
    """).df()
    observado = dict(zip(conteos.foto_mes, conteos.filas))
    if observado != cfg.FILAS_POR_MES_ESPERADAS:
        raise AssertionError(
            "GATE conteos: el dataset no es el esperado.\n"
            f"  esperado: {cfg.FILAS_POR_MES_ESPERADAS}\n"
            f"  observado: {observado}"
        )
    print("[01] GATE conteos por mes OK", flush=True)

    # ----------------------------------------------------------------------
    # 4. Diagnostico: clientes con huecos en el medio
    # ----------------------------------------------------------------------
    # Esto justifica el CROSS JOIN del paso 5. Si un cliente esta en marzo,
    # falta en abril y vuelve en mayo, un lead() sobre las filas EXISTENTES
    # leeria mayo como "el mes siguiente de marzo" y lo llamaria CONTINUA,
    # cuando en realidad desaparecio en abril y eso lo hace BAJA+1 en marzo.
    huecos = con.execute("""
        WITH presencia AS (
            SELECT numero_de_cliente,
                   count(*)                                        AS meses_presente,
                   (max(foto_mes) // 100) * 12 + (max(foto_mes) % 100)
                 - ((min(foto_mes) // 100) * 12 + (min(foto_mes) % 100)) + 1
                                                                   AS meses_abarcados
            FROM crudo GROUP BY numero_de_cliente
        )
        SELECT count(*) FROM presencia WHERE meses_presente < meses_abarcados;
    """).fetchone()[0]
    print(f"[01] clientes con hueco en el medio (desaparecen y vuelven): "
          f"{huecos:,}", flush=True)

    # ----------------------------------------------------------------------
    # 5. Construccion del target
    # ----------------------------------------------------------------------
    # La idea: materializar explicitamente la AUSENCIA. Se arma el producto
    # cartesiano cliente x mes para tener una fila por cada combinacion
    # posible, se marca con mes_0 = 1 donde el cliente realmente aparecio y 0
    # donde no, y recien entonces se aplica lead(). Asi lead() camina por el
    # calendario completo y no por las filas que existen, y los huecos del
    # paso 4 se leen bien.
    con.execute("""
        CREATE OR REPLACE TABLE base AS

        WITH periodos AS (
            SELECT DISTINCT foto_mes FROM crudo
        ),
        clientes AS (
            SELECT DISTINCT numero_de_cliente FROM crudo
        ),
        -- Una fila por cada (cliente, mes) concebible, exista o no.
        calendario AS (
            SELECT c.numero_de_cliente, p.foto_mes
            FROM clientes AS c CROSS JOIN periodos AS p
        ),
        -- LEFT JOIN: donde el cliente no aparecio, todas las columnas del
        -- crudo quedan NULL y mes_0 queda en 0. Esa es la ausencia explicita.
        presencia AS (
            SELECT
                k.numero_de_cliente,
                k.foto_mes,
                d.* EXCLUDE (numero_de_cliente, foto_mes),
                CASE WHEN d.numero_de_cliente IS NULL THEN 0 ELSE 1 END AS mes_0
            FROM calendario AS k
            LEFT JOIN crudo AS d
                ON  d.numero_de_cliente = k.numero_de_cliente
                AND d.foto_mes          = k.foto_mes
        ),
        -- Ahora si: el futuro del cliente, mes a mes.
        -- WINDOW w se define una vez y se reusa; es la sintaxis que ensena
        -- z402 para no repetir el OVER (PARTITION BY ...) en cada linea.
        etiquetado AS (
            SELECT
                *,
                LEAD(mes_0, 1) OVER w AS presente_en_m1,
                LEAD(mes_0, 2) OVER w AS presente_en_m2
            FROM presencia
            WINDOW w AS (PARTITION BY numero_de_cliente ORDER BY foto_mes)
        )

        SELECT
            * EXCLUDE (mes_0, presente_en_m1, presente_en_m2),
            CASE
                -- No esta el mes que viene: ya se fue. Llegamos tarde.
                WHEN presente_en_m1 = 0                      THEN 'BAJA+1'
                -- Esta el mes que viene pero no el siguiente: ESTE es el target.
                WHEN presente_en_m1 = 1 AND presente_en_m2 = 0 THEN 'BAJA+2'
                -- Esta en los dos: sigue siendo cliente.
                WHEN presente_en_m1 = 1 AND presente_en_m2 = 1 THEN 'CONTINUA'
                -- Fuera del borde de la ventana: no se puede saber. Queda NULL
                -- a proposito, nunca imputado, para que no entre a entrenar.
                ELSE NULL
            END AS clase_ternaria
        FROM etiquetado
        -- Solo las filas que existen de verdad. Las de ausencia cumplieron su
        -- funcion (hacer que lead() camine el calendario) y se descartan.
        WHERE mes_0 = 1
        ORDER BY foto_mes, numero_de_cliente;
    """)

    # ----------------------------------------------------------------------
    # 6. GATE del target
    # ----------------------------------------------------------------------
    resumen = con.execute("""
        SELECT foto_mes,
               count(*)                                           AS filas,
               count(*) FILTER (clase_ternaria IS NULL)            AS sin_clase,
               count(*) FILTER (clase_ternaria = 'BAJA+1')         AS baja1,
               count(*) FILTER (clase_ternaria = 'BAJA+2')         AS baja2,
               count(*) FILTER (clase_ternaria = 'CONTINUA')       AS continua,
               round(100.0 * count(*) FILTER (clase_ternaria = 'BAJA+2')
                     / count(*), 3)                               AS tasa_baja2_pct
        FROM base GROUP BY foto_mes ORDER BY foto_mes;
    """).df()
    print("\n[01] Resumen del target por mes:")
    print(resumen.to_string(index=False), flush=True)

    # Gate A: BAJA+2 reproduce los valores ya validados en el EDA.
    obs_b2 = {int(r.foto_mes): int(r.baja2) for r in resumen.itertuples()
              if int(r.foto_mes) in cfg.BAJA2_POR_MES_ESPERADOS}
    if obs_b2 != cfg.BAJA2_POR_MES_ESPERADOS:
        raise AssertionError(
            f"GATE target: BAJA+2 no reproduce el EDA.\n"
            f"  esperado: {cfg.BAJA2_POR_MES_ESPERADOS}\n  observado: {obs_b2}"
        )
    print("[01] GATE BAJA+2 por mes OK (reproduce el EDA)", flush=True)

    # Gate B: los meses maduros no pueden tener NULL, y 202108 tiene que ser
    # todo NULL. Si 202108 tuviera alguna etiqueta, habria fuga de futuro.
    for fila in resumen.itertuples():
        mes, sin_clase, filas_mes = int(fila.foto_mes), int(fila.sin_clase), int(fila.filas)
        if mes in cfg.MESES_TARGET_MADURO and sin_clase != 0:
            raise AssertionError(
                f"GATE madurez: {mes} es mes maduro y tiene {sin_clase} sin clase."
            )
        if mes == cfg.MES_SCORING and sin_clase != filas_mes:
            raise AssertionError(
                f"GATE scoring: {mes} deberia estar 100% sin clase, "
                f"tiene {filas_mes - sin_clase} etiquetadas -> FUGA DE FUTURO."
            )
    print("[01] GATE madurez OK: maduros completos, 202108 sin etiqueta", flush=True)

    # 202107: se puede identificar BAJA+1 (desaparece en 202108) pero no se
    # puede separar BAJA+2 de CONTINUA sin ver 202109. Queda inutilizable como
    # conjunto supervisado, aunque SI aporta historia para los lags de 202108.
    b1_julio = int(resumen.loc[resumen.foto_mes == 202107, "baja1"].iloc[0])
    print(f"[01] nota: 202107 tiene {b1_julio:,} BAJA+1 identificables pero "
          f"0 BAJA+2 separables -> no es conjunto supervisado, solo historia",
          flush=True)

    resumen.to_csv(cfg.SALIDAS / "p01_resumen_target.csv", index=False)
    con.close()
    print(f"\n[01] base lista en {cfg.BASE_DUCKDB.name} "
          f"({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
