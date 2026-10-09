"""Etapa 02 - Gates de fuga de informacion.

LAS SEIS FUGAS POSIBLES EN ESTE PROBLEMA, Y QUE LAS ATAJA
---------------------------------------------------------
F1. Ventanas de resultado solapadas. La etiqueta del mes M codifica la
    presencia en M+1 y M+2. Si el ultimo mes de train esta a 1 mes de la
    validacion, las dos etiquetas comparten un mes de observacion.
    -> lo ataja p00_config.verificar_gap()

F2. Etiqueta en el mes de scoring. Si 202108 tuviera clase, el modelo
    entrenaria sobre el futuro que debe predecir.
    -> lo ataja el GATE scoring de p01_base.py

F3. Target propagado entre meses del mismo cliente. Copiar la baja de junio
    hacia abril le entrega al modelo informacion del futuro.
    -> lo ataja gate_target_no_propagado()

F4. Columnas prohibidas como predictores. numero_de_cliente no es una
    magnitud, es un identificador: si entra, el modelo memoriza clientes.
    foto_mes es el calendario mismo.
    -> lo ataja gate_columnas_prohibidas()

F5. Feature calculada con datos del futuro. Es la mas sutil y la mas grave.
    Un percent_rank() OVER (PARTITION BY foto_mes) es legitimo: rankea dentro
    del mes. Un percent_rank() OVER () sin particion rankea contra TODA la
    tabla, incluido 202108, y mete el futuro en cada fila. Lo mismo cualquier
    normalizacion global (z-score, min-max) o un avg() sobre la historia
    completa del cliente sin marco de ventana.
    -> lo ataja gate_sin_fuga_de_futuro(), que es un test EXACTO, no estadistico

F6. Fuga de seleccion. Elegir features, hiperparametros, cantidad de
    iteraciones o punto de corte mirando el mismo conjunto donde despues se
    reporta. No ensucia los datos: ensucia la ESTIMACION. El Experimento 01 la
    midio en +35,88 M en junio para el corte.
    -> lo atajan los folds cruzados y la regla de decision fijada de antemano

EXIGENCIA DE ARQUITECTURA QUE SE DESPRENDE DE F5
------------------------------------------------
Para poder testear F5 de forma exacta, la fabrica de features (etapa 03) NO
puede escribir su SQL contra un nombre de tabla fijo. Tiene que recibir la
tabla de origen como parametro. Asi el gate puede reconstruir exactamente las
mismas features sobre una tabla truncada en el mes M y comparar: si alguna
columna cambia de valor en las filas <= M, esa columna estaba leyendo meses
posteriores a M. Es una prueba, no un indicio.
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p00_config as cfg

# Nunca pueden ser predictores.
COLUMNAS_PROHIBIDAS = frozenset({
    "numero_de_cliente",   # identificador, no magnitud
    "foto_mes",            # el calendario mismo
    "clase_ternaria",      # el target
    "target_baja2",        # el target binarizado
})


def gate_columnas_prohibidas(features) -> None:
    """F4: ninguna columna prohibida entre los predictores."""
    encontradas = sorted(COLUMNAS_PROHIBIDAS.intersection(features))
    if encontradas:
        raise AssertionError(
            "GATE F4: columnas prohibidas entre los predictores: "
            + ", ".join(encontradas)
        )
    print(f"[02] GATE F4 OK: {len(features)} features, ninguna prohibida")


def gate_target_no_propagado(con) -> None:
    """F3: la clase pertenece al par (cliente, mes), no al cliente.

    Si la etiqueta estuviera propagada, cada cliente tendria exactamente una
    clase distinta. Exigimos que una parte de los clientes tenga mas de una:
    son los que transitaron CONTINUA -> BAJA+2 -> BAJA+1.
    """
    con_varias = con.execute("""
        SELECT count(*) FROM (
            SELECT numero_de_cliente
            FROM base WHERE clase_ternaria IS NOT NULL
            GROUP BY numero_de_cliente
            HAVING count(DISTINCT clase_ternaria) > 1);
    """).fetchone()[0]
    if con_varias == 0:
        raise AssertionError(
            "GATE F3: ningun cliente cambia de clase entre meses. "
            "Eso solo pasa si la etiqueta fue propagada."
        )
    print(f"[02] GATE F3 OK: {con_varias:,} clientes cambian de clase "
          f"entre meses (la etiqueta es del par cliente-mes)")


def gate_consistencia_baja2(con) -> None:
    """Control de construccion: todo BAJA+2 en M tiene que ser BAJA+1 en M+1.

    Se deduce de la definicion: BAJA+2 en M significa presente en M+1 y
    ausente en M+2; visto desde M+1, eso es exactamente 'ausente el mes que
    viene', o sea BAJA+1. Si falla, el lead() esta desalineado.
    """
    filas = con.execute("""
        WITH pares AS (
            SELECT b.clase_ternaria AS clase_siguiente
            FROM base a
            JOIN base b ON b.numero_de_cliente = a.numero_de_cliente
                       AND (b.foto_mes//100)*12+(b.foto_mes%100)
                         = (a.foto_mes//100)*12+(a.foto_mes%100) + 1
            WHERE a.clase_ternaria = 'BAJA+2')
        SELECT clase_siguiente, count(*) AS casos FROM pares
        GROUP BY clase_siguiente;
    """).df()
    malas = filas[filas.clase_siguiente != "BAJA+1"]
    if len(malas):
        raise AssertionError(
            "GATE consistencia: hay BAJA+2 cuyo mes siguiente no es BAJA+1:\n"
            + malas.to_string(index=False)
        )
    print(f"[02] GATE consistencia OK: los {int(filas.casos.sum()):,} BAJA+2 "
          f"con mes siguiente son BAJA+1 el 100%")


def gate_sin_fuga_de_futuro(
    con,
    constructor_sql,
    features,
    mes_corte: int,
    tolerancia: float = 0.0,
) -> pd.DataFrame:
    """F5: test EXACTO de que ninguna feature lee meses posteriores a mes_corte.

    Procedimiento
    -------------
    1. Construye las features sobre la tabla completa.
    2. Construye las MISMAS features sobre una tabla truncada en mes_corte.
    3. Compara, fila por fila, las filas con foto_mes <= mes_corte.

    Si una feature solo usa informacion de meses <= M, truncar en M no puede
    cambiarla. Cualquier diferencia es prueba de que leyo el futuro.

    Parameters
    ----------
    constructor_sql : callable
        Recibe el nombre de la tabla de origen y devuelve el SQL que
        construye las features. Esta es la razon por la que la fabrica tiene
        que estar parametrizada por tabla.
    """
    con.execute(f"""
        CREATE OR REPLACE TABLE _truncada AS
        SELECT * FROM base WHERE foto_mes <= {mes_corte};
    """)
    con.execute(f"CREATE OR REPLACE TABLE _fe_completa AS {constructor_sql('base')}")
    con.execute(f"CREATE OR REPLACE TABLE _fe_truncada AS {constructor_sql('_truncada')}")

    filas = []
    for feature in features:
        # COALESCE con un centinela para que NULL == NULL cuente como igual:
        # sin esto, dos nulos se comparan como distintos y todo da diferencia.
        distintas = con.execute(f"""
            SELECT count(*) FROM _fe_completa AS c
            JOIN _fe_truncada AS t
              ON t.numero_de_cliente = c.numero_de_cliente
             AND t.foto_mes = c.foto_mes
            WHERE c.foto_mes <= {mes_corte}
              AND coalesce(c."{feature}", -999999999) 
               != coalesce(t."{feature}", -999999999);
        """).fetchone()[0]
        filas.append({"feature": feature, "filas_distintas": int(distintas)})

    reporte = pd.DataFrame(filas)
    culpables = reporte[reporte.filas_distintas > tolerancia]

    con.execute("DROP TABLE IF EXISTS _truncada")
    con.execute("DROP TABLE IF EXISTS _fe_completa")
    con.execute("DROP TABLE IF EXISTS _fe_truncada")

    if len(culpables):
        raise AssertionError(
            f"GATE F5: {len(culpables)} features cambian al truncar en "
            f"{mes_corte}, o sea leen el futuro:\n"
            + culpables.to_string(index=False)
        )
    print(f"[02] GATE F5 OK: las {len(features)} features son identicas al "
          f"truncar en {mes_corte} (no leen el futuro)")
    return reporte


def diagnostico_proxy_de_calendario(base: pd.DataFrame, features, semilla=None):
    """F5 complementario, ESTADISTICO y no concluyente.

    Entrena un modelo para predecir foto_mes desde las features. Si acierta
    mucho, las features codifican el calendario.

    POR QUE NO ES UN GATE: parte de esa senal es legitima e inevitable. La
    inflacion mueve las variables de monto, cliente_antiguedad y cliente_edad
    avanzan un mes por mes, y hay tendencias reales de cartera (el uso de la
    app movil crece todo el periodo). Un AUC alto puede ser drift honesto.
    Sirve para RANKEAR sospechosas, no para condenar: la feature que aparece
    arriba hay que mirarla con el test exacto de gate_sin_fuga_de_futuro.
    """
    import lightgbm as lgb

    semilla = semilla or cfg.SEMILLAS_ENTREGA[0]
    meses = sorted(base["foto_mes"].unique())
    codigo = {m: i for i, m in enumerate(meses)}
    y = base["foto_mes"].map(codigo).to_numpy()
    X = base[list(features)].to_numpy(dtype=np.float32)

    modelo = lgb.train(
        params={
            "objective": "multiclass", "num_class": len(meses),
            "learning_rate": 0.1, "num_leaves": 31, "max_bin": 31,
            "verbosity": -1, "seed": semilla, "feature_pre_filter": False,
        },
        train_set=lgb.Dataset(X, label=y),
        num_boost_round=60,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    acierto = float((modelo.predict(X).argmax(axis=1) == y).mean())
    importancia = pd.DataFrame({
        "feature": list(features),
        "importancia_calendario": modelo.feature_importance("gain"),
    }).sort_values("importancia_calendario", ascending=False)

    print(f"[02] diagnostico calendario: acierto en el mes = {acierto:.3f} "
          f"(azar = {1/len(meses):.3f})")
    print("     top 10 features que mas delatan el mes:")
    print(importancia.head(10).to_string(index=False))
    return acierto, importancia


def imprimir_estrategia_de_validacion() -> None:
    """Deja escrita la estrategia, porque es una decision de modelado y hay
    que poder defenderla en el examen."""
    print("""
[02] ESTRATEGIA DE VALIDACION (y por que NO es k-fold)

  k-fold aleatorio es INVALIDO en este problema, por dos razones distintas:

    a) Mezcla meses. Un fold aleatorio pone 202106 en train y 202104 en test,
       o sea entrena con el futuro para predecir el pasado. La estimacion que
       sale de ahi no se parece a nada de lo que va a pasar en 202108.

    b) Mezcla clientes. El mismo cliente aparece en train y en test con
       vectores de features casi identicos de un mes al otro. El modelo lo
       reconoce en vez de generalizar.

  Lo que la catedra ensena en z301 es Montecarlo / ShuffleSplit, pero notar
  COMO lo aplica: remuestrea DENTRO de un mes (202103) para elegir
  hiperparametros, y se guarda 202105 entero como 'el futuro'. Incluso ahi la
  validacion que decide es temporal.

  Lo que usamos:

    1. VALIDACION TEMPORAL con gap de maduracion. Dos folds:
         [202103]          -> 202105
         [202103, 202104]  -> 202106
       Son los dos unicos que esta ventana admite con gap 2. Un brazo gana
       solo si gana en los DOS. El Experimento 01 mostro por que: el corte
       optimo y la cantidad de iteraciones apuntan en direcciones OPUESTAS en
       los dos meses, las dos con p < 0,01.

    2. REPETICION POR SEMILLA, apareada. El ruido que domina la comparacion
       entre dos juegos de features NO es el de la muestra de evaluacion
       (es el mismo conjunto de clientes en los dos brazos): es el de la
       semilla del arbol. Desvio entre semillas en junio: 13,18 M. Por eso se
       corren 20 semillas, se compara apareado y se testea con Wilcoxon.
       Con 5 semillas solo se podian resolver efectos de ~20 M.

    3. MONTECARLO, para otra cosa. Remuestrear el conjunto de evaluacion no
       reduce el ruido de la comparacion, porque es el mismo. Sirve para dos
       usos: cuantificar la incertidumbre del leaderboard (IC95 de ancho
       ~96 M) y, si se aparea el remuestreo entre brazos, reducir 4,6x el
       ruido de una comparacion sobre predicciones ya calculadas.

  En la Ultima Competencia, con 18 meses, habra muchos folds temporales y ahi
  si corresponde walk-forward completo. Con 6 fotos, no existe.
""")


if __name__ == "__main__":
    con = duckdb.connect(str(cfg.BASE_DUCKDB), read_only=True)
    print("[02] Gates que se pueden correr sobre la base de la etapa 01:\n")
    gate_target_no_propagado(con)
    gate_consistencia_baja2(con)
    gate_columnas_prohibidas(["ctrx_quarter", "mrentabilidad", "cpayroll_trx"])
    for fold in cfg.FOLDS:
        cfg.verificar_gap(fold)
    print(f"[02] GATE F1 OK: los {len(cfg.FOLDS)} folds tienen gap >= "
          f"{cfg.MESES_MADURACION}")
    con.close()
    imprimir_estrategia_de_validacion()
    print("[02] GATE F5 queda pendiente: necesita la fabrica de la etapa 03,\n"
          "     que debe recibir la tabla de origen como parametro.")
