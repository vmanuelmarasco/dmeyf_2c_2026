"""verscla09.py — script único que genera la entrega final verscla09 (público 101,09).

Qué hace, en orden:
  1. Lee el crudo y arma la clase ternaria (BAJA+1 / BAJA+2 / CONTINUA), como en z101.
  2. Feature engineering en DuckDB:
       - lag 1 y delta 1 de todas las variables numéricas (historia por cliente, z402);
       - historia puntual: lags 2 y 3, promedios de 3 meses;
       - cambio relativo contra el mes anterior y manejo de la tarjeta ("Amex");
       - señales de desvinculación (cayó la actividad, dejó de cobrar el sueldo, préstamo por terminar, etc.).
  3. Entrena LightGBM con 202103-202106 (los 4 meses con clase completa), target BAJA+1 + BAJA+2,
     min_data_in_leaf escalado por la cantidad de filas (z494), semillerío de 10 semillas.
  4. Ordena los clientes de 202108 por probabilidad y entrega los 13.000 primeros:
     un CSV de una columna, sin encabezado, ids enteros. Además un archivo por semilla.

Uso:
  python verscla09.py                      genera la entrega
  python verscla09.py --validar            además mide en 202106 entrenando con 202103-202104
  python verscla09.py --crudo /ruta/competencia_01_crudo.csv

Si ya existe entregas/verscla09_k13000.csv (la entrega enviada), al final compara contra ella.
"""
import argparse
import os
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

# ============================================================================ configuración
CRUDO = ('/Users/manuelmarasco/Library/CloudStorage/GoogleDrive-hola@manuelmarasco.com/'
         'Mi unidad/DMEyF/2026/notebooks/data/competencia_01_crudo.csv')
MESES_FINAL = [202103, 202104, 202105, 202106]   # todos los meses con clase completa
MES_FUTURO = 202108
MES_REFERENCIA = 202104                           # los hiperparámetros se buscaron con un mes (abril)
K = 13000                                         # cantidad de envíos

GANANCIA, COSTO = 1_072_500, 27_500

# Hiperparámetros de verscla05 (Optuna sobre abril, z601), 655 árboles
PARAMS = {'objective': 'binary', 'boosting_type': 'gbdt', 'first_metric_only': True,
          'boost_from_average': True, 'feature_pre_filter': False, 'max_bin': 31,
          'bagging_freq': 1, 'verbose': -1, 'num_leaves': 76,
          'learning_rate': 0.009414528604172152, 'min_data_in_leaf': 691,
          'feature_fraction': 0.8075276247459833, 'bagging_fraction': 0.5141471765041344}
N_ARBOLES = 655
SEMILLAS = [100129, 100183, 100151, 500041, 900007, 273311, 750209, 219281, 484609, 496259]

NO_FEATURES = {'numero_de_cliente', 'foto_mes', 'clase_ternaria', 'clase_binaria2'}
# Fuera del modelo, con todo lo derivado: fechas con drift de datos y monto de sueldo (aguinaldo)
FUERA = ('fultimo_cierre', 'Finiciomora', 'mpayroll')
AMEX = ['ctrx_quarter', 'mcuentas_saldo', 'mcaja_ahorro', 'mpasivos_margen', 'mactivos_margen',
        'mrentabilidad', 'mcomisiones', 'mcomisiones_mantenimiento', 'mtarjeta_visa_consumo',
        'mtarjeta_master_consumo', 'ctarjeta_visa_transacciones', 'ctarjeta_master_transacciones',
        'chomebanking_transacciones', 'mtransferencias_recibidas', 'mtransferencias_emitidas',
        'mprestamos_personales', 'Visa_msaldototal', 'Master_msaldototal', 'Visa_mpagado', 'Master_mpagado']

T0 = time.time()


def log(*a):
    print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)


# ============================================================================ 1. datos y clase
def cargar_y_clase(con, crudo):
    # numero_de_cliente a entero: el CSV trae algún id en notación científica (48000000)
    con.sql(f"""create table crudo as
                select * replace (cast(numero_de_cliente as bigint) as numero_de_cliente)
                from read_csv_auto('{crudo}')""")
    # ccajas_depositos vale 0 para todos los clientes en 202105: error de datos
    if 'ccajas_depositos' in [r[0] for r in con.sql('describe crudo').fetchall()]:
        con.sql("update crudo set ccajas_depositos = null where foto_mes = 202105")

    # Clase ternaria: se mira si el cliente está en t+1 y t+2
    con.sql("""
    create table base as
    with periodos as (select distinct foto_mes from crudo),
         clientes as (select distinct numero_de_cliente from crudo),
         todo as (select numero_de_cliente, foto_mes from clientes cross join periodos),
         marca as (
            select t.numero_de_cliente, t.foto_mes,
                   if(c.numero_de_cliente is null, 0, 1) as mes_0
            from todo t left join crudo c using (numero_de_cliente, foto_mes)),
         clase as (
            select numero_de_cliente, foto_mes, mes_0,
              case
                when lead(mes_0, 1) over w = 0 then 'BAJA+1'
                when lead(mes_0, 1) over w = 1 and lead(mes_0, 2) over w = 0 then 'BAJA+2'
                when lead(mes_0, 1) over w = 1 and lead(mes_0, 2) over w = 1 then 'CONTINUA'
                else null end as clase_ternaria
            from marca window w as (partition by numero_de_cliente order by foto_mes))
    select c.*, k.clase_ternaria
    from crudo c join clase k using (numero_de_cliente, foto_mes)
    where k.mes_0 = 1
    """)
    log('clase armada:', con.sql("""select foto_mes, clase_ternaria, count(*) from base
                                    group by all order by all""").fetchall()[:3], '...')


# ============================================================================ 2. variables
def armar_variables(con):
    tipos = {r[0]: r[1] for r in con.sql('describe base').fetchall()}
    numericas = [c for c, t in tipos.items()
                 if c not in NO_FEATURES and any(x in t for x in ('INT', 'DOUBLE', 'FLOAT', 'DECIMAL'))]

    def q(c):
        return f'"{c}"'

    exprs = []
    # lag 1 y delta 1 de todas las variables numéricas
    for c in numericas:
        exprs.append(f'lag({q(c)}, 1) over w as {q(c + "_lag1")}')
        exprs.append(f'{q(c)} - lag({q(c)}, 1) over w as {q(c + "_delta1")}')
    # historia puntual que usan las señales de desvinculación
    puntual = {
        'ctrx_quarter_lag2': 'lag(ctrx_quarter, 2) over w',
        'ctrx_quarter_delta2': 'ctrx_quarter - lag(ctrx_quarter, 2) over w',
        'cproductos_delta2': 'cproductos - lag(cproductos, 2) over w',
        'ctrx_quarter_avg3': 'avg(ctrx_quarter) over w3',
        'consumo_tc_avg3': 'avg(coalesce(mtarjeta_visa_consumo, 0) + coalesce(mtarjeta_master_consumo, 0)) over w3',
        'mprestamos_personales_lag3': 'lag(mprestamos_personales, 3) over w',
        'cpayroll_trx_lag2': 'lag(cpayroll_trx, 2) over w',
        'cpayroll_trx_lag3': 'lag(cpayroll_trx, 3) over w',
    }
    exprs += [f'{e} as {n}' for n, e in puntual.items()]
    # cambio relativo contra el mes anterior: -0,5 = cayó a la mitad, sin importar la inflación
    for c in [a for a in AMEX if a in numericas]:
        exprs.append(f'({q(c)} - lag({q(c)}, 1) over w) / nullif(abs(lag({q(c)}, 1) over w), 0) as {q(c + "_var1")}')
    # manejo de la tarjeta
    for t in ['Visa', 'Master']:
        s, p, m, l = f'{t}_msaldototal', f'{t}_mpagado', f'{t}_mpagominimo', f'{t}_mlimitecompra'
        if s in numericas and p in numericas:
            exprs.append(f'{s} - {p} as {t}_deuda_tras_pago')
            exprs.append(f'{p} / nullif({s}, 0) as {t}_pago_sobre_saldo')
        if p in numericas and m in numericas:
            exprs.append(f'{p} / nullif({m}, 0) as {t}_pago_sobre_minimo')
        if s in numericas and l in numericas:
            exprs.append(f'{s} / nullif({l}, 0) as {t}_uso_limite')

    con.sql(f"""create table historia as
        select *, {', '.join(exprs)}
        from base
        window w as (partition by numero_de_cliente order by foto_mes),
               w3 as (partition by numero_de_cliente order by foto_mes rows between 2 preceding and current row)""")

    # señales de desvinculación (respaldadas por el EDA)
    con.sql("""create table fe as
    with s as (
      select *
        , if(ctrx_quarter < 0.5 * ctrx_quarter_avg3, 1, 0) as cayo_actividad
        , if(coalesce(mtarjeta_visa_consumo, 0) + coalesce(mtarjeta_master_consumo, 0) < 0.5 * consumo_tc_avg3, 1, 0) as cayo_consumo_tc
        , if(cproductos < cproductos_lag1, 1, 0) as bajo_productos
        , if(cpayroll_trx_lag1 > 0 and coalesce(cpayroll_trx, 0) = 0, 1, 0) as dejo_de_cobrar_sueldo
        , if(coalesce(cpayroll_trx, 0) = 0 and (cpayroll_trx_lag1 > 0 or cpayroll_trx_lag2 > 0 or cpayroll_trx_lag3 > 0), 1, 0) as dejo_de_cobrar_reciente
        , if(coalesce(cpayroll_trx, 0) > 0, 1, 0) as cobra_sueldo
        , if(cproductos <= 5, 1, 0) as pocos_productos
        , if(mprestamos_personales_lag3 > 0 and coalesce(mprestamos_personales, 0) = 0, 1, 0) as termino_prestamo
        , if(mprestamos_personales_lag3 > 0 and mprestamos_personales > 0
             and mprestamos_personales < 0.25 * mprestamos_personales_lag3, 1, 0) as prestamo_por_terminar
      from historia)
    select *, cayo_actividad + cayo_consumo_tc + bajo_productos + dejo_de_cobrar_reciente as indice_desvinculacion
    from s""")


def meses(con, lista):
    df = con.sql(f"select * from fe where foto_mes in ({', '.join(map(str, lista))}) "
                 f"order by foto_mes, numero_de_cliente").df()
    if 'clase_ternaria' in df.columns:
        df['clase_binaria2'] = df['clase_ternaria'].isin(['BAJA+1', 'BAJA+2']).astype(int)
    return df


def variables_del_modelo(df):
    return [c for c in df.columns if c not in NO_FEATURES and not any(p in c for p in FUERA)]


# ============================================================================ 3. modelo
def semillerio(train, aplicar, feats, filas_ref):
    """Un LightGBM por semilla (mismos hiperparámetros) y promedio de las probabilidades."""
    p = dict(PARAMS)
    # min_data_in_leaf es un mínimo absoluto de filas: se escala con el tamaño del entrenamiento
    p['min_data_in_leaf'] = max(1, int(round(PARAMS['min_data_in_leaf'] * len(train) / filas_ref)))
    log(f'entreno {len(SEMILLAS)} semillas | {len(train):,} filas | {len(feats)} variables | '
        f'min_data_in_leaf {PARAMS["min_data_in_leaf"]} -> {p["min_data_in_leaf"]}')
    preds = []
    for i, s in enumerate(SEMILLAS):
        d = lgb.Dataset(train[feats], label=train['clase_binaria2'], params={'max_bin': p['max_bin'], 'seed': s})
        m = lgb.train({**p, 'seed': s, 'verbose': -1}, d, num_boost_round=N_ARBOLES)
        preds.append(m.predict(aplicar[feats]))
        log(f'  semilla {i + 1}/{len(SEMILLAS)} ({s}) lista')
    return np.mean(preds, axis=0), preds


def ganancia_por_envios(prob, clase, cortes=range(10000, 16001, 500)):
    orden = np.argsort(-prob, kind='stable')
    acum = np.cumsum(np.where(clase[orden] == 'BAJA+2', GANANCIA, -COSTO))
    return pd.Series({k: acum[k - 1] / 1e6 for k in cortes}, name='ganancia_M')


# ============================================================================ 4. entrega
def escribir(prob, ids, archivo):
    elegidos = ids[np.argsort(-prob, kind='stable')][:K].astype('int64')
    assert len(set(elegidos)) == K
    os.makedirs(os.path.dirname(archivo), exist_ok=True)
    pd.Series(elegidos).to_csv(archivo, index=False, header=False)
    return archivo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crudo', default=CRUDO)
    ap.add_argument('--validar', action='store_true', help='mide en 202106 entrenando con 202103-202104')
    ap.add_argument('--salida', default='entregas_verscla09')
    a = ap.parse_args()

    con = duckdb.connect()
    cargar_y_clase(con, a.crudo)
    armar_variables(con)
    filas_ref = con.sql(f'select count(*) from fe where foto_mes = {MES_REFERENCIA}').fetchone()[0]
    log('variables listas')

    if a.validar:
        # Mismo hueco de dos meses que la entrega: se aprende hasta abril y se mide en junio
        train, junio = meses(con, [202103, 202104]), meses(con, [202106])
        feats = variables_del_modelo(train)
        prob, _ = semillerio(train, junio, feats, filas_ref)
        print('\nValidación 202103-202104 -> 202106 (millones):')
        print(ganancia_por_envios(prob, junio['clase_ternaria'].values).round(1).to_string())
        del train, junio

    train, agosto = meses(con, MESES_FINAL), meses(con, [MES_FUTURO])
    feats = variables_del_modelo(train)
    prob, preds = semillerio(train, agosto, feats, filas_ref)

    ids = agosto['numero_de_cliente'].values
    final = escribir(prob, ids, f'{a.salida}/verscla09_k{K}.csv')
    for s, p in zip(SEMILLAS, preds):
        escribir(p, ids, f'{a.salida}/semillas/verscla09_seed{s}_k{K}.csv')
    log(f'listo: {final} y {len(SEMILLAS)} archivos por semilla en {a.salida}/semillas/')
    print(open(final).read()[:60])

    enviada = 'entregas/verscla09_k13000.csv'
    if os.path.exists(enviada):
        a_ = set(pd.read_csv(final, header=None)[0])
        b_ = set(pd.read_csv(enviada, header=None)[0])
        print(f'Coincidencia con la entrega enviada ({enviada}): {len(a_ & b_) / K * 100:.2f} %')


if __name__ == '__main__':
    main()
