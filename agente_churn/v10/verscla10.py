"""verscla10.py — script único: verscla09 + variables de dominio bancario + feature_fraction 0,35 + 20 semillas.

Qué hace, en orden:
  1. Lee el crudo y arma la clase ternaria (BAJA+1 / BAJA+2 / CONTINUA), como en z101.
  2. Feature engineering en DuckDB:
       - lag 1 y delta 1 de todas las variables numéricas (historia por cliente, z402);
       - historia puntual: lags 2 y 3, promedios de 3 meses;
       - cambio relativo contra el mes anterior y manejo de la tarjeta;
       - señales de desvinculación (cayó la actividad, dejó de cobrar el sueldo, préstamo por terminar, etc.);
       - NUEVO: variables de dominio bancario (vinculación, compromisos domiciliados, flujo de fondos,
         utilización de tarjeta, precio efectivo, desaceleración de la rentabilidad, canales) y su
         dinámica contra la media de los 3 meses anteriores, más eventos (apagó débitos, se quedó
         sin movimientos, bajó tarjetas, meses inactivo, etc.).
  3. Entrena LightGBM con 202103-202106 (los 4 meses con clase completa), target BAJA+1 + BAJA+2,
     min_data_in_leaf escalado por filas (z494). NUEVO: feature_fraction 0,35 (con ~560 variables,
     cada árbol ve un subconjunto distinto). Semillerío de 20 semillas.
  4. Entrega los 13.000 clientes de 202108 con mayor probabilidad: CSV de una columna, sin encabezado.

Validación (entrenar 202103-202104, medir 202106, 5 semillas pareadas, meseta 12.000-15.000):
  verscla09 398,3 M  ->  + dominio 405,6  ->  + feature_fraction 0,35 409,5  ->  ambos 413,8 (+15,5, 5/5 semillas)

Uso:
  python verscla10.py                      genera la entrega
  python verscla10.py --validar            además mide en 202106 (5 semillas, entrenando con 202103-202104)
  python verscla10.py --crudo /ruta/competencia_01_crudo.csv
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

# Hiperparámetros de verscla05 (Optuna sobre abril, z601), 655 árboles; feature_fraction bajado a 0,35
PARAMS = {'objective': 'binary', 'boosting_type': 'gbdt', 'first_metric_only': True,
          'boost_from_average': True, 'feature_pre_filter': False, 'max_bin': 31,
          'bagging_freq': 1, 'verbose': -1, 'num_leaves': 76,
          'learning_rate': 0.009414528604172152, 'min_data_in_leaf': 691,
          'feature_fraction': 0.35, 'bagging_fraction': 0.5141471765041344}
N_ARBOLES = 655
SEMILLAS = [100129, 100183, 100151, 500041, 900007, 273311, 750209, 219281, 484609, 496259,
            847201, 345673, 546173, 385811, 983911, 518597, 885943, 129893, 606653, 123499]

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


# ---------------------------------------------------------------- variables de dominio bancario

def z(c):
    return f'ifnull({c}, 0)'

def sumz(cols):
    return ' + '.join(z(c) for c in cols)

TRX = ['ctarjeta_debito_transacciones', 'ctarjeta_visa_transacciones', 'ctarjeta_master_transacciones',
       'chomebanking_transacciones', 'cmobile_app_trx', 'ccajas_transacciones', 'catm_trx', 'catm_trx_other',
       'cextraccion_autoservicio', 'ccallcenter_transacciones']
COMPROMISOS = ['ccuenta_debitos_automaticos', 'ctarjeta_visa_debitos_automaticos',
               'ctarjeta_master_debitos_automaticos', 'cpagomiscuentas', 'cpagodeservicios']
M_COMPROMISOS = ['mcuenta_debitos_automaticos', 'mttarjeta_visa_debitos_automaticos',
                 'mttarjeta_master_debitos_automaticos', 'mpagomiscuentas', 'mpagodeservicios']
PRODUCTOS = ['ccuenta_corriente', 'ccaja_ahorro', 'ctarjeta_debito', 'ctarjeta_visa', 'ctarjeta_master',
             'cprestamos_personales', 'cprestamos_prendarios', 'cprestamos_hipotecarios', 'cplazo_fijo',
             'cinversion1', 'cinversion2', 'cseguro_vida', 'cseguro_auto', 'cseguro_vivienda',
             'cseguro_accidentes_personales', 'ccaja_seguridad', 'cpayroll_trx', 'ccuenta_debitos_automaticos']

NIVEL = f"""
  , {sumz(TRX)} as d_trx_total
  , {sumz(COMPROMISOS)} as d_compromisos
  , {sumz(M_COMPROMISOS)} as d_m_compromisos
  , ({z('chomebanking_transacciones')} + {z('cmobile_app_trx')}) / nullif({sumz(TRX)}, 0) as d_share_digital
  , {z('ccajas_transacciones')} / nullif({sumz(TRX)}, 0) as d_share_sucursal
  , (chomebanking_transacciones > 0)::int + (cmobile_app_trx > 0)::int + (catm_trx > 0)::int
    + (ccajas_transacciones > 0)::int + (cextraccion_autoservicio > 0)::int + (ccallcenter_transacciones > 0)::int as d_canales
  , {' + '.join(f'({c} > 0)::int' for c in PRODUCTOS)} as d_productos_activos
  , (ctarjeta_visa > 0)::int + (ctarjeta_master > 0)::int as d_cant_tarjetas
  , ({z('Visa_msaldototal')} + {z('Master_msaldototal')}) / nullif({z('Visa_mlimitecompra')} + {z('Master_mlimitecompra')}, 0) as d_utilizacion_tc
  , ({z('Visa_mconsumototal')} + {z('Master_mconsumototal')}) / nullif({z('Visa_mlimitecompra')} + {z('Master_mlimitecompra')}, 0) as d_intensidad_tc
  , ({z('Visa_madelantopesos')} + {z('Master_madelantopesos')}) / nullif({z('Visa_mconsumototal')} + {z('Master_mconsumototal')}, 0) as d_adelantos_tc
  , {z('mtransferencias_recibidas')} + {z('mcheques_depositados')} as d_ingresos_sin_sueldo
  , {sumz(['mtransferencias_emitidas', 'mextraccion_autoservicio', 'matm', 'matm_other', 'mcheques_emitidos', 'mpagomiscuentas', 'mpagodeservicios'])} as d_egresos
  , {z('mtransferencias_emitidas')} / nullif({sumz(['mpayroll', 'mtransferencias_recibidas', 'mcheques_depositados'])}, 0) as d_fuga_ratio
  , {sumz(['mpayroll', 'mtransferencias_recibidas', 'mcheques_depositados'])}
    - {sumz(['mtransferencias_emitidas', 'mextraccion_autoservicio', 'matm', 'matm_other', 'mcheques_emitidos', 'mpagomiscuentas', 'mpagodeservicios'])} as d_flujo_neto
  , matm_other / nullif({z('matm')} + {z('matm_other')}, 0) as d_uso_atm_ajeno
  , {sumz(['mcuentas_saldo', 'mplazo_fijo_pesos', 'mplazo_fijo_dolares', 'minversion1_pesos', 'minversion2'])} as d_aum
  , mcomisiones_mantenimiento / nullif({sumz(['mcuentas_saldo', 'mplazo_fijo_pesos', 'mplazo_fijo_dolares', 'minversion1_pesos', 'minversion2'])}, 0) as d_precio_efectivo
  , mrentabilidad - mrentabilidad_annual / 12.0 as d_desaceleracion
  , mrentabilidad / nullif(mrentabilidad_annual / 12.0, 0) as d_rentab_vs_anual
  , mactivos_margen / nullif({z('mactivos_margen')} + {z('mpasivos_margen')}, 0) as d_perfil_deudor
  , mcomisiones / nullif(mrentabilidad, 0) as d_peso_comisiones
  , greatest({z('Visa_delinquency')}, {z('Master_delinquency')}) as d_mora
"""

# columnas de nivel sobre las que se calcula la dinámica
DINAMICA = ['d_trx_total', 'd_compromisos', 'd_m_compromisos', 'd_share_digital', 'd_utilizacion_tc',
            'd_productos_activos', 'd_canales', 'd_aum', 'ctrx_quarter', 'mcuentas_saldo', 'mrentabilidad',
            'chomebanking_transacciones', 'ctarjeta_debito_transacciones', 'mtarjeta_visa_consumo',
            'mcaja_ahorro', 'mtransferencias_emitidas', 'd_egresos']


def sql_dominio():
    din = []
    for c in DINAMICA:
        prev = f'avg({c}) over (partition by numero_de_cliente order by foto_mes rows between 3 preceding and 1 preceding)'
        din.append(f', {c} / nullif({prev}, 0) as {c}_vs_media3')
        din.append(f', {c} - {prev} as {c}_menos_media3')
    l = lambda c, n: f'lag({c}, {n}) over w'
    eventos = f"""
      , if({l('d_compromisos', 1)} > 0 and d_compromisos = 0, 1, 0) as e_apago_debitos
      , if({l('d_compromisos', 1)} > d_compromisos, 1, 0) as e_bajo_debitos
      , if({l('cplazo_fijo', 1)} > 0 and {z('cplazo_fijo')} = 0, 1, 0) as e_apago_plazo_fijo
      , if({l('cinversion1', 1)} + {l('cinversion2', 1)} > 0 and {z('cinversion1')} + {z('cinversion2')} = 0, 1, 0) as e_apago_inversion
      , if({l('chomebanking_transacciones', 1)} > 0 and {z('chomebanking_transacciones')} = 0, 1, 0) as e_apago_homebanking
      , if({l('d_cant_tarjetas', 1)} > d_cant_tarjetas, 1, 0) as e_bajo_tarjetas
      , if({l('d_trx_total', 1)} > 0 and d_trx_total = 0, 1, 0) as e_sin_movimientos
      , (d_trx_total = 0)::int + ifnull(({l('d_trx_total', 1)} = 0)::int, 0) + ifnull(({l('d_trx_total', 2)} = 0)::int, 0) as e_meses_inactivo3
      , ifnull((ctrx_quarter < {l('ctrx_quarter', 1)})::int, 0) + ifnull(({l('ctrx_quarter', 1)} < {l('ctrx_quarter', 2)})::int, 0)
        + ifnull(({l('ctrx_quarter', 2)} < {l('ctrx_quarter', 3)})::int, 0) as e_racha_caidas_ctrx
      , if(catm_trx < {l('catm_trx', 1)} and catm_trx_other > {l('catm_trx_other', 1)}, 1, 0) as e_migra_atm_ajeno
      , if(d_mora > {l('d_mora', 1)}, 1, 0) as e_mora_empeora
      , if({z('ccallcenter_transacciones')} > 0 and ifnull({l('ccallcenter_transacciones', 1)}, 0) = 0
           and ifnull({l('ccallcenter_transacciones', 2)}, 0) = 0, 1, 0) as e_llamo_nuevo
      , d_utilizacion_tc - {l('d_utilizacion_tc', 1)} as d_utilizacion_tc_delta1
      , d_share_digital - {l('d_share_digital', 1)} as d_share_digital_delta1
      , d_compromisos - {l('d_compromisos', 1)} as d_compromisos_delta1
      , d_productos_activos - {l('d_productos_activos', 1)} as d_productos_activos_delta1
    """
    return f"""
    create or replace table dom as
    with n as (select numero_de_cliente, foto_mes, ctrx_quarter, mcuentas_saldo, mrentabilidad,
                      chomebanking_transacciones, ctarjeta_debito_transacciones, mtarjeta_visa_consumo,
                      mcaja_ahorro, mtransferencias_emitidas, cplazo_fijo, cinversion1, cinversion2,
                      catm_trx, catm_trx_other, ccallcenter_transacciones {NIVEL}
               from base)
    select numero_de_cliente, foto_mes,
           d_trx_total, d_compromisos, d_m_compromisos, d_share_digital, d_share_sucursal, d_canales,
           d_productos_activos, d_cant_tarjetas, d_utilizacion_tc, d_intensidad_tc, d_adelantos_tc,
           d_ingresos_sin_sueldo, d_egresos, d_fuga_ratio, d_flujo_neto, d_uso_atm_ajeno, d_aum,
           d_precio_efectivo, d_desaceleracion, d_rentab_vs_anual, d_perfil_deudor, d_peso_comisiones, d_mora
           {''.join(din)} {eventos}
    from n
    window w as (partition by numero_de_cliente order by foto_mes)
    """


def meses(con, lista):
    cols_fe = [r[0] for r in con.sql('describe fe').fetchall()]
    cols_dom = [r[0] for r in con.sql('describe dom').fetchall()][2:]
    claves = ('numero_de_cliente', 'foto_mes', 'clase_ternaria')
    sel = [f'f."{c}"' if c in claves else f'cast(f."{c}" as float) as "{c}"' for c in cols_fe]
    sel += [f'cast(d."{c}" as float) as "{c}"' for c in cols_dom]
    partes = []
    for mes in sorted(lista):   # de a un mes: baja el pico de memoria
        partes.append(con.sql(f"""select {', '.join(sel)} from fe f join dom d using (numero_de_cliente, foto_mes)
                                  where f.foto_mes = {mes} order by f.numero_de_cliente""").df())
    df = pd.concat(partes, ignore_index=True) if len(partes) > 1 else partes[0]
    del partes
    if 'clase_ternaria' in df.columns:
        df['clase_binaria2'] = df['clase_ternaria'].isin(['BAJA+1', 'BAJA+2']).astype(int)
    return df


def variables_del_modelo(df):
    return [c for c in df.columns if c not in NO_FEATURES and not any(p in c for p in FUERA)]


# ============================================================================ 3. modelo
def matrices(con, meses_train, mes_aplicar):
    """Carga los meses y los deja como matrices float32 (una sola copia en memoria)."""
    train = meses(con, meses_train)
    feats = variables_del_modelo(train)
    X, y = train[feats].to_numpy(np.float32), train['clase_binaria2'].to_numpy()
    del train
    aplicar = meses(con, [mes_aplicar])
    Xa = aplicar[feats].to_numpy(np.float32)
    info = aplicar[['numero_de_cliente'] + (['clase_ternaria'] if 'clase_ternaria' in aplicar else [])].copy()
    del aplicar
    return X, y, Xa, info, feats


def semillerio(X, y, Xa, feats, filas_ref, semillas=None):
    """Un LightGBM por semilla (mismos hiperparámetros) y promedio de las probabilidades."""
    p = dict(PARAMS)
    # min_data_in_leaf es un mínimo absoluto de filas: se escala con el tamaño del entrenamiento
    p['min_data_in_leaf'] = max(1, int(round(PARAMS['min_data_in_leaf'] * len(X) / filas_ref)))
    semillas = semillas or SEMILLAS
    log(f'entreno {len(semillas)} semillas | {len(X):,} filas | {len(feats)} variables | '
        f'min_data_in_leaf {PARAMS["min_data_in_leaf"]} -> {p["min_data_in_leaf"]}')
    preds = []
    for i, s in enumerate(semillas):
        d = lgb.Dataset(X, label=y, feature_name=feats, params={'max_bin': p['max_bin'], 'seed': s})
        m = lgb.train({**p, 'seed': s, 'verbose': -1}, d, num_boost_round=N_ARBOLES)
        preds.append(m.predict(Xa))
        del d, m
        log(f'  semilla {i + 1}/{len(semillas)} ({s}) lista')
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
    ap.add_argument('--salida', default='entregas_verscla10')
    ap.add_argument('--db', default=':memory:', help='archivo DuckDB en disco (ahorra RAM en máquinas chicas)')
    ap.add_argument('--memoria', default=None, help="límite de memoria de DuckDB, ej. '2GB'")
    a = ap.parse_args()

    con = duckdb.connect(a.db, config={'memory_limit': a.memoria} if a.memoria else {})
    for t in ('fe', 'dom', 'historia', 'base', 'crudo'):
        con.sql(f'drop table if exists {t}')
    cargar_y_clase(con, a.crudo)
    armar_variables(con)
    con.sql(sql_dominio())
    filas_ref = con.sql(f'select count(*) from fe where foto_mes = {MES_REFERENCIA}').fetchone()[0]
    log('variables listas')

    if a.validar:
        # Mismo hueco de dos meses que la entrega: se aprende hasta abril y se mide en junio
        X, y, Xa, junio, feats = matrices(con, [202103, 202104], 202106)
        prob, _ = semillerio(X, y, Xa, feats, filas_ref, SEMILLAS[:5])
        print('\nValidación 202103-202104 -> 202106 (millones):')
        print(ganancia_por_envios(prob, junio['clase_ternaria'].values).round(1).to_string())
        del X, y, Xa, junio

    X, y, Xa, agosto, feats = matrices(con, MESES_FINAL, MES_FUTURO)
    con.close()   # libera la memoria de DuckDB antes de entrenar
    prob, preds = semillerio(X, y, Xa, feats, filas_ref)

    ids = agosto['numero_de_cliente'].values
    os.makedirs(a.salida, exist_ok=True)
    np.save(f'{a.salida}/prob_agosto_semillerio.npy', prob)
    final = escribir(prob, ids, f'{a.salida}/verscla10_k{K}.csv')
    for s, p in zip(SEMILLAS, preds):
        escribir(p, ids, f'{a.salida}/semillas/verscla10_seed{s}_k{K}.csv')
    log(f'listo: {final} y {len(SEMILLAS)} archivos por semilla en {a.salida}/semillas/')
    print(open(final).read()[:60])

    previa = 'entregas/verscla09_k13000.csv'
    if os.path.exists(previa):
        a_ = set(pd.read_csv(final, header=None)[0])
        b_ = set(pd.read_csv(previa, header=None)[0])
        print(f'Clientes en común con verscla09 ({previa}): {len(a_ & b_) / K * 100:.1f} %')


if __name__ == '__main__':
    main()
