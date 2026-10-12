"""pipeline_v8.py — verscla08 (Amex con 4 meses) y verscla09 (+ lag_1/delta_1 de todas).

Cambios respecto de verscla05, cada uno con su porqué:
  1. Variables: las de verscla05 (señales de desvinculación + Amex) MÁS lag_1 y delta_1 de TODAS las
     variables numéricas (historia por cliente, z402).
  2. Modelo final con 202103-202106 (4 meses, ~4 veces más bajas para aprender) en lugar de solo
     202106, con min_data_in_leaf escalado por filas (param_normalizado de z494).
  3. Semillerío final de 20 semillas y un archivo por semilla para leer media y desvío en el bot.
  4. ccajas_depositos vale 0 para todos en 202105: error de datos -> NULL.
  5. Fuera del modelo: *_fultimo_cierre, *_Finiciomora (drift de datos) y el monto de sueldo
     (mpayroll*, aguinaldo), con todo lo derivado de ellas.

Validación (mismo hueco de 2 meses que la entrega):
  A: entrenar 202104 -> medir 202106            (como verscla05)
  B: entrenar 202103 + 202104 -> medir 202106   (como el modelo final: varios meses)
Se registran en resultados/registro.csv con lab.evaluar.

Uso (desde la carpeta dmeyf402, con lab.py al lado):
  python pipeline_v8.py --solo-final                       verscla08: variables de verscla05, final con 4 meses
  python pipeline_v8.py --solo-final --features todas      verscla09: + lag_1 y delta_1 de todas
  python pipeline_v8.py                                    validación + final + entrega (k 13000)
  python pipeline_v8.py --solo-validacion
  python pipeline_v8.py --solo-final --k 13000
  python pipeline_v8.py --crudo /ruta/competencia_01_crudo.csv
"""
import argparse
import json
import os
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

import lab

CRUDO = ('/Users/manuelmarasco/Library/CloudStorage/GoogleDrive-hola@manuelmarasco.com/'
         'Mi unidad/DMEyF/2026/notebooks/data/competencia_01_crudo.csv')
NO_FEATURES = {'numero_de_cliente', 'foto_mes', 'clase_ternaria', 'clase_binaria2'}
FUERA_PATRONES = ('fultimo_cierre', 'Finiciomora', 'mpayroll')
AMEX = ['ctrx_quarter', 'mcuentas_saldo', 'mcaja_ahorro', 'mpasivos_margen', 'mactivos_margen',
        'mrentabilidad', 'mcomisiones', 'mcomisiones_mantenimiento', 'mtarjeta_visa_consumo',
        'mtarjeta_master_consumo', 'ctarjeta_visa_transacciones', 'ctarjeta_master_transacciones',
        'chomebanking_transacciones', 'mtransferencias_recibidas', 'mtransferencias_emitidas',
        'mprestamos_personales', 'Visa_msaldototal', 'Master_msaldototal', 'Visa_mpagado', 'Master_mpagado']
T0 = time.time()


def log(*a):
    print(f'[{(time.time() - T0) / 60:5.1f} min]', *a, flush=True)


# ----------------------------------------------------------------------------- datos
def construir(crudo):
    con = duckdb.connect()
    con.sql(f"""create table crudo as
                select * replace (cast(numero_de_cliente as bigint) as numero_de_cliente)
                from read_csv_auto('{crudo}')""")
    cols_crudo = [r[0] for r in con.sql('describe crudo').fetchall()]
    if 'ccajas_depositos' in cols_crudo:
        con.sql("update crudo set ccajas_depositos = null where foto_mes = 202105")

    # Clase ternaria (z101): igual que la celda 3 del notebook
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
    tipos = {r[0]: r[1] for r in con.sql('describe base').fetchall()}
    numericas = [c for c, t in tipos.items()
                 if c not in NO_FEATURES and any(x in t for x in ('INT', 'DOUBLE', 'FLOAT', 'DECIMAL'))]
    log(len(numericas), 'variables numéricas en el crudo')

    def q(c):
        return f'"{c}"'

    exprs = []
    # lag_1 y delta_1 de todas
    for c in numericas:
        exprs.append(f'lag({q(c)}, 1) over w as {q(c + "_lag1")}')
        exprs.append(f'{q(c)} - lag({q(c)}, 1) over w as {q(c + "_delta1")}')
    # historia específica del notebook (celda 5) que no está en lo anterior
    extra = {
        'ctrx_quarter_lag2': 'lag(ctrx_quarter, 2) over w',
        'ctrx_quarter_delta2': 'ctrx_quarter - lag(ctrx_quarter, 2) over w',
        'cproductos_delta2': 'cproductos - lag(cproductos, 2) over w',
        'ctrx_quarter_avg3': 'avg(ctrx_quarter) over w3',
        'consumo_tc_avg3': 'avg(coalesce(mtarjeta_visa_consumo, 0) + coalesce(mtarjeta_master_consumo, 0)) over w3',
        'mprestamos_personales_lag3': 'lag(mprestamos_personales, 3) over w',
        'cpayroll_trx_lag2': 'lag(cpayroll_trx, 2) over w',
        'cpayroll_trx_lag3': 'lag(cpayroll_trx, 3) over w',
    }
    exprs += [f'{e} as {n}' for n, e in extra.items()]
    # Amex: cambio relativo contra el mes anterior
    for c in [a for a in AMEX if a in numericas]:
        exprs.append(f'({q(c)} - lag({q(c)}, 1) over w) / nullif(abs(lag({q(c)}, 1) over w), 0) as {q(c + "_var1")}')
    # Amex: manejo de la tarjeta
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

    # Señales de desvinculación (celda 6 del notebook)
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
    return con


def features_de(df):
    return [c for c in df.columns
            if c not in NO_FEATURES and not any(p in c for p in FUERA_PATRONES)]


def meses(con, lista):
    df = con.sql(f"select * from fe where foto_mes in ({', '.join(map(str, lista))}) "
                 f"order by foto_mes, numero_de_cliente").df()
    if 'clase_ternaria' in df.columns:
        df['clase_binaria2'] = df['clase_ternaria'].isin(['BAJA+1', 'BAJA+2']).astype(int)
    return df


# ----------------------------------------------------------------------------- modelos
def semillas_20(base):
    """Las 5 semillas de siempre + 15 primos de 6 dígitos elegidos con semilla fija."""
    cand = np.arange(100001, 999999, 2)
    primos = [n for n in np.random.default_rng(1791).choice(cand, 4000, replace=False)
              if all(n % d for d in range(3, int(n ** .5) + 1, 2))]
    return list(base) + [int(p) for p in primos if p not in base][:20 - len(base)]


def params_escalados(params, filas, filas_ref):
    p = dict(params)
    p['min_data_in_leaf'] = max(1, int(round(params['min_data_in_leaf'] * filas / filas_ref)))
    return p


def semillerio(tr, pred, feats, params, n_iter, semillas, filas_ref, guardar=None):
    p = params_escalados(params, len(tr), filas_ref)
    log(f'  entreno {len(semillas)} semillas | {len(tr):,} filas | {len(feats)} variables | '
        f'min_data_in_leaf {params["min_data_in_leaf"]} -> {p["min_data_in_leaf"]}')
    preds = []
    for i, s in enumerate(semillas):
        d = lgb.Dataset(tr[feats], label=tr['clase_binaria2'], params={'max_bin': p.get('max_bin', 31), 'seed': s})
        m = lgb.train({**p, 'seed': s, 'verbose': -1}, d, num_boost_round=n_iter)
        if guardar:
            os.makedirs(os.path.dirname(guardar), exist_ok=True)
            m.save_model(f'{guardar}_seed{s}.txt')
        preds.append(m.predict(pred[feats]))
        log(f'    semilla {i + 1}/{len(semillas)} ({s}) lista')
    return np.mean(preds, axis=0), preds


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crudo', default=CRUDO)
    ap.add_argument('--k', type=int, default=13000)
    ap.add_argument('--solo-validacion', action='store_true')
    ap.add_argument('--solo-final', action='store_true')
    ap.add_argument('--params', choices=['v05', 'v06'], default='v05')
    ap.add_argument('--semillas-final', type=int, default=20)
    ap.add_argument('--features', choices=['v05', 'todas'], default='v05',
                    help="v05: las 203 variables de verscla05 (Amex); todas: + lag_1 y delta_1 de todas")
    ap.add_argument('--nombre', default=None, help='prefijo de la entrega (por defecto verscla08 / verscla09)')
    a = ap.parse_args()
    nombre = a.nombre or ('verscla08' if a.features == 'v05' else 'verscla09')

    cfg = json.load(open(lab.BASE + 'config.json'))
    params, n_iter = (cfg['params'], cfg['best_iter']) if a.params == 'v05' else (cfg['params_v2'], cfg['iter_v2'])
    params = {k: v for k, v in params.items() if k not in ('seed', 'metric')}
    log('params', a.params, params, '| árboles', n_iter)

    con = construir(a.crudo)
    log('features listas')

    abril = meses(con, [202104])
    filas_ref = len(abril)            # los hiperparámetros se buscaron con un mes (abril)
    feats = features_de(abril)
    if a.features == 'v05':
        faltan = [f for f in cfg['features_a'] if f not in abril.columns]
        assert not faltan, f'faltan variables de verscla05: {faltan[:10]}'
        feats = list(cfg['features_a'])
    log(len(feats), 'variables en el modelo |', a.features, '| entrega:', nombre)
    os.makedirs('resultados', exist_ok=True)
    json.dump(feats, open(f'resultados/features_{nombre}.json', 'w'))

    if not a.solo_final:
        junio = meses(con, [202106])
        marzo = meses(con, [202103])
        log('VALIDACIÓN A: 202104 -> 202106')
        pA, _ = semillerio(abril, junio, feats, params, n_iter, cfg['mis_semillas'], filas_ref)
        lab.evaluar(f'{nombre} A: abril -> junio', pA, junio)
        log('VALIDACIÓN B: 202103 + 202104 -> 202106')
        pB, _ = semillerio(pd.concat([marzo, abril]), junio, feats, params, n_iter, cfg['mis_semillas'], filas_ref)
        lab.evaluar(f'{nombre} B: marzo+abril -> junio', pB, junio)
        ref = np.load(lab.BASE + 'prob_junio_v05.npy')
        te_ref = duckdb.sql(f"select numero_de_cliente from read_parquet('{lab.BASE}test_a.parquet')").df()
        ref_al = lab.mismo_orden(te_ref.assign(p=ref), junio[['numero_de_cliente']])['p'].values
        lab.evaluar(f'{nombre} ens A + verscla05', lab.ensamble(pA, ref_al), junio)
        print('B contra A, por corte:', lab.gana_en_cortes(pB, pA, junio))
        del marzo, junio

    if not a.solo_validacion:
        log('MODELO FINAL: 202103-202106 -> 202108')
        tr = meses(con, [202103, 202104, 202105, 202106])
        agosto = meses(con, [202108])
        sem = semillas_20(cfg['mis_semillas'])[:a.semillas_final]
        prob, preds = semillerio(tr, agosto, feats, params, n_iter, sem, filas_ref,
                                 guardar=f'resultados/modelos_{nombre}/lgbm')
        os.makedirs(lab.PROBS, exist_ok=True)
        np.save(lab.PROBS + f'{nombre}_agosto_semillerio.npy', prob)
        np.save(lab.PROBS + f'{nombre}_agosto_por_semilla.npy', np.array(preds))
        lab.entrega(prob, agosto, a.k, nombre)
        for s, p in zip(sem, preds):
            lab.entrega(p, agosto, a.k, f'{nombre}_seed{s}', carpeta=f'entregas/{nombre}_semillas')
        log(f'listo: entregas/{nombre}_k{a.k}.csv (semillerío) y {len(sem)} archivos en entregas/{nombre}_semillas/')


if __name__ == '__main__':
    main()
