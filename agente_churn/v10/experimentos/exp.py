"""Experimentos pareados: entrenar 202103-202104 -> medir 202106, mismas semillas por variante.
Uso: python exp.py VARIANTE [VARIANTE ...]"""
import os, sys, json, time
import duckdb, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, '.')
import verscla09 as v

SEMILLAS = v.SEMILLAS[:5]
OUT = 'probs'
os.makedirs(OUT, exist_ok=True)
con = duckdb.connect('fe.duckdb', read_only=True, config={'memory_limit': '2GB', 'threads': 2})
DOM = [r[0] for r in con.sql('describe dom').fetchall()][2:]
MPAGADO_TRAMPA = ['Visa_mpagado_delta1', 'Master_mpagado_delta1', 'Visa_mpagado_var1', 'Master_mpagado_var1',
                  'Visa_mpagospesos_delta1', 'Master_mpagospesos_delta1', 'ccajas_depositos_lag1', 'ccajas_depositos_delta1']


FE_COLS = [r[0] for r in con.sql('describe fe').fetchall()]


DIN = [c for c in con.sql("describe select * from read_parquet('din.parquet')").df()['column_name'] if c not in ('numero_de_cliente', 'foto_mes')]


def cargar(meses, con_dom, con_din=False):
    cols = [f'f."{c}"' if c in ('numero_de_cliente', 'foto_mes', 'clase_ternaria') else f'cast(f."{c}" as float) as "{c}"'
            for c in FE_COLS]
    if con_dom:
        cols += [f'cast(d."{c}" as float) as "{c}"' for c in DOM]
    if con_din:
        cols += [f'cast(s."{c}" as float) as "{c}"' for c in DIN]
    q = f"""select {', '.join(cols)}
            from fe f {'join dom d using (numero_de_cliente, foto_mes)' if con_dom else ''}
                 {"join read_parquet('din.parquet') s using (numero_de_cliente, foto_mes)" if con_din else ''}
            where f.foto_mes in ({', '.join(map(str, meses))}) order by f.foto_mes, f.numero_de_cliente"""
    df = con.sql(q).df()
    df['clase_binaria2'] = df['clase_ternaria'].isin(['BAJA+1', 'BAJA+2']).astype(int)
    return df


CON_DIN = {'din', 'dom_din', 'dom_din_ff035'}


def variante(nombre):
    """Devuelve (usa_dom, filtro_features, overrides de params)."""
    if nombre == 'din':
        return False, lambda f: f, {}
    if nombre == 'dom_din':
        return True, lambda f: f, {}
    if nombre == 'dom_din_ff035':
        return True, lambda f: f, {'feature_fraction': 0.35}
    base_excl = []
    if nombre == 'base':
        return False, lambda f: f, {}
    if nombre == 'dom':
        return True, lambda f: f, {}
    if nombre == 'sin_trampas':
        return False, lambda f: [c for c in f if c not in MPAGADO_TRAMPA], {}
    if nombre == 'ff035':
        return False, lambda f: f, {'feature_fraction': 0.35}
    if nombre == 'dom_ff035':
        return True, lambda f: f, {'feature_fraction': 0.35}
    if nombre == 'dom_sin_trampas':
        return True, lambda f: [c for c in f if c not in MPAGADO_TRAMPA], {}
    if nombre == 'dom_ff035_sin_trampas':
        return True, lambda f: [c for c in f if c not in MPAGADO_TRAMPA], {'feature_fraction': 0.35}
    if nombre.startswith('p_'):   # p_<clave>=<valor>,... sobre dom con feature_fraction 0.35 por defecto
        ov = {'feature_fraction': 0.35}
        for kv in nombre[2:].split(','):
            k, val = kv.split('=')
            ov[k] = float(val) if '.' in val else int(val)
        return True, lambda f: f, ov
    raise ValueError(nombre)


def correr(nombre):
    usa_dom, filtro, ov = variante(nombre)
    faltan = [s for s in SEMILLAS if not os.path.exists(f'{OUT}/{nombre}_{s}.npy')]
    if not faltan:
        print(nombre, 'ya está', flush=True); return
    tr = cargar([202103, 202104], usa_dom, nombre in CON_DIN)
    te = cargar([202106], usa_dom, nombre in CON_DIN)
    feats = filtro(v.variables_del_modelo(tr))
    n_iter = ov.pop('num_iterations', v.N_ARBOLES)
    p = {**v.PARAMS, **ov}
    p['min_data_in_leaf'] = max(1, int(round(p['min_data_in_leaf'] * len(tr) / 163000)))
    print(f'{nombre}: {len(feats)} variables | params {ov} | iter {n_iter} | mdl {p["min_data_in_leaf"]}', flush=True)
    np.save(f'{OUT}/clase_junio.npy', te['clase_ternaria'].values.astype(str))
    for s in faltan:
        t = time.time()
        d = lgb.Dataset(tr[feats], label=tr['clase_binaria2'], params={'max_bin': 31, 'seed': s})
        m = lgb.train({**p, 'seed': s, 'verbose': -1}, d, num_boost_round=n_iter)
        np.save(f'{OUT}/{nombre}_{s}.npy', m.predict(te[feats]))
        if s == faltan[0]:
            imp = pd.Series(m.feature_importance('gain'), index=feats)
            (imp / imp.sum()).sort_values(ascending=False).head(40).to_csv(f'{OUT}/{nombre}_importancia.csv')
        print(f'  {nombre} semilla {s} {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    for n in sys.argv[1:]:
        correr(n)
