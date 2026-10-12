"""lab.py — funciones reutilizables para experimentar sobre la base de verscla07.

Uso desde un notebook en la misma carpeta:
    import lab
    tr, te, fu, cfg, probs = lab.cargar()
    lab.evaluar('mi experimento', prob_junio, te)

Todo lo que se evalúa queda registrado en resultados/registro.csv (y la probabilidad en
resultados/probs/), así que nada se pierde si se reinicia el kernel.

Protocolo (el mismo de la competencia):
  - Se entrena con abril (tr) y se mide en junio (te): mes sin usar en el medio.
  - La entrega se entrena con junio (te) y predice agosto (fu).
  - Métrica de decisión: meseta = ganancia promedio entre 12.500 y 15.000 envíos.
"""
import os
import json
import datetime as _dt

import numpy as np
import pandas as pd
import lightgbm as lgb
import duckdb
from scipy.stats import rankdata

BASE = 'resultados/base/'
PROBS = 'resultados/probs/'
REGISTRO = 'resultados/registro.csv'
GANANCIA, COSTO = 1_072_500, 27_500
CORTES = np.arange(4000, 19001, 500)
MESETA = (12500, 15000)
REFERENCIA = 432.8   # meseta de verscla07 en junio


# ----------------------------------------------------------------------------- datos
def cargar():
    """Tablas de abril, junio y agosto con las 203 variables de verscla05, config y probabilidades guardadas."""
    t = {n: duckdb.sql(f"select * from read_parquet('{BASE}{n}.parquet')").df()
         for n in ['train_a', 'test_a', 'futuro_a']}
    for df in t.values():
        if 'clase_ternaria' in df.columns:
            df['clase_binaria2'] = df['clase_ternaria'].isin(['BAJA+1', 'BAJA+2']).astype(int)
    cfg = json.load(open(BASE + 'config.json'))
    probs = {f[:-4]: np.load(BASE + f) for f in sorted(os.listdir(BASE)) if f.endswith('.npy')}
    print(f"abril {t['train_a'].shape} | junio {t['test_a'].shape} | agosto {t['futuro_a'].shape} | "
          f"{len(cfg['features_a'])} variables | probabilidades: {list(probs)}")
    return t['train_a'], t['test_a'], t['futuro_a'], cfg, probs


def mismo_orden(df_otro, df_ref, clave='numero_de_cliente'):
    """Reordena df_otro con el orden de clientes de df_ref (para comparar o ensamblar cliente a cliente)."""
    out = df_ref[[clave]].merge(df_otro, on=clave, how='left')
    assert len(out) == len(df_ref), 'hay clientes duplicados'
    return out


# ----------------------------------------------------------------------------- medición
def curva(prob, test, cortes=CORTES):
    """Ganancia en junio para cada cantidad de envíos (ordenando de mayor a menor score)."""
    orden = np.argsort(-np.asarray(prob), kind='stable')
    gan = np.where(test['clase_ternaria'].values[orden] == 'BAJA+2', GANANCIA, -COSTO)
    return np.cumsum(gan)[cortes - 1]


def meseta(c, cortes=CORTES):
    m = (cortes >= MESETA[0]) & (cortes <= MESETA[1])
    return float(np.mean(c[m]))


def evaluar(nombre, prob, test, referencia=REFERENCIA, guardar=True):
    """Imprime meseta, diferencia con verscla07 y mejor corte; lo agrega al registro en disco."""
    c = curva(prob, test) / 1e6
    ms = meseta(c)
    linea = {'fecha': _dt.datetime.now().strftime('%Y-%m-%d %H:%M'), 'experimento': nombre,
             'meseta': round(ms, 1), 'vs_ref': round(ms - referencia, 1),
             'max': round(float(c.max()), 1), 'k_max': int(CORTES[c.argmax()]),
             **{f'k{k}': round(float(v), 1) for k, v in zip(CORTES, c) if 11000 <= k <= 16000}}
    if guardar:
        os.makedirs(PROBS, exist_ok=True)
        np.save(PROBS + _slug(nombre) + '.npy', np.asarray(prob))
        pd.DataFrame([linea]).to_csv(REGISTRO, mode='a', index=False, header=not os.path.exists(REGISTRO))
    print(f"{nombre:42s} meseta {ms:6.1f} | vs ref {ms - referencia:+6.1f} | máx {c.max():6.1f} en {CORTES[c.argmax()]}")
    return ms


def registro(ultimos=None):
    """Tabla con todos los experimentos evaluados (los más recientes abajo)."""
    r = pd.read_csv(REGISTRO)
    cols = ['fecha', 'experimento', 'meseta', 'vs_ref', 'max', 'k_max']
    return r[cols].tail(ultimos) if ultimos else r[cols]


def gana_en_cortes(prob, prob_ref, test, desde=11000, hasta=16000):
    """En cuántos cortes de la franja un modelo le gana / empata / pierde a la referencia."""
    m = (CORTES >= desde) & (CORTES <= hasta)
    d = (curva(prob, test) - curva(prob_ref, test))[m]
    return {'gana': int((d > 0).sum()), 'empata': int((d == 0).sum()), 'pierde': int((d < 0).sum())}


def _slug(s):
    return ''.join(ch if ch.isalnum() else '_' for ch in s).strip('_')[:80]


# ----------------------------------------------------------------------------- modelos
def ensamble(*probs, pesos=None):
    """Promedio de rankings (no de probabilidades): cada modelo aporta su orden de clientes."""
    pesos = pesos or [1] * len(probs)
    return sum(w * rankdata(p) for w, p in zip(pesos, probs)) / sum(pesos)


def _mascara_us(df, ratio, seed):
    if ratio >= 1:
        return np.ones(len(df), dtype=bool)
    r = np.random.default_rng(seed)
    return (r.random(len(df)) < ratio) | (df['clase_ternaria'].values != 'CONTINUA')


def semillerio_lgb(df_train, df_pred, feats, params, n_iter, semillas, ratio=1.0, peso_b2=1.0,
                   categoricas='auto', target='clase_binaria2', guardar_modelos=None, por_semilla=False):
    """Un LightGBM por semilla, promedio de probabilidades.
    ratio < 1: undersampling de CONTINUA distinto por semilla. peso_b2: peso de BAJA+2 al entrenar.
    guardar_modelos: prefijo de archivo para guardar cada modelo (ej. 'resultados/modelos/v08')."""
    preds = []
    for seed in semillas:
        m = _mascara_us(df_train, ratio, seed)
        d_tr = df_train.loc[m]
        w = np.where(d_tr['clase_ternaria'].values == 'BAJA+2', peso_b2, 1.0) if peso_b2 != 1 else None
        d = lgb.Dataset(d_tr[feats], label=d_tr[target], weight=w, categorical_feature=categoricas,
                        free_raw_data=False)
        mod = lgb.train({**params, 'seed': seed, 'verbose': -1}, d, num_boost_round=n_iter)
        if guardar_modelos:
            os.makedirs(os.path.dirname(guardar_modelos), exist_ok=True)
            mod.save_model(f'{guardar_modelos}_seed{seed}.txt')
        preds.append(mod.predict(df_pred[feats]))
    media = np.mean(preds, axis=0)
    return (media, preds) if por_semilla else media


def semillerio_xgb(df_train, df_pred, feats, params, n_iter, semillas, target='clase_binaria2'):
    import xgboost as xgb
    X_tr = df_train[feats].to_numpy(dtype='float32', na_value=np.nan)
    X_pr = df_pred[feats].to_numpy(dtype='float32', na_value=np.nan)
    d_tr = xgb.DMatrix(X_tr, label=df_train[target].values, missing=np.nan)
    d_pr = xgb.DMatrix(X_pr, missing=np.nan)
    return np.mean([xgb.train({**params, 'seed': s}, d_tr, num_boost_round=n_iter).predict(d_pr)
                    for s in semillas], axis=0)


def etiqueta_ranking(df):
    """BAJA+2 = 2, BAJA+1 = 1, CONTINUA = 0: le dice al ranking que BAJA+2 vale más."""
    return df['clase_ternaria'].map({'BAJA+2': 2, 'BAJA+1': 1, 'CONTINUA': 0}).values


def semillerio_rank(df_train, df_pred, feats, params, n_iter, semillas, tam_grupo=1000,
                    objetivo='rank_xendcg'):
    """Objetivo de ranking (industria): optimiza el ORDEN de los clientes, no la probabilidad.
    Los clientes se reparten al azar en grupos de tam_grupo (una 'consulta' por grupo)."""
    y = etiqueta_ranking(df_train)
    preds = []
    for seed in semillas:
        perm = np.random.default_rng(seed).permutation(len(df_train))
        n = len(perm)
        grupos = [tam_grupo] * (n // tam_grupo) + ([n % tam_grupo] if n % tam_grupo else [])
        d = lgb.Dataset(df_train[feats].iloc[perm], label=y[perm], group=grupos)
        p = {k: v for k, v in params.items() if k not in ('objective', 'metric')}
        mod = lgb.train({**p, 'objective': objetivo, 'seed': seed, 'verbose': -1}, d, num_boost_round=n_iter)
        s = mod.predict(df_pred[feats])
        preds.append(rankdata(s) / len(s))   # los scores de ranking no son probabilidades: promedio rankings
    return np.mean(preds, axis=0)


# ----------------------------------------------------------------------------- selección de variables
def null_importance(df_train, feats, params, n_iter, semilla, n_null=5, target='clase_binaria2'):
    """Importancia real contra importancia con el target mezclado al azar (n_null veces).
    Devuelve una tabla con la importancia real, el máximo nulo y el cociente."""
    def imp(y, s):
        d = lgb.Dataset(df_train[feats], label=y)
        m = lgb.train({**params, 'seed': s, 'verbose': -1}, d, num_boost_round=n_iter)
        return pd.Series(m.feature_importance('gain'), index=feats)
    real = imp(df_train[target].values, semilla)
    rng = np.random.default_rng(semilla)
    nulas = pd.concat([imp(rng.permutation(df_train[target].values), semilla + i + 1)
                       for i in range(n_null)], axis=1)
    t = pd.DataFrame({'real': real, 'nula_max': nulas.max(axis=1), 'nula_media': nulas.mean(axis=1)})
    t['cociente'] = (t['real'] + 1) / (t['nula_max'] + 1)
    return t.sort_values('cociente', ascending=False)


# ----------------------------------------------------------------------------- entrega
def entrega(prob_agosto, futuro, k, nombre, carpeta='entregas'):
    """CSV de una columna, sin encabezado, ids enteros: los k clientes de agosto con mayor score."""
    os.makedirs(carpeta, exist_ok=True)
    ids = futuro['numero_de_cliente'].values[np.argsort(-np.asarray(prob_agosto), kind='stable')][:k]
    ids = ids.astype('int64')
    archivo = f'{carpeta}/{nombre}_k{k}.csv'
    pd.Series(ids).to_csv(archivo, index=False, header=False)
    assert len(set(ids)) == k
    print(archivo, '->', k, 'ids |', open(archivo).read()[:40].replace('\n', ' '))
    return archivo


def coincidencia(archivo_a, archivo_b):
    a = set(pd.read_csv(archivo_a, header=None)[0])
    b = set(pd.read_csv(archivo_b, header=None)[0])
    return round(len(a & b) / max(len(a), len(b)) * 100, 1)
