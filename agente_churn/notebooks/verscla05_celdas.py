# verscla05 — celdas agregadas a competencia_v5.ipynb después de la celda 21.
# Requieren lo que dejan en memoria las celdas 0-17: data, train, test, futuro, features,
# params, best_iter, mis_semillas, cortes, curva, prob_semillerio2, rankdata, np, pd, lgb, os.
# Se corren en el notebook (usa %sql de jupysql en la celda 27).

# ---------------------------------------------------------------- celda 22
# Undersampling al 25% y 50% con los hiperparámetros de verscla03 (sin Optuna nuevo)
def semillerio_ratio(df_train, df_pred, ratio):
    preds = []
    for seed in mis_semillas:
        r = np.random.default_rng(seed)
        m = (r.random(len(df_train)) < ratio) | (df_train['clase_ternaria'] != 'CONTINUA')
        frac = m.mean()
        p = {**params, 'seed': seed,
             'min_data_in_leaf': max(1, int(params['min_data_in_leaf'] * frac))}
        d = lgb.Dataset(df_train.loc[m, features], label=df_train.loc[m, 'clase_binaria2'])
        mod = lgb.train(p, d, num_boost_round=best_iter)
        preds.append(mod.predict(df_pred[features]))
    return np.mean(preds, axis=0)
# Resultado (meseta 12.500-15.000): verscla03 408,2 · us0.25 408,9 · ens0.25 410,0 · us0.5 405,3 · ens0.5 407,5

# ---------------------------------------------------------------- celdas 23-24
# Marzo + abril -> junio. Resultado: mar+abr 381,8 (marzo no tiene historia: lags vacíos). Descartado.

# ---------------------------------------------------------------- celda 25
# Validación adversarial
def adversarial(df_a, df_b, nombre):
    X = pd.concat([df_a[features], df_b[features]])
    y = np.r_[np.zeros(len(df_a)), np.ones(len(df_b))]
    p = {'objective': 'binary', 'learning_rate': 0.05, 'num_leaves': 31,
         'feature_fraction': 0.5, 'verbose': -1, 'seed': semillas[0]}
    cv = lgb.cv(p, lgb.Dataset(X, label=y), num_boost_round=200, nfold=3,
                stratified=True, metrics='auc', seed=semillas[0])
    clave = [c for c in cv if c.endswith('auc-mean')][0]
    m = lgb.train(p, lgb.Dataset(X, label=y), num_boost_round=200)
    imp = pd.Series(m.feature_importance('gain'), index=features)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    print(f'{nombre}: AUC adversarial = {cv[clave][-1]:.3f}')
    print(imp.head(15).round(3), '\n')
    return imp
# abril vs junio: AUC 1,000 (lags 2 y 3 vacíos en abril: falta de historia, no drift)
# junio vs agosto: AUC 0,993 (fechas de tarjeta, montos inflados, conteo de sueldo por aguinaldo)

# ---------------------------------------------------------------- celda 26
# Sacar fechas / rank por mes / sacar conteo de sueldo. Ninguna mejora:
# sin fechas 407,1 · rank 7 montos 399,6 · rank todos los montos 380,3 · sin conteo sueldo 405,6

# ---------------------------------------------------------------- celda 27 (GANADORA)
# Variables estilo American Express: cambio relativo contra el mes anterior + manejo de la tarjeta
previo = %sql select * from competencia_01 where foto_mes in (202103, 202105, 202107)
previo['foto_mes'] = previo['foto_mes'] + 1

candidatas = ['ctrx_quarter', 'mcuentas_saldo', 'mcaja_ahorro', 'mpasivos_margen', 'mactivos_margen',
              'mrentabilidad', 'mcomisiones', 'mcomisiones_mantenimiento',
              'mtarjeta_visa_consumo', 'mtarjeta_master_consumo',
              'ctarjeta_visa_transacciones', 'ctarjeta_master_transacciones',
              'chomebanking_transacciones', 'mtransferencias_recibidas', 'mtransferencias_emitidas',
              'mprestamos_personales', 'Visa_msaldototal', 'Master_msaldototal',
              'Visa_mpagado', 'Master_mpagado']
base = [v for v in candidatas if v in data.columns]
ant = previo[['numero_de_cliente', 'foto_mes'] + base].rename(columns={v: v + '_ant' for v in base})

def div(a, b):
    return a / b.where(b != 0)

def amex(df):
    out = df.merge(ant, on=['numero_de_cliente', 'foto_mes'], how='left')
    nuevas = {}
    for v in base:
        nuevas[v + '_var1'] = div(out[v] - out[v + '_ant'], out[v + '_ant'].abs())
    for t in ['Visa', 'Master']:
        s, p, m, l = f'{t}_msaldototal', f'{t}_mpagado', f'{t}_mpagominimo', f'{t}_mlimitecompra'
        if {s, p} <= set(out.columns):
            nuevas[f'{t}_deuda_tras_pago'] = out[s] - out[p]
            nuevas[f'{t}_pago_sobre_saldo'] = div(out[p], out[s])
        if {p, m} <= set(out.columns):
            nuevas[f'{t}_pago_sobre_minimo'] = div(out[p], out[m])
        if {s, l} <= set(out.columns):
            nuevas[f'{t}_uso_limite'] = div(out[s], out[l])
    out = out.drop(columns=[v + '_ant' for v in base])
    return pd.concat([out, pd.DataFrame(nuevas, index=out.index)], axis=1)

def semillerio_feats(df_train, df_pred, feats):
    d = lgb.Dataset(df_train[feats], label=df_train['clase_binaria2'])
    preds = []
    for seed in mis_semillas:
        mod = lgb.train({**params, 'seed': seed}, d, num_boost_round=best_iter)
        preds.append(mod.predict(df_pred[feats]))
    return np.mean(preds, axis=0)

train_a, test_a, futuro_a = amex(train), amex(test), amex(futuro)
nuevas_cols = [c for c in train_a.columns if c not in train.columns]
features_a = features + nuevas_cols          # 175 + 28 = 203 variables
prob_amex = semillerio_feats(train_a, test_a, features_a)
# Junio: verscla03 meseta 408,2 (máx 412,0) · amex meseta 426,9 (máx 431,2 en 13.000)

# ---------------------------------------------------------------- celda 28 (entrega)
k = 13500
d_final = lgb.Dataset(test_a[features_a], label=test_a['clase_binaria2'])
preds_fut = []
os.makedirs('entregas', exist_ok=True)
for seed in mis_semillas:
    m = lgb.train({**params, 'seed': seed}, d_final, num_boost_round=best_iter)
    p = m.predict(futuro_a[features_a])
    preds_fut.append(p)
    ids_seed = futuro_a['numero_de_cliente'].values[np.argsort(p)[::-1]][:k].astype('int64')
    pd.Series(ids_seed).to_csv(f'entregas/verscla05_seed{seed}_k{k}.csv', index=False, header=False)

prob_fut_amex = np.mean(preds_fut, axis=0)
elegidos = futuro_a['numero_de_cliente'].values[np.argsort(prob_fut_amex)[::-1]][:k].astype('int64')
archivo = f'entregas/verscla05_amex_k{k}.csv'
pd.Series(elegidos).to_csv(archivo, index=False, header=False)
