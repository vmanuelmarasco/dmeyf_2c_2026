"""mi_bot.py — un "Kaggle" propio con junio (202106) como mes de la verdad.

Cómo funciona (igual que el bot de la cátedra, pero con la respuesta conocida):
  - Los modelos se entrenan con 202103-202104 y predicen 202106 (mismo hueco de 2 meses que la entrega).
  - Junio se parte al azar en PÚBLICO (25 %) y PRIVADO (75 %), estratificado por clase, como hace la
    cátedra en z494 (realidad_inicializar). La ganancia se informa en millones, sin escalar.
  - A diferencia del bot real, acá se puede ver el privado y repetir el sorteo miles de veces.

Usos:
  python mi_bot.py archivo.csv [archivo2.csv ...]     como el bot: ganancia pública media y desvío
  python mi_bot.py --analisis                          ¿el público elige bien? (usa probs/ de exp.py)
"""
import sys, glob, os
import numpy as np, pandas as pd, duckdb

G, C = 1_072_500, 27_500
SEMILLA_SORTEO = 314159   # la semilla_kaggle de z494


def junio():
    con = duckdb.connect('fe.duckdb', read_only=True, config={'threads': 1})
    return con.sql("select numero_de_cliente, clase_ternaria from base where foto_mes = 202106 "
                   "order by numero_de_cliente").df()


def sorteo(clase, semilla, prop_publico=0.25):
    """Máscara booleana del público, estratificada por clase."""
    rng = np.random.default_rng(semilla)
    pub = np.zeros(len(clase), dtype=bool)
    for c in np.unique(clase):
        idx = np.where(clase == c)[0]
        pub[rng.choice(idx, int(round(len(idx) * prop_publico)), replace=False)] = True
    return pub


def ganancia_envio(ids_enviados, ref, pub):
    env = ref['numero_de_cliente'].isin(set(ids_enviados)).values
    g = np.where(ref['clase_ternaria'].values == 'BAJA+2', G, -C)
    return (g * env * pub).sum() / 1e6, (g * env * ~pub).sum() / 1e6


def modo_bot(archivos):
    ref = junio()
    pub = sorteo(ref['clase_ternaria'].values, SEMILLA_SORTEO)
    gp = [ganancia_envio(pd.read_csv(a, header=None)[0], ref, pub)[0] for a in archivos]
    print(f'{len(archivos)} candidate model(s) | Public gain -> mean: {np.mean(gp):.4f}, std dev: {np.std(gp):.4f}')


def curvas(prob, clase, pub, cortes):
    """Ganancia pública y privada (millones) para cada cantidad de envíos."""
    o = np.argsort(-prob, kind='stable')
    g = np.where(clase[o] == 'BAJA+2', G, -C)
    p = pub[o]
    return np.cumsum(g * p)[cortes - 1] / 1e6, np.cumsum(g * ~p)[cortes - 1] / 1e6


def modo_analisis(n_sorteos=500):
    """Con los semilleríos de exp.py: ¿cuán seguido el público elige el mismo modelo y el mismo k que el total?"""
    clase = np.load('probs/clase_junio.npy', allow_pickle=True)
    variantes = sorted({os.path.basename(f).rsplit('_', 1)[0] for f in glob.glob('probs/*_[0-9]*.npy')})
    probs = {}
    for v in variantes:
        ps = [np.load(f) for f in sorted(glob.glob(f'probs/{v}_[0-9]*.npy'))]
        if len(ps) == 5:
            probs[v] = np.mean(ps, axis=0)
    cortes = np.arange(10000, 16001, 500)
    total = {v: sum(curvas(p, clase, np.ones(len(clase), bool), cortes)) for v, p in probs.items()}
    mejor_total = max(total, key=lambda v: total[v][(cortes >= 12000) & (cortes <= 15000)].mean())

    acierta_modelo_pico, acierta_modelo_meseta, picos, k_total = 0, 0, [], []
    for s in range(n_sorteos):
        pub = sorteo(clase, 1000 + s)
        cp = {v: curvas(p, clase, pub, cortes)[0] for v, p in probs.items()}
        # criterio A (el que critica Denicolay): el modelo con el pico más alto del público
        a = max(cp, key=lambda v: cp[v].max())
        # criterio B: el modelo con mejor promedio en la meseta del público
        b = max(cp, key=lambda v: cp[v][(cortes >= 12000) & (cortes <= 15000)].mean())
        acierta_modelo_pico += a == mejor_total
        acierta_modelo_meseta += b == mejor_total
        # para el mejor modelo: ¿qué tan lejos queda el pico del público del pico del total?
        # para el mejor modelo: ¿cuánto se pierde en el total por elegir k en el pico del público?
        tot = total[mejor_total]
        i = cp[mejor_total].argmax()
        picos.append((cortes[i], tot[i] - tot[cortes == 13000][0]))
    picos = np.array(picos)
    print(f'Modelos comparados: {len(probs)} | mejor según junio completo: {mejor_total}')
    print(f'Elegir por el PICO del público acierta el mejor modelo en {acierta_modelo_pico / n_sorteos:.0%} de los sorteos')
    print(f'Elegir por la MESETA del público acierta el mejor modelo en {acierta_modelo_meseta / n_sorteos:.0%} de los sorteos')
    print(f'Elegir k en el pico del público, en vez de fijarlo en 13.000 (centro de la meseta), cambia la ganancia '
          f'TOTAL de junio en {picos[:, 1].mean():+.1f} M en promedio (peor caso {picos[:, 1].min():+.1f} M)')
    print('Cortes elegidos por el pico del público:', pd.Series(picos[:, 0].astype(int)).value_counts().sort_index().to_dict())


if __name__ == '__main__':
    if sys.argv[1:] == ['--analisis']:
        modo_analisis()
    else:
        modo_bot(sys.argv[1:])
