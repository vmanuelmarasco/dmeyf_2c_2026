"""Compara variantes contra una referencia, semilla por semilla (pareado).
Regla fijada antes de medir: adoptar si la meseta media mejora >= 5 M y gana en >= 4 de 5 semillas."""
import sys, glob, os
import numpy as np, pandas as pd
sys.path.insert(0, '.')
import exp

G, C = 1_072_500, 27_500
CORTES = np.arange(9000, 17001, 500)
clase = np.load('probs/clase_junio.npy', allow_pickle=True)


def curva(p):
    o = np.argsort(-p, kind='stable')
    return np.cumsum(np.where(clase[o] == 'BAJA+2', G, -C))[CORTES - 1] / 1e6


def meseta(c, lo=12000, hi=15000):
    return c[(CORTES >= lo) & (CORTES <= hi)].mean()


def resumen(nombre):
    ps = [np.load(f'probs/{nombre}_{s}.npy') for s in exp.SEMILLAS if os.path.exists(f'probs/{nombre}_{s}.npy')]
    if len(ps) < len(exp.SEMILLAS):
        return None
    mes = [meseta(curva(p)) for p in ps]
    semillerio = curva(np.mean(ps, axis=0))
    return np.array(mes), semillerio


ref = sys.argv[1]
r_mes, r_sem = resumen(ref)
filas = []
for n in [ref] + sys.argv[2:]:
    r = resumen(n)
    if r is None:
        continue
    mes, sem = r
    d = mes - r_mes
    filas.append({'variante': n, 'meseta_media': round(mes.mean(), 1), 'desvio': round(mes.std(), 1),
                  'vs_ref': round(d.mean(), 1), 'gana_semillas': f'{(d > 0).sum()}/{len(d)}',
                  'semillerio_meseta': round(meseta(sem), 1), 'k_max_semillerio': int(CORTES[sem.argmax()]),
                  'adopta': (d.mean() >= 5) and ((d > 0).sum() >= 4)})
print(pd.DataFrame(filas).to_string(index=False))
