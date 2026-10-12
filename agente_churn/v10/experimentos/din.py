import duckdb, time
con = duckdb.connect('fe.duckdb', read_only=True, config={'memory_limit': '1500MB', 'threads': 1})
V = ['ctrx_quarter', 'mcuentas_saldo', 'mrentabilidad', 'mtarjeta_visa_consumo', 'mtarjeta_master_consumo',
     'chomebanking_transacciones', 'ctarjeta_debito_transacciones', 'mcaja_ahorro', 'mpasivos_margen',
     'mactivos_margen', 'mcomisiones', 'cproductos', 'mtransferencias_emitidas', 'mtransferencias_recibidas',
     'ctarjeta_visa_transacciones', 'cmobile_app_trx', 'mautoservicio', 'Visa_msaldototal']
ex = []
for c in V:
    ex.append(f'regr_slope({c}, cliente_antiguedad) over w3 as s_{c}_slope3')
    ex.append(f'min({c}) over w3 as s_{c}_min3')
    ex.append(f'max({c}) over w3 as s_{c}_max3')
ex.append('(mcuentas_saldo < 0)::int as s_saldo_negativo')
ex.append('sum((mcuentas_saldo < 0)::int) over w3 as s_meses_saldo_negativo3')
ex.append('count(*) over w3 as s_meses_historia3')
t = time.time()
con.sql(f"""copy (select numero_de_cliente, foto_mes, {', '.join(ex)} from base
            window w3 as (partition by numero_de_cliente order by foto_mes rows between 2 preceding and current row))
            to 'din.parquet' (format parquet)""")
print('ok', time.time() - t)
