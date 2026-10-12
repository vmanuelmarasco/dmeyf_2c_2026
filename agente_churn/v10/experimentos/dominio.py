"""Variables de dominio bancario (catálogo INFORME_FEATURES_CHURN.md) sobre la tabla base."""

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
