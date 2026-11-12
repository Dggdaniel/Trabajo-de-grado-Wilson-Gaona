import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import os
import warnings

warnings.filterwarnings('ignore')

# RUTAS PORTÁTILES
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.abspath(os.path.join(RUTA_BASE, '..', '..'))

print('=== MODELOS ESTADÍSTICOS TRADICIONALES DE REFERENCIA (CON Y SIN DOMINIOS) ===')
print('1. Cargando microdatos depurados de la ECV 2024...')
df = pd.read_excel(os.path.join(RUTA_DATOS, 'validacion_final.xlsx'))
base_raw = pd.read_csv(
    os.path.join(RUTA_DATOS, 'base_final_embedings.csv'),
    usecols=['DIRECTORIO', 'SECUENCIA_P', 'felicidad_ayer', 'preocupacion_ayer_inv', 'tristeza_ayer_inv']
)
df['cluster_hogar'] = base_raw['DIRECTORIO'].astype(str) + '_' + base_raw['SECUENCIA_P'].astype(str)
# Índice puramente afectivo de 3 ítems
df['bienestar_experiencial'] = (
    base_raw['felicidad_ayer'] + base_raw['preocupacion_ayer_inv'] + base_raw['tristeza_ayer_inv']
) / 3.0

# Especificación base (sin satisfacciones de dominio para probar el ingreso directo)
cols_sin_dominios = [
    'ingreso_total_log', 'sexo_nacimiento', 'grupo_edad', 
    'nivel_educativo_alcanzado', 'estado_conyugal', 'P1_DEPARTAMENTO'
]

# Especificación completa (con satisfacciones de dominio: salud y trabajo)
cols_con_dominios = cols_sin_dominios + ['satisfaccion_salud', 'satisfaccion_trabajo']

pesos = df['FEX_C']
y_eval = df['satisfaccion_vida'].astype(int)
y_exp = df['bienestar_experiencial']

print('\n' + '='*70)
print('1. ESTIMACIÓN LOGIT ORDENADO (MÉTODO SIN DOMINIOS VS. CON DOMINIOS)')
print('Nota técnica: OrderedModel en statsmodels se ajusta no ponderado, con errores')
print('estándar agrupados por hogar (conglomerados).')
print('='*70)

# 1A. Logit sin dominios
X_eval_sin = pd.get_dummies(df[cols_sin_dominios], drop_first=True, dtype=float)
print(f'-> Ajustando Logit sin satisfacciones de salud ni trabajo (p = {X_eval_sin.shape[1]})...')
mod_logit_sin = OrderedModel(y_eval, X_eval_sin, distr='logit')
res_logit_sin = mod_logit_sin.fit(method='bfgs', cov_type='cluster', cov_kwds={'groups': df['cluster_hogar']}, disp=False)
coef_ing_sin = res_logit_sin.params['ingreso_total_log']
pval_ing_sin = res_logit_sin.pvalues['ingreso_total_log']
print(f'   Ingreso total log (sin dominios): Coef = {coef_ing_sin:.4f}, p = {pval_ing_sin:.4e}')

# 1B. Logit con dominios
X_eval_con = pd.get_dummies(df[cols_con_dominios], drop_first=True, dtype=float)
print(f'-> Ajustando Logit con satisfacciones de salud y trabajo (p = {X_eval_con.shape[1]})...')
mod_logit_con = OrderedModel(y_eval, X_eval_con, distr='logit')
res_logit_con = mod_logit_con.fit(method='bfgs', cov_type='cluster', cov_kwds={'groups': df['cluster_hogar']}, disp=False)
coef_ing_con = res_logit_con.params['ingreso_total_log']
pval_ing_con = res_logit_con.pvalues['ingreso_total_log']
print(f'   Ingreso total log (con dominios): Coef = {coef_ing_con:.4f}, p = {pval_ing_con:.4f}')

print('\n' + '='*70)
print('2. ESTIMACIÓN WLS PONDERADA (FEX_C + CLÚSTER HOGAR): SIN DOMINIOS VS. CON DOMINIOS')
print('='*70)

# 2A. WLS sin dominios
X_exp_sin = sm.add_constant(X_eval_sin)
mod_wls_sin = sm.WLS(y_exp, X_exp_sin, weights=pesos)
res_wls_sin = mod_wls_sin.fit(cov_type='cluster', cov_kwds={'groups': df['cluster_hogar']})
print(f'-> WLS sin dominios (p = {X_exp_sin.shape[1]-1}):')
print(f'   Ingreso total log: Coef = {res_wls_sin.params["ingreso_total_log"]:.4f}, p = {res_wls_sin.pvalues["ingreso_total_log"]:.4e}, R2 = {res_wls_sin.rsquared:.4f}')

# 2B. WLS con dominios
X_exp_con = sm.add_constant(X_eval_con)
mod_wls_con = sm.WLS(y_exp, X_exp_con, weights=pesos)
res_wls_con = mod_wls_con.fit(cov_type='cluster', cov_kwds={'groups': df['cluster_hogar']})
print(f'-> WLS con dominios (p = {X_exp_con.shape[1]-1}):')
print(f'   Ingreso total log: Coef = {res_wls_con.params["ingreso_total_log"]:.4f}, p = {res_wls_con.pvalues["ingreso_total_log"]:.4e}, R2 = {res_wls_con.rsquared:.4f}')

# Exportar tablas a Excel
excel_path = os.path.join(RUTA_DATOS, 'resultados_modelos_tradicionales_completos.xlsx')
with pd.ExcelWriter(excel_path) as writer:
    pd.DataFrame(res_logit_con.summary().tables[1]).to_excel(writer, sheet_name='Logit_Con_Dominios')
    pd.DataFrame(res_logit_sin.summary().tables[1]).to_excel(writer, sheet_name='Logit_Sin_Dominios')
    pd.DataFrame(res_wls_con.summary().tables[1]).to_excel(writer, sheet_name='WLS_Con_Dominios')
    pd.DataFrame(res_wls_sin.summary().tables[1]).to_excel(writer, sheet_name='WLS_Sin_Dominios')

print(f'\n¡Comparaciones econométricas guardadas exitosamente en: {excel_path}!')
