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

print('=== MODELOS ESTADÍSTICOS TRADICIONALES DE REFERENCIA (ESPECIFICACIÓN Y PONDERACIÓN HOMOGENIZADA) ===')
print('1. Cargando microdatos depurados de la ECV 2024...')
df = pd.read_excel(os.path.join(RUTA_DATOS, 'validacion_final.xlsx'))
base_raw = pd.read_csv(
    os.path.join(RUTA_DATOS, 'base_final_embedings.csv'),
    usecols=['DIRECTORIO', 'SECUENCIA_P', 'felicidad_ayer', 'preocupacion_ayer', 'tristeza_ayer']
)
df['cluster_hogar'] = base_raw['DIRECTORIO'].astype(str) + '_' + base_raw['SECUENCIA_P'].astype(str)
df['bienestar_experiencial'] = (
    base_raw['felicidad_ayer'] + (10.0 - base_raw['preocupacion_ayer']) + (10.0 - base_raw['tristeza_ayer'])
) / 3.0

cols_explicativas = [
    'ingreso_total_log', 'sexo_nacimiento', 'grupo_edad', 
    'nivel_educativo_alcanzado', 'estado_conyugal', 
    'satisfaccion_salud', 'satisfaccion_trabajo', 'P1_DEPARTAMENTO'
]

print('\n-----------------------------------------------------------------------------')
print('MODELO 1: LOGIT ORDENADO PONDERADO (Bienestar Evaluativo - Satisfaccion Vida)')
print('-----------------------------------------------------------------------------')
X_eval = pd.get_dummies(df[cols_explicativas], drop_first=True, dtype=float)
y_eval = df['satisfaccion_vida'].astype(int)
pesos_eval = df['FEX_C']

print(f'  - Regresores en matriz de diseño X: {X_eval.shape[1]} covariables explicativas')
print(f'  - Ajustando Logit Ordenado Ponderado (FEX_C) con Clúster de Hogar ({len(df):,} obs)...')

# Ajuste con OrderedModel y errores estándar agrupados por hogar
mod_logit = OrderedModel(y_eval, X_eval, distr='logit')
res_logit = mod_logit.fit(method='bfgs', cov_type='cluster', cov_kwds={'groups': df['cluster_hogar']}, disp=False)

print(f'  - Log-Likelihood: {res_logit.llf:.2f}')
print(f'  - Pseudo R2 (McFadden): {1 - (res_logit.llf / res_logit.llnull):.4f}')
print('\n=== RESUMEN COMPLETO MODELO LOGIT ORDENADO ===')
summary_logit = res_logit.summary()
print(summary_logit)

print('\n-----------------------------------------------------------------------------')
print('MODELO 2: REGRESIÓN WLS PONDERADA (Bienestar Experiencial - Afecto)')
print('-----------------------------------------------------------------------------')
X_exp = pd.get_dummies(df[cols_explicativas], drop_first=True, dtype=float)
X_exp = sm.add_constant(X_exp)
y_exp = df['bienestar_experiencial']
pesos_wls = df['FEX_C']

print(f'  - Regresores en matriz de diseño X: {X_exp.shape[1] - 1} covariables + Constante')
print(f'  - Ajustando WLS Ponderado (FEX_C) con Clúster de Hogar ({len(df):,} obs)...')

mod_wls = sm.WLS(y_exp, X_exp, weights=pesos_wls)
res_wls = mod_wls.fit(cov_type='cluster', cov_kwds={'groups': df['cluster_hogar']})

print(f'  - R2 WLS: {res_wls.rsquared:.4f}')
print(f'  - R2 Ajustado: {res_wls.rsquared_adj:.4f}')
print(f'  - F-statistic: {res_wls.fvalue:.2f} (p-value: {res_wls.f_pvalue:.4e})')
print('\n=== RESUMEN COMPLETO MODELO WLS ===')
summary_wls = res_wls.summary()
print(summary_wls)

# Exportar automáticamente a Excel para facilidad de lectura
excel_path = os.path.join(RUTA_DATOS, 'resultados_modelos_tradicionales_completos.xlsx')
with pd.ExcelWriter(excel_path) as writer:
    pd.DataFrame(res_logit.summary().tables[1]).to_excel(writer, sheet_name='Logit_Ordenado')
    pd.DataFrame(res_wls.summary().tables[1]).to_excel(writer, sheet_name='WLS_Experiencial')

print(f'\n¡Resultados completos guardados automáticamente en: {excel_path}!')
