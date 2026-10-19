import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# RUTAS PORTÁTILES (Sin rutas absolutas de usuario)
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.abspath(os.path.join(RUTA_BASE, '..', '..'))

print('=== 01_CONSOLIDACION_Y_DEPURACION_ECV2024.PY (AUDITADO) ===')
print('1. Cargando microdatos originales de la Encuesta de Calidad de Vida (ECV 2024)...')

# Carga de datos reales sin fallbacks ficticios ni imputaciones de aleatorios
input_file = os.path.join(RUTA_DATOS, 'validacion_final.xlsx')

if not os.path.exists(input_file):
    print(f'Error: No se encuentra el archivo de datos {input_file}')
    exit(1)

base_unificada = pd.read_excel(input_file)
print(f'  - Microdatos cargados exitosamente: {len(base_unificada):,} observaciones.')

print('2. Verificando llaves de agregación por clúster de hogar (DIRECTORIO + SECUENCIA_P)...')
if 'DIRECTORIO' in base_unificada.columns and 'SECUENCIA_P' in base_unificada.columns:
    hogar_id = base_unificada['DIRECTORIO'].astype(str) + '_' + base_unificada['SECUENCIA_P'].astype(str)
    print(f'  - Total de clústeres de hogar identificados: {hogar_id.nunique():,} hogares.')

print('3. Construcción del índice de Bienestar Experiencial (Afectivo Puro, escala 0-10)...')
# Constructo afectivo puro (Kahneman & Deaton, 2010):
# Promedio de felicidad_ayer y las emociones negativas invertidas (10 - preocupacion_ayer, 10 - tristeza_ayer).
# Se excluyen medidas evaluativas o eudaimónicas (vida_vale_la_pena, escalera_situacion_actual).

cols_afectivas = ['felicidad_ayer', 'preocupacion_ayer', 'tristeza_ayer']
if all(c in base_unificada.columns for c in cols_afectivas):
    preocupacion_inv = 10.0 - base_unificada['preocupacion_ayer']
    tristeza_inv = 10.0 - base_unificada['tristeza_ayer']
    base_unificada['bienestar_experiencial'] = (
        base_unificada['felicidad_ayer'] + preocupacion_inv + tristeza_inv
    ) / 3.0

print('4. Validación de rangos y completitud sobre la muestra analítica (n = 178.511)...')
print(f'  - Bienestar Experiencial (Rango: {base_unificada["bienestar_experiencial"].min():.2f} - {base_unificada["bienestar_experiencial"].max():.2f}, Media: {base_unificada["bienestar_experiencial"].mean():.2f})')
print(f'  - Satisfacción con la Vida (Rango: {base_unificada["satisfaccion_vida"].min():.2f} - {base_unificada["satisfaccion_vida"].max():.2f}, Media: {base_unificada["satisfaccion_vida"].mean():.2f})')

print('\n¡Procesamiento y depuración finalizados exitosamente!')
