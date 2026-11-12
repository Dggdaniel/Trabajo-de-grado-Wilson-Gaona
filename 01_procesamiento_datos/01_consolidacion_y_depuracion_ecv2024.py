import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# RUTAS PORTÁTILES (Sin rutas absolutas de usuario)
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.abspath(os.path.join(RUTA_BASE, '..', '..'))

print('=== 01_CONSOLIDACION_Y_DEPURACION_ECV2024.PY ===')
print('1. Cargando microdatos de la Encuesta de Calidad de Vida (ECV 2024)...')

input_file = os.path.join(RUTA_DATOS, 'validacion_final.xlsx')

if not os.path.exists(input_file):
    print(f'Error: No se encuentra el archivo de datos {input_file}')
    exit(1)

base_unificada = pd.read_excel(input_file)
print(f'  - Microdatos cargados exitosamente: {len(base_unificada):,} observaciones.')

print('2. Verificando llaves de agregación por clúster de hogar (DIRECTORIO + SECUENCIA_P)...')
if 'DIRECTORIO' in base_unificada.columns and 'SECUENCIA_P' in base_unificada.columns:
    hogar_id = base_unificada['DIRECTORIO'].astype(str) + '_' + base_unificada['SECUENCIA_P'].astype(str)
    n_hogares = hogar_id.nunique()
    print(f'  - Total de clústeres de hogar identificados: {n_hogares:,} hogares.')
    print(f'  - Promedio de personas por hogar: {len(base_unificada) / n_hogares:.2f}')

print('3. Construcción y auditoría psicométrica del índice de Bienestar Experiencial (Afectivo Puro)...')
# Constructo afectivo puro (Kahneman & Deaton, 2010):
# Promedio de felicidad_ayer y las emociones negativas invertidas (10 - preocupacion_ayer, 10 - tristeza_ayer).
cols_afectivas = ['felicidad_ayer', 'preocupacion_ayer', 'tristeza_ayer']
if all(c in base_unificada.columns for c in cols_afectivas):
    preocupacion_inv = 10.0 - base_unificada['preocupacion_ayer']
    tristeza_inv = 10.0 - base_unificada['tristeza_ayer']
    base_unificada['bienestar_experiencial'] = (
        base_unificada['felicidad_ayer'] + preocupacion_inv + tristeza_inv
    ) / 3.0
    
    # Cálculo de Consistencia Interna (Alfa de Cronbach)
    df_items = pd.DataFrame({
        'felicidad': base_unificada['felicidad_ayer'],
        'no_preocupacion': preocupacion_inv,
        'no_tristeza': tristeza_inv
    })
    k = df_items.shape[1]
    var_items = df_items.var(axis=0, ddof=1).sum()
    var_total = df_items.sum(axis=1).var(ddof=1)
    cronbach_alpha = (k / (k - 1.0)) * (1.0 - (var_items / var_total))
    
    corr_spearman = base_unificada[['bienestar_experiencial', 'satisfaccion_vida']].corr(method='spearman').iloc[0, 1]
    corr_pearson = base_unificada[['bienestar_experiencial', 'satisfaccion_vida']].corr(method='pearson').iloc[0, 1]
    
    print(f'  - Consistencia Interna (Alfa de Cronbach): {cronbach_alpha:.4f}')
    print(f'  - Correlación con Satisfacción con la Vida: Pearson = {corr_pearson:.4f}, Spearman = {corr_spearman:.4f}')

print('\n4. Validación de completitud muestral:')
print(f'  - Total registros analíticos: {len(base_unificada):,}')
print(f'  - Bienestar Experiencial (Media: {base_unificada["bienestar_experiencial"].mean():.2f}, DE: {base_unificada["bienestar_experiencial"].std():.2f})')
print(f'  - Satisfacción con la Vida (Media: {base_unificada["satisfaccion_vida"].mean():.2f}, DE: {base_unificada["satisfaccion_vida"].std():.2f})')

print('\n¡Procesamiento y depuración finalizados exitosamente!')
