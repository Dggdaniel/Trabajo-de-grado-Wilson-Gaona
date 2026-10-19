import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# RUTAS PORTÁTILES (Sin rutas absolutas de usuario)
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.abspath(os.path.join(RUTA_BASE, '..', '..'))

print('=== 02_SELECCION_Y_DEPURACION_VARIABLES.PY (AUDITADO) ===')
print('1. Cargando base unificada consolidada de la ECV 2024...')
input_file = os.path.join(RUTA_DATOS, 'validacion_final.xlsx')

if not os.path.exists(input_file):
    print(f'Error: No se encuentra el archivo de datos {input_file}')
    exit(1)

df = pd.read_excel(input_file)
print(f'  - Base cargada: {df.shape[0]:,} observaciones y {df.shape[1]} variables iniciales.')

print('2. Aplicando filtro de selección de características (Feature Selection)...')
# Mantenimiento estricto de identificadores de hogar, ponderadores y variables dependientes
cols_obligatorias = ['satisfaccion_vida', 'bienestar_experiencial', 'FEX_C', 'DIRECTORIO', 'SECUENCIA_P']

# Criterio de depuración metodológica:
# - Eliminar variables con más del 80% de datos faltantes (nulos)
# - Eliminar variables constantes sin varianza (nunique <= 1)
cols_validas = []
for col in df.columns:
    if col in cols_obligatorias:
        cols_validas.append(col)
    else:
        pct_missing = df[col].isnull().mean()
        if pct_missing < 0.80 and df[col].nunique() > 1:
            cols_validas.append(col)

df_filtrado = df[cols_validas].copy()

print(f'  - Variables seleccionadas tras filtrado de varianza y completitud: {df_filtrado.shape[1]} columnas.')
print(f'  - Muestra final lista para serialización y embeddings: {len(df_filtrado):,} observaciones.')

print('3. Exportando archivo consolidado final (base_para_embeddings_filtrada_final.csv)...')
out_file_csv = os.path.join(RUTA_DATOS, 'base_para_embeddings_filtrada_final.csv')
df_filtrado.to_csv(out_file_csv, index=False, encoding='utf-8')

print(f'¡Selección de variables finalizada con 100% de rigor empírico! Exportado a: {out_file_csv}')
