import pandas as pd
import numpy as np
import os
import re
import sys

# RUTA RELATIVA Y PORTÁTIL (Sin rutas absolutas de usuario)
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.abspath(os.path.join(RUTA_BASE, '..', '..'))

# Uso: python 01_preparacion_datos_serializacion.py [n_filas_prueba]
LIMITE = int(sys.argv[1]) if len(sys.argv) > 1 else None

# =============================================================================
# VARIABLES EXCLUIDAS DEL TEXTO (DROP ANTES DE GENERAR EMBEDDINGS)
# El embedding solo debe contener las variables RESTANTES del hogar.
# Las variables principales van en la matriz tabular del modelo tradicional.
# =============================================================================

# 1. Identificadores, orden de encuesta y ponderador
EXCLUIR_IDENTIFICADORES = [
    'Unnamed: 0', 'DIRECTORIO', 'SECUENCIA_P', 'SECUENCIA_ENCUESTA', 'ORDEN',
    'orden_informante_x', 'orden_informante_y', 'FEX_C', 'FEX', 'PROB_IAG',
    'tipo_documento', 'quien_contesta_capitulo'
]

# 2. Variables dependientes y sus componentes (evita fuga de información)
EXCLUIR_DEPENDIENTES = [
    'satisfaccion_vida',
    'bienestar_experiencial', 'bienestar_experiencial_entero',
    'felicidad_ayer', 'tristeza_ayer', 'preocupacion_ayer',
    'vida_vale_la_pena', 'escalera_situacion_actual', 'auto_percepcion',
    'afecto_positivo', 'afecto_negativo',
    'preocupacion_ayer_inv', 'tristeza_ayer_inv', 'afecto_negativo_inv',
]

# 3. Variables principales del modelo tradicional (02_evaluacion_group_kfold.py)
EXCLUIR_TRADICIONALES = [
    'sexo_nacimiento', 'nivel_educativo_alcanzado', 'ingreso_total_log',
    'estado_conyugal', 'satisfaccion_trabajo', 'satisfaccion_salud',
    'P1_DEPARTAMENTO', 'grupo_edad', 'satisfaccion_ingresos', 
    'satisfaccion_tiempo_libre', 'satisfaccion_seguridad'
]

# 4. Derivaciones directas de las variables principales
#    (repetirían la misma información con otro nombre)
EXCLUIR_DERIVADAS = [
    # Ingreso (ingreso_total_log se construye a partir de estos montos)
    'ingreso', 'ingreso_mensual_hogar', 'ingreso_mensual_unidad_gasto',
    'ingreso_mensual_empleo', 'ingreso_mensual_otra_unidad',
    'ingreso_total', 'ingreso_per_capita',
    # Edad (grupo_edad se construye a partir de estas)
    'edad', 'fecha_nacimiento',
    # Educación (nivel_educativo_alcanzado resume el grado aprobado)
    'grado_aprobado',
    # Departamento (REGION es una agregación de P1_DEPARTAMENTO)
    'REGION',
]

COLUMNAS_EXCLUIR = set(
    EXCLUIR_IDENTIFICADORES + EXCLUIR_DEPENDIENTES
    + EXCLUIR_TRADICIONALES + EXCLUIR_DERIVADAS
)

# Componentes MCA / PCA: nunca deben entrar al texto
PATRON_MCA_PCA = re.compile(r'^(MC|PC)\d+$')

def excluida(clave):
    return clave in COLUMNAS_EXCLUIR or bool(PATRON_MCA_PCA.match(clave))

print('=== 01. PREPARACIÓN DE DATOS Y SERIALIZACIÓN DE HOGARES ===')
# Fuente: textos con las respuestas ETIQUETADAS ("Sí", "Nunca", "Baldosa"...).
# Los códigos numéricos de la ECV no tienen significado para un modelo de lenguaje.
input_file = os.path.join(RUTA_DATOS, 'base_para_embendings.xlsx')
print(f'Cargando textos etiquetados: {input_file}')
df = pd.read_excel(input_file, nrows=LIMITE)
textos_originales = df['texto_embedding'].fillna('').astype(str)
print(f'  - {len(df):,} observaciones cargadas.')

# Llaves de hogar (mismo orden de filas que validacion_final.xlsx y los embeddings)
ids = pd.read_csv(os.path.join(RUTA_DATOS, 'base_final_embedings.csv'),
                  usecols=['DIRECTORIO', 'SECUENCIA_P'], nrows=LIMITE)
assert len(ids) == len(df), f'Filas no coinciden: textos={len(df)} vs llaves={len(ids)}'

claves_vistas, claves_quitadas = set(), set()

def filtrar_texto(texto):
    partes_ok = []
    for parte in texto.split('; '):
        if ': ' not in parte:
            continue
        clave = parte.split(': ', 1)[0].strip()
        claves_vistas.add(clave)
        if excluida(clave):
            claves_quitadas.add(clave)
        else:
            partes_ok.append(parte)
    return '; '.join(partes_ok)

print('Eliminando variables principales, dependientes y MCA/PCA del texto...')
textos_serializados = textos_originales.apply(filtrar_texto)

claves_en_texto = sorted(claves_vistas - claves_quitadas)
print(f'  - Variables quitadas del texto: {len(claves_quitadas)}')
print(f'  - Variables que quedan en el texto: {len(claves_en_texto)}')

# Auditoría final: ninguna variable excluida puede quedar en el texto
patron_auditoria = '|'.join(rf'(?:^|; ){re.escape(k)}: ' for k in sorted(COLUMNAS_EXCLUIR))
fuga = textos_serializados.str.contains(patron_auditoria, regex=True)
assert not fuga.any(), f'{fuga.sum()} textos aún contienen variables excluidas.'
print('  - Auditoría OK: ninguna variable principal, dependiente ni MCA/PCA aparece en el texto.')

df_out = pd.DataFrame({
    'DIRECTORIO': ids['DIRECTORIO'],
    'SECUENCIA_P': ids['SECUENCIA_P'],
    'texto_serializado': textos_serializados,
})

# Registro de variables incluidas y excluidas (para el anexo metodológico)
pd.DataFrame({
    'variable': claves_en_texto + sorted(claves_quitadas),
    'en_texto': [True] * len(claves_en_texto) + [False] * len(claves_quitadas),
}).to_csv(os.path.join(RUTA_DATOS, 'variables_serializacion.csv'), index=False, encoding='utf-8')

nombre_salida = 'base_textos_serializados.csv' if LIMITE is None else f'base_textos_serializados_prueba_{LIMITE}.csv'
output_file = os.path.join(RUTA_DATOS, nombre_salida)
df_out.to_csv(output_file, index=False, encoding='utf-8')
print(f'¡Preprocesamiento finalizado! Textos serializados guardados en: {output_file}')
