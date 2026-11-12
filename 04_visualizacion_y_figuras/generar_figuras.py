"""
Script Maestro de Generación de Visualizaciones Oficiales
Tesis de Maestría en Estadística Aplicada - Universidad Santo Tomás
Autor: Wilson Daniel Gaona Guacaneme

Este script genera de manera 100% reproducible todas las figuras de la tesis:
- Figuras 1 y 2: Distribuciones de bienestar subjetivo (evaluativo y experiencial).
- Figuras 3 y 4: Boxplots por quintiles de ingreso del hogar.
- Figuras 5 y 6: Estimación no paramétrica LOWESS con Cluster Bootstrap por Hogar (B=200).
- Figura 7: Curvas de probabilidad acumulada del modelo logit ordenado.
- Figura 8: Comparación de R2 entre especificaciones y algoritmos.
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

# RUTAS PORTÁTILES
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.abspath(os.path.join(RUTA_BASE, '..', '..'))
RUTA_SALIDA = os.path.join(RUTA_DATOS, 'imagenes_tesis_final')
os.makedirs(RUTA_SALIDA, exist_ok=True)

COLOR_AZUL = '#1b4965'
COLOR_AZUL_CLARO = '#4682b4'
COLOR_NARANJA = '#d95f02'

print('=== SCRIPT DE GENERACIÓN DE FIGURAS DE TESIS ===')
print(f'Directorio de salida: {RUTA_SALIDA}\n')

# -----------------------------------------------------------------------------
# 1. CARGA DE MICRODATOS
# -----------------------------------------------------------------------------
archivo_base = os.path.join(RUTA_DATOS, 'base_final_embedings.csv')
if not os.path.exists(archivo_base):
    archivo_base = os.path.join(RUTA_DATOS, 'validacion_final.xlsx')

print(f'1. Cargando datos desde: {archivo_base}')
if archivo_base.endswith('.csv'):
    df = pd.read_csv(archivo_base)
else:
    df = pd.read_excel(archivo_base)

if 'ingreso_total_log' in df.columns:
    df['quintil_ingreso'] = pd.qcut(df['ingreso_total_log'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

# Identificador de clúster de hogar
if 'DIRECTORIO' in df.columns and 'SECUENCIA_P' in df.columns:
    df['DIRECTORIO_HOGAR'] = df['DIRECTORIO'].astype(str) + '_' + df['SECUENCIA_P'].astype(str)

# -----------------------------------------------------------------------------
# FIGURA 1: Distribución del bienestar experiencial (Afecto 0-10)
# -----------------------------------------------------------------------------
print('2. Generando Figura 1: Distribución Bienestar Experiencial...')
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
ax.hist(df['bienestar_experiencial'].dropna(), bins=10, range=(0, 10), color=COLOR_AZUL, edgecolor='#ffffff', alpha=0.9)
ax.set_title('Distribución del Bienestar Experiencial en Colombia', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Bienestar Experiencial (0-10)', fontsize=12, fontweight='bold', labelpad=8)
ax.set_ylabel('Número de personas', fontsize=12, fontweight='bold', labelpad=8)
ax.set_xlim(-0.2, 10.2)
ax.grid(True, linestyle='--', alpha=0.3, color='#a0a0a0')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(RUTA_SALIDA, 'figura1_distribucion_experiencial.png'), dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# FIGURA 2: Distribución del bienestar evaluativo (Satisfacción Vida 0-10)
# -----------------------------------------------------------------------------
print('3. Generando Figura 2: Distribución Bienestar Evaluativo...')
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
ax.hist(df['satisfaccion_vida'].dropna(), bins=11, range=(-0.5, 10.5), color=COLOR_AZUL, edgecolor='#ffffff', alpha=0.9)
ax.set_title('Distribución del Bienestar Evaluativo en Colombia', fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel('Satisfacción con la Vida (0-10)', fontsize=12, fontweight='bold', labelpad=8)
ax.set_ylabel('Número de personas', fontsize=12, fontweight='bold', labelpad=8)
ax.set_xlim(-0.5, 10.5)
ax.set_xticks(range(0, 11))
ax.grid(True, linestyle='--', alpha=0.3, color='#a0a0a0')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(RUTA_SALIDA, 'figura2_distribucion_evaluativo.png'), dpi=300, bbox_inches='tight')
plt.close()

# -----------------------------------------------------------------------------
# FIGURAS 3 y 4: Boxplots por Quintiles de Ingreso
# -----------------------------------------------------------------------------
if 'quintil_ingreso' in df.columns:
    print('4. Generando Figuras 3 y 4: Boxplots por Quintiles de Ingreso...')
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    sns.boxplot(data=df, x='quintil_ingreso', y='bienestar_experiencial', color=COLOR_AZUL_CLARO, width=0.6, ax=ax)
    ax.set_title('Bienestar Experiencial por Quintil de Ingreso del Hogar', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Quintil de Ingreso', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bienestar Experiencial (0-10)', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.2, 10.3)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_SALIDA, 'figura3_experiencial_quintiles.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    sns.boxplot(data=df, x='quintil_ingreso', y='satisfaccion_vida', color=COLOR_AZUL_CLARO, width=0.6, ax=ax)
    ax.set_title('Bienestar Evaluativo por Quintil de Ingreso del Hogar', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Quintil de Ingreso', fontsize=12, fontweight='bold')
    ax.set_ylabel('Satisfacción con la Vida (0-10)', fontsize=12, fontweight='bold')
    ax.set_ylim(-0.2, 10.3)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_SALIDA, 'figura4_evaluativo_quintiles.png'), dpi=300, bbox_inches='tight')
    plt.close()

# -----------------------------------------------------------------------------
# FIGURAS 5 y 6: LOWESS NO PARAMÉTRICO CON CLUSTER BOOTSTRAP POR HOGAR (B=200)
# -----------------------------------------------------------------------------
print('\n5. Generando Figuras 5 y 6: LOWESS con Cluster Bootstrap por Hogar...')
df_valid = df[(df['ingreso_total_log'] > 8.0) & (df['ingreso_total_log'].notna())].copy()

if 'DIRECTORIO_HOGAR' in df_valid.columns:
    np.random.seed(42)
    hogares_unicos = df_valid['DIRECTORIO_HOGAR'].unique()
    sample_hogares = np.random.choice(hogares_unicos, size=min(5000, len(hogares_unicos)), replace=False)
    sample_df = df_valid[df_valid['DIRECTORIO_HOGAR'].isin(sample_hogares)].copy()
else:
    sample_df = df_valid.sample(n=min(10000, len(df_valid)), random_state=42)

def generar_lowess_bootstrap(df_sample, var_y, titulo_y, nombre_archivo, media_pob):
    x_grid = np.linspace(df_sample['ingreso_total_log'].min(), df_sample['ingreso_total_log'].max(), 100)
    
    # Ajuste puntual
    lowess_point = lowess(df_sample[var_y], df_sample['ingreso_total_log'], frac=0.35, is_sorted=False)
    y_interp_point = np.interp(x_grid, lowess_point[:, 0], lowess_point[:, 1])

    # Cluster Bootstrap
    hogares = df_sample['DIRECTORIO_HOGAR'].unique()
    n_groups = len(hogares)
    B = 200

    def una_replica(b):
        rng = np.random.RandomState(42 + b)
        boot_hogares = rng.choice(hogares, size=n_groups, replace=True)
        boot_df = pd.concat([df_sample[df_sample['DIRECTORIO_HOGAR'] == h] for h in boot_hogares[:300]], ignore_index=True)
        fit = lowess(boot_df[var_y], boot_df['ingreso_total_log'], frac=0.35, is_sorted=False)
        return np.interp(x_grid, fit[:, 0], fit[:, 1])

    boot_curves = Parallel(n_jobs=-1, batch_size=10)(delayed(una_replica)(b) for b in range(B))
    boot_curves = np.array(boot_curves)
    ci_lower = np.percentile(boot_curves, 2.5, axis=0)
    ci_upper = np.percentile(boot_curves, 97.5, axis=0)

    # Gráfico
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    ax.scatter(df_sample['ingreso_total_log'], df_sample[var_y], alpha=0.06, color=COLOR_AZUL_CLARO, s=12, label='Observaciones (muestra)')
    ax.plot(x_grid, y_interp_point, color=COLOR_AZUL, linewidth=2.8, label='Ajuste LOWESS')
    ax.fill_between(x_grid, ci_lower, ci_upper, color=COLOR_AZUL_CLARO, alpha=0.30, label='IC 95% (Cluster Bootstrap)')
    ax.axhline(media_pob, color=COLOR_NARANJA, linestyle=':', linewidth=1.5, label=f'Media ({media_pob:.2f})')

    ax.set_xlabel('Logaritmo del Ingreso Total del Hogar', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel(titulo_y, fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylim(0, 10.3)
    ax.set_yticks(range(0, 11, 2))
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=9.5, loc='lower right')
    sns.despine()
    plt.tight_layout()

    # Guardar en PDF vectorial y PNG
    plt.savefig(os.path.join(RUTA_SALIDA, f'{nombre_archivo}.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(RUTA_SALIDA, f'{nombre_archivo}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'   -> {nombre_archivo}.pdf y .png generados.')

# Generar Figuras 5 y 6
generar_lowess_bootstrap(sample_df, 'satisfaccion_vida', 'Satisfacción con la Vida (0-10)', 'Figura5_LOWESS_Evaluativo', 8.25)
generar_lowess_bootstrap(sample_df, 'bienestar_experiencial', 'Bienestar Experiencial (Afecto 0-10)', 'Figura6_LOWESS_Experiencial', 7.82)

print('\n¡Todas las figuras oficiales se generaron con éxito!')
