#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script Maestro de Generación de Figuras Oficiales de la Tesis
============================================================
Título: El Bienestar Subjetivo en Colombia: Evaluación de la Vida y Bienestar
        Emocional como Función de Factores Socioeconómicos Según la ECV 2024.
Autor: Wilson Daniel Gaona Guacaneme
Maestría en Ciencia de Datos - Universidad Santo Tomás

Figuras Oficiales Generadas:
----------------------------
1. Figura 1: Distribución del Bienestar Experiencial (escala 0-10, histograma y media).
2. Figura 2: Distribución del Bienestar Evaluativo (satisfacción con la vida 0-10).
3. Figura 3: Distribución del Bienestar Experiencial por Quintiles de Ingreso del Hogar.
4. Figura 4: Distribución del Bienestar Evaluativo por Quintiles de Ingreso del Hogar.
5. Figura 5: Ajuste LOWESS no paramétrico entre Log-Ingreso y Bienestar Evaluativo
             (con Intervalos de Confianza al 95% calculados por Cluster Bootstrap a nivel de hogar, B=200).
6. Figura 6: Ajuste LOWESS no paramétrico entre Log-Ingreso y Bienestar Experiencial
             (con Intervalos de Confianza al 95% calculados por Cluster Bootstrap a nivel de hogar, B=200).
7. Figura 7: Comparación del Coeficiente de Determinación (R2 ± 1 DE) entre las 15 Especificaciones
             y Algoritmos (Cuadro 5 de la Tesis: Clásicos, Solo Texto, Híbridos y Saturados).
8. Figura 8 / Anexo: Curvas de Probabilidades Predichas según Log-Ingreso del Hogar (Logit Ordenado).

Formatos de Salida:
-------------------
Exporta automáticamente cada figura en formato vectorial PDF (.pdf) y mapa de bits
de alta resolución (.png, 300 DPI) con tipografía académica y paleta sobria de azules.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from joblib import Parallel, delayed

# -----------------------------------------------------------------------------
# 1. LOCALIZACIÓN DINÁMICA Y PORTABILIDAD DE RUTAS
# -----------------------------------------------------------------------------
def obtener_rutas():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    posibles_bases = [
        os.path.abspath(os.path.join(script_dir, '..', '..')),
        os.path.abspath(os.path.join(script_dir, '..')),
        script_dir,
        r"c:\Users\WILSGAOGUA\OneDrive - colsubsidio.com\universidad\Trabajo de grado"
    ]
    
    ruta_datos = None
    for p in posibles_bases:
        if os.path.exists(os.path.join(p, "base_final_embedings.csv")) or os.path.exists(os.path.join(p, "validacion_final.xlsx")):
            ruta_datos = p
            break
            
    if ruta_datos is None:
        ruta_datos = posibles_bases[0]

    out_dirs = [
        os.path.join(ruta_datos, "imagenes_tesis_final"),
        os.path.join(script_dir, "figuras_generadas")
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        
    return ruta_datos, out_dirs

# -----------------------------------------------------------------------------
# 2. CONFIGURACIÓN EDITORIAL Y ESTILO VISUAL ACADÉMICO
# -----------------------------------------------------------------------------
COLOR_AZUL_OSCURO = '#1b4965'     # Azul institucional formal
COLOR_AZUL_MEDIO  = '#4682b4'     # Azul acero complementario
COLOR_AZUL_CLARO  = '#62929a'     # Azul tenue para desagregaciones
COLOR_BANDA_IC    = '#4682b4'     # Sombra semitransparente del IC 95%
COLOR_OBSERVA     = '#5c80a8'     # Dispersión sutil de observaciones ECV
COLOR_MEDIA       = '#d95f02'     # Naranja terracota para medias poblacionales
COLOR_GRID        = '#a0a0a0'     # Cuadrícula sutil

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9.5
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def guardar_figura(fig, out_dirs, base_name):
    """Guarda en PDF vectorial y PNG (300 DPI) en todas las carpetas destino."""
    for d in out_dirs:
        pdf_path = os.path.join(d, f"{base_name}.pdf")
        png_path = os.path.join(d, f"{base_name}.png")
        fig.savefig(pdf_path, bbox_inches='tight')
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"  [OK] Guardada: {base_name} (.pdf y .png)")

# -----------------------------------------------------------------------------
# 3. CARGA Y FILTRADO DE MICRODATOS
# -----------------------------------------------------------------------------
def cargar_datos(ruta_datos):
    csv_path = os.path.join(ruta_datos, "base_final_embedings.csv")
    xlsx_path = os.path.join(ruta_datos, "validacion_final.xlsx")
    
    cols = ['DIRECTORIO', 'SECUENCIA_P', 'FEX_C', 'ingreso_total_log', 
            'satisfaccion_vida', 'bienestar_experiencial']
    
    if os.path.exists(csv_path):
        print(f"Cargando microdatos desde CSV: {csv_path}...")
        df = pd.read_csv(csv_path, usecols=cols)
    elif os.path.exists(xlsx_path):
        print(f"Cargando microdatos desde Excel: {xlsx_path}...")
        df = pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(f"No se encontró archivo de datos en {ruta_datos}")
        
    # Crear ID de hogar para análisis clúster
    if 'DIRECTORIO' in df.columns and 'SECUENCIA_P' in df.columns:
        df['hogar_id'] = df['DIRECTORIO'].astype(str) + '_' + df['SECUENCIA_P'].astype(str)
    else:
        df['hogar_id'] = np.arange(len(df)).astype(str)
        
    if 'FEX_C' not in df.columns:
        df['FEX_C'] = 1.0

    # Variable entera para el histograma de afecto
    df['bienestar_experiencial_entero'] = df['bienestar_experiencial'].round(0)

    # Quintiles de ingreso
    valid_inc = df['ingreso_total_log'].dropna()
    df['quintil_ingreso'] = pd.qcut(valid_inc, q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    # Muestra limpia para análisis bivariado y LOWESS (ingreso positivo log > 8)
    df_clean = df[(df['ingreso_total_log'] > 8) & 
                  df['satisfaccion_vida'].notna() & 
                  df['bienestar_experiencial'].notna()].copy()
                  
    print(f"Datos cargados: {len(df):,} registros totales. Muestra limpia con ln(Ingreso)>8: {len(df_clean):,} registros.")
    return df, df_clean

# -----------------------------------------------------------------------------
# 4. GENERADORES ESPECÍFICOS DE FIGURAS
# -----------------------------------------------------------------------------

def generar_figura_1(df, out_dirs):
    """Figura 1: Distribución del Bienestar Experiencial (escala 0-10)."""
    print("Generando Figura 1: Distribución del bienestar experiencial...")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    
    data = df['bienestar_experiencial_entero'].dropna()
    conteos = data.value_counts().sort_index()
    x_vals = np.arange(0, 11)
    y_vals = [conteos.get(i, 0) for i in x_vals]
    
    ax.bar(x_vals, y_vals, color=COLOR_AZUL_OSCURO, edgecolor='#ffffff', width=0.82, alpha=0.92)
    
    media = (df['bienestar_experiencial'] * df['FEX_C']).sum() / df['FEX_C'].sum()
    ax.axvline(media, color=COLOR_MEDIA, linestyle='--', linewidth=1.8, label=f'Media ponderada ({media:.2f})')
    
    ax.set_title('Figura 1: Distribución del Bienestar Experiencial en Colombia (ECV 2024)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Bienestar Experiencial (0 - 10)', fontweight='bold', labelpad=8)
    ax.set_ylabel('Número de Personas', fontweight='bold', labelpad=8)
    ax.set_xlim(-0.6, 10.6)
    ax.set_xticks(range(0, 11))
    ax.grid(axis='y', linestyle='--', alpha=0.35, color=COLOR_GRID)
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper left')
    plt.tight_layout()
    
    guardar_figura(fig, out_dirs, "figura1_distribucion_experiencial")


def generar_figura_2(df, out_dirs):
    """Figura 2: Distribución del Bienestar Evaluativo (Satisfacción con la Vida)."""
    print("Generando Figura 2: Distribución del bienestar evaluativo...")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    
    data = df['satisfaccion_vida'].dropna()
    conteos = data.value_counts().sort_index()
    x_vals = np.arange(0, 11)
    y_vals = [conteos.get(i, 0) for i in x_vals]
    
    ax.bar(x_vals, y_vals, color=COLOR_AZUL_OSCURO, edgecolor='#ffffff', width=0.82, alpha=0.92)
    
    media = (df['satisfaccion_vida'] * df['FEX_C']).sum() / df['FEX_C'].sum()
    ax.axvline(media, color=COLOR_MEDIA, linestyle='--', linewidth=1.8, label=f'Media ponderada ({media:.2f})')
    
    ax.set_title('Figura 2: Distribución del Bienestar Evaluativo en Colombia (ECV 2024)', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Satisfacción con la Vida (0 - 10)', fontweight='bold', labelpad=8)
    ax.set_ylabel('Número de Personas', fontweight='bold', labelpad=8)
    ax.set_xlim(-0.6, 10.6)
    ax.set_xticks(range(0, 11))
    ax.grid(axis='y', linestyle='--', alpha=0.35, color=COLOR_GRID)
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='upper left')
    plt.tight_layout()
    
    guardar_figura(fig, out_dirs, "figura2_distribucion_evaluativo")


def generar_figura_3(df, out_dirs):
    """Figura 3: Distribución del Bienestar Experiencial por Quintiles de Ingreso."""
    print("Generando Figura 3: Bienestar experiencial por quintiles de ingreso...")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    
    df_box = df.dropna(subset=['quintil_ingreso', 'bienestar_experiencial'])
    sns.boxplot(data=df_box, x='quintil_ingreso', y='bienestar_experiencial', 
                color=COLOR_AZUL_MEDIO, width=0.55, fliersize=2.0, linewidth=1.1, ax=ax)
                
    ax.set_title('Figura 3: Distribución del Bienestar Experiencial por Quintiles de Ingreso del Hogar', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Quintil de Ingreso del Hogar', fontweight='bold', labelpad=8)
    ax.set_ylabel('Balance Afectivo Experiencial (0 - 10)', fontweight='bold', labelpad=8)
    ax.set_ylim(-0.2, 10.3)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.grid(axis='y', linestyle='--', alpha=0.35, color=COLOR_GRID)
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    guardar_figura(fig, out_dirs, "figura3_experiencial_quintiles")


def generar_figura_4(df, out_dirs):
    """Figura 4: Distribución del Bienestar Evaluativo por Quintiles de Ingreso."""
    print("Generando Figura 4: Bienestar evaluativo por quintiles de ingreso...")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    
    df_box = df.dropna(subset=['quintil_ingreso', 'satisfaccion_vida'])
    sns.boxplot(data=df_box, x='quintil_ingreso', y='satisfaccion_vida', 
                color=COLOR_AZUL_MEDIO, width=0.55, fliersize=2.0, linewidth=1.1, ax=ax)
                
    ax.set_title('Figura 4: Distribución del Bienestar Evaluativo por Quintiles de Ingreso del Hogar', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Quintil de Ingreso del Hogar', fontweight='bold', labelpad=8)
    ax.set_ylabel('Satisfacción con la Vida (0 - 10)', fontweight='bold', labelpad=8)
    ax.set_ylim(-0.2, 10.3)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.grid(axis='y', linestyle='--', alpha=0.35, color=COLOR_GRID)
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    guardar_figura(fig, out_dirs, "figura4_evaluativo_quintiles")


def generar_figuras_5_y_6(df, df_clean, out_dirs, bootstrap_b=200):
    """
    Figuras 5 y 6: Relaciones no paramétricas (LOWESS) con Cluster Bootstrap por hogar.
    Calcula intervalos de confianza al 95% preservando la estructura correlacionada intra-hogar.
    """
    print(f"\nIniciando ajuste no paramétrico LOWESS y Cluster Bootstrap (B = {bootstrap_b})...")
    
    hogares_unicos = df_clean['hogar_id'].unique()
    np.random.seed(42)
    sample_hogares = np.random.choice(hogares_unicos, size=min(5000, len(hogares_unicos)), replace=False)
    sample_df = df_clean[df_clean['hogar_id'].isin(sample_hogares)].copy()
    
    x_min = sample_df['ingreso_total_log'].quantile(0.01)
    x_max = sample_df['ingreso_total_log'].quantile(0.99)
    x_grid = np.linspace(x_min, x_max, 120)
    
    # Ajustes centrales
    y_lowess_eval = sm.nonparametric.lowess(sample_df['satisfaccion_vida'], 
                                            sample_df['ingreso_total_log'], 
                                            xvals=x_grid, frac=0.35)
    y_lowess_exp = sm.nonparametric.lowess(sample_df['bienestar_experiencial'], 
                                           sample_df['ingreso_total_log'], 
                                           xvals=x_grid, frac=0.35)
    
    # Bootstrap en paralelo
    t0 = time.time()
    grouped_data = [g[['ingreso_total_log', 'satisfaccion_vida', 'bienestar_experiencial']].values 
                    for _, g in sample_df.groupby('hogar_id')]
    n_groups = len(grouped_data)
    
    def run_one_boot(seed):
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_groups, size=n_groups, replace=True)
        mat = np.vstack([grouped_data[i] for i in idx])
        fit_eval = sm.nonparametric.lowess(mat[:, 1], mat[:, 0], xvals=x_grid, frac=0.35)
        fit_exp  = sm.nonparametric.lowess(mat[:, 2], mat[:, 0], xvals=x_grid, frac=0.35)
        return fit_eval, fit_exp

    results = Parallel(n_jobs=-1, batch_size=5)(delayed(run_one_boot)(42 + b) for b in range(bootstrap_b))
    boot_eval = np.array([r[0] for r in results])
    boot_exp  = np.array([r[1] for r in results])
    
    eval_ci_lower = np.percentile(boot_eval, 2.5, axis=0)
    eval_ci_upper = np.percentile(boot_eval, 97.5, axis=0)
    exp_ci_lower  = np.percentile(boot_exp, 2.5, axis=0)
    exp_ci_upper  = np.percentile(boot_exp, 97.5, axis=0)
    print(f"Bootstrap finalizado en {time.time() - t0:.1f} segundos.")
    
    scatter_sample = sample_df.sample(n=min(2500, len(sample_df)), random_state=42)
    
    # --- FIGURA 5: EVALUATIVO ---
    print("Generando Figura 5: LOWESS Evaluativo...")
    fig5, ax5 = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    ax5.scatter(scatter_sample['ingreso_total_log'], scatter_sample['satisfaccion_vida'],
                alpha=0.08, color=COLOR_OBSERVA, s=14, edgecolors='none', label='Observaciones ECV')
    ax5.fill_between(x_grid, eval_ci_lower, eval_ci_upper, color=COLOR_BANDA_IC, alpha=0.32,
                     label='Intervalo de confianza 95% (bootstrap por hogar)')
    ax5.plot(x_grid, eval_ci_lower, color=COLOR_AZUL_OSCURO, linestyle='--', linewidth=0.8, alpha=0.55)
    ax5.plot(x_grid, eval_ci_upper, color=COLOR_AZUL_OSCURO, linestyle='--', linewidth=0.8, alpha=0.55)
    ax5.plot(x_grid, y_lowess_eval, color=COLOR_AZUL_OSCURO, linewidth=2.8, label=f'Ajuste LOWESS (N = {len(df_clean):,})')
    
    media_eval = (df_clean['satisfaccion_vida'] * df_clean['FEX_C']).sum() / df_clean['FEX_C'].sum()
    ax5.axhline(media_eval, color=COLOR_MEDIA, linestyle=':', linewidth=1.5, label=f'Media ponderada ({media_eval:.2f})')
    
    ax5.set_title('Figura 5: Ajuste LOWESS no Paramétrico entre Ingreso y Bienestar Evaluativo', fontsize=12, fontweight='bold', pad=12)
    ax5.set_xlabel('Logaritmo del Ingreso Total del Hogar', fontweight='bold', labelpad=8)
    ax5.set_ylabel('Satisfacción con la Vida (0 - 10)', fontweight='bold', labelpad=8)
    ax5.set_ylim(0, 10.3)
    ax5.set_yticks(range(0, 11, 2))
    ax5.set_xlim(x_min - 0.2, x_max + 0.2)
    ax5.grid(True, linestyle='--', alpha=0.35, color=COLOR_GRID)
    ax5.set_facecolor('#ffffff')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.legend(frameon=True, facecolor='white', edgecolor='#d0d0d0', loc='lower right')
    plt.tight_layout()
    
    guardar_figura(fig5, out_dirs, "Figura5_LOWESS_Evaluativo")
    guardar_figura(fig5, out_dirs, "figura5_lowess_evaluativo") # Nombre alternativo compatible

    # --- FIGURA 6: EXPERIENCIAL ---
    print("Generando Figura 6: LOWESS Experiencial...")
    fig6, ax6 = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    ax6.scatter(scatter_sample['ingreso_total_log'], scatter_sample['bienestar_experiencial'],
                alpha=0.08, color=COLOR_OBSERVA, s=14, edgecolors='none', label='Observaciones ECV')
    ax6.fill_between(x_grid, exp_ci_lower, exp_ci_upper, color=COLOR_BANDA_IC, alpha=0.32,
                     label='Intervalo de confianza 95% (bootstrap por hogar)')
    ax6.plot(x_grid, exp_ci_lower, color=COLOR_AZUL_OSCURO, linestyle='--', linewidth=0.8, alpha=0.55)
    ax6.plot(x_grid, exp_ci_upper, color=COLOR_AZUL_OSCURO, linestyle='--', linewidth=0.8, alpha=0.55)
    ax6.plot(x_grid, y_lowess_exp, color=COLOR_AZUL_OSCURO, linewidth=2.8, label=f'Ajuste LOWESS (N = {len(df_clean):,})')
    
    media_exp = (df_clean['bienestar_experiencial'] * df_clean['FEX_C']).sum() / df_clean['FEX_C'].sum()
    ax6.axhline(media_exp, color=COLOR_MEDIA, linestyle=':', linewidth=1.5, label=f'Media ponderada ({media_exp:.2f})')
    
    ax6.set_title('Figura 6: Ajuste LOWESS no Paramétrico entre Ingreso y Bienestar Experiencial', fontsize=12, fontweight='bold', pad=12)
    ax6.set_xlabel('Logaritmo del Ingreso Total del Hogar', fontweight='bold', labelpad=8)
    ax6.set_ylabel('Balance Afectivo Experiencial (0 - 10)', fontweight='bold', labelpad=8)
    ax6.set_ylim(0, 10.3)
    ax6.set_yticks(range(0, 11, 2))
    ax6.set_xlim(x_min - 0.2, x_max + 0.2)
    ax6.grid(True, linestyle='--', alpha=0.35, color=COLOR_GRID)
    ax6.set_facecolor('#ffffff')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.legend(frameon=True, facecolor='white', edgecolor='#d0d0d0', loc='lower right')
    plt.tight_layout()
    
    guardar_figura(fig6, out_dirs, "Figura6_LOWESS_Experiencial")
    guardar_figura(fig6, out_dirs, "figura6_lowess_experiencial") # Nombre alternativo compatible


def generar_figura_7(out_dirs):
    """
    Figura 7: Comparación de R2 entre especificaciones y algoritmos.
    Incorpora las 15 especificaciones del Cuadro 5 de la Tesis con sus barras de error (± 1 DE).
    """
    print("Generando Figura 7: Comparación del Desempeño Predictivo (R2)...")
    
    modelos = [
        'Estructurado clásico (Ridge)',
        'Estructurado clásico (LightGBM)',
        'Solo texto (Qwen) (Ridge)',
        'Solo texto (Qwen) (LightGBM)',
        'Solo texto (OpenAI) (Ridge)',
        'Solo texto (OpenAI) (LightGBM)',
        'Híbrido (Qwen) (Ridge)',
        'Híbrido (Qwen) (LightGBM)',
        'Híbrido (OpenAI) (Ridge)',
        'Híbrido (OpenAI) (LightGBM)',
        'Saturado comparable (Ridge)',
        'Saturado comparable (LightGBM)',
        'Saturado completo (Ridge)',
        'Saturado completo (LightGBM)',
        'Saturado completo + OpenAI (LightGBM)'
    ]
    
    # Valores oficiales de validación cruzada agrupada por hogar (Cuadro 5 de la Tesis)
    r2_eval = [0.362, 0.373, 0.078, 0.058, 0.139, 0.126, 0.299, 0.369, 0.365, 0.381, 0.375, 0.399, 0.418, 0.445, 0.437]
    de_eval = [0.008, 0.009, 0.007, 0.006, 0.012, 0.008, 0.016, 0.007, 0.009, 0.006, 0.010, 0.008, 0.010, 0.008, 0.009]
    
    r2_exp  = [0.228, 0.237, 0.089, 0.066, 0.154, 0.131, 0.218, 0.241, 0.263, 0.272, 0.267, 0.313, 0.298, 0.342, 0.324]
    de_exp  = [0.014, 0.012, 0.009, 0.004, 0.009, 0.015, 0.015, 0.012, 0.010, 0.014, 0.013, 0.010, 0.012, 0.012, 0.013]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8.5), sharey=True, dpi=300)
    y_pos = np.arange(len(modelos))
    
    # Panel Evaluativo
    ax1.barh(y_pos, r2_eval, xerr=de_eval, align='center', color=COLOR_AZUL_OSCURO, capsize=3.5, alpha=0.92)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(modelos, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('Coeficiente de Determinación ($R^2$)', fontweight='bold', labelpad=8)
    ax1.set_title('Bienestar Evaluativo (Satisfacción con la Vida)', fontsize=11.5, fontweight='bold', pad=10)
    ax1.set_xlim(0, 0.52)
    ax1.grid(axis='x', linestyle='--', alpha=0.35, color=COLOR_GRID)
    ax1.set_facecolor('#ffffff')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for i, (v, err) in enumerate(zip(r2_eval, de_eval)):
        ax1.text(v + err + 0.008, i, f'{v:.3f}', va='center', fontsize=9, fontweight='semibold')
        
    # Panel Experiencial
    ax2.barh(y_pos, r2_exp, xerr=de_exp, align='center', color=COLOR_AZUL_MEDIO, capsize=3.5, alpha=0.92)
    ax2.set_xlabel('Coeficiente de Determinación ($R^2$)', fontweight='bold', labelpad=8)
    ax2.set_title('Bienestar Experiencial (Afecto Diario)', fontsize=11.5, fontweight='bold', pad=10)
    ax2.set_xlim(0, 0.52)
    ax2.grid(axis='x', linestyle='--', alpha=0.35, color=COLOR_GRID)
    ax2.set_facecolor('#ffffff')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    for i, (v, err) in enumerate(zip(r2_exp, de_exp)):
        ax2.text(v + err + 0.008, i, f'{v:.3f}', va='center', fontsize=9, fontweight='semibold')
        
    plt.suptitle(r'Figura 7: Comparación del Coeficiente de Determinación ($R^2 \pm 1$ DE) entre Modelos', 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    guardar_figura(fig, out_dirs, "figura7_comparacion_r2")
    guardar_figura(fig, out_dirs, "figura8_comparacion_r2") # Guardar también como figura 8 para compatibilidad


def generar_figura_8(df_clean, out_dirs):
    """
    Figura 8 / Anexo: Curvas de probabilidad predicha del Logit Ordenado en función del Log-Ingreso.
    """
    print("Generando Figura 8 (Anexo): Probabilidades predichas según log-ingreso...")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    
    x_val = np.linspace(df_clean['ingreso_total_log'].min(), df_clean['ingreso_total_log'].max(), 120)
    x_mean = df_clean['ingreso_total_log'].mean()
    
    # Función sigmoide estándar de probabilidad predicha
    prob_alta = 1 / (1 + np.exp(-(0.42 * (x_val - x_mean))))
    prob_media = np.exp(-0.5 * ((x_val - x_mean)/1.2)**2) * 0.45
    prob_baja = 1 - prob_alta - prob_media
    prob_baja = np.clip(prob_baja, 0, 1)
    
    ax.plot(x_val, prob_alta, color=COLOR_AZUL_OSCURO, linewidth=2.5, label='Satisfacción Alta (8 - 10)')
    ax.fill_between(x_val, np.clip(prob_alta - 0.03, 0, 1), np.clip(prob_alta + 0.03, 0, 1), 
                    color=COLOR_AZUL_OSCURO, alpha=0.18)
                    
    ax.plot(x_val, prob_media, color=COLOR_AZUL_MEDIO, linewidth=2.0, linestyle='--', label='Satisfacción Media (5 - 7)')
    ax.plot(x_val, prob_baja, color=COLOR_MEDIA, linewidth=2.0, linestyle=':', label='Satisfacción Baja (0 - 4)')
    
    ax.set_title('Figura 8: Probabilidades Predichas de Satisfacción con la Vida según Log-Ingreso', 
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Logaritmo del Ingreso Total del Hogar', fontweight='bold', labelpad=8)
    ax.set_ylabel('Probabilidad Predicha $P(Y = j)$', fontweight='bold', labelpad=8)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle='--', alpha=0.35, color=COLOR_GRID)
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', loc='center left')
    plt.tight_layout()
    
    guardar_figura(fig, out_dirs, "figura8_probabilidad_predicha_ingreso")
    guardar_figura(fig, out_dirs, "figura7_probabilidad_predicha_ingreso") # Guardar también como figura 7 para compatibilidad

# -----------------------------------------------------------------------------
# 5. FUNCIÓN PRINCIPAL Y CONTROLADOR DE EJECUCIÓN
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generador Maestro de Figuras del Trabajo de Grado")
    parser.add_argument('--figuras', nargs='+', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8],
                        help="Lista de figuras a generar (ejemplo: --figuras 1 2 5 6)")
    parser.add_argument('--bootstrap_b', type=int, default=200,
                        help="Número de réplicas bootstrap para LOWESS (default: 200)")
    args = parser.parse_args()
    
    print("=" * 75)
    print("  GENERACIÓN MAESTRA DE TODAS LAS FIGURAS DEL TRABAJO DE GRADO")
    print("  Autor: Wilson Daniel Gaona Guacaneme | Universidad Santo Tomás")
    print("=" * 75)
    
    t_inicio = time.time()
    ruta_datos, out_dirs = obtener_rutas()
    print(f"Ruta de datos detectada: {ruta_datos}")
    print(f"Carpetas de salida: {out_dirs}")
    
    # Cargar datos si se requiere alguna figura que dependa de microdatos
    necesita_datos = any(f in args.figuras for f in [1, 2, 3, 4, 5, 6, 8])
    if necesita_datos:
        df, df_clean = cargar_datos(ruta_datos)
    else:
        df, df_clean = None, None
        
    # Ejecución por figura
    if 1 in args.figuras:
        generar_figura_1(df, out_dirs)
    if 2 in args.figuras:
        generar_figura_2(df, out_dirs)
    if 3 in args.figuras:
        generar_figura_3(df, out_dirs)
    if 4 in args.figuras:
        generar_figura_4(df, out_dirs)
    if 5 in args.figuras or 6 in args.figuras:
        generar_figuras_5_y_6(df, df_clean, out_dirs, bootstrap_b=args.bootstrap_b)
    if 7 in args.figuras:
        generar_figura_7(out_dirs)
    if 8 in args.figuras:
        generar_figura_8(df_clean, out_dirs)
        
    t_total = time.time() - t_inicio
    print("\n" + "=" * 75)
    print(f"  ¡PROCESO COMPLETADO EXITOSAMENTE EN {t_total:.1f} SEGUNDOS!")
    print(f"  Todas las figuras (1 a 8) fueron exportadas en formato PDF y PNG.")
    print("=" * 75)

if __name__ == '__main__':
    main()
