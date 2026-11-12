import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from joblib import Parallel, delayed

def main():
    print("=== GENERACIÓN DE FIGURAS 5 Y 6 (LOWESS CON BOOTSTRAP POR HOGAR) ===")
    
    # 1. Definir rutas
    ruta_base = r"c:\Users\WILSGAOGUA\OneDrive - colsubsidio.com\universidad\Trabajo de grado"
    out_dir_tesis = os.path.join(ruta_base, "imagenes_tesis_final")
    out_dir_downloads = r"C:\Users\WILSGAOGUA\Downloads"
    os.makedirs(out_dir_tesis, exist_ok=True)
    
    csv_path = os.path.join(ruta_base, "base_final_embedings.csv")
    print(f"Cargando microdatos desde: {csv_path}")
    
    cols = ['DIRECTORIO', 'SECUENCIA_P', 'FEX_C', 'ingreso_total_log', 'satisfaccion_vida', 'bienestar_experiencial']
    df = pd.read_csv(csv_path, usecols=cols)
    
    # 2. Filtrado riguroso: ingreso positivo (log > 8) y sin valores faltantes
    df_clean = df[(df['ingreso_total_log'] > 8) & 
                  df['satisfaccion_vida'].notna() & 
                  df['bienestar_experiencial'].notna()].copy()
    
    df_clean['hogar_id'] = df_clean['DIRECTORIO'].astype(str) + '_' + df_clean['SECUENCIA_P'].astype(str)
    
    total_personas = len(df_clean)
    hogares_unicos = df_clean['hogar_id'].unique()
    total_hogares = len(hogares_unicos)
    print(f"Muestra limpia filtrada: {total_personas:,} personas en {total_hogares:,} hogares únicos.")
    
    # 3. Submuestra representativa de 5.000 hogares para el ajuste no paramétrico
    # (Como recomienda el tutor para garantizar estabilidad y balance computacional)
    np.random.seed(42)
    sample_hogares = np.random.choice(hogares_unicos, size=5000, replace=False)
    sample_df = df_clean[df_clean['hogar_id'].isin(sample_hogares)].copy()
    n_sample_personas = len(sample_df)
    print(f"Submuestra de análisis: {n_sample_personas} personas de 5.000 hogares seleccionados aleatoriamente.")
    
    # 4. Rejilla fija de evaluación para el LOWESS (cuantil 1% a 99%)
    x_min = sample_df['ingreso_total_log'].quantile(0.01)
    x_max = sample_df['ingreso_total_log'].quantile(0.99)
    x_grid = np.linspace(x_min, x_max, 120)
    print(f"Rejilla de evaluación de ingresos: ln(ingreso) de {x_min:.2f} a {x_max:.2f}")
    
    # Ajuste principal sobre la submuestra
    print("Calculando ajuste LOWESS central...")
    y_lowess_eval = sm.nonparametric.lowess(sample_df['satisfaccion_vida'], 
                                            sample_df['ingreso_total_log'], 
                                            xvals=x_grid, frac=0.35)
    
    y_lowess_exp = sm.nonparametric.lowess(sample_df['bienestar_experiencial'], 
                                           sample_df['ingreso_total_log'], 
                                           xvals=x_grid, frac=0.35)
    
    # 5. Bootstrap por conglomerados a nivel de hogar (B = 200)
    B = 200
    print(f"Iniciando Bootstrap por Hogar con {B} remuestreos en paralelo...")
    t0 = time.time()
    
    # Agrupar datos por hogar en matrices numpy para máxima velocidad
    grouped_data = [g[['ingreso_total_log', 'satisfaccion_vida', 'bienestar_experiencial']].values 
                    for _, g in sample_df.groupby('hogar_id')]
    n_groups = len(grouped_data)
    
    def run_one_bootstrap(seed):
        rng = np.random.RandomState(seed)
        idx = rng.choice(n_groups, size=n_groups, replace=True)
        # Combinar personas de los hogares remuestreados
        mat = np.vstack([grouped_data[i] for i in idx])
        x_b = mat[:, 0]
        y_eval_b = mat[:, 1]
        y_exp_b = mat[:, 2]
        
        fit_eval = sm.nonparametric.lowess(y_eval_b, x_b, xvals=x_grid, frac=0.35)
        fit_exp = sm.nonparametric.lowess(y_exp_b, x_b, xvals=x_grid, frac=0.35)
        return fit_eval, fit_exp
    
    # Ejecución paralela
    results = Parallel(n_jobs=-1, batch_size=5)(delayed(run_one_bootstrap)(42 + b) for b in range(B))
    
    boot_eval = np.array([r[0] for r in results])
    boot_exp = np.array([r[1] for r in results])
    
    eval_ci_lower = np.percentile(boot_eval, 2.5, axis=0)
    eval_ci_upper = np.percentile(boot_eval, 97.5, axis=0)
    
    exp_ci_lower = np.percentile(boot_exp, 2.5, axis=0)
    exp_ci_upper = np.percentile(boot_exp, 97.5, axis=0)
    
    print(f"Bootstrap finalizado en {time.time() - t0:.1f} segundos.")
    
    # 6. Configuración de estilo editorial académico
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 9.5
    
    COLOR_LINEA = '#1b4965'     # Azul oscuro formal
    COLOR_BANDA = '#4682b4'     # Azul medio para la sombra de confianza
    COLOR_SCATTER = '#5c80a8'   # Azul suave para observaciones
    COLOR_MEDIA = '#d95f02'     # Naranja tenue para la media poblacional
    
    # ----------------------------------------------------
    # FIGURA 5: BIENESTAR EVALUATIVO (SATISFACCIÓN CON LA VIDA)
    # ----------------------------------------------------
    print("Graficando Figura 5: Bienestar Evaluativo...")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    
    # Scatter de observaciones (submuestra aleatoria para visualización limpia)
    scatter_sample = sample_df.sample(n=2500, random_state=42)
    ax.scatter(scatter_sample['ingreso_total_log'], scatter_sample['satisfaccion_vida'],
               alpha=0.08, color=COLOR_SCATTER, s=14, edgecolors='none', label='Observaciones ECV')
    
    # Banda de confianza 95% mediante Bootstrap
    ax.fill_between(x_grid, eval_ci_lower, eval_ci_upper, color=COLOR_BANDA, alpha=0.32,
                    label='Intervalo de confianza 95% (bootstrap por hogar)')
    ax.plot(x_grid, eval_ci_lower, color=COLOR_LINEA, linestyle='--', linewidth=0.8, alpha=0.55)
    ax.plot(x_grid, eval_ci_upper, color=COLOR_LINEA, linestyle='--', linewidth=0.8, alpha=0.55)
    
    # Curva central LOWESS
    ax.plot(x_grid, y_lowess_eval, color=COLOR_LINEA, linewidth=2.8, label='Ajuste LOWESS (N = 175 164)')
    
    # Línea de media poblacional ponderada
    media_eval = (df_clean['satisfaccion_vida'] * df_clean['FEX_C']).sum() / df_clean['FEX_C'].sum()
    ax.axhline(media_eval, color=COLOR_MEDIA, linestyle=':', linewidth=1.5, label=f'Media ponderada ({media_eval:.2f})')
    
    ax.set_xlabel('Logaritmo del Ingreso Total del Hogar', fontweight='bold', labelpad=8)
    ax.set_ylabel('Satisfacción con la Vida (0-10)', fontweight='bold', labelpad=8)
    ax.set_ylim(0, 10.3)
    ax.set_yticks(range(0, 11, 2))
    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.grid(True, linestyle='--', alpha=0.35, color='#b0b0b0')
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, facecolor='white', edgecolor='#d0d0d0', loc='lower right')
    plt.tight_layout()
    
    # Guardar en PDF vectorial y PNG de alta resolución
    f5_pdf_tesis = os.path.join(out_dir_tesis, "Figura5_LOWESS_Evaluativo.pdf")
    f5_png_tesis = os.path.join(out_dir_tesis, "Figura5_LOWESS_Evaluativo.png")
    f5_pdf_down = os.path.join(out_dir_downloads, "Figura5_LOWESS_Evaluativo.pdf")
    f5_png_down = os.path.join(out_dir_downloads, "Figura5_LOWESS_Evaluativo.png")
    
    fig.savefig(f5_pdf_tesis, bbox_inches='tight')
    fig.savefig(f5_png_tesis, bbox_inches='tight', dpi=300)
    fig.savefig(f5_pdf_down, bbox_inches='tight')
    fig.savefig(f5_png_down, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Figura 5 guardada exitosamente en PDF y PNG.")
    
    # ----------------------------------------------------
    # FIGURA 6: BIENESTAR EXPERIENCIAL (AFECTO DIARIO)
    # ----------------------------------------------------
    print("Graficando Figura 6: Bienestar Experiencial...")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
    
    ax.scatter(scatter_sample['ingreso_total_log'], scatter_sample['bienestar_experiencial'],
               alpha=0.08, color=COLOR_SCATTER, s=14, edgecolors='none', label='Observaciones ECV')
    
    ax.fill_between(x_grid, exp_ci_lower, exp_ci_upper, color=COLOR_BANDA, alpha=0.32,
                    label='Intervalo de confianza 95% (bootstrap por hogar)')
    ax.plot(x_grid, exp_ci_lower, color=COLOR_LINEA, linestyle='--', linewidth=0.8, alpha=0.55)
    ax.plot(x_grid, exp_ci_upper, color=COLOR_LINEA, linestyle='--', linewidth=0.8, alpha=0.55)
    
    ax.plot(x_grid, y_lowess_exp, color=COLOR_LINEA, linewidth=2.8, label='Ajuste LOWESS (N = 175 164)')
    
    media_exp = (df_clean['bienestar_experiencial'] * df_clean['FEX_C']).sum() / df_clean['FEX_C'].sum()
    ax.axhline(media_exp, color=COLOR_MEDIA, linestyle=':', linewidth=1.5, label=f'Media ponderada ({media_exp:.2f})')
    
    ax.set_xlabel('Logaritmo del Ingreso Total del Hogar', fontweight='bold', labelpad=8)
    ax.set_ylabel('Balance Afectivo Experiencial (0-10)', fontweight='bold', labelpad=8)
    ax.set_ylim(0, 10.3)
    ax.set_yticks(range(0, 11, 2))
    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.grid(True, linestyle='--', alpha=0.35, color='#b0b0b0')
    ax.set_facecolor('#ffffff')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=True, facecolor='white', edgecolor='#d0d0d0', loc='lower right')
    plt.tight_layout()
    
    f6_pdf_tesis = os.path.join(out_dir_tesis, "Figura6_LOWESS_Experiencial.pdf")
    f6_png_tesis = os.path.join(out_dir_tesis, "Figura6_LOWESS_Experiencial.png")
    f6_pdf_down = os.path.join(out_dir_downloads, "Figura6_LOWESS_Experiencial.pdf")
    f6_png_down = os.path.join(out_dir_downloads, "Figura6_LOWESS_Experiencial.png")
    
    fig.savefig(f6_pdf_tesis, bbox_inches='tight')
    fig.savefig(f6_png_tesis, bbox_inches='tight', dpi=300)
    fig.savefig(f6_pdf_down, bbox_inches='tight')
    fig.savefig(f6_png_down, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print("Figura 6 guardada exitosamente en PDF y PNG.")
    
    print("\n=== RESUMEN DE ARCHIVOS GENERADOS ===")
    print(f"1. En carpeta de la tesis: {out_dir_tesis}")
    print(f"   - Figura5_LOWESS_Evaluativo.pdf / .png")
    print(f"   - Figura6_LOWESS_Experiencial.pdf / .png")
    print(f"2. En carpeta de Descargas: {out_dir_downloads}")
    print(f"   - Figura5_LOWESS_Evaluativo.pdf / .png")
    print(f"   - Figura6_LOWESS_Experiencial.pdf / .png")

if __name__ == '__main__':
    main()
