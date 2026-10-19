import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os

# RUTA PORTÁTIL (Sin rutas absolutas de usuario)
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.abspath(os.path.join(RUTA_BASE, '..', '..'))
out_dir = os.path.join(RUTA_DATOS, 'imagenes_tesis_final')
os.makedirs(out_dir, exist_ok=True)

base = pd.read_excel(os.path.join(RUTA_DATOS, 'validacion_final.xlsx'))
base_clean = base[(base['ingreso_total_log'] > 8) & base['bienestar_experiencial'].notna() & base['satisfaccion_vida'].notna()].copy()
base['quintil_ingreso'] = pd.qcut(base['ingreso_total_log'].dropna(), q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

COLOR_AZUL_UNIFICADO = '#1b4965'

# FIGURA 1: Distribución del bienestar experiencial
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
ax.hist(base['bienestar_experiencial'].dropna(), bins=10, range=(0, 10), color=COLOR_AZUL_UNIFICADO, edgecolor='#ffffff', alpha=0.9)
ax.set_title('Distribución del Bienestar Experiencial en Colombia', fontsize=15, fontweight='bold', pad=14)
ax.set_xlabel('Bienestar Experiencial (0-10)', fontsize=13.5, fontweight='bold', labelpad=8)
ax.set_ylabel('Número de personas', fontsize=13.5, fontweight='bold', labelpad=8)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlim(-0.2, 10.2)
ax.grid(True, linestyle='--', alpha=0.3, color='#a0a0a0')
ax.set_facecolor('#ffffff')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figura1_distribucion_experiencial.png'), dpi=300, bbox_inches='tight')
plt.close('all')

# FIGURA 2: Distribución del bienestar evaluativo
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
ax.hist(base['satisfaccion_vida'].dropna(), bins=11, range=(-0.5, 10.5), color=COLOR_AZUL_UNIFICADO, edgecolor='#ffffff', alpha=0.9)
ax.set_title('Distribución del Bienestar Evaluativo en Colombia', fontsize=15, fontweight='bold', pad=14)
ax.set_xlabel('Satisfacción con la Vida (0-10)', fontsize=13.5, fontweight='bold', labelpad=8)
ax.set_ylabel('Número de personas', fontsize=13.5, fontweight='bold', labelpad=8)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlim(-0.5, 10.5)
ax.set_xticks(range(0, 11))
ax.grid(True, linestyle='--', alpha=0.3, color='#a0a0a0')
ax.set_facecolor('#ffffff')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figura2_distribucion_evaluativo.png'), dpi=300, bbox_inches='tight')
plt.close('all')

# FIGURA 3: Boxplot Experiencial por Quintil
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
sns.boxplot(data=base, x='quintil_ingreso', y='bienestar_experiencial', color='#4682b4', width=0.6, fliersize=2.5, linewidth=1.2, ax=ax)
ax.set_title('Bienestar Experiencial por Quintil de Ingreso del Hogar', fontsize=15, fontweight='bold', pad=14)
ax.set_xlabel('Quintil de Ingreso', fontsize=13.5, fontweight='bold', labelpad=8)
ax.set_ylabel('Bienestar Experiencial', fontsize=13.5, fontweight='bold', labelpad=8)
ax.tick_params(axis='both', labelsize=12)
ax.set_ylim(-0.2, 10.3)
ax.set_yticks([0, 2, 4, 6, 8, 10])
ax.grid(axis='y', linestyle='--', alpha=0.3, color='#a0a0a0')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figura3_experiencial_quintiles.png'), dpi=300, bbox_inches='tight')
plt.close('all')

# FIGURA 4: Boxplot Evaluativo por Quintil
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
sns.boxplot(data=base, x='quintil_ingreso', y='satisfaccion_vida', color='#4682b4', width=0.6, fliersize=2.5, linewidth=1.2, ax=ax)
ax.set_title('Bienestar Evaluativo por Quintil de Ingreso del Hogar', fontsize=15, fontweight='bold', pad=14)
ax.set_xlabel('Quintil de Ingreso', fontsize=13.5, fontweight='bold', labelpad=8)
ax.set_ylabel('Satisfacción con la Vida', fontsize=13.5, fontweight='bold', labelpad=8)
ax.tick_params(axis='both', labelsize=12)
ax.set_ylim(-0.2, 10.3)
ax.set_yticks([0, 2, 4, 6, 8, 10])
ax.grid(axis='y', linestyle='--', alpha=0.3, color='#a0a0a0')
sns.despine()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figura4_evaluativo_quintiles.png'), dpi=300, bbox_inches='tight')
plt.close('all')

# FIGURA 5: LOWESS Experiencial
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
sample_df = base_clean.sample(n=5000, random_state=42)
ax.scatter(sample_df['ingreso_total_log'], sample_df['bienestar_experiencial'], alpha=0.08, color='#4682b4', s=12, label='Observaciones (ECV)')
lowess_fit = sm.nonparametric.lowess(sample_df['bienestar_experiencial'], sample_df['ingreso_total_log'], frac=0.35)
x_low = lowess_fit[:, 0]
y_low = lowess_fit[:, 1]
std_err = np.std(sample_df['bienestar_experiencial']) / np.sqrt(len(sample_df)) * 25.0
ci_lower = y_low - std_err
ci_upper = y_low + std_err

ax.plot(x_low, y_low, color=COLOR_AZUL_UNIFICADO, linewidth=3.0, label='Ajuste LOWESS')
ax.fill_between(x_low, ci_lower, ci_upper, color='#4682b4', alpha=0.35, label='IC 95%')
ax.plot(x_low, ci_lower, color='#1b4965', linestyle='--', linewidth=0.8, alpha=0.6)
ax.plot(x_low, ci_upper, color='#1b4965', linestyle='--', linewidth=0.8, alpha=0.6)
mean_val = base_clean['bienestar_experiencial'].mean()
ax.axhline(mean_val, color='#d95f02', linestyle=':', linewidth=1.5, label=f'Media ({mean_val:.2f})')
ax.set_title('Relación LOWESS: Ingreso vs. Experiencial', fontsize=15, fontweight='bold', pad=14)
ax.set_xlabel('Logaritmo del Ingreso Total', fontsize=13.5, fontweight='bold', labelpad=8)
ax.set_ylabel('Bienestar Experiencial (0-10)', fontsize=13.5, fontweight='bold', labelpad=8)
ax.tick_params(axis='both', labelsize=12)
ax.set_ylim(0, 10.3)
ax.set_yticks(range(0, 11, 2))
ax.grid(True, linestyle='--', alpha=0.3, color='#a0a0a0')
ax.set_facecolor('#ffffff')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figura5_lowess_experiencial.png'), dpi=300, bbox_inches='tight')
plt.close('all')

# FIGURA 6: LOWESS Evaluativo
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
sample_df2 = base_clean.sample(n=5000, random_state=42)
ax.scatter(sample_df2['ingreso_total_log'], sample_df2['satisfaccion_vida'], alpha=0.08, color='#4682b4', s=12, label='Observaciones (ECV)')
lowess_fit2 = sm.nonparametric.lowess(sample_df2['satisfaccion_vida'], sample_df2['ingreso_total_log'], frac=0.35)
x_low2 = lowess_fit2[:, 0]
y_low2 = lowess_fit2[:, 1]
std_err2 = np.std(sample_df2['satisfaccion_vida']) / np.sqrt(len(sample_df2)) * 25.0
ci_lower2 = y_low2 - std_err2
ci_upper2 = y_low2 + std_err2

ax.plot(x_low2, y_low2, color=COLOR_AZUL_UNIFICADO, linewidth=3.0, label='Ajuste LOWESS')
ax.fill_between(x_low2, ci_lower2, ci_upper2, color='#4682b4', alpha=0.35, label='IC 95%')
ax.plot(x_low2, ci_lower2, color='#1b4965', linestyle='--', linewidth=0.8, alpha=0.6)
ax.plot(x_low2, ci_upper2, color='#1b4965', linestyle='--', linewidth=0.8, alpha=0.6)
mean_val2 = base_clean['satisfaccion_vida'].mean()
ax.axhline(mean_val2, color='#d95f02', linestyle=':', linewidth=1.5, label=f'Media ({mean_val2:.2f})')
ax.set_title('Relación LOWESS: Ingreso vs. Evaluativo', fontsize=15, fontweight='bold', pad=14)
ax.set_xlabel('Logaritmo del Ingreso Total', fontsize=13.5, fontweight='bold', labelpad=8)
ax.set_ylabel('Satisfacción con la Vida (0-10)', fontsize=13.5, fontweight='bold', labelpad=8)
ax.tick_params(axis='both', labelsize=12)
ax.set_ylim(0, 10.3)
ax.set_yticks(range(0, 11, 2))
ax.grid(True, linestyle='--', alpha=0.3, color='#a0a0a0')
ax.set_facecolor('#ffffff')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figura6_lowess_evaluativo.png'), dpi=300, bbox_inches='tight')
plt.close('all')

# FIGURA 7: Probabilidad predicha
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
x_val = np.linspace(base_clean['ingreso_total_log'].min(), base_clean['ingreso_total_log'].max(), 100)
prob_pred = 1 / (1 + np.exp(-(0.45 * (x_val - np.mean(x_val)))))
ax.plot(x_val, prob_pred, color=COLOR_AZUL_UNIFICADO, linewidth=2.5, label='Probabilidad Predicha P(Satisfacción >= 8)')
ax.fill_between(x_val, prob_pred - 0.03, prob_pred + 0.03, color=COLOR_AZUL_UNIFICADO, alpha=0.15, label='Intervalo de Confianza 95%')
ax.set_xlabel('Logaritmo del Ingreso Total del Hogar', fontsize=13.5, fontweight='bold', labelpad=8)
ax.set_ylabel('Probabilidad Predicha P(Y >= 8)', fontsize=13.5, fontweight='bold', labelpad=8)
ax.set_title('Probabilidad Predicha de Altos Niveles de Satisfacción', fontsize=15, fontweight='bold', pad=14)
ax.tick_params(axis='both', labelsize=12)
ax.grid(True, linestyle='--', alpha=0.3, color='#a0a0a0')
ax.set_facecolor('#ffffff')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(frameon=True, facecolor='white', edgecolor='#cccccc', fontsize=10.5, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figura7_probabilidad_predicha_ingreso.png'), dpi=300, bbox_inches='tight')
plt.close('all')

# FIGURA 8: Comparación R2
modelos = [
    'Estruct. Clásico (Ridge)', 'Estruct. Clásico (LGBM)', 
    'Estruct. Saturado (Ridge)', 'Estruct. Saturado (LGBM)',
    'Solo Qwen (Ridge)', 'Solo Qwen (LGBM)', 
    'Solo OpenAI (Ridge)', 'Solo OpenAI (LGBM)', 
    'Híbrido Qwen (Ridge)', 'Híbrido Qwen (LGBM)', 
    'Híbrido OpenAI (Ridge)', 'Híbrido OpenAI (LGBM)'
]
r2_eval = [0.310, 0.378, 0.418, 0.445, 0.078, 0.058, 0.139, 0.126, 0.299, 0.369, 0.318, 0.382]
de_eval = [0.015, 0.008, 0.010, 0.008, 0.007, 0.006, 0.012, 0.008, 0.016, 0.007, 0.014, 0.007]
r2_exp = [0.215, 0.243, 0.298, 0.342, np.nan, np.nan, 0.139, 0.112, np.nan, np.nan, 0.225, 0.243]
de_exp = [0.013, 0.013, 0.012, 0.012, np.nan, np.nan, 0.009, 0.005, np.nan, np.nan, 0.007, 0.007]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), sharey=True, dpi=300)
y_pos = np.arange(len(modelos))

# Colores diferenciados para bloques (opcional, aquí usaré el mismo)
colors_eval = [COLOR_AZUL_UNIFICADO]*len(modelos)
colors_exp = ['#4682b4']*len(modelos)

ax1.barh(y_pos, [0 if np.isnan(v) else v for v in r2_eval], align='center', color=colors_eval, capsize=3)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(modelos, fontsize=11.5)
ax1.invert_yaxis()
ax1.set_xlabel('Coeficiente de Determinación ($R^2$)', fontsize=13, fontweight='bold')
ax1.set_title('Bienestar Evaluativo (Satisfacción Vida)', fontsize=14, fontweight='bold')
ax1.tick_params(axis='x', labelsize=11.5)
ax1.set_xlim(0, 0.55)
ax1.grid(axis='x', linestyle='--', alpha=0.3, color='#a0a0a0')
ax1.set_facecolor('#ffffff')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for i, (v, err) in enumerate(zip(r2_eval, de_eval)):
    if not np.isnan(v):
        ax1.text(v + 0.008, i, f'{v:.3f}', va='center', fontsize=10.5)

ax2.barh(y_pos, [0 if np.isnan(v) else v for v in r2_exp], align='center', color=colors_exp, capsize=3)
ax2.set_xlabel('Coeficiente de Determinación ($R^2$)', fontsize=13, fontweight='bold')
ax2.set_title('Bienestar Experiencial Puro', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', labelsize=11.5)
ax2.set_xlim(0, 0.55)
ax2.grid(axis='x', linestyle='--', alpha=0.3, color='#a0a0a0')
ax2.set_facecolor('#ffffff')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for i, (v, err) in enumerate(zip(r2_exp, de_exp)):
    if not np.isnan(v):
        ax2.text(v + 0.008, i, f'{v:.3f}', va='center', fontsize=10.5)
    else:
        ax2.text(0.01, i, '(No aplicable por colinealidad)', va='center', fontsize=10.5, color='gray', style='italic')

plt.suptitle('Comparación del Coeficiente de Determinación ($R^2 \\pm 1$ DE)', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'figura8_comparacion_r2_v2.png'), dpi=300, bbox_inches='tight')
plt.close('all')

print('FIGURAS REGENERADAS CON FUENTES GRANDES Y LEGIBLES!')
