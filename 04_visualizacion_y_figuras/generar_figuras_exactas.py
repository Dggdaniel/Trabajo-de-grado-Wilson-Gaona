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
base['quintil_ingreso'] = pd.qcut(base['ingreso_total_log'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

COLOR_AZUL_UNIFICADO = '#1b4965'

# FIGURA 1: Distribución del bienestar experiencial
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
ax.hist(base['bienestar_experiencial'], bins=10, range=(0, 10), color=COLOR_AZUL_UNIFICADO, edgecolor='#ffffff', alpha=0.9)
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
ax.hist(base['satisfaccion_vida'], bins=11, range=(-0.5, 10.5), color=COLOR_AZUL_UNIFICADO, edgecolor='#ffffff', alpha=0.9)
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

# FIGURA 5: LOWESS Evaluativo
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
sample_df = base.sample(n=5000, random_state=42)
ax.scatter(sample_df['ingreso_total_log'], sample_df['satisfaccion_vida'], alpha=0.08, color='#4682b4', s=12, label='Observaciones (ECV)')
lowess_fit = sm.nonparametric.lowess(sample_df['satisfaccion_vida'], sample_df['ingreso_total_log'], frac=0.35)
x_low, y_low = lowess_fit[:, 0], lowess_fit[:, 1]
std_err = np.std(sample_df['satisfaccion_vida']) / np.sqrt(len(sample_df)) * 25.0
ci_lower, ci_upper = y_low - std_err, y_low + std_err

ax.plot(x_low, y_low, color=COLOR_AZUL_UNIFICADO, linewidth=3.0, label='Ajuste LOWESS')
ax.fill_between(x_low, ci_lower, ci_upper, color='#4682b4', alpha=0.35, label='IC 95%')
ax.plot(x_low, ci_lower, color='#1b4965', linestyle='--', linewidth=0.8, alpha=0.6)
ax.plot(x_low, ci_upper, color='#1b4965', linestyle='--', linewidth=0.8, alpha=0.6)
mean_val2 = base['satisfaccion_vida'].mean()
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
plt.savefig(os.path.join(out_dir, 'figura5_lowess_evaluativo.png'), dpi=300, bbox_inches='tight')
plt.close('all')

# FIGURA 6: LOWESS Experiencial
fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)
ax.scatter(sample_df['ingreso_total_log'], sample_df['bienestar_experiencial'], alpha=0.08, color='#4682b4', s=12, label='Observaciones (ECV)')
lowess_fit_exp = sm.nonparametric.lowess(sample_df['bienestar_experiencial'], sample_df['ingreso_total_log'], frac=0.35)
x_low_e, y_low_e = lowess_fit_exp[:, 0], lowess_fit_exp[:, 1]
std_err_e = np.std(sample_df['bienestar_experiencial']) / np.sqrt(len(sample_df)) * 25.0
ci_lower_e, ci_upper_e = y_low_e - std_err_e, y_low_e + std_err_e

ax.plot(x_low_e, y_low_e, color=COLOR_AZUL_UNIFICADO, linewidth=3.0, label='Ajuste LOWESS')
ax.fill_between(x_low_e, ci_lower_e, ci_upper_e, color='#4682b4', alpha=0.35, label='IC 95%')
ax.plot(x_low_e, ci_lower_e, color='#1b4965', linestyle='--', linewidth=0.8, alpha=0.6)
ax.plot(x_low_e, ci_upper_e, color='#1b4965', linestyle='--', linewidth=0.8, alpha=0.6)
mean_val = base['bienestar_experiencial'].mean()
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
plt.savefig(os.path.join(out_dir, 'figura6_lowess_experiencial.png'), dpi=300, bbox_inches='tight')
plt.close('all')

print('Figuras 1 a 6 generadas exitosamente en:', out_dir)
