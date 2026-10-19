import os
import re
import pandas as pd
import matplotlib.pyplot as plt

RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.abspath(os.path.join(RUTA_BASE, '..', '..'))
EXCEL = os.path.join(RUTA_DATOS, 'resultados_group_kfold_y_holdout.xlsx')
SALIDA = os.path.join(RUTA_DATOS, 'imagenes_tesis_final', 'figura8_comparacion_r2_final.png')
os.makedirs(os.path.dirname(SALIDA), exist_ok=True)

orden = [
    ('01.', 'Estruct. Clásico (Ridge)', '02.'), ('', 'Estruct. Clásico (LGBM)', '01.'),
    ('', 'Estruct. Saturado (Ridge)', '04.'), ('', 'Estruct. Saturado (LGBM)', '03.'),
    ('', 'Solo Qwen (Ridge)', '06.'), ('', 'Solo Qwen (LGBM)', '05.'),
    ('', 'Solo OpenAI (Ridge)', '08.'), ('', 'Solo OpenAI (LGBM)', '07.'),
    ('', 'Híbrido Qwen (Ridge)', '10.'), ('', 'Híbrido Qwen (LGBM)', '09.'),
    ('', 'Híbrido OpenAI (Ridge)', '12.'), ('', 'Híbrido OpenAI (LGBM)', '11.'),
]
grupo_color = {'Estruct. Cl': '#1f3b57', 'Estruct. Sa': '#2e6f95', 'Solo': '#9aa5b1', 'Híbrido': '#c97b3c'}

def color(lbl):
    for k, c in grupo_color.items():
        if lbl.startswith(k):
            return c

def leer(hoja):
    df = pd.read_excel(EXCEL, sheet_name=hoja)
    m, s = [], []
    for _, lbl, pref in orden:
        v = df.loc[df['Modelo'].str.startswith(pref), 'R2 (10-Folds Val)'].iloc[0]
        a, b = [float(x) for x in re.findall(r'\d+\.\d+', str(v))[:2]]
        m.append(a); s.append(b)
    return m, s

paneles = [('Evaluativo', 'Bienestar evaluativo (satisfacción con la vida)'),
           ('Experiencial Puro', 'Bienestar experiencial (afecto)')]
labels = [o[1] for o in orden]
cols = [color(l) for l in labels]

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
for ax, (hoja, titulo) in zip(axes, paneles):
    m, s = leer(hoja)
    y = range(len(labels))
    ax.barh(y, m, xerr=s, color=cols, capsize=3, error_kw={'elinewidth': 1, 'ecolor': 'black'})
    for i, (a, b) in enumerate(zip(m, s)):
        ax.text(a + b + 0.006, i, f'{a:.3f}', va='center', fontsize=10)
    ax.set_title(titulo, fontweight='bold')
    ax.set_xlim(0, 0.52)
    ax.set_xlabel('$R^2$ (validación cruzada, media ± 1 DE)')
    ax.grid(axis='x', ls='--', alpha=.4)
    ax.spines[['top', 'right']].set_visible(False)
    ax.invert_yaxis() if ax is axes[0] else None
axes[0].set_yticks(range(len(labels)))
axes[0].set_yticklabels(labels)
plt.tight_layout()
plt.savefig(SALIDA, dpi=300, bbox_inches='tight')
print('OK', SALIDA)
