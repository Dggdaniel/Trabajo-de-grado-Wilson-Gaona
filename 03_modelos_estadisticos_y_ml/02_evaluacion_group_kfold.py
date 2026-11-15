import pandas as pd
import numpy as np
import re
import os
import warnings
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import kendalltau
import lightgbm as lgb
from patsy import dmatrix

warnings.filterwarnings('ignore')

RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.abspath(os.path.join(RUTA_BASE, '..', '..'))

print('=== PASO 1: CARGANDO MICRODATOS Y LLAVES DE HOGAR POR LLAVE COMPUESTA ===')
base_trabajo = pd.read_excel(os.path.join(RUTA_DATOS, 'validacion_final.xlsx'))
if 'satisfaccion_trabajo' in base_trabajo.columns and 'satisfaccion_ocupacion' not in base_trabajo.columns:
    base_trabajo['satisfaccion_ocupacion'] = base_trabajo['satisfaccion_trabajo']

base_raw = pd.read_csv(os.path.join(RUTA_DATOS, 'base_final_embedings.csv'))
if 'satisfaccion_trabajo' in base_raw.columns and 'satisfaccion_ocupacion' not in base_raw.columns:
    base_raw['satisfaccion_ocupacion'] = base_raw['satisfaccion_trabajo']

# Llave única compuesta para evitar desalineaciones por posición de fila
hogar_id = base_raw['DIRECTORIO'].astype(str) + '_' + base_raw['SECUENCIA_P'].astype(str)

print('\n=== PASO 2: CARGANDO REPRESENTACIONES VECTORIALES (EMBEDDINGS) ===')
openai_path = os.path.join(RUTA_DATOS, 'embeding generados con text-embedding-3-small.csv')
openai_df = pd.read_csv(openai_path).drop(columns=['Unnamed: 0'], errors='ignore')
openai_df.columns = [f'openai_{c}' for c in openai_df.columns]
openai_df.index = base_trabajo.index

qwen_path = os.path.join(RUTA_DATOS, 'embeding generados con qwen.csv')
qwen_df = pd.read_csv(qwen_path).drop(columns=['Unnamed: 0'], errors='ignore')

if 'embedding' in qwen_df.columns:
    qwen_df['embedding'] = qwen_df['embedding'].apply(
        lambda x: np.fromstring(x.replace('\n', ' ').strip('[]'), sep=' ') if isinstance(x, str) else x
    )
    qwen_mat = pd.DataFrame(qwen_df['embedding'].tolist())
else:
    qwen_mat = qwen_df

qwen_mat.columns = [f'qwen_{c}' for c in qwen_mat.columns]
qwen_mat.index = base_trabajo.index

print('\n=== PASO 3: CONSTRUYENDO MATRIZ CLÁSICA / TRADICIONAL ===')
variables_tradicionales = [
    'sexo_nacimiento', 'FEX_C', 'nivel_educativo_alcanzado',
    'ingreso_total_log', 'estado_conyugal', 'satisfaccion_ocupacion',
    'satisfaccion_salud', 'P1_DEPARTAMENTO', 'grupo_edad'
]

var_eval = 'satisfaccion_vida'
var_exp_puro = 'afecto_puro'

base_modelo_ml = base_trabajo[variables_tradicionales + [var_eval]].copy()
base_modelo_ml['sexo_nacimiento'] = base_modelo_ml['sexo_nacimiento'].astype(str)
base_modelo_ml['nivel_educativo_alcanzado'] = base_modelo_ml['nivel_educativo_alcanzado'].astype(str)

# Construcción del índice de afecto puro (3 ítems de balance afectivo)
base_modelo_ml[var_exp_puro] = (
    base_raw['felicidad_ayer'] + base_raw['preocupacion_ayer_inv'] + base_raw['tristeza_ayer_inv']
) / 3.0

categoricas = ['sexo_nacimiento', 'nivel_educativo_alcanzado', 'estado_conyugal', 'satisfaccion_ocupacion', 'satisfaccion_salud', 'grupo_edad', 'P1_DEPARTAMENTO']
splines = dmatrix('bs(ingreso_total_log, df=4, degree=3, include_intercept=False) - 1', {'ingreso_total_log': base_modelo_ml['ingreso_total_log']}, return_type='dataframe')
X_dummies = pd.get_dummies(base_modelo_ml[categoricas], drop_first=True)
X_trad = pd.concat([X_dummies, splines], axis=1)
clean_cols = [re.sub(r'[^\w]', '_', str(c)) for c in X_trad.columns]
X_trad.columns = clean_cols

X_hib_openai = pd.concat([X_trad, openai_df], axis=1)
X_hib_qwen = pd.concat([X_trad, qwen_mat], axis=1)

print('\n=== PASO 3.1: CONSTRUYENDO MATRIZ SATURADA COMPLETA Y SATURADA COMPARABLE ===')
cols_to_drop = [
    'Unnamed: 0', 'DIRECTORIO', 'SECUENCIA_P', 'SECUENCIA_ENCUESTA', 'ORDEN',
    'orden_informante_x', 'orden_informante_y', 'FEX_C', 'FEX', 'PROB_IAG',
    'tipo_documento', 'quien_contesta_capitulo',
    'satisfaccion_vida', 'bienestar_experiencial', 'bienestar_experiencial_entero',
    'felicidad_ayer', 'tristeza_ayer', 'preocupacion_ayer',
    'vida_vale_la_pena', 'escalera_situacion_actual', 'auto_percepcion',
    'afecto_positivo', 'afecto_negativo',
    'preocupacion_ayer_inv', 'tristeza_ayer_inv', 'afecto_negativo_inv',
]

# Exclusiones específicas para el "Saturado Comparable" (mismo universo de información que el híbrido)
# Excluye las satisfacciones adicionales y las derivadas excluidas de la serialización
EXCLUIR_COMPARABLE = [
    'satisfaccion_ingresos', 'satisfaccion_tiempo_libre', 'satisfaccion_seguridad',
    'ingreso', 'ingreso_mensual_hogar', 'ingreso_mensual_unidad_gasto',
    'ingreso_mensual_empleo', 'ingreso_mensual_otra_unidad',
    'ingreso_total', 'ingreso_per_capita',
    'edad', 'fecha_nacimiento', 'grado_aprobado', 'REGION'
]

cols_keep_full = [c for c in base_raw.columns if c not in cols_to_drop]
cols_keep_comp = [c for c in base_raw.columns if c not in cols_to_drop and c not in EXCLUIR_COMPARABLE]

def dummificar_matriz(columnas):
    X_sub = base_raw[columnas].copy()
    cat_c, num_c = [], []
    for c in X_sub.columns:
        if X_sub[c].dtype == 'object' or X_sub[c].nunique() < 15:
            cat_c.append(c)
        else:
            num_c.append(c)
    for c in num_c:
        X_sub[c] = pd.to_numeric(X_sub[c], errors='coerce').fillna(0)
    for c in cat_c:
        X_sub[c] = X_sub[c].fillna('No aplica').astype(str)
    
    X_d = pd.get_dummies(X_sub[cat_c], drop_first=True)
    X_out = pd.concat([X_sub[num_c], X_d], axis=1)
    X_out.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X_out.columns]
    return X_out

print('Dummificando Matriz Saturada Completa...')
X_trad_ampliada = dummificar_matriz(cols_keep_full)

print('Dummificando Matriz Saturada Comparable (Mismo universo del híbrido)...')
X_sat_comparable = dummificar_matriz(cols_keep_comp)

print('Construyendo Matriz Saturada + Embeddings OpenAI...')
X_sat_plus_openai = pd.concat([X_trad_ampliada, openai_df], axis=1)

print(f'Dimensiones reales (p columnas):')
print(f'  - Clásico: p = {X_trad.shape[1]}')
print(f'  - Híbrido OpenAI: p = {X_hib_openai.shape[1]}')
print(f'  - Saturado Comparable: p = {X_sat_comparable.shape[1]}')
print(f'  - Saturado Completo: p = {X_trad_ampliada.shape[1]}')
print(f'  - Saturado + OpenAI: p = {X_sat_plus_openai.shape[1]}')

pesos = base_modelo_ml['FEX_C']

# Partición 90% Dev / 10% Test Holdout agrupado por hogar
gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
dev_idx, test_idx = next(gss.split(base_modelo_ml, groups=hogar_id))

def evaluar_modelo(X_mat, var_target, tipo_modelo='lgb'):
    y = base_modelo_ml[var_target]
    
    X_dev, y_dev, w_dev, hogar_dev = X_mat.iloc[dev_idx], y.iloc[dev_idx], pesos.iloc[dev_idx], hogar_id.iloc[dev_idx]
    X_te, y_te, w_te = X_mat.iloc[test_idx], y.iloc[test_idx], pesos.iloc[test_idx]
    
    gkf = GroupKFold(n_splits=10)
    r2_folds, rmse_folds, mae_folds, tau_folds = [], [], [], []
    
    for train_idx, val_idx in gkf.split(X_dev, y_dev, groups=hogar_dev):
        X_tr, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_tr, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
        w_tr, w_val = w_dev.iloc[train_idx], w_dev.iloc[val_idx]
        
        if tipo_modelo == 'lgb':
            model = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=4)
            model.fit(X_tr, y_tr, sample_weight=w_tr)
            y_pred_val = model.predict(X_val)
        else:
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)
            model = RidgeCV(alphas=np.logspace(-3, 3, 50))
            model.fit(X_tr_sc, y_tr, sample_weight=w_tr)
            y_pred_val = model.predict(X_val_sc)
            
        r2_folds.append(r2_score(y_val, y_pred_val, sample_weight=w_val))
        rmse_folds.append(np.sqrt(mean_squared_error(y_val, y_pred_val, sample_weight=w_val)))
        mae_folds.append(mean_absolute_error(y_val, y_pred_val, sample_weight=w_val))
        t_val, _ = kendalltau(y_val, y_pred_val)
        tau_folds.append(t_val)
        
    # Ajuste en Dev (90%) y predicción en Test (10%)
    if tipo_modelo == 'lgb':
        model_final = lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, n_jobs=4)
        model_final.fit(X_dev, y_dev, sample_weight=w_dev)
        y_pred_te = model_final.predict(X_te)
    else:
        scaler_final = StandardScaler()
        X_dev_sc = scaler_final.fit_transform(X_dev)
        X_te_sc = scaler_final.transform(X_te)
        model_final = RidgeCV(alphas=np.logspace(-3, 3, 50))
        model_final.fit(X_dev_sc, y_dev, sample_weight=w_dev)
        y_pred_te = model_final.predict(X_te_sc)
        
    r2_te = r2_score(y_te, y_pred_te, sample_weight=w_te)
    rmse_te = np.sqrt(mean_squared_error(y_te, y_pred_te, sample_weight=w_te))
    mae_te = mean_absolute_error(y_te, y_pred_te, sample_weight=w_te)
    tau_te, _ = kendalltau(y_te, y_pred_te)
    
    res_resumen = {
        'R2 (10-Folds Val)': f"{np.mean(r2_folds):.3f} \\pm {np.std(r2_folds):.3f}",
        'R2 (Test 10% Holdout)': f"{r2_te:.3f}",
        'Kendall Tau (Test)': f"{tau_te:.3f}",
        'RMSE (Test)': f"{rmse_te:.3f}",
        'MAE (Test)': f"{mae_te:.3f}",
        'p_dimension': X_mat.shape[1]
    }
    
    return res_resumen, r2_folds, y_pred_te

# Definición de modelos a evaluar
modelos = [
    ('01. Estructurado Clásico (LightGBM)', X_trad, 'lgb'),
    ('02. Estructurado Clásico (Ridge)', X_trad, 'ridge'),
    ('03. Estructurado Saturado Comparable (LightGBM)', X_sat_comparable, 'lgb'),
    ('04. Estructurado Saturado Comparable (Ridge)', X_sat_comparable, 'ridge'),
    ('05. Estructurado Saturado Completo (LightGBM)', X_trad_ampliada, 'lgb'),
    ('06. Estructurado Saturado Completo (Ridge)', X_trad_ampliada, 'ridge'),
    ('07. Solo Qwen (LightGBM)', qwen_mat, 'lgb'),
    ('08. Solo Qwen (Ridge)', qwen_mat, 'ridge'),
    ('09. Solo OpenAI (LightGBM)', openai_df, 'lgb'),
    ('10. Solo OpenAI (Ridge)', openai_df, 'ridge'),
    ('11. Híbrido Qwen (LightGBM)', X_hib_qwen, 'lgb'),
    ('12. Híbrido Qwen (Ridge)', X_hib_qwen, 'ridge'),
    ('13. Híbrido OpenAI (LightGBM)', X_hib_openai, 'lgb'),
    ('14. Híbrido OpenAI (Ridge)', X_hib_openai, 'ridge'),
    ('15. Saturado Completo + OpenAI (LightGBM)', X_sat_plus_openai, 'lgb'),
    ('16. Saturado Completo + OpenAI (Ridge)', X_sat_plus_openai, 'ridge')
]

out_excel = os.path.join(RUTA_DATOS, 'resultados_group_kfold_y_holdout.xlsx')
res_eval, res_exp = [], []
folds_eval_dict, folds_exp_dict = {}, {}
preds_test_eval, preds_test_exp = {}, {}

# Submuestra test para OE5 (desglose)
df_test = base_trabajo.iloc[test_idx].copy()
df_test['FEX_C_test'] = pesos.iloc[test_idx]

print('\n=== INICIANDO EVALUACIÓN CON 10-FOLDS GROUP KFOLD + TEST HOLDOUT 10% ===')
for nombre, X_m, m_type in modelos:
    print(f'--> Evaluando: {nombre} | Evaluativo...')
    res_m, r2_f, y_pr = evaluar_modelo(X_m, var_eval, tipo_modelo=m_type)
    res_m['Modelo'] = nombre
    res_eval.append(res_m)
    folds_eval_dict[nombre] = r2_f
    preds_test_eval[nombre] = y_pr

    print(f'--> Evaluando: {nombre} | Experiencial (Afectivo Puro)...')
    res_m_exp, r2_f_exp, y_pr_exp = evaluar_modelo(X_m, var_exp_puro, tipo_modelo=m_type)
    res_m_exp['Modelo'] = nombre
    res_exp.append(res_m_exp)
    folds_exp_dict[nombre] = r2_f_exp
    preds_test_exp[nombre] = y_pr_exp

# Cálculo de diferencias pareadas pliegue a pliegue (Fold-by-Fold Paired Differences)
def calcular_diferencias_pareadas(folds_dict, target_label):
    pares = [
        ('Híbrido OpenAI (LGBM) vs Clásico (LGBM)', '13. Híbrido OpenAI (LightGBM)', '01. Estructurado Clásico (LightGBM)'),
        ('Híbrido OpenAI (Ridge) vs Clásico (Ridge)', '14. Híbrido OpenAI (Ridge)', '02. Estructurado Clásico (Ridge)'),
        ('Híbrido Qwen (LGBM) vs Clásico (LGBM)', '11. Híbrido Qwen (LightGBM)', '01. Estructurado Clásico (LightGBM)'),
        ('Híbrido Qwen (Ridge) vs Clásico (Ridge)', '12. Híbrido Qwen (Ridge)', '02. Estructurado Clásico (Ridge)'),
        ('Saturado Comp (LGBM) vs Híbrido OpenAI (LGBM)', '03. Estructurado Saturado Comparable (LightGBM)', '13. Híbrido OpenAI (LightGBM)'),
        ('Saturado Comp (Ridge) vs Híbrido OpenAI (Ridge)', '04. Estructurado Saturado Comparable (Ridge)', '14. Híbrido OpenAI (Ridge)'),
        ('Saturado+OpenAI (LGBM) vs Saturado Completo (LGBM)', '15. Saturado Completo + OpenAI (LightGBM)', '05. Estructurado Saturado Completo (LightGBM)')
    ]
    diff_records = []
    for comp_name, m_a, m_b in pares:
        if m_a in folds_dict and m_b in folds_dict:
            arr_a = np.array(folds_dict[m_a])
            arr_b = np.array(folds_dict[m_b])
            diff = arr_a - arr_b
            diff_records.append({
                'Comparación': comp_name,
                'Variable': target_label,
                'Delta_Media (R2)': np.mean(diff),
                'Delta_DE (R2)': np.std(diff),
                'IC_95_Inf': np.mean(diff) - 1.96 * np.std(diff) / np.sqrt(len(diff)),
                'IC_95_Sup': np.mean(diff) + 1.96 * np.std(diff) / np.sqrt(len(diff)),
                'P-valor (t pareada)': np.nan if np.std(diff)==0 else (np.mean(diff) / (np.std(diff)/np.sqrt(len(diff))))
            })
    return pd.DataFrame(diff_records)

df_diff_eval = calcular_diferencias_pareadas(folds_eval_dict, 'Evaluativo')
df_diff_exp = calcular_diferencias_pareadas(folds_exp_dict, 'Experiencial Puro')
df_diff_total = pd.concat([df_diff_eval, df_diff_exp], ignore_index=True)

# Desempeño por Subgrupos (Sexo y Edad) en Test Set para OE5
subgrupos_records = []
for var_name, y_true, preds_dict in [('Evaluativo', df_test['satisfaccion_vida'], preds_test_eval),
                                      ('Experiencial', (base_raw.iloc[test_idx]['felicidad_ayer'] + base_raw.iloc[test_idx]['preocupacion_ayer_inv'] + base_raw.iloc[test_idx]['tristeza_ayer_inv'])/3.0, preds_test_exp)]:
    modelos_oe5 = [
        ('Clásico (LGBM)', '01. Estructurado Clásico (LightGBM)'),
        ('Híbrido OpenAI (LGBM)', '13. Híbrido OpenAI (LightGBM)'),
        ('Saturado Comp (LGBM)', '03. Estructurado Saturado Comparable (LightGBM)')
    ]
    for mod_label, mod_key in modelos_oe5:
        if mod_key in preds_dict:
            y_p = preds_dict[mod_key]
            for sexo in df_test['sexo_nacimiento'].unique():
                m_s = (df_test['sexo_nacimiento'] == sexo)
                r2_s = r2_score(y_true[m_s], y_p[m_s], sample_weight=df_test.loc[m_s, 'FEX_C_test'])
                subgrupos_records.append({'Variable': var_name, 'Modelo': mod_label, 'Subgrupo': 'Sexo', 'Categoria': str(sexo), 'R2_Test': r2_s})
            for grupo_e in df_test['grupo_edad'].unique():
                m_e = (df_test['grupo_edad'] == grupo_e)
                r2_e = r2_score(y_true[m_e], y_p[m_e], sample_weight=df_test.loc[m_e, 'FEX_C_test'])
                subgrupos_records.append({'Variable': var_name, 'Modelo': mod_label, 'Subgrupo': 'Grupo_Edad', 'Categoria': str(grupo_e), 'R2_Test': r2_e})

df_oe5 = pd.DataFrame(subgrupos_records)

# Guardar todos los resultados consolidados en Excel
with pd.ExcelWriter(out_excel) as writer:
    pd.DataFrame(res_eval).to_excel(writer, sheet_name='Evaluativo', index=False)
    pd.DataFrame(res_exp).to_excel(writer, sheet_name='Experiencial Puro', index=False)
    df_diff_total.to_excel(writer, sheet_name='Diferencias_Pareadas', index=False)
    pd.DataFrame(folds_eval_dict).to_excel(writer, sheet_name='Folds_Raw_Evaluativo', index=False)
    pd.DataFrame(folds_exp_dict).to_excel(writer, sheet_name='Folds_Raw_Experiencial', index=False)
    df_oe5.to_excel(writer, sheet_name='Desempeño_Subgrupos_OE5', index=False)

print(f'\n¡Evaluación integral finalizada con éxito!')
print(f'Resultados consolidados guardados en: {out_excel}')
