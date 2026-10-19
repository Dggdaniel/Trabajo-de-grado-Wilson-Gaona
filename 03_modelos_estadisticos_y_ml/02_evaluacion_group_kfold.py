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

print('=== PASO 1: CARGANDO MICRODATOS Y LLAVES DE HOGAR (ECV 2024) ===')
base_trabajo = pd.read_excel(os.path.join(RUTA_DATOS, 'validacion_final.xlsx'))
base_raw = pd.read_csv(os.path.join(RUTA_DATOS, 'base_final_embedings.csv'))
hogar_id = base_raw['DIRECTORIO'].astype(str) + '_' + base_raw['SECUENCIA_P'].astype(str)

print('\n=== PASO 2: CARGANDO VECTORES LATENTES (OPENAI VS. QWEN3 d=1536) ===')
openai_path = os.path.join(RUTA_DATOS, 'embeding generados con text-embedding-3-small.csv')
openai_df = pd.read_csv(openai_path).drop(columns=['Unnamed: 0'], errors='ignore')
openai_df.columns = [f'openai_{c}' for c in openai_df.columns]
openai_df.index = base_trabajo.index

qwen_path = os.path.join(RUTA_DATOS, 'embeding generados con qwen.csv')
qwen_df = pd.read_csv(qwen_path).drop(columns=['Unnamed: 0'], errors='ignore')

if 'embedding' in qwen_df.columns:
    qwen_df['embedding'] = qwen_df['embedding'].apply(lambda x: np.fromstring(x.replace('\n', ' ').strip('[]'), sep=' ') if isinstance(x, str) else x)
    qwen_mat = pd.DataFrame(qwen_df['embedding'].tolist())
else:
    qwen_mat = qwen_df

qwen_mat.columns = [f'qwen_{c}' for c in qwen_mat.columns]
qwen_mat.index = base_trabajo.index

print('\n=== PASO 3: CONSTRUYENDO MATRIZ DE VARIABLES ESTRUCTURADAS TRADICIONALES (p=25) ===')
variables_tradicionales = [
    'sexo_nacimiento', 'FEX_C', 'nivel_educativo_alcanzado',
    'ingreso_total_log', 'estado_conyugal', 'satisfaccion_trabajo',
    'satisfaccion_salud', 'P1_DEPARTAMENTO', 'grupo_edad'
]

var_eval = 'satisfaccion_vida'
var_exp_puro = 'afecto_puro'

base_modelo_ml = base_trabajo[variables_tradicionales + [var_eval]].copy()
base_modelo_ml['sexo_nacimiento'] = base_modelo_ml['sexo_nacimiento'].astype(str)
base_modelo_ml['nivel_educativo_alcanzado'] = base_modelo_ml['nivel_educativo_alcanzado'].astype(str)

# Construcción del índice de bienestar experiencial (afectivo puro: emociones del día anterior)
base_modelo_ml[var_exp_puro] = (
    base_raw['felicidad_ayer'] + base_raw['preocupacion_ayer_inv'] + base_raw['tristeza_ayer_inv']
) / 3.0

categoricas = ['sexo_nacimiento', 'nivel_educativo_alcanzado', 'estado_conyugal', 'satisfaccion_trabajo', 'satisfaccion_salud', 'grupo_edad', 'P1_DEPARTAMENTO']
splines = dmatrix('bs(ingreso_total_log, df=4, degree=3, include_intercept=False) - 1', {'ingreso_total_log': base_modelo_ml['ingreso_total_log']}, return_type='dataframe')
X_dummies = pd.get_dummies(base_modelo_ml[categoricas], drop_first=True)
X_trad = pd.concat([X_dummies, splines], axis=1)
clean_cols = [re.sub(r'[^\w]', '_', str(c)) for c in X_trad.columns]
X_trad.columns = clean_cols

X_hib_openai = pd.concat([X_trad, openai_df], axis=1)
X_hib_qwen = pd.concat([X_trad, qwen_mat], axis=1)

print('\n=== PASO 3.1: CONSTRUYENDO MATRIZ ESTRUCTURADA SATURADA (p=523, SIN FUGA DE DATOS) ===')
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
cols_keep = [c for c in base_raw.columns if c not in cols_to_drop]
X_raw = base_raw[cols_keep].copy()

cat_cols, num_cols = [], []
for c in X_raw.columns:
    if X_raw[c].dtype == 'object' or X_raw[c].nunique() < 15:
        cat_cols.append(c)
    else:
        num_cols.append(c)

for c in num_cols:
    X_raw[c] = pd.to_numeric(X_raw[c], errors='coerce')

for c in cat_cols:
    X_raw[c] = X_raw[c].fillna('No aplica').astype(str)

X_dummies_sat = pd.get_dummies(X_raw[cat_cols], drop_first=True)
X_trad_ampliada = pd.concat([X_raw[num_cols], X_dummies_sat], axis=1)
X_trad_ampliada.columns = [re.sub(r'[^\w]', '_', str(c)) for c in X_trad_ampliada.columns]

pesos = base_modelo_ml['FEX_C']

def evaluar_modelo_10_folds_con_holdout(X_mat, var_target, tipo_modelo='lgb'):
    y = base_modelo_ml[var_target]
    
    # 1. Reservar 10% de Test Holdout agrupado por hogar
    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    dev_idx, test_idx = next(gss.split(X_mat, y, groups=hogar_id))
    
    X_dev, y_dev, w_dev, hogar_dev = X_mat.iloc[dev_idx], y.iloc[dev_idx], pesos.iloc[dev_idx], hogar_id.iloc[dev_idx]
    X_te, y_te, w_te = X_mat.iloc[test_idx], y.iloc[test_idx], pesos.iloc[test_idx]
    
    # 2. Ejecutar 10-Fold GroupKFold sobre el 90% de desarrollo
    gkf = GroupKFold(n_splits=10)
    r2_val_l, rmse_val_l, mae_val_l, tau_val_l = [], [], [], []
    
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
            
        r2_val_l.append(r2_score(y_val, y_pred_val, sample_weight=w_val))
        rmse_val_l.append(np.sqrt(mean_squared_error(y_val, y_pred_val, sample_weight=w_val)))
        mae_val_l.append(mean_absolute_error(y_val, y_pred_val, sample_weight=w_val))
        
        t_val, _ = kendalltau(y_val, y_pred_val)
        tau_val_l.append(t_val)
        
    # 3. Entrenamiento en el 90% de desarrollo y evaluación en Test Holdout (10%)
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
    
    r2_val_str = f"{np.mean(r2_val_l):.3f} \\pm {np.std(r2_val_l):.3f}"
    rmse_val_str = f"{np.mean(rmse_val_l):.3f} \\pm {np.std(rmse_val_l):.3f}"
    mae_val_str = f"{np.mean(mae_val_l):.3f} \\pm {np.std(mae_val_l):.3f}"
    tau_val_str = f"{np.mean(tau_val_l):.3f} \\pm {np.std(tau_val_l):.3f}"
    
    return r2_val_str, f"{r2_te:.3f}", tau_val_str, rmse_val_str, mae_val_str

modelos = [
    ('01. Tradicional (LightGBM)', X_trad, 'lgb'),
    ('02. Tradicional (Ridge)', X_trad, 'ridge'),
    ('03. Estructurado Saturado (LightGBM)', X_trad_ampliada, 'lgb'),
    ('04. Estructurado Saturado (Ridge)', X_trad_ampliada, 'ridge'),
    ('05. Solo Qwen (LightGBM)', qwen_mat, 'lgb'),
    ('06. Solo Qwen (Ridge)', qwen_mat, 'ridge'),
    ('07. Solo OpenAI (LightGBM)', openai_df, 'lgb'),
    ('08. Solo OpenAI (Ridge)', openai_df, 'ridge'),
    ('09. Híbrido Qwen (LightGBM)', X_hib_qwen, 'lgb'),
    ('10. Híbrido Qwen (Ridge)', X_hib_qwen, 'ridge'),
    ('11. Híbrido OpenAI (LightGBM)', X_hib_openai, 'lgb'),
    ('12. Híbrido OpenAI (Ridge)', X_hib_openai, 'ridge')
]

out_excel = os.path.join(RUTA_DATOS, 'resultados_group_kfold_y_holdout.xlsx')
res_eval, res_exp_puro = [], []

print('\n=== INICIANDO EVALUACIÓN CON 10-FOLDS GROUP KFOLD + TEST HOLDOUT 10% (POR HOGAR) ===')
for nombre, X_m, m_type in modelos:
    print(f'Evaluando: {nombre} | Bienestar Evaluativo...')
    r2_v, r2_te, tau, rmse, mae = evaluar_modelo_10_folds_con_holdout(X_m, var_eval, tipo_modelo=m_type)
    res_eval.append({'Modelo': nombre, 'R2 (10-Folds Val)': r2_v, 'R2 (Test 10% Holdout)': r2_te, 'Kendall Tau (Test)': tau, 'RMSE (Test)': rmse, 'MAE (Test)': mae})

    print(f'Evaluando: {nombre} | Bienestar Experiencial (Afectivo Puro)...')
    r2_v, r2_te, tau, rmse, mae = evaluar_modelo_10_folds_con_holdout(X_m, var_exp_puro, tipo_modelo=m_type)
    res_exp_puro.append({'Modelo': nombre, 'R2 (10-Folds Val)': r2_v, 'R2 (Test 10% Holdout)': r2_te, 'Kendall Tau (Test)': tau, 'RMSE (Test)': rmse, 'MAE (Test)': mae})

    with pd.ExcelWriter(out_excel) as writer:
        pd.DataFrame(res_eval).to_excel(writer, sheet_name='Evaluativo', index=False)
        pd.DataFrame(res_exp_puro).to_excel(writer, sheet_name='Experiencial Puro', index=False)

df_eval_res = pd.DataFrame(res_eval)
df_exp_puro_res = pd.DataFrame(res_exp_puro)

print('\n=================================================================================')
print('CUADRO DE RESULTADOS DEFINITIVOS (10-FOLDS + HOLDOUT): BIENESTAR EVALUATIVO')
print('=================================================================================')
print(df_eval_res.to_string(index=False))

print('\n=================================================================================')
print('CUADRO DE RESULTADOS DEFINITIVOS (10-FOLDS + HOLDOUT): BIENESTAR EXPERIENCIAL')
print('=================================================================================')
print(df_exp_puro_res.to_string(index=False))

print(f'\n¡Evaluación finalizada exitosamente! Resultados guardados en: {out_excel}')
