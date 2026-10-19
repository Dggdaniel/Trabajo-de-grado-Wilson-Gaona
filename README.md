# Tesis de Maestría/Grado: Análisis del Bienestar Subjetivo en Colombia (ECV 2024)

Este repositorio contiene la secuencia completa de código, preprocesamiento, generación de vectores latentes, modelos estadísticos, modelos de Machine Learning y scripts de generación de figuras para garantizar la **reproducibilidad total de la investigación**.

---

## Descarga de Microdatos Oficiales (DANE)

Los datos primarios utilizados en esta investigación provienen de la **Encuesta de Calidad de Vida (ECV) 2024** realizada por el Departamento Administrativo Nacional de Estadística (DANE) de Colombia.

* **Fuente oficial de descarga:** [Portal de Microdatos del DANE - ECV 2024](https://microdatos.dane.gov.co/index.php/catalog/824)
* **Módulos requeridos para la reconstrucción del dataset:**
  1. **Características y composición del hogar** (`Características y composición del hogar.csv`)
  2. **Servicios de la vivienda y tenencia de bienes** (`Datos de la vivienda.csv`)
  3. **Salud** (`Salud.csv`)
  4. **Educación** (`Educación.csv`)
  5. **Fuerza de trabajo** (`Fuerza de trabajo.csv`)

> **Instrucción de ubicación:** Descargue los archivos en formato `.csv` o `.parquet` desde el portal del DANE y colóquelos en el directorio raíz del proyecto antes de ejecutar el pipeline de código.

---

## NOTA ACLARATORIA SOBRE REQUISITOS DE API DE OPENAI (PAGO Y CLAVES)

> **Requisito obligatorio para el script de OpenAI (`02_generar_embeddings_small_api_openai.py`):**
> 
> 1. **Cuenta comercial y API Key:** El modelo `text-embedding-3-small` de OpenAI es un servicio propietario comercial que **requiere una cuenta activa con créditos/saldo de pago habilitado** en la plataforma [OpenAI Platform](https://platform.openai.com/).
> 2. **Configuración de variable de entorno:** El script **NO** contiene ni almacenará claves de API hardcodeadas por motivos de seguridad. Para ejecutar este paso, debe crear su API Key personal y exportarla en su entorno de trabajo antes de correr el código:
>    ```bash
>    # En Linux / macOS:
>    export OPENAI_API_KEY="sk-tu-api-key-aqui"
>    
>    # En Windows (PowerShell):
>    $env:OPENAI_API_KEY="sk-tu-api-key-aqui"
>    ```
> 3. **Alternativa Gratuita Local:** Si no dispone de créditos de OpenAI o desea una ejecución 100% gratuita y privada, utilice directamente el script local **`03_generar_embeddings_qwen3_local.py`**, el cual descarga y ejecuta el modelo abierto `Qwen3-Embedding` en su propia GPU/CPU sin costo alguno ni necesidad de registros externos.

---

## Estructura del Repositorio (`/git/`)

```text
git/
├── 01_procesamiento_datos/
│   ├── 01_consolidacion_y_depuracion_ecv2024.py      # Paso 1: Unificación de módulos ECV 2024 y saltos lógicos
│   └── 02_seleccion_y_depuracion_variables.py       # Paso 2: Selección y filtrado de variables explicativas (188 vars)
│
├── 02_generacion_embeddings/
│   ├── 01_preparacion_datos_serializacion.py        # Paso 1: Serialización de observaciones y exclusión (drop) de variables principales/dependientes (evita fuga de datos)
│   ├── 02_generar_embeddings_small_api_openai.py     # Paso 2: Generación vía API de OpenAI (requiere saldo y API Key)
│   └── 03_generar_embeddings_qwen3_local.py          # Paso 3: Generación local de Qwen3 con PyTorch (Gratuito/Local)
│
├── 03_modelos_estadisticos_y_ml/
│   ├── 01_modelos_estadisticos_referencia.py        # SCRIPT DE ESTADÍSTICA TRADICIONAL: Logit Ordenado & Regresión WLS
│   ├── 02_evaluacion_group_kfold.py                 # SCRIPT DE ML: Evaluación 10-Folds GroupKFold + Holdout 10% (Ridge & LightGBM)
│   └── 04_ejecutar_seleccion.py                     # SCRIPT CON CHECKPOINT: Ejecución con reanudación automática por modelo
│
├── 04_visualizacion_y_figuras/
│   ├── generar_figuras_exactas.py                   # SCRIPT DE FIGURAS: Distribuciones, boxplots por quintil y curvas LOWESS (300 DPI)
│   └── figura_r2_final.py                           # SCRIPT DE FIGURA 7: Comparación de R² (±1 DE) para los 24 modelos (300 DPI)
│
└── documentacion/
    └── mas fuentes para citar.txt                   # Referencias y citas bibliográficas adicionales
```

---

## Guía de Ejecución Rápida (Reproducibilidad)

### 1. Requisitos Previos
Asegúrese de contar con Python 3.10+ e instalar las dependencias necesarias:

```bash
pip install pandas numpy scikit-learn lightgbm statsmodels matplotlib seaborn patsy openpyxl transformers torch openai
```

### 2. Procesamiento de Datos (Fase 1)
```bash
python git/01_procesamiento_datos/01_consolidacion_y_depuracion_ecv2024.py
python git/01_procesamiento_datos/02_seleccion_y_depuracion_variables.py
```

### 3. Generación de Embeddings (Fase 2)
```bash
# Paso 2.1: Serialización (Este script aplica los 'drops' necesarios para evitar fuga de datos)
python git/02_generacion_embeddings/01_preparacion_datos_serializacion.py

# Paso 2.2: OpenAI API (Requiere saldo y variable OPENAI_API_KEY)
python git/02_generacion_embeddings/02_generar_embeddings_small_api_openai.py

# Paso 2.3: Qwen3 Local (100% Gratuito y de ejecución local en GPU/CPU)
python git/02_generacion_embeddings/03_generar_embeddings_qwen3_local.py
```

### 4. Evaluación de Modelos (Fase 3)

#### A. Modelos Estadísticos Tradicionales de Referencia (Logit Ordenado & WLS con errores corregidos por Clúster de Hogar):
```bash
python git/03_modelos_estadisticos_y_ml/01_modelos_estadisticos_referencia.py
```

#### B. Evaluación de Machine Learning (GroupKFold 10-Folds por Hogar + Test Holdout 10% con Ridge & LightGBM):
```bash
python git/03_modelos_estadisticos_y_ml/02_evaluacion_group_kfold.py
```

### 5. Generar todas las Figuras Oficiales de la Tesis (Fase 4 - 300 DPI)
```bash
python git/04_visualizacion_y_figuras/generar_figuras_exactas.py
python git/04_visualizacion_y_figuras/figura_r2_final.py
```
