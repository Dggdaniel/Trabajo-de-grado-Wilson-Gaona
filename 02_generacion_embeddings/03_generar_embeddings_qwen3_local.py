"""
=============================================================================
SCRIPT PARA GENERAR EMBEDDINGS LOCALES CON MODELO QWEN (gte-Qwen / Qwen-Embedding)
=============================================================================
Objetivo:
  Generar los vectores de embedding para las 178,511 observaciones de
  'base_para_embendings.xlsx' (o .csv / .parquet) procesando en lotes (batches),
  con soporte para GPU (CUDA) / CPU, guardado de puntos de control (checkpoints)
  para recuperacion ante interrupciones y exportacion en los mismos formatos
  utilizados para OpenAI text-embedding-3-small (PKL y CSV).

Instrucciones de uso:
  1. Instalar dependencias si no las tienes:
     pip install sentence-transformers torch pandas openpyxl tqdm

  2. Ejecutar el script:
     python generar_embeddings_qwen_local.py
=============================================================================
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

# =============================================================================
# 1. CONFIGURACION Y PARAMETROS
# =============================================================================
import argparse

# Rutas de entrada y salida
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.abspath(os.path.join(RUTA_BASE, '..', '..'))

INPUT_FILE = os.path.join(RUTA_DATOS, "base_textos_serializados.csv")
COL_TEXTO = "texto_serializado"

OUTPUT_PKL = os.path.join(RUTA_DATOS, "embeddings_qwen.pkl")
OUTPUT_CSV = os.path.join(RUTA_DATOS, "embeding generados con qwen.csv")
CHECKPOINT_FILE = os.path.join(RUTA_BASE, "checkpoint_qwen_embeddings.pkl")

# Modelo Qwen local (gte-Qwen2 / Qwen3 / Qwen-Embedding)
# Opciones recomendadas:
#   - "Alibaba-NLP/gte-Qwen2-1.5B-instruct" (SentenceTransformers - muy rapido y ligero)
#   - "Alibaba-NLP/gte-Qwen2-7B-instruct"   (SentenceTransformers - maximo rendimiento)
#   - "Qwen/Qwen3-Embedding-8B" 
DEFAULT_MODEL = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

# Parametros de ejecucion en lotes
DEFAULT_BATCH_SIZE = 64     # Ajustar segun VRAM: 32 (4-6GB), 64 (8-12GB), 128 (16GB+)
SAVE_EVERY_N_BATCHES = 50   # Guardar checkpoint cada N lotes
MAX_LENGTH = 512            # Longitud maxima de tokens por texto
NORMALIZE_EMBEDDINGS = True # Normalizar vectores L2 (estandar para similitud coseno / Ridge)

# =============================================================================
# PROTECCION TERMICA Y WATCHDOG DE HARDWARE (GPU / CPU)
# =============================================================================
MAX_GPU_TEMP = 70.0        # Temperatura maxima permitida para pausar (°C)
COOL_GPU_TEMP = 50.0       # Temperatura objetivo para reanudar la inferencia (°C)
EMERGENCY_GPU_TEMP = 75.0  # Temperatura critica de parada de emergencia inmediata (°C)

import subprocess

def consultar_temperatura_gpu():
    """Consulta la temperatura actual del chip grafico mediante nvidia-smi."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"]
        res = subprocess.check_output(cmd, encoding="utf-8", stderr=subprocess.DEVNULL)
        return float(res.strip().split("\n")[0])
    except Exception:
        return None

def aplicar_guardia_termico(checkpoint_path, embeddings_list, pbar=None):
    """
    Verifica la temperatura de la GPU. Si supera los 70°C:
      1. Guarda checkpoint inmediato en disco.
      2. Pausa el cálculo y espera a que baje a 50°C.
      3. Si supera 75°C, detiene el proceso por seguridad de hardware.
    """
    temp = consultar_temperatura_gpu()
    if temp is None:
        return

    # 1. Parada de Emergencia Absoluta
    if temp >= EMERGENCY_GPU_TEMP:
        msg = f"\n[PARADA DE EMERGENCIA] GPU supero {EMERGENCY_GPU_TEMP} °C ({temp:.1f} °C). Guardando checkpoint y saliendo..."
        if pbar:
            pbar.write(msg)
        else:
            print(msg)
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"embeddings": embeddings_list}, f)
        sys.exit(0)

    # 2. Pausa y Enfriamiento Controlado
    if temp >= MAX_GPU_TEMP:
        alerta = f"\n[ALERTA TERMICA] GPU alcanzo {temp:.1f} °C (Limite: {MAX_GPU_TEMP} °C)."
        aviso = f"[*] Guardando checkpoint y pausando inferencia hasta que la GPU baje a {COOL_GPU_TEMP} °C..."
        if pbar:
            pbar.write(alerta)
            pbar.write(aviso)
        else:
            print(alerta)
            print(aviso)

        # Guardar inmediatamente por seguridad
        with open(checkpoint_path, "wb") as f:
            pickle.dump({"embeddings": embeddings_list}, f)

        # Esperar a que la GPU se enfrie a 50°C
        while True:
            time.sleep(5)
            t_actual = consultar_temperatura_gpu()
            if t_actual is None or t_actual <= COOL_GPU_TEMP:
                resumen = f"[+] GPU enfriada exitosamente a {t_actual:.1f} °C. Reanudando procesamiento..."
                if pbar:
                    pbar.write(resumen)
                else:
                    print(resumen)
                break



# =============================================================================
# 2. CARGA DE DATOS
# =============================================================================
def cargar_textos(ruta_archivo, columna_texto, limit=None):
    """Carga los textos desde Excel, CSV o Parquet con validaciones y limite opcional."""
    print(f"[*] Cargando datos desde: {ruta_archivo}")
    start_time = time.time()
    
    if ruta_archivo.endswith(".xlsx") or ruta_archivo.endswith(".xls"):
        df = pd.read_excel(ruta_archivo, nrows=limit)
    elif ruta_archivo.endswith(".csv"):
        df = pd.read_csv(ruta_archivo, nrows=limit)
    elif ruta_archivo.endswith(".parquet"):
        df = pd.read_parquet(ruta_archivo)
        if limit:
            df = df.head(limit)
    else:
        raise ValueError(f"Formato no soportado: {ruta_archivo}")

    
    if columna_texto not in df.columns:
        # Si la columna no tiene el nombre exacto, buscar columnas con texto
        posibles = [c for c in df.columns if "texto" in c.lower() or "emb" in c.lower()]
        if posibles:
            columna_texto = posibles[0]
            print(f"[!] Columna seleccionada automaticamente: '{columna_texto}'")
        else:
            raise KeyError(f"No se encontro la columna '{columna_texto}' en {df.columns.tolist()}")

    # Rellenar valores nulos o vacios con string vacio para evitar errores
    textos = df[columna_texto].fillna("").astype(str).tolist()
    print(f"[+] {len(textos):,} textos cargados exitosamente en {time.time() - start_time:.2f} s.")
    return df, textos


# =============================================================================
# 3. INICIALIZACION DEL MODELO
# =============================================================================
def inicializar_modelo(model_name, use_8bit=False, use_4bit=False):
    """Carga el modelo SentenceTransformer con deteccion automatica de GPU/CPU y precision optimizada."""
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        print("[ERROR] Requiere 'sentence-transformers' y 'torch'.")
        print("Instalalos con: pip install sentence-transformers torch")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Dispositivo detectado para inferencia: {device.upper()}")
    
    model_kwargs = {}
    if device == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        
        # En GPUs modernas (RTX 5060 Ti / Serie 40/50), bfloat16 reduce memoria a la mitad sin perder precision
        if torch.cuda.is_bf16_supported():
            model_kwargs["torch_dtype"] = torch.bfloat16
            print("    Precision: bfloat16 (activada)")
        else:
            model_kwargs["torch_dtype"] = torch.float16
            print("    Precision: float16 (activada)")

        # Cuantizacion opcional con bitsandbytes para modelos 7B
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4"
                )
                print("    Cuantizacion: 4-bit NF4 activada (consumo ~5 GB VRAM)")
            except ImportError:
                print("    [!] Para cuantizacion 4-bit se requiere: pip install bitsandbytes")
        elif use_8bit:
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                print("    Cuantizacion: 8-bit activada (consumo ~8 GB VRAM)")
            except ImportError:
                print("    [!] Para cuantizacion 8-bit se requiere: pip install bitsandbytes")

    print(f"[*] Cargando modelo local: '{model_name}' ...")
    
    # En transformers modernos, Qwen2 esta soportado nativamente sin requerir remote code
    model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=False,
        model_kwargs=model_kwargs
    )
    
    # Configurar longitud de contexto maxima
    model.max_seq_length = MAX_LENGTH
    dim = model.get_embedding_dimension() if hasattr(model, 'get_embedding_dimension') else model.get_sentence_embedding_dimension()
    print(f"[+] Modelo listo. Dimension de embeddings: {dim}")
    return model, device




# =============================================================================
# 4. GENERACION POR LOTES CON CHECKPOINTS
# =============================================================================
def generar_embeddings_por_lotes(model, textos, batch_size=64, checkpoint_path=CHECKPOINT_FILE):
    """
    Genera embeddings en lotes asegurando:
      - Reanudacion automatica desde checkpoint si el proceso se interrumpe.
      - Guardado periodico cada N lotes.
      - Barra de progreso con tqdm (o fallback basico).
    """
    total_textos = len(textos)
    embeddings_list = []
    start_idx = 0

    # Verificar si existe checkpoint previo
    if os.path.exists(checkpoint_path):
        print(f"[!] Se encontro un checkpoint previo en '{checkpoint_path}'. Intentando reanudar...")
        try:
            with open(checkpoint_path, "rb") as f:
                chk = pickle.load(f)
                embeddings_list = chk.get("embeddings", [])
                start_idx = len(embeddings_list)
                print(f"[+] Reanudando desde el indice {start_idx:,} de {total_textos:,} ({start_idx/total_textos*100:.1f}%)")
        except Exception as e:
            print(f"[!] Error al leer checkpoint ({e}). Se iniciara desde cero.")
            embeddings_list = []
            start_idx = 0

    if start_idx >= total_textos:
        print("[+] Todos los embeddings ya estaban generados en el checkpoint.")
        return embeddings_list

    # Intentar usar tqdm para barra de progreso
    try:
        from tqdm import tqdm
        has_tqdm = True
    except ImportError:
        has_tqdm = False

    num_batches = (total_textos - start_idx + batch_size - 1) // batch_size
    batch_indices = range(start_idx, total_textos, batch_size)

    if has_tqdm:
        pbar = tqdm(total=total_textos, initial=start_idx, desc="Generando embeddings", unit="textos")
    else:
        pbar = None
        print(f"[*] Procesando {num_batches} lotes de tamano {batch_size}...")

    t0 = time.time()
    batches_since_save = 0

    for i in batch_indices:
        lote_textos = textos[i : i + batch_size]
        
        # Generar embeddings para el lote actual
        lote_emb = model.encode(
            lote_textos,
            batch_size=len(lote_textos),
            show_progress_bar=False,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True
        )

        embeddings_list.extend(lote_emb.tolist())
        batches_since_save += 1

        if pbar:
            pbar.update(len(lote_textos))
        else:
            current_count = len(embeddings_list)
            if batches_since_save % 10 == 0 or current_count == total_textos:
                elapsed = time.time() - t0
                rate = (current_count - start_idx) / (elapsed + 1e-6)
                print(f"    Progreso: {current_count:,}/{total_textos:,} ({current_count/total_textos*100:.1f}%) - {rate:.1f} textos/s")

        # Guardar checkpoint periodico
        if batches_since_save >= SAVE_EVERY_N_BATCHES:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"embeddings": embeddings_list}, f)
            batches_since_save = 0

        # Guardia termico activo en cada lote
        aplicar_guardia_termico(checkpoint_path, embeddings_list, pbar)


    if pbar:
        pbar.close()

    total_time = time.time() - t0
    print(f"\n[+] Proceso completado en {total_time:.2f} s ({len(embeddings_list)/(total_time+1e-6):.1f} textos/s).")
    return embeddings_list


# =============================================================================
# 5. GUARDADO EN FORMATOS OFICIALES (PKL Y CSV)
# =============================================================================
def guardar_resultados(embeddings_list, output_pkl=OUTPUT_PKL, output_csv=OUTPUT_CSV):
    """
    Guarda los embeddings exactamente en los formatos usados para OpenAI Small:
      - Archivo .pkl: Lista de listas / Matriz de numpy serializada con pickle.
      - Archivo .csv: Columnas indexadas '0', '1', ..., 'D-1' con 'Unnamed: 0' como indice.
    """
    print(f"\n[*] Guardando embeddings...")
    matriz_emb = np.array(embeddings_list, dtype=np.float32)
    n_filas, n_dim = matriz_emb.shape
    print(f"    Forma final de la matriz: {n_filas:,} filas x {n_dim} dimensiones")

    # 1. Guardar archivo .pkl
    print(f"[*] Guardando en '{output_pkl}' ...")
    with open(output_pkl, "wb") as f:
        pickle.dump(embeddings_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[+] Archivo PKL guardado exitosamente.")

    # 2. Guardar archivo .csv (formato identico a embeding generados con text-embedding-3-small.csv)
    print(f"[*] Generando DataFrame y exportando a '{output_csv}' ...")
    df_out = pd.DataFrame(matriz_emb)
    df_out.columns = [str(i) for i in range(n_dim)]
    df_out.to_csv(output_csv, index=True) # index=True genera 'Unnamed: 0' como en el archivo original
    print(f"[+] Archivo CSV guardado exitosamente.")

    # Opcional: Eliminar checkpoint una vez completado exitosamente
    if os.path.exists(CHECKPOINT_FILE):
        try:
            os.remove(CHECKPOINT_FILE)
            print(f"[+] Checkpoint temporal eliminado.")
        except Exception:
            pass

    print("\n=============================================================================")
    print(f"[EXITO] Embeddings Qwen generados y guardados en:")
    print(f"  - PKL: {os.path.abspath(output_pkl)}")
    print(f"  - CSV: {os.path.abspath(output_csv)}")
    print("=============================================================================")


# =============================================================================
# 6. BLOQUE PRINCIPAL
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generar embeddings locales con Qwen en lotes.")
    parser.add_argument("--input", type=str, default=INPUT_FILE, help=f"Ruta al archivo de entrada (default: {INPUT_FILE})")
    parser.add_argument("--col-text", type=str, default=COL_TEXTO, help=f"Columna de texto (default: {COL_TEXTO})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"Nombre del modelo (default: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Tamano del lote (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--output-pkl", type=str, default=OUTPUT_PKL, help=f"Archivo PKL de salida (default: {OUTPUT_PKL})")
    parser.add_argument("--output-csv", type=str, default=OUTPUT_CSV, help=f"Archivo CSV de salida (default: {OUTPUT_CSV})")
    parser.add_argument("--8bit", action="store_true", dest="use_8bit", help="Cargar en 8-bit (reduce VRAM para 7B a ~8 GB)")
    parser.add_argument("--4bit", action="store_true", dest="use_4bit", help="Cargar en 4-bit (reduce VRAM para 7B a ~5 GB)")
    parser.add_argument("--limit", type=int, default=None, help="Limitar el numero de observaciones para pruebas (ej. --limit 10)")
    args = parser.parse_args()


    print("=" * 75)
    print("   GENERADOR DE EMBEDDINGS LOCALES - MODELO QWEN (POR LOTES)")
    print("=" * 75)
    print(f"[*] Modelo seleccionado: {args.model}")
    print(f"[*] Tamano de lote: {args.batch_size}")
    print(f"[*] Archivo de entrada: {args.input}")
    if args.limit:
        print(f"[*] Limite de observaciones (Modo Prueba): {args.limit}")
    
    # 1. Cargar datos
    _, textos = cargar_textos(args.input, args.col_text, limit=args.limit)

    
    # 2. Cargar modelo local
    model, _ = inicializar_modelo(args.model, use_8bit=args.use_8bit, use_4bit=args.use_4bit)

    
    # 3. Procesar en lotes
    embeddings = generar_embeddings_por_lotes(
        model=model,
        textos=textos,
        batch_size=args.batch_size,
        checkpoint_path=CHECKPOINT_FILE
    )
    
    # 4. Guardar resultados
    guardar_resultados(embeddings, args.output_pkl, args.output_csv)


if __name__ == "__main__":
    main()

