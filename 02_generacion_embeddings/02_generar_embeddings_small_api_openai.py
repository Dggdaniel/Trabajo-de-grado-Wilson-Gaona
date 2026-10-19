import pandas as pd
import numpy as np
import os
from openai import OpenAI

# RUTA RELATIVA Y PORTÁTIL (Sin información sensible del usuario)
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.abspath(os.path.join(RUTA_BASE, '..', '..'))

print('=== 02. GENERACIÓN DE EMBEDDINGS VÍA API DE OPENAI (text-embedding-3-small) ===')

# Se requiere la variable de entorno OPENAI_API_KEY previamente configurada
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    print('ADVERTENCIA: No se detectó la variable de entorno OPENAI_API_KEY.')
    print('Por favor configure la clave mediante: export OPENAI_API_KEY="tu_api_key"')
    client = None
else:
    client = OpenAI(api_key=api_key)

input_file = os.path.join(RUTA_DATOS, 'base_textos_serializados.csv')
if not os.path.exists(input_file):
    print(f'Error: No se encuentra el archivo {input_file}. Ejecute primero el paso 01.')
    exit(1)

df_textos = pd.read_csv(input_file)
textos = df_textos['texto_serializado'].tolist()

print(f'Cargados {len(textos)} textos serializados. Iniciando llamados por lotes a la API...')

batch_size = 500
embeddings_totales = []

for i in range(0, len(textos), batch_size):
    batch = textos[i:i+batch_size]
    if client is not None:
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-small"
        )
        batch_emb = [data.embedding for data in response.data]
        embeddings_totales.extend(batch_emb)
    
    if (i // batch_size) % 10 == 0:
        print(f'  - Procesadas {min(i + batch_size, len(textos))} / {len(textos)} observaciones...')

if len(embeddings_totales) > 0:
    df_out = pd.DataFrame(embeddings_totales)
    output_file = os.path.join(RUTA_DATOS, 'embeding generados con text-embedding-3-small.csv')
    df_out.to_csv(output_file, index=False)
    print(f'¡Embeddings de OpenAI guardados exitosamente en: {output_file}!')
