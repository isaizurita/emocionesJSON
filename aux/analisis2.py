import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

# Ruta de los JSON
carpeta_json = "/Users/isaizurita/Downloads/JSONS"

# Búsqueda de los JSON
archivos_json = glob.glob(os.path.join(carpeta_json, "*.json")) + glob.glob(os.path.join(carpeta_json, "*.JSON"))
print(f"\nSe encontraron {len(archivos_json)} archivos JSON.")

# DataFrame general para juntar todo
df_general = pd.DataFrame()

# Carpeta de gráficos
output_folder = os.path.join(carpeta_json, "graficas2.1")
os.makedirs(output_folder, exist_ok=True)

# Contador global para IDs
global_id = 0

# Procesar cada archivo
for archivo in archivos_json:
    try:
        with open(archivo, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠️ Archivo {archivo} NO es un JSON válido. Error: {e}")
        continue  # Salta archivo inválido
    
    # Extraer datos
    initials = data.get("initials", "unknown")
    solution_emotion = data["Solution_Experiment_Simulated"]["emotion_value"]
    history = data["History_Experiment_Simulated"][0]

    df = pd.DataFrame(history)

    # Convertir sliders a numéricos
    df['sliderDissatisfiedSatisfied'] = pd.to_numeric(df['sliderDissatisfiedSatisfied'])
    df['sliderBoredExcited'] = pd.to_numeric(df['sliderBoredExcited'])

    # Agregar columnas necesarias
    df['initials'] = initials
    df['target_valence'] = solution_emotion[0]
    df['target_arousal'] = solution_emotion[1]
    df['archivo'] = os.path.basename(archivo)

    # Reasignar ronda por usuario
    df['ronda'] = range(len(df))

    # Agregar ID global
    df['global_id'] = range(global_id, global_id + len(df))
    global_id += len(df)

    # Agregar al DataFrame general
    df_general = pd.concat([df_general, df], ignore_index=True)

    # Gráfica individual
    plt.figure(figsize=(8, 4))
    plt.plot(df['ronda'], df['sliderDissatisfiedSatisfied'], marker='o', label='Dissatisfied-Satisfied')
    plt.plot(df['ronda'], df['sliderBoredExcited'], marker='o', label='Bored-Excited')
    plt.hlines(solution_emotion[0], xmin=0, xmax=df['ronda'].max(), colors='blue', linestyles='dashed', label='Target Valence')
    plt.hlines(solution_emotion[1], xmin=0, xmax=df['ronda'].max(), colors='orange', linestyles='dashed', label='Target Arousal')
    plt.title(f"Evolución sliders - Usuario {initials}")
    plt.xlabel("Iteración / Ronda")
    plt.ylabel("Valor Slider")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{initials}_sliders.png"))
    plt.close()

# Exportar a CSV
csv_path = os.path.join(carpeta_json, "historial_con_id_y_ronda.csv")
df_general.to_csv(csv_path, index=False)

# Exportar también como JSON (lista de diccionarios)
json_output_path = os.path.join(carpeta_json, "historial_con_id_y_ronda.json")
df_general.to_json(json_output_path, orient="records", indent=2)

print(f"\n✅ Análisis y conversión completados.")
print(f"📄 Historial guardado como: {csv_path}")
print(f"🖼️ Gráficas guardadas en: {output_folder}")
