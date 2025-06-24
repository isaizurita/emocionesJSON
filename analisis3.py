import json
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import itertools

# === Configuración ===
carpeta_json = "/Users/isaizurita/Downloads/JSONS"

# Buscar archivos .json y .JSON
archivos_json = glob.glob(os.path.join(carpeta_json, "*.json")) + glob.glob(os.path.join(carpeta_json, "*.JSON"))
print(f"\n📂 Se encontraron {len(archivos_json)} archivos JSON.")

# Crear carpeta para gráficas
output_folder = os.path.join(carpeta_json, "graficas")
os.makedirs(output_folder, exist_ok=True)

# DataFrame general
df_general = pd.DataFrame()
usuario_id = 0

# Procesar cada archivo JSON
for archivo in archivos_json:
    try:
        with open(archivo, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            print(f"⚠️ Archivo {archivo} no tiene formato de diccionario. Se salta.")
            continue

        usuario = data.get("initials", "unknown")
        rondas = data.get("History_Experiment_Simulated", [])

        for ronda_idx, ronda in enumerate(rondas):
            df_ronda = pd.DataFrame(ronda)

            # Convertir sliders a numéricos
            df_ronda['sliderDissatisfiedSatisfied'] = pd.to_numeric(df_ronda['sliderDissatisfiedSatisfied'], errors='coerce')
            df_ronda['sliderBoredExcited'] = pd.to_numeric(df_ronda['sliderBoredExcited'], errors='coerce')

            # Añadir columnas
            df_ronda['usuario'] = usuario
            df_ronda['ronda'] = ronda_idx
            df_ronda['iteracion'] = range(len(df_ronda))
            df_ronda['global_id'] = usuario_id
            usuario_id += 0  # solo cambia por archivo, no por fila

            # Agregar al general (solo columnas pedidas en CSV)
            df_general = pd.concat([
                df_general,
                df_ronda[['usuario', 'ronda', 'iteracion', 'sliderDissatisfiedSatisfied',
                          'sliderBoredExcited', 'time', 'risk', 'arrival', 'global_id']]
            ], ignore_index=True)

        usuario_id += 1  # avanzar al siguiente usuario solo una vez por archivo

    except json.JSONDecodeError as e:
        print(f"⚠️ Archivo inválido: {archivo} | Error: {e}")
        continue
    except Exception as e:
        print(f"❌ Error inesperado en archivo {archivo}: {e}")
        continue

# === Gráficas por usuario ===
usuarios = df_general['usuario'].unique()
colors = plt.cm.get_cmap('tab10', 10)
markers = itertools.cycle(('o', 's', 'D', '^', 'v', '<', '>', 'x', '*', '+'))

for user in usuarios:
    df_user = df_general[df_general['usuario'] == user]
    plt.figure(figsize=(10, 5))

    for i, ronda in enumerate(sorted(df_user['ronda'].unique())):
        df_r = df_user[df_user['ronda'] == ronda]
        marker = next(markers)
        color = colors(i % 10)  # ciclo de colores

        plt.plot(df_r['iteracion'], df_r['sliderDissatisfiedSatisfied'],
                 label=f'R{ronda} - Valence', color=color, marker=marker)
        plt.plot(df_r['iteracion'], df_r['sliderBoredExcited'],
                 label=f'R{ronda} - Arousal', color=color, marker=marker, linestyle='dashed')

    plt.title(f'Evolución emocional - Usuario {user}')
    plt.xlabel("Iteración")
    plt.ylabel("Valor del Slider")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{user}_rondas.png"))
    plt.close()

# === Exportar datos ===
csv_path = os.path.join(carpeta_json, "rondas_usuarios.csv")
json_path = os.path.join(carpeta_json, "rondas_usuarios.json")
df_general.to_csv(csv_path, index=False)
df_general.drop(columns=['time', 'risk', 'arrival']).to_json(json_path, orient="records", indent=2)

print(f"\n✅ Proceso finalizado.")
print(f"📄 CSV exportado: {csv_path}")
print(f"📄 JSON exportado (sin time/risk/arrival): {json_path}")
print(f"🖼️ Gráficas guardadas en: {output_folder}")
